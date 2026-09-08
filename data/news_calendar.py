"""Economic calendar and news blackout - soul file 5.6.

    Major scheduled news events (RBI policy, Fed announcements, major macro data
    releases, geopolitical shocks) - **no new entries** in the window around the
    event. [...] Beast maintains an economic calendar for both markets; if the
    calendar feed is unavailable, it treats the blackout as **active** for the
    known standing windows rather than assuming clear.

The fail-closed behaviour is the important part and is why this module exists at
all rather than being a lookup in the gate. A missing calendar must not read as
"no events today" - that is exactly the day an unmonitored release runs the stop.

The blackout blocks *entries only*. Section 6 exits continue to run regardless
(5.6, final bullet).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pandas as pd
import yaml

from core.config import Config, get_config


@dataclass
class CalendarEvent:
    """One scheduled high-impact release.

    Attributes:
        name: e.g. ``"US CPI"``.
        market: ``"GOLD"``, ``"INDIAN"`` or ``"ALL"``.
        when: Event time in IST.
        impact: ``"high"`` gates trading; anything else is recorded only.
    """

    name: str
    market: str
    when: datetime
    impact: str = "high"

    def applies_to(self, family: str) -> bool:
        """True when this event blacks out ``family``."""
        token = self.market.upper()
        if token == "ALL":
            return True
        if token == "INDIAN":
            return family == "indian"
        if token in ("GOLD", "XAUUSD"):
            return family == "gold"
        return False


class NewsCalendar:
    """Loads scheduled events and answers the G2 blackout question.

    Args:
        config: Injected for tests.

    Attributes:
        loaded: False when the calendar file was missing or unreadable, which
            switches the fail-closed standing windows on.
    """

    def __init__(self, config: Config | None = None) -> None:
        self.cfg = config or get_config()
        self.timezone = str(self.cfg.get("sessions.timezone"))
        self.events: list[CalendarEvent] = []
        self.loaded = False
        self.load_error = ""
        self.reload()

    # -- loading -------------------------------------------------------------

    def reload(self) -> bool:
        """Re-read the calendar file. Returns True when it loaded cleanly."""
        path = Path(str(self.cfg.get("news.calendar_file")))
        if not path.is_absolute():
            path = Path(__file__).resolve().parent.parent / path

        self.events = []
        if not path.exists():
            self.loaded = False
            self.load_error = f"calendar file not found: {path}"
            return False

        try:
            with path.open("r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle) or []
        except Exception as error:  # a malformed calendar is a missing calendar
            self.loaded = False
            self.load_error = f"calendar unreadable: {error}"
            return False

        for entry in raw if isinstance(raw, list) else raw.get("events", []):
            try:
                when = pd.Timestamp(entry["when"])
                if when.tz is None:
                    when = when.tz_localize(self.timezone)
                self.events.append(
                    CalendarEvent(
                        name=str(entry["name"]),
                        market=str(entry.get("market", "ALL")),
                        when=when.tz_convert(self.timezone).to_pydatetime(),
                        impact=str(entry.get("impact", "high")),
                    )
                )
            except (KeyError, ValueError) as error:
                self.load_error = f"skipped malformed calendar entry {entry!r}: {error}"

        self.loaded = True
        return True

    def add_event(self, event: CalendarEvent) -> None:
        """Add an event at runtime - used by tests and by manual operator entry."""
        self.events.append(event)

    # -- the gate ------------------------------------------------------------

    def blackout(self, market: str, moment: datetime) -> tuple[bool, str]:
        """Is ``market`` inside a news blackout at ``moment``?

        Returns:
            ``(blocked, reason)``. When the calendar failed to load and
            ``news.fail_closed`` is true, the standing windows in config are
            applied instead - Beast treats the blackout as active rather than
            assuming clear.
        """
        family = self.cfg.market_family(market)
        blackout = self.cfg.get("entry.news_blackout_min")
        before = int(blackout["before"])
        after = int(blackout["after"])
        local = _to_ist(moment, self.timezone)

        for event in self.events:
            if event.impact.lower() != "high" or not event.applies_to(family):
                continue
            start = event.when - timedelta(minutes=before)
            end = event.when + timedelta(minutes=after)
            if start <= local <= end:
                return True, (
                    f"news blackout: {event.name} at "
                    f"{event.when.strftime('%H:%M')} IST "
                    f"({before}m before / {after}m after)"
                )

        if not self.loaded and bool(self.cfg.get("news.fail_closed")):
            blocked, reason = self._standing_window(family, local, before, after)
            if blocked:
                return True, f"{reason} (calendar unavailable: {self.load_error})"

        return False, "no scheduled high-impact event nearby"

    def _standing_window(self, family: str, moment: datetime, before: int,
                         after: int) -> tuple[bool, str]:
        """Fail-closed fallback using the standing windows in config."""
        for window in self.cfg.get("news.standing_windows", []) or []:
            market = str(window.get("market", "ALL")).upper()
            if market == "INDIAN" and family != "indian":
                continue
            if market in ("GOLD", "XAUUSD") and family != "gold":
                continue

            weekday = window.get("weekday")
            if weekday is not None and moment.weekday() != int(weekday):
                continue

            hour, minute = (int(part) for part in str(window["time_ist"]).split(":"))
            event_time = moment.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if event_time - timedelta(minutes=before) <= moment <= event_time + timedelta(minutes=after):
                return True, f"standing blackout window: {window.get('name', 'unnamed')}"
        return False, ""

    def next_event(self, market: str, moment: datetime) -> CalendarEvent | None:
        """The next high-impact event for ``market`` after ``moment``."""
        family = self.cfg.market_family(market)
        local = _to_ist(moment, self.timezone)
        upcoming = [
            event
            for event in self.events
            if event.when > local and event.impact.lower() == "high"
            and event.applies_to(family)
        ]
        return min(upcoming, key=lambda event: event.when) if upcoming else None

    def event_in_window(self, market: str, moment: datetime,
                        minutes: int) -> CalendarEvent | None:
        """A high-impact event inside the next ``minutes``.

        Used for the ``IV_CRUSH_RISK`` flag (4.7.4): a scheduled event sitting
        inside the trade's expected holding window is what destroys a long
        option's premium while the underlying goes nowhere.
        """
        event = self.next_event(market, moment)
        if event is None:
            return None
        local = _to_ist(moment, self.timezone)
        return event if event.when <= local + timedelta(minutes=minutes) else None


def _to_ist(moment: datetime, timezone: str) -> datetime:
    """Coerce ``moment`` into IST, localising it when naive."""
    stamp = pd.Timestamp(moment)
    if stamp.tz is None:
        stamp = stamp.tz_localize(timezone)
    return stamp.tz_convert(timezone).to_pydatetime()


def write_example_calendar(path: str | Path) -> Path:
    """Write a starter calendar file so the fail-closed path is not the default.

    Returns:
        The path written.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sample = [
        {"name": "RBI Monetary Policy", "market": "INDIAN",
         "when": "2026-10-07 10:00:00", "impact": "high"},
        {"name": "US CPI", "market": "GOLD",
         "when": "2026-10-13 18:00:00", "impact": "high"},
        {"name": "FOMC Rate Decision", "market": "GOLD",
         "when": "2026-10-29 23:30:00", "impact": "high"},
    ]
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(sample, handle, sort_keys=False)
    return target
