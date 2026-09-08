"""Session windows and the trading clock - soul file section 3 and 6.7.

Every time in the soul file is IST, including the XAUUSD window, so this module
works exclusively in ``Asia/Kolkata`` and converts on the way in. Mixing naive
and aware timestamps is the classic way a session boundary gets missed by an
hour twice a year, so :meth:`SessionClock.localise` is the only entry point.

The session state machine has four phases:

===============  ==========================================================
``CLOSED``       Outside the window. No entries, no positions.
``GUARD``        Inside the window but within the opening-range guard.
``OPEN``         Normal trading. New entries permitted.
``NO_ENTRY``     Past the last-entry cutoff. Positions run; no new entries.
``FLATTEN``      Past ``flatten_begin``. Managed for exit (6.7 step 2).
``HARD_FLAT``    Past ``hard_flat``. Everything closes at market.
===============  ==========================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from enum import Enum

import pandas as pd

from core.config import Config, get_config


class SessionPhase(str, Enum):
    """Where the clock currently sits inside a market's window."""

    CLOSED = "CLOSED"
    GUARD = "GUARD"
    OPEN = "OPEN"
    NO_ENTRY = "NO_ENTRY"
    FLATTEN = "FLATTEN"
    HARD_FLAT = "HARD_FLAT"

    @property
    def entries_allowed(self) -> bool:
        """Only ``OPEN`` permits a new entry."""
        return self is SessionPhase.OPEN


@dataclass
class SessionWindow:
    """Parsed session times for one market family, all IST."""

    family: str
    open_time: time
    close_time: time
    last_entry: time
    flatten_begin: time
    hard_flat: time
    opening_guard_min: int

    @property
    def guard_end(self) -> time:
        """End of the opening-range guard."""
        base = datetime.combine(date(2000, 1, 1), self.open_time)
        return (base + pd.Timedelta(minutes=self.opening_guard_min)).time()


class SessionClock:
    """Answers "may Beast act right now?" for one market.

    Args:
        market: e.g. ``"NIFTY50"`` or ``"XAUUSD"``.
        config: Injected for tests.
    """

    def __init__(self, market: str, config: Config | None = None) -> None:
        self.cfg = config or get_config()
        self.market = market
        self.family = self.cfg.market_family(market)
        self.timezone = str(self.cfg.get("sessions.timezone"))
        raw = self.cfg.session(market)
        self.window = SessionWindow(
            family=self.family,
            open_time=_parse(raw["open"]),
            close_time=_parse(raw["close"]),
            last_entry=_parse(raw["last_entry"]),
            flatten_begin=_parse(raw["flatten_begin"]),
            hard_flat=_parse(raw["hard_flat"]),
            opening_guard_min=int(raw["opening_guard_min"]),
        )

    # -- time handling -------------------------------------------------------

    def localise(self, moment: datetime) -> datetime:
        """Convert ``moment`` to IST, attaching the zone if it is naive."""
        stamp = pd.Timestamp(moment)
        if stamp.tz is None:
            stamp = stamp.tz_localize(self.timezone)
        else:
            stamp = stamp.tz_convert(self.timezone)
        return stamp.to_pydatetime()

    def session_day(self, moment: datetime) -> date:
        """Which session ``moment`` belongs to.

        Bars before the window open belong to the previous session - this is what
        lets the Gold overnight range (pre-05:00 IST) be attributed correctly.
        """
        local = self.localise(moment)
        if local.time() < self.window.open_time:
            return (local - pd.Timedelta(days=1)).date()
        return local.date()

    # -- phase ---------------------------------------------------------------

    def phase(self, moment: datetime) -> SessionPhase:
        """Return the current session phase (soul file 3, 6.7)."""
        local = self.localise(moment)
        now = local.time()
        window = self.window

        if now < window.open_time or now >= window.close_time:
            return SessionPhase.CLOSED
        if now >= window.hard_flat:
            return SessionPhase.HARD_FLAT
        if now >= window.flatten_begin:
            return SessionPhase.FLATTEN
        if now >= window.last_entry:
            return SessionPhase.NO_ENTRY
        if window.opening_guard_min > 0 and now < window.guard_end:
            return SessionPhase.GUARD
        return SessionPhase.OPEN

    def may_enter(self, moment: datetime, expiry_day_cutoff: time | None = None) -> tuple[bool, str]:
        """Gate G0.

        Args:
            moment: Now.
            expiry_day_cutoff: The earlier expiry-day cutoff from 5.7.4, when the
                instrument is an option expiring today. On expiry day the final
                ninety minutes is a decay race, not a directional edge.

        Returns:
            ``(allowed, reason)``.
        """
        phase = self.phase(moment)
        local = self.localise(moment)

        if phase is SessionPhase.CLOSED:
            return False, f"outside the {self.family} window"
        if phase is SessionPhase.GUARD:
            return False, (
                f"inside the {self.window.opening_guard_min}-minute opening-range guard "
                f"(until {self.window.guard_end.strftime('%H:%M')})"
            )
        if phase is not SessionPhase.OPEN:
            return False, (
                f"past the {self.window.last_entry.strftime('%H:%M')} last-entry cutoff "
                f"(phase {phase.value})"
            )
        if expiry_day_cutoff is not None and local.time() >= expiry_day_cutoff:
            return False, (
                f"past the {expiry_day_cutoff.strftime('%H:%M')} expiry-day entry cutoff"
            )
        return True, f"{self.family} session open"

    def must_flatten(self, moment: datetime) -> bool:
        """True once the hard-flat time has passed (6.7 step 3)."""
        return self.phase(moment) is SessionPhase.HARD_FLAT

    def in_flatten_window(self, moment: datetime) -> bool:
        """True once positions are being managed for exit (6.7 step 2)."""
        return self.phase(moment) in (SessionPhase.FLATTEN, SessionPhase.HARD_FLAT)

    def is_open(self, moment: datetime) -> bool:
        """True whenever the market window is live, entries permitted or not."""
        return self.phase(moment) is not SessionPhase.CLOSED


def _parse(text: str) -> time:
    """Parse ``"HH:MM"`` into a :class:`datetime.time`."""
    hour, minute = (int(part) for part in str(text).split(":"))
    return time(hour=hour, minute=minute)
