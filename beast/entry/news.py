"""Section 5.6 - the news blackout, and the rest of the hard no-trade conditions.

Default blackout: no entries **15 minutes before to 15 minutes after** a known high-impact
release. The important half of this rule is the failure mode:

> "if the calendar feed is unavailable, it treats the blackout as **active** for the known
> standing windows rather than assuming clear."

So an unavailable feed fails closed. Missing information is never read as permission.

**Existing positions are never affected.** No-trade blocks *entries* only; Section 6 exits
continue to run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from beast.constants import Market


@dataclass(frozen=True)
class NewsEvent:
    """A scheduled high-impact release (RBI policy, Fed, major macro, geopolitical)."""

    when: datetime
    title: str
    markets: tuple[Market, ...]
    impact: str = "high"


@dataclass
class EconomicCalendar:
    """The economic calendar Beast maintains for both markets (5.6).

    ``available=False`` means the feed could not be reached. Blackout then evaluates as
    active - across ``standing_windows`` if the operator has supplied them, and otherwise
    unconditionally, because Beast has no basis to declare the session clear.
    """

    events: list[NewsEvent] = field(default_factory=list)
    standing_windows: list[tuple[datetime, datetime, str]] = field(default_factory=list)
    available: bool = True

    def in_blackout(self, market: Market, now: datetime, cfg) -> tuple[bool, str]:
        before = int(cfg.get("entry.news_blackout_min.before"))
        after = int(cfg.get("entry.news_blackout_min.after"))

        if not self.available:
            for start, end, label in self.standing_windows:
                if start <= now <= end:
                    return True, f"calendar feed unavailable - standing window {label} treated as active"
            if not self.standing_windows:
                return True, "calendar feed unavailable - blackout treated as active (5.6)"

        for event in self.events:
            if market not in event.markets or event.impact != "high":
                continue
            if event.when - timedelta(minutes=before) <= now <= event.when + timedelta(minutes=after):
                return True, f"news blackout: {event.title} at {event.when:%H:%M}"
        return False, ""

    def next_event(self, market: Market, now: datetime) -> Optional[NewsEvent]:
        upcoming = [e for e in self.events if market in e.markets and e.when >= now]
        return min(upcoming, key=lambda e: e.when) if upcoming else None

    def event_inside_window(self, market: Market, now: datetime, minutes: int) -> bool:
        """4.7.4 ``IV_CRUSH_RISK`` - a scheduled event inside the expected holding window."""
        event = self.next_event(market, now)
        return event is not None and event.when <= now + timedelta(minutes=minutes)
