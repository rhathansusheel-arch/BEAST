"""Section 7 portfolio limits - daily loss, consecutive losses, concurrency, correlation.

"The moment either the % loss cap OR the 3-consecutive-loss trigger is hit (whichever comes
first), Beast stops trading that market for the remainder of the session. No exceptions, no
'one more trade to win it back.'" The pause is **per-market**: an Indian-session pause does
not stop XAUUSD, and vice versa.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

from beast.constants import Direction, Market


@dataclass
class MarketDayState:
    """One market's state for one session."""

    session_date: date
    realised_pnl: float = 0.0
    consecutive_losses: int = 0
    trades: int = 0
    paused: bool = False
    pause_reason: str = ""
    last_loss_time: Optional[datetime] = None
    blacklisted_levels: set[str] = field(default_factory=set)
    level_attempts: dict[str, int] = field(default_factory=dict)


class RiskState:
    """Session-scoped risk bookkeeping across all markets.

    Holds only what Section 7 and 5.5 require: the daily loss state per market, the
    post-loss cooldown, the per-level attempt counter, and the open-position book used by
    the concurrency and correlation rules.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._days: dict[str, MarketDayState] = {}
        self.open_positions: dict[Market, Direction] = {}

    # -- per-market day state --------------------------------------------------

    def day(self, market: Market, now: datetime) -> MarketDayState:
        key = market.session_key
        state = self._days.get(key)
        if state is None or state.session_date != now.date():
            state = MarketDayState(session_date=now.date())
            self._days[key] = state
        return state

    def record_trade_result(self, market: Market, now: datetime, pnl: float) -> None:
        """Update the daily loss state after a closed trade (Section 7).

        Either trigger - the % cap or three consecutive losses - pauses that market for the
        rest of the session.
        """
        state = self.day(market, now)
        state.realised_pnl += pnl
        state.trades += 1
        if pnl < 0:
            state.consecutive_losses += 1
            state.last_loss_time = now
        else:
            state.consecutive_losses = 0

        capital = self.cfg.get("capital", None)
        cap_pct = self.cfg.daily_loss_cap(market)
        if capital and state.realised_pnl <= -abs(capital * cap_pct):
            state.paused = True
            state.pause_reason = f"daily loss cap {cap_pct:.0%} hit"
        trigger = int(self.cfg.get("risk.consecutive_loss_trigger"))
        if state.consecutive_losses >= trigger:
            state.paused = True
            state.pause_reason = f"{trigger} consecutive losing trades"

    def is_paused(self, market: Market, now: datetime) -> tuple[bool, str]:
        state = self.day(market, now)
        return state.paused, state.pause_reason

    def in_cooldown(self, market: Market, now: datetime) -> tuple[bool, str]:
        """5.5 post-loss cooldown - no new entry on that instrument for 15 minutes.

        Winning changes nothing: "Post-win behaviour: no cooldown. Winning does not change
        the rules either."
        """
        state = self.day(market, now)
        if state.last_loss_time is None:
            return False, ""
        minutes = int(self.cfg.get("entry.post_loss_cooldown_min"))
        until = state.last_loss_time + timedelta(minutes=minutes)
        if now < until:
            return True, f"post-loss cooldown until {until.strftime('%H:%M')}"
        return False, ""

    # -- 5.5 level re-entry cap ------------------------------------------------

    def note_level_attempt(self, market: Market, now: datetime, level_ref: str) -> None:
        state = self.day(market, now)
        state.level_attempts[level_ref] = state.level_attempts.get(level_ref, 0) + 1
        if state.level_attempts[level_ref] >= int(self.cfg.get("entry.level_reentry_cap")):
            state.blacklisted_levels.add(level_ref)

    def level_blacklisted(self, market: Market, now: datetime, level_ref: str) -> bool:
        """After 2 failed attempts at the same level in one session it is blacklisted (5.5)."""
        return level_ref in self.day(market, now).blacklisted_levels

    # -- concurrency and correlation ------------------------------------------

    def can_open(self, market: Market, direction: Direction) -> tuple[bool, str]:
        """Concurrency cap and the Nifty/Sensex correlation rule (Section 7).

        Nifty and Sensex are highly correlated: simultaneous **same-direction** positions
        count as one against the cap, and simultaneous **opposite-direction** positions are
        not permitted at all.
        """
        if market in self.open_positions:
            return False, f"already holding a {market.value} position - Beast never adds to a trade"

        correlated = bool(self.cfg.get("risk.nifty_sensex_correlated"))
        if correlated and market in (Market.NIFTY, Market.SENSEX):
            twin = Market.SENSEX if market is Market.NIFTY else Market.NIFTY
            twin_dir = self.open_positions.get(twin)
            if twin_dir is not None and twin_dir is not direction:
                return False, (
                    f"opposite-direction {twin.value} position is open - "
                    "Nifty/Sensex opposing exposure is not permitted"
                )

        cap = self.cfg.max_concurrent(market)
        if self._effective_count(market.session_key, correlated) >= cap:
            return False, f"max concurrent positions for {market.session_key} ({cap}) reached"
        return True, "within concurrency and correlation limits"

    def _effective_count(self, session_key: str, correlated: bool) -> int:
        """Count open positions, collapsing a correlated Nifty+Sensex pair into one."""
        same_session = [m for m in self.open_positions if m.session_key == session_key]
        count = len(same_session)
        if correlated and Market.NIFTY in same_session and Market.SENSEX in same_session:
            if self.open_positions[Market.NIFTY] is self.open_positions[Market.SENSEX]:
                count -= 1
        return count

    def correlated_risk_budget(self, market: Market, direction: Direction) -> float:
        """Fraction of a single trade's risk still available to a correlated pair.

        Section 7: same-direction Nifty and Sensex positions "count as one position against
        the concurrent cap and their combined risk may not exceed a single trade's risk
        allocation". So the second leg of a correlated pair gets what the first left over.
        """
        if not self.cfg.get("risk.nifty_sensex_correlated") or market is Market.GOLD:
            return 1.0
        twin = Market.SENSEX if market is Market.NIFTY else Market.NIFTY
        if self.open_positions.get(twin) is direction:
            return 0.5
        return 1.0

    def open_position(self, market: Market, direction: Direction) -> None:
        self.open_positions[market] = direction

    def close_position(self, market: Market) -> None:
        self.open_positions.pop(market, None)
