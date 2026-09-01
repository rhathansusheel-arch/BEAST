"""Sections 6.3 - 6.10 - managing a live position to its exit.

The rules that shape this module, in the Soul File's own terms:

* Stops, targets and trailing levels evaluate on **live price**, not candle close (4.2
  exception). "A stop is a stop."
* The trailing stop only ever **tightens** (6.3 ratchet). Immutable Rule 7 makes widening
  impossible rather than merely discouraged.
* Stop and target inside the same candle: the **stop** filled first unless tick data proves
  otherwise (6.8). Never assume the favourable fill.
* A gap through the stop records the **actual** R-multiple, worse than -1R, not clamped
  (6.8) - Section 9's expectancy must see real slippage.
* High spread blocks entries, never exits (6.8, 6.10).
* For options the underlying-level stop stays primary; the premium hard stop is a backstop
  that can only cut risk short, never widen it (6.10).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Optional

import pandas as pd

from beast.constants import Direction, ExitReason, Flag, Market
from beast.ops.immutable import ratchet_stop
from beast.schemas import Signal, TradeRecord


class SessionPhase(str):
    NORMAL = "NORMAL"
    NO_NEW_ENTRIES = "NO_NEW_ENTRIES"
    FLATTEN = "FLATTEN"
    HARD_FLAT = "HARD_FLAT"
    CLOSED = "CLOSED"


def session_phase(market: Market, now: datetime, cfg) -> str:
    """Where we are in the Section 3 / 6.7 session sequence.

    ``NO_NEW_ENTRIES`` after the last-entry cutoff, ``FLATTEN`` once the flatten window
    opens (manage for exit, stop honouring the fixed target as a hold condition),
    ``HARD_FLAT`` at the hard-flat time (close at market, reason ``SESSION``).
    """
    s = cfg.session(market)
    t = now.time()

    def hhmm(key: str) -> time:
        return time.fromisoformat(s[key])

    if t < hhmm("open") or t >= hhmm("close"):
        return SessionPhase.CLOSED
    if t >= hhmm("hard_flat"):
        return SessionPhase.HARD_FLAT
    if t >= hhmm("flatten_begin"):
        return SessionPhase.FLATTEN
    if t >= hhmm("last_entry"):
        return SessionPhase.NO_NEW_ENTRIES
    return SessionPhase.NORMAL


def entries_allowed(market: Market, now: datetime, cfg, expiry_day: bool = False) -> tuple[bool, str]:
    """G0 - inside the window, after the opening-range guard, before the last-entry cutoff.

    On option expiry day the cutoff moves earlier (5.7.4: 1:30 PM instead of 3:00 PM) -
    "the final ninety minutes of an expiry session is a decay race, not a directional edge".
    """
    s = cfg.session(market)
    t = now.time()
    open_t = time.fromisoformat(s["open"])
    if t < open_t or t >= time.fromisoformat(s["close"]):
        return False, "outside the trading window"

    guard_min = int(s["opening_guard_min"])
    if guard_min:
        guard_end = (datetime.combine(now.date(), open_t) + pd.Timedelta(minutes=guard_min)).time()
        if t < guard_end:
            return False, f"inside the {guard_min}-minute opening-range guard"

    cutoff = time.fromisoformat(s["last_entry"])
    if expiry_day and market.is_option_market and cfg.get("options.expiry_day.enabled"):
        cutoff = min(cutoff, time.fromisoformat(str(cfg.get("options.expiry_day.last_entry"))))
    if t >= cutoff:
        return False, f"past the last-entry cutoff {cutoff.isoformat(timespec='minutes')}"
    return True, "inside the trading window"


@dataclass
class Position:
    """A live position, managed entirely by the plan computed before entry."""

    signal: Signal
    record: TradeRecord
    direction: Direction
    entry_price: float
    stop_price: float
    target_price: float
    r_points: float
    trail_method: str
    trail_mult: float
    trail_activate_at: float
    entry_time: datetime
    entry_premium: Optional[float] = None
    premium_stop: Optional[float] = None
    trail_activated: bool = False
    bars_held: int = 0
    high_since_entry: float = field(default=float("-inf"))
    low_since_entry: float = field(default=float("inf"))
    mae_r: float = 0.0
    mfe_r: float = 0.0
    mae_premium: Optional[float] = None
    mfe_premium: Optional[float] = None
    flags: list[str] = field(default_factory=list)

    # -- bookkeeping -----------------------------------------------------------

    def r_at(self, price: float) -> float:
        return self.direction.sign * (price - self.entry_price) / self.r_points

    def observe(self, high: float, low: float, premium: Optional[float] = None) -> None:
        """Record excursions in R (6.9) on each trigger-TF close."""
        self.high_since_entry = max(self.high_since_entry, high)
        self.low_since_entry = min(self.low_since_entry, low)
        favourable = high if self.direction is Direction.LONG else low
        adverse = low if self.direction is Direction.LONG else high
        self.mfe_r = max(self.mfe_r, self.r_at(favourable))
        self.mae_r = min(self.mae_r, self.r_at(adverse))
        if premium is not None:
            self.mfe_premium = premium if self.mfe_premium is None else max(self.mfe_premium, premium)
            self.mae_premium = premium if self.mae_premium is None else min(self.mae_premium, premium)
        self.bars_held += 1

    # -- 6.3 trailing stop -----------------------------------------------------

    def update_trail(self, atr_value: float, cfg, recent_swing: Optional[float] = None) -> None:
        """Activate and ratchet the trailing stop (6.3).

        On activation the stop moves to breakeven + estimated costs in one step - "from
        that moment the trade cannot produce a loss". After that it trails by the
        configured method, and a wider computed value is discarded.
        """
        if not self.trail_activated:
            reached = (
                self.high_since_entry >= self.trail_activate_at
                if self.direction is Direction.LONG
                else self.low_since_entry <= self.trail_activate_at
            )
            if not reached:
                return
            self.trail_activated = True
            costs = float(cfg.get("exit.breakeven_cost_points"))
            breakeven = self.entry_price + self.direction.sign * costs
            self.stop_price = ratchet_stop(self.direction, self.stop_price, breakeven)

        if self.trail_method == "structure_trail" and recent_swing is not None:
            buffer = float(cfg.get("exit.stop_buffer_atr")) * atr_value
            candidate = recent_swing - self.direction.sign * buffer
        else:
            offset = self.trail_mult * atr_value
            candidate = (
                self.high_since_entry - offset
                if self.direction is Direction.LONG
                else self.low_since_entry + offset
            )
        self.stop_price = ratchet_stop(self.direction, self.stop_price, candidate)

    def tighten_for_flatten(self, recent_trigger_swing: float, atr_value: float, cfg) -> None:
        """6.7 step 2 - in the flatten window the trail tightens to the most recent
        trigger-TF swing and the fixed target stops being a hold condition.
        """
        buffer = float(cfg.get("exit.stop_buffer_atr")) * atr_value
        candidate = recent_trigger_swing - self.direction.sign * buffer
        self.stop_price = ratchet_stop(self.direction, self.stop_price, candidate)
        self.trail_activated = True

    # -- 6.10 option overlays --------------------------------------------------

    def check_theta_guard(self, now: datetime, cfg) -> bool:
        """6.10 theta guard - flag only, no forced action.

        Held longer than ``theta_guard_minutes`` without ``theta_guard_min_r`` in favour
        gets tagged ``THETA_DRAG``. This is the data that decides whether the 6.5 time stop
        should ever be switched on for options.
        """
        minutes = (now - self.entry_time).total_seconds() / 60.0
        if minutes <= float(cfg.get("options.theta_guard_minutes")):
            return False
        if self.mfe_r >= float(cfg.get("options.theta_guard_min_r")):
            return False
        if Flag.THETA_DRAG.value not in self.flags:
            self.flags.append(Flag.THETA_DRAG.value)
        return True


@dataclass
class ExitDecision:
    reason: ExitReason
    price: float
    detail: str
    premium: Optional[float] = None


def evaluate(
    position: Position,
    high: float,
    low: float,
    close: float,
    now: datetime,
    cfg,
    market: Market,
    premium: Optional[float] = None,
    gap_price: Optional[float] = None,
) -> Optional[ExitDecision]:
    """Resolve this bar's exit, applying the 6.8 priority rules.

    Ordering, in the order the Soul File states it:

    1. **Session hard flat** - nothing survives it (6.7 step 3).
    2. **Premium hard stop** (options) - whichever comes first, underlying or premium
       (6.10). It can only cut the trade short, never widen its risk.
    3. **Stop / trail** - the tighter of the two governs, and if the target was hit in the
       same candle the stop is assumed to have filled first (6.8).
    4. **Target** - a resting level evaluated on live price (6.2).
    5. **Time stop** - only if explicitly enabled (6.5).
    """
    phase = session_phase(market, now, cfg)
    if phase in (SessionPhase.HARD_FLAT, SessionPhase.CLOSED):
        return ExitDecision(ExitReason.SESSION, close, "hard flat - position closed at market", premium)

    if premium is not None and position.premium_stop is not None:
        if premium <= position.premium_stop:
            return ExitDecision(
                ExitReason.PREMIUM_STOP,
                close,
                f"premium {premium:.2f} at or below hard stop {position.premium_stop:.2f} (6.10)",
                premium,
            )

    stop_hit = low <= position.stop_price if position.direction is Direction.LONG else high >= position.stop_price
    target_hit = (
        high >= position.target_price
        if position.direction is Direction.LONG
        else low <= position.target_price
    )

    if stop_hit:
        # 6.8: a gap through the stop exits at the gap price and records the real,
        # worse-than--1R multiple rather than clamping it.
        fill = gap_price if gap_price is not None else position.stop_price
        reason = ExitReason.TRAIL if position.trail_activated else ExitReason.SL
        detail = "trailing stop triggered" if position.trail_activated else "stop-loss hit"
        if target_hit:
            detail += " - stop assumed filled first (6.8), never the favourable fill"
        return ExitDecision(reason, fill, detail, premium)

    if target_hit and phase is not SessionPhase.FLATTEN:
        return ExitDecision(ExitReason.TP, position.target_price, "fixed R:R target hit", premium)

    if cfg.get("exit.time_stop_enabled"):
        bars = int(cfg.get("exit.time_stop_bars"))
        if position.bars_held >= bars and position.mfe_r < 0.5:
            return ExitDecision(
                ExitReason.TIME, close, f"time stop - no +0.5R within {bars} trigger-TF candles", premium
            )
    return None


def r_multiples(position: Position, exit_price: float, exit_premium: Optional[float]) -> dict[str, float]:
    """The R-multiples recorded at exit (6.9).

    For options the **premium-based** R is the R of record - that is the actual money -
    with the underlying R stored alongside it. The gap between the two is the cost of the
    instrument choice, and Section 9 reads it to tell whether strike selection (5.7) is
    leaking edge rather than the analysis being wrong.
    """
    underlying_r = position.r_at(exit_price)
    out = {"underlying_r_multiple": underlying_r, "r_multiple": underlying_r}
    if exit_premium is not None and position.entry_premium:
        risk = position.entry_premium - (position.premium_stop or 0.0)
        premium_r = (exit_premium - position.entry_premium) / risk if risk else 0.0
        out["premium_r_multiple"] = premium_r
        out["r_multiple"] = premium_r
    return out


def hypothetical_r_if_held(position: Position, bars_after: pd.DataFrame) -> Optional[float]:
    """6.9 / Section 8 - what the plan would have produced if it had been left alone.

    This is the field that turns the operator's early-exit habit into visible data.
    """
    if bars_after.empty:
        return None
    for _, row in bars_after.iterrows():
        high, low = float(row["high"]), float(row["low"])
        if position.direction is Direction.LONG:
            if low <= position.stop_price:
                return position.r_at(position.stop_price)
            if high >= position.target_price:
                return position.r_at(position.target_price)
        else:
            if high >= position.stop_price:
                return position.r_at(position.stop_price)
            if low <= position.target_price:
                return position.r_at(position.target_price)
    return position.r_at(float(bars_after["close"].iloc[-1]))
