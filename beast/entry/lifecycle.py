"""Section 5.5 - signal lifecycle and anti-overtrading.

Four rules, all of them about *not* firing:

* **Validity window** - a detected setup must trigger within 5 setup-TF candles or it
  expires and must be re-detected from scratch.
* **One instance, one signal** - a specific trendline, zone, OB or pullback leg produces at
  most one signal; re-arming needs a fresh detection.
* **Duplicate suppression** - identical signals (same instrument, direction, setup type,
  entry within ``0.25 x ATR``) inside the validity window are suppressed, not re-emitted.
* **One signal per bar** - if two setups qualify on the same trigger-TF close, take the one
  with the tighter structural stop; ties break toward higher confluence, then toward the
  better trailing 30-trade expectancy (5.1).

The post-loss cooldown and the level blacklist also come from 5.5 but live in
``beast.risk.limits`` with the rest of the session state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from beast.constants import Direction, Market


@dataclass
class EmittedSignal:
    """Just enough of a past signal to suppress its duplicate."""

    when: datetime
    market: Market
    direction: Direction
    setup_type: int
    entry_price: float
    setup_ref: str


@dataclass
class Lifecycle:
    """Per-session memory of what has been detected, is still live, and has been emitted.

    Setups live here rather than being re-derived each cycle because 5.5 gives them a
    lifetime: a setup detected on the setup TF stays armed for 5 setup-TF candles while
    Beast waits for its trigger on the trigger TF. Re-detecting from scratch every cycle
    would collapse that window to a single bar and silently discard most valid triggers.
    """

    emitted: list[EmittedSignal] = field(default_factory=list)
    traded_refs: set[str] = field(default_factory=set)
    detected_at: dict[str, datetime] = field(default_factory=dict)
    armed: dict[str, object] = field(default_factory=dict)

    def register(self, setups: list, ctx) -> None:
        """Arm newly detected instances. A ref already armed or traded is not re-armed."""
        for setup in setups:
            if setup.ref in self.traded_refs or setup.ref in self.armed:
                continue
            self.armed[setup.ref] = setup
            self.detected_at[setup.ref] = setup.detected_ts

    def active(self, ctx, cfg) -> list:
        """Armed setups still inside their validity window, freshest first (5.5)."""
        from beast.analysis.indicators import tf_minutes

        minutes = tf_minutes(ctx.timeframes["setup"])
        limit = int(cfg.get("entry.signal_validity_candles"))
        now_bar = ctx.setup_ind.index[-1]
        live = []
        for ref, setup in list(self.armed.items()):
            elapsed = (now_bar - self.detected_at[ref]).total_seconds() / 60.0 / minutes
            if elapsed > limit or ref in self.traded_refs:
                self.armed.pop(ref, None)
                continue
            live.append(setup)
        return sorted(live, key=lambda s: self.detected_at[s.ref], reverse=True)

    def expired(self, ref: str, current_ts, cfg, setup_tf_minutes: int) -> bool:
        first = self.detected_at.get(ref)
        if first is None:
            return False
        elapsed = (current_ts - first).total_seconds() / 60.0 / setup_tf_minutes
        return elapsed > int(cfg.get("entry.signal_validity_candles"))

    def already_traded(self, ref: str) -> bool:
        """One instance, one signal (5.5)."""
        return ref in self.traded_refs

    def is_duplicate(
        self,
        market: Market,
        direction: Direction,
        setup_type: int,
        entry_price: float,
        atr_value: float,
        now: datetime,
        cfg,
        setup_tf_minutes: int,
    ) -> bool:
        """Identical signal inside the validity window (5.5)."""
        tol = float(cfg.get("entry.duplicate_suppression_atr")) * atr_value
        window_min = int(cfg.get("entry.signal_validity_candles")) * setup_tf_minutes
        for past in self.emitted:
            if past.market is not market or past.direction is not direction:
                continue
            if past.setup_type != setup_type:
                continue
            if (now - past.when).total_seconds() / 60.0 > window_min:
                continue
            if abs(past.entry_price - entry_price) <= tol:
                return True
        return False

    def record_emission(self, signal_market: Market, direction: Direction, setup_type: int,
                        entry_price: float, setup_ref: str, now: datetime) -> None:
        self.emitted.append(
            EmittedSignal(now, signal_market, direction, setup_type, entry_price, setup_ref)
        )
        self.traded_refs.add(setup_ref)
        self.armed.pop(setup_ref, None)

    def reset_session(self) -> None:
        self.emitted.clear()
        self.traded_refs.clear()
        self.detected_at.clear()
        self.armed.clear()


def choose_one(candidates: list[tuple], expectancy_lookup=None) -> Optional[tuple]:
    """5.1 "One signal per bar".

    ``candidates`` are ``(setup, plan, confluence)`` tuples that have passed every gate.
    The winner has the **tighter structural stop** - better R per unit risk. Ties break
    toward the higher confluence count, then toward the setup type with the better trailing
    30-trade expectancy from Section 9.
    """
    if not candidates:
        return None

    def key(item):
        setup, plan, confluence = item
        expectancy = 0.0
        if expectancy_lookup is not None:
            expectancy = expectancy_lookup(setup.setup_type) or 0.0
        return (plan.r_points, -confluence.aligned, -expectancy)

    return min(candidates, key=key)
