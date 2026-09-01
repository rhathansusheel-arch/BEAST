"""Section 4.6 - the data integrity gate, run before any evaluation cycle.

Each check either suppresses *new entries* or degrades context. None of them touch open
positions: "Existing positions are never affected by a no-trade condition" (5.6) and "High
spread blocks entries, never exits" (6.8).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from beast.analysis.indicators import tf_minutes
from beast.constants import Flag, Market


@dataclass
class IntegrityState:
    """Outcome of the 4.6 checks for one instrument on one cycle."""

    ok: bool
    reasons: list[str] = field(default_factory=list)
    flags: list[Flag] = field(default_factory=list)
    chain_stale: bool = False
    gap_detected: bool = False

    def detail(self) -> str:
        return "; ".join(self.reasons)


def check(
    market: Market,
    trigger_bars: pd.DataFrame,
    setup_bars: pd.DataFrame,
    now: datetime,
    cfg,
    spread: Optional[float] = None,
    chain_age_sec: Optional[float] = None,
    atr_value: Optional[float] = None,
) -> IntegrityState:
    """Run every 4.6 check.

    * **Staleness** - last close older than ``2 x`` the trigger-TF interval suppresses new
      entries until fresh data resumes.
    * **Sensex delay** - tagged on every emission; Sensex is excluded from trigger-TF logic
      tighter than the configured floor.
    * **XAUUSD spread** - above ``gold_spread_max`` emits HIGH SPREAD ALERT and blocks
      entry. An unset ceiling is a *blocker*, not a licence to trade (Open Item 15).
    * **Gap detection** - a session opening more than ``1.0 x ATR`` from the prior close
      forces a level recompute before the first entry.
    * **Option chain freshness** - a snapshot older than ``chain_max_age_sec`` drops
      OI-derived levels from the Tier A pool for that cycle but does not halt trading.
    """
    state = IntegrityState(ok=True)
    tfs = cfg.timeframes(market)

    if trigger_bars.empty or setup_bars.empty:
        state.ok = False
        state.reasons.append("no closed trigger-TF or setup-TF candles yet")
        state.flags.append(Flag.STALE_FEED)
        return state

    interval = timedelta(minutes=tf_minutes(tfs["trigger"]))
    age = now - (trigger_bars.index[-1] + interval)
    if age > interval * int(cfg.get("data.stale_feed_multiplier")):
        state.ok = False
        state.reasons.append(f"feed STALE - last close {age} beyond tolerance")
        state.flags.append(Flag.STALE_FEED)

    if market is Market.SENSEX:
        state.flags.append(Flag.SENSEX_DELAY)
        floor = tf_minutes(str(cfg.get("data.sensex_min_trigger_tf")))
        if tf_minutes(tfs["trigger"]) < floor:
            state.ok = False
            state.reasons.append(
                f"Sensex excluded from trigger TF tighter than {cfg.get('data.sensex_min_trigger_tf')} "
                f"(feed delay {cfg.get('data.sensex_delay_min')} min)"
            )

    if market is Market.GOLD and spread is not None:
        if not cfg.is_set("data.gold_spread_max"):
            state.ok = False
            state.reasons.append(
                "data.gold_spread_max is unset - the high-spread rule cannot be enforced (Open Item 15)"
            )
        elif spread > float(cfg.get("data.gold_spread_max")):
            state.ok = False
            state.reasons.append(f"HIGH SPREAD ALERT - spread {spread} above ceiling")
            state.flags.append(Flag.HIGH_SPREAD)

    if atr_value and not setup_bars.empty:
        gap = _session_gap(setup_bars, cfg, market)
        if gap is not None and abs(gap) > float(cfg.get("data.gap_recompute_atr")) * atr_value:
            state.gap_detected = True
            state.flags.append(Flag.GAP_OPEN)

    if market.is_option_market and chain_age_sec is not None:
        if chain_age_sec > float(cfg.get("options.chain_max_age_sec")):
            state.chain_stale = True
            state.flags.append(Flag.STALE_CHAIN)

    return state


def _session_gap(setup_bars: pd.DataFrame, cfg, market: Market) -> Optional[float]:
    """Distance between this session's open and the prior session's close."""
    from beast.analysis.indicators import session_ids

    sid = session_ids(setup_bars.index, cfg.session(market)["open"])
    sessions = sid.unique()
    if len(sessions) < 2:
        return None
    current = setup_bars[sid == sessions[-1]]
    prior = setup_bars[sid == sessions[-2]]
    if current.empty or prior.empty:
        return None
    return float(current["open"].iloc[0]) - float(prior["close"].iloc[-1])
