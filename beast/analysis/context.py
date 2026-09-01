"""``build_context`` - the analysis layer, assembled once per cycle (4.2 - 4.7).

Sections 5 and 6 do not compute levels themselves; they consume what this produces. That is
the whole point of building the level/context engine once rather than inline per setup.

**The instrument architecture rule (3.1) is enforced here.** Everything in a context is
computed on the underlying index or spot price. The context carries the option chain as
*context and levels only* (4.7); the premium never reaches an indicator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import pandas as pd

from beast.analysis import indicators, integrity, levels, regime
from beast.analysis.option_chain import ChainSnapshot, chain_flags, oi_tag, oi_zones
from beast.constants import Direction, Flag, Market, Tier
from beast.schemas import ChainContext


@dataclass
class MarketFeed:
    """Everything the data layer supplies for one instrument on one cycle.

    ``bias``/``setup``/``trigger`` are raw underlying OHLCV frames indexed by bar open
    time. The forming candle may be present - :func:`build_context` drops it (4.2).
    """

    market: Market
    bias: pd.DataFrame
    setup: pd.DataFrame
    trigger: pd.DataFrame
    chain: Optional[ChainSnapshot] = None
    next_chain: Optional[ChainSnapshot] = None
    spread: Optional[float] = None
    futures_contract: Optional[dict] = None
    atr_median: Optional[float] = None
    session_price_change: Optional[float] = None
    session_oi_change: Optional[float] = None


@dataclass
class Context:
    """The measurement layer for one instrument at one moment."""

    cfg: Any
    market: Market
    now: datetime
    timeframes: dict[str, str]
    bias_ind: pd.DataFrame
    setup_ind: pd.DataFrame
    trigger_ind: pd.DataFrame
    atr: float
    atr_median: Optional[float]
    regime: regime.RegimeState
    swings: list[levels.SwingPoint]
    zones: list[levels.Zone]
    trendlines: list[levels.Trendline]
    order_blocks: list[levels.OrderBlock]
    integrity: integrity.IntegrityState
    chain: Optional[ChainSnapshot] = None
    next_chain: Optional[ChainSnapshot] = None
    chain_context: Optional[ChainContext] = None
    futures_contract: Optional[dict] = None
    spread: Optional[float] = None
    flags: list[str] = field(default_factory=list)
    pending_direction: Optional[Direction] = None

    @property
    def last_price(self) -> float:
        return float(self.trigger_ind["close"].iloc[-1])

    @property
    def expiry_day(self) -> bool:
        return self.chain is not None and self.chain.dte(self.now) == 0

    def tier_a_zones(self) -> list[levels.Zone]:
        return [z for z in self.zones if z.tier is Tier.A]


class LevelState:
    """State the level engine must carry between cycles (4.5).

    Order-block freshness and expiry are lifetime properties - recomputing them from
    scratch each cycle would resurrect blocks that price has already used up.
    """

    def __init__(self) -> None:
        self.order_blocks: dict[str, levels.OrderBlock] = {}

    def merge(self, found: list[levels.OrderBlock]) -> list[levels.OrderBlock]:
        for block in found:
            key = f"{block.direction.value}:{block.created_ts.isoformat()}"
            if key not in self.order_blocks:
                self.order_blocks[key] = block
        return [b for b in self.order_blocks.values() if not b.dead]

    def reset(self) -> None:
        self.order_blocks.clear()


def build_context(
    feed: MarketFeed,
    cfg,
    now: datetime,
    level_state: Optional[LevelState] = None,
    calendar=None,
) -> Context:
    """Assemble the full 4.2 - 4.7 picture for one instrument.

    Closed candles only (4.2). Indicators used for confluence come from the **setup**
    timeframe; the bias frame is used for the regime and Tier A levels, and the trigger
    frame only for entry triggers and invalidation.
    """
    market = feed.market
    tfs = cfg.timeframes(market)
    session_open = cfg.session(market)["open"]

    bias = indicators.closed_only(feed.bias, now, tfs["bias"])
    setup = indicators.closed_only(feed.setup, now, tfs["setup"])
    trigger = indicators.closed_only(feed.trigger, now, tfs["trigger"])

    bias_ind = indicators.compute(bias, cfg, session_open)
    setup_ind = indicators.compute(setup, cfg, session_open)
    trigger_ind = indicators.compute(trigger, cfg, session_open)

    atr_value = float(setup_ind["atr"].iloc[-1]) if not setup_ind.empty else 0.0

    state = integrity.check(
        market,
        trigger,
        setup,
        now,
        cfg,
        spread=feed.spread,
        chain_age_sec=feed.chain.age_seconds(now) if feed.chain else None,
        atr_value=atr_value,
    )

    regime_state = regime.classify(bias_ind, cfg)

    fractal_n = int(cfg.get("levels.fractal_n"))
    lookback = int(cfg.get("levels.swing_lookback"))
    setup_swings = levels.find_swings(setup_ind, fractal_n, lookback)
    bias_swings = levels.find_swings(bias_ind, fractal_n, lookback)

    zones: list[levels.Zone] = []
    for kind in ("high", "low"):
        zones += levels.cluster_zones(bias_swings, kind, atr_value, cfg, Tier.A)
        zones += levels.cluster_zones(setup_swings, kind, atr_value, cfg, Tier.B)
    zones += levels.session_zones(
        setup_ind, atr_value, cfg, session_open, include_overnight=market is Market.GOLD
    )
    zones = levels.update_zones(zones, setup_ind, atr_value, cfg)

    flags = [f.value for f in state.flags]
    chain_ctx: Optional[ChainContext] = None
    converged = False

    if market.is_option_market and feed.chain is not None:
        # 4.7.1 - OI levels join the same Tier A pool, unless the snapshot is stale (4.6).
        if not state.chain_stale:
            zones, converged = levels.merge_convergent(
                zones, oi_zones(feed.chain, cfg), atr_value, cfg
            )
        news_near = (
            calendar.event_inside_window(market, now, 60) if calendar is not None else False
        )
        flags += [f.value for f in chain_flags(feed.chain, cfg, now, news_near)]
        tag = oi_tag(feed.session_price_change or 0.0, feed.session_oi_change or 0.0)
        chain_ctx = ChainContext(
            max_call_oi_strike=getattr(feed.chain.max_call_oi(), "strike", None),
            max_put_oi_strike=getattr(feed.chain.max_put_oi(), "strike", None),
            max_oi_change_strike=getattr(feed.chain.max_oi_change("CE"), "strike", None),
            pcr=feed.chain.pcr(),
            iv_percentile=feed.chain.iv_percentile,
            oi_tag=tag.value if tag else None,
            level_convergence=converged,
        )

    trendlines = levels.find_trendlines(setup_ind, setup_swings, atr_value, cfg)

    blocks = levels.find_order_blocks(
        setup_ind,
        setup_swings,
        atr_value,
        cfg,
        session_open,
        str(cfg.get("levels.ob_expiry")),
    )
    if level_state is not None:
        blocks = level_state.merge(blocks)
    blocks = levels.update_order_blocks(blocks, setup_ind, now, cfg)

    return Context(
        cfg=cfg,
        market=market,
        now=now,
        timeframes=tfs,
        bias_ind=bias_ind,
        setup_ind=setup_ind,
        trigger_ind=trigger_ind,
        atr=atr_value,
        atr_median=feed.atr_median,
        regime=regime_state,
        swings=setup_swings,
        zones=zones,
        trendlines=trendlines,
        order_blocks=blocks,
        integrity=state,
        chain=feed.chain,
        next_chain=feed.next_chain,
        chain_context=chain_ctx,
        futures_contract=feed.futures_contract,
        spread=feed.spread,
        flags=sorted(set(flags)),
    )
