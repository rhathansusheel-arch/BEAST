"""Section 5.2 - the four setup types.

Each setup specifies four things and this module implements exactly those four: how it is
*detected* on the setup TF, what *triggers* the entry on the trigger TF, where the *stop*
sits, and what *disqualifies* it.

The cascade rule (4.3) is what binds them: a setup is only actionable if it was identified
on the setup TF, is not contradicted by the bias TF, and is triggered on the trigger TF.
A trigger-TF signal with no setup-TF setup behind it is noise and is discarded without
logging - which is why nothing here starts from the trigger frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from beast.analysis.levels import (
    OrderBlock,
    Trendline,
    Zone,
    is_rejection_candle,
    last_swing,
    trendline_broken,
)
from beast.constants import Direction, Regime, SETUP_MODES, Tier, ZoneKind


def _flip_zone(direction: Direction, low: float, high: float, source: str) -> Zone:
    """A transient zone used only as a rejection-candle reference (4.5)."""
    return Zone(
        kind=ZoneKind.SUPPORT if direction is Direction.LONG else ZoneKind.RESISTANCE,
        low=low,
        high=high,
        tier=Tier.B,
        source=source,
    )


@dataclass
class SetupInstance:
    """One detected setup instance - a specific trendline, zone, OB or pullback leg.

    "One instance, one signal" (5.5): each instance can produce at most one signal, and
    re-arming requires a fresh detection.
    """

    setup_type: int
    direction: Direction
    ref: str
    detected_idx: int
    detected_ts: object
    structural_price: float
    stop_source: str
    counter_bias: bool = False
    zone: Optional[Zone] = None
    order_block: Optional[OrderBlock] = None
    trendline: Optional[Trendline] = None
    pullback_ref: Optional[float] = None
    pullback_extreme: Optional[float] = None
    detail: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def mode(self):
        return SETUP_MODES[self.setup_type]

    def expired(self, current_idx: int, cfg) -> bool:
        """Validity window (5.5): the trigger must fire within 5 setup-TF candles."""
        return current_idx - self.detected_idx > int(cfg.get("entry.signal_validity_candles"))

    def stop_price(self, atr_value: float, cfg) -> float:
        """Structural point plus the 0.25 x ATR buffer (5.2, 6.1)."""
        buffer = float(cfg.get("exit.stop_buffer_atr")) * atr_value
        if self.direction is Direction.LONG:
            return self.structural_price - buffer
        return self.structural_price + buffer


@dataclass
class TriggerResult:
    fired: bool
    entry_price: Optional[float] = None
    detail: str = ""


# ---------------------------------------------------------------------------
# Setup 1 - Trendline Breakout/Breakdown
# ---------------------------------------------------------------------------


def detect_setup1(ctx) -> list[SetupInstance]:
    """Detect: a valid trendline (>= 3 anchors, unbroken) and a setup-TF close beyond it
    by >= 0.10 x ATR, with the momentum requirement satisfied.

    Momentum (5.2): the breaking candle's range is >= 1.2 x ATR **or** the next candle
    continues in the break direction. Volatility expansion stands in for volume, which is
    not reliable on index derivatives or spot gold.
    """
    cfg, bars, atr_value = ctx.cfg, ctx.setup_ind, ctx.atr
    out: list[SetupInstance] = []
    idx = len(bars) - 1
    min_touches = int(cfg.get("levels.trendline_min_touches"))
    momentum_atr = float(cfg.get("entry.setup1_momentum_atr"))

    for line in ctx.trendlines:
        if line.retired or len(line.anchors) < min_touches:
            continue

        # The break candle is either this one - which must then carry the range - or the
        # previous one, in which case this candle continuing the break supplies the
        # momentum instead. That is the "or the next candle continues" branch of 5.2.
        direction = trendline_broken(line, bars, idx, atr_value, cfg)
        break_idx = idx
        if direction is not None:
            row = bars.iloc[idx]
            if float(row["high"]) - float(row["low"]) < momentum_atr * atr_value:
                direction = None
        if direction is None and idx >= 1:
            prior = trendline_broken(line, bars, idx - 1, atr_value, cfg)
            if prior is not None:
                closes_on = float(bars["close"].iloc[idx]) - float(bars["close"].iloc[idx - 1])
                if (prior is Direction.LONG and closes_on > 0) or (
                    prior is Direction.SHORT and closes_on < 0
                ):
                    direction, break_idx = prior, idx - 1
        if direction is None:
            continue

        pivot_kind = "low" if direction is Direction.LONG else "high"
        pivot = last_swing(ctx.swings, pivot_kind, before_idx=idx)
        if pivot is None:
            continue
        out.append(
            SetupInstance(
                setup_type=1,
                direction=direction,
                ref=line.ref,
                detected_idx=idx,
                detected_ts=bars.index[idx],
                structural_price=pivot.price,
                stop_source=f"opposing_swing({pivot.price:.2f}) + {cfg.get('exit.stop_buffer_atr')}ATR",
                trendline=line,
                detail=f"{line.kind}-trendline broken with {len(line.anchors)} anchors",
                meta={"break_idx": break_idx, "break_level": line.value_at(break_idx)},
            )
        )
    return out


def _trigger_setup1(setup: SetupInstance, ctx) -> TriggerResult:
    """Trigger modes (5.2), config-selected, default ``break_close``.

    ``break_close`` - enter on the close of the first trigger-TF candle that closes beyond
    the line following the setup-TF break.
    ``retest`` - wait for price to return to the broken line (now flipped S<->R) and print
    a rejection candle there; enter on that candle's close.
    """
    cfg, tbars = ctx.cfg, ctx.trigger_ind
    mode = str(cfg.get("entry.setup1_trigger_mode"))
    level = float(setup.meta["break_level"])
    close = float(tbars["close"].iloc[-1])

    if mode == "break_close":
        beyond = close > level if setup.direction is Direction.LONG else close < level
        if beyond:
            return TriggerResult(True, close, f"trigger-TF close {close:.2f} beyond broken line")
        return TriggerResult(False, detail="trigger-TF candle has not closed beyond the line")

    zone = _flip_zone(
        setup.direction, level - 0.1 * ctx.atr, level + 0.1 * ctx.atr, "trendline_retest"
    )
    idx = len(tbars) - 1
    if is_rejection_candle(tbars, idx, zone, setup.direction, tbars["rsi"], cfg):
        return TriggerResult(True, close, f"rejection candle at retested line {level:.2f}")
    return TriggerResult(False, detail="no rejection candle at the retested line yet")


# ---------------------------------------------------------------------------
# Setup 2 - Reversal at Support/Resistance
# ---------------------------------------------------------------------------


def detect_setup2(ctx) -> list[SetupInstance]:
    """Detect: price trades into a Tier A or Tier B zone and prints a rejection candle on
    the setup TF, closing back inside the zone.

    Disqualifier applied here (5.2): price arrived at the zone in a single impulse candle
    greater than 2.5 x ATR - momentum blowthrough risk, wait for a second test. The level
    re-entry cap and the counter-bias Tier B rule are applied by the pipeline, which is
    where session state and regime live.
    """
    cfg, bars, atr_value = ctx.cfg, ctx.setup_ind, ctx.atr
    idx = len(bars) - 1
    row = bars.iloc[idx]
    blowthrough = float(cfg.get("entry.setup2_blowthrough_atr")) * atr_value
    arrival = float(row["high"]) - float(row["low"])
    out: list[SetupInstance] = []

    for zone in ctx.zones:
        touched = float(row["low"]) <= zone.high and float(row["high"]) >= zone.low
        if not touched:
            continue
        direction = Direction.LONG if zone.kind.value == "SUPPORT" else Direction.SHORT
        if not is_rejection_candle(bars, idx, zone, direction, bars["rsi"], cfg):
            continue
        if arrival > blowthrough:
            continue

        extreme = float(row["low"]) if direction is Direction.LONG else float(row["high"])
        out.append(
            SetupInstance(
                setup_type=2,
                direction=direction,
                ref=zone.ref,
                detected_idx=idx,
                detected_ts=bars.index[idx],
                structural_price=extreme,
                stop_source=f"rejection_wick({extreme:.2f}) + {cfg.get('exit.stop_buffer_atr')}ATR",
                zone=zone,
                detail=(
                    f"rejection at Tier {zone.tier.value} {zone.kind.value.lower()} "
                    f"{zone.center:.2f} (strength {zone.score()})"
                ),
                meta={"rejection_extreme": extreme, "zone_edge": zone.edge(direction)},
            )
        )
    return out


def _trigger_setup2(setup: SetupInstance, ctx) -> TriggerResult:
    """Trigger: on the trigger TF, price closes back above (bullish) / below (bearish) the
    zone edge **with the rejection candle's extreme intact** - the extreme has not been
    exceeded before the trigger closes.
    """
    tbars = ctx.trigger_ind
    edge = float(setup.meta["zone_edge"])
    extreme = float(setup.meta["rejection_extreme"])
    close = float(tbars["close"].iloc[-1])
    since = tbars[tbars.index >= setup.detected_ts]

    if setup.direction is Direction.LONG:
        if not since.empty and float(since["low"].min()) < extreme:
            return TriggerResult(False, detail=f"rejection low {extreme:.2f} was breached")
        if close > edge:
            return TriggerResult(True, close, f"closed back above zone edge {edge:.2f}")
    else:
        if not since.empty and float(since["high"].max()) > extreme:
            return TriggerResult(False, detail=f"rejection high {extreme:.2f} was breached")
        if close < edge:
            return TriggerResult(True, close, f"closed back below zone edge {edge:.2f}")
    return TriggerResult(False, detail="no close back through the zone edge yet")


# ---------------------------------------------------------------------------
# Setup 3 - Order Block Retest
# ---------------------------------------------------------------------------


def detect_setup3(ctx) -> list[SetupInstance]:
    """Detect: a fresh, unexpired order block in the direction of the impulse that created
    it, and price re-entering its zone.

    Disqualifiers (5.2): OB already marked *used*, expired, dead (a setup-TF close beyond
    the far edge), or price entering the zone against the impulse with a structure break in
    between - all of which are carried on the block itself by the 4.5 level engine.
    """
    cfg, bars = ctx.cfg, ctx.setup_ind
    idx = len(bars) - 1
    row = bars.iloc[idx]
    out: list[SetupInstance] = []

    for block in ctx.order_blocks:
        if not block.tradable():
            continue
        re_entered = float(row["low"]) <= block.high and float(row["high"]) >= block.low
        if not re_entered:
            continue
        out.append(
            SetupInstance(
                setup_type=3,
                direction=block.direction,
                ref=block.ref,
                detected_idx=idx,
                detected_ts=bars.index[idx],
                structural_price=block.far_edge,
                stop_source=f"ob_far_edge({block.far_edge:.2f}) + {cfg.get('exit.stop_buffer_atr')}ATR",
                order_block=block,
                detail=f"fresh {block.direction.value.lower()} order block {block.low:.2f}-{block.high:.2f}",
            )
        )
    return out


def _trigger_setup3(setup: SetupInstance, ctx) -> TriggerResult:
    """Trigger: a rejection or continuation candle closes inside the zone in the impulse
    direction on the trigger TF.
    """
    cfg, tbars = ctx.cfg, ctx.trigger_ind
    block = setup.order_block
    idx = len(tbars) - 1
    row = tbars.iloc[idx]
    close, open_ = float(row["close"]), float(row["open"])
    if not (block.low <= close <= block.high):
        return TriggerResult(False, detail="trigger-TF close is not inside the OB zone")

    continuation = (close > open_) if setup.direction is Direction.LONG else (close < open_)
    zone = _flip_zone(setup.direction, block.low, block.high, "order_block")
    rejection = is_rejection_candle(tbars, idx, zone, setup.direction, tbars["rsi"], cfg)
    if continuation or rejection:
        kind = "continuation" if continuation else "rejection"
        return TriggerResult(True, close, f"{kind} candle closed inside the OB zone")
    return TriggerResult(False, detail="no continuation or rejection candle inside the zone")


# ---------------------------------------------------------------------------
# Setup 4 - Indicator Confluence Trend Continuation
# ---------------------------------------------------------------------------


def detect_setup4(ctx) -> list[SetupInstance]:
    """Detect: ADX >= threshold **and rising**, DI aligned with the regime, and price in a
    pullback - a retracement to Session VWAP, the BB middle band, or the most recent
    setup-TF swing, without breaking the prior structural swing in the trend direction.

    Disqualifiers (5.2): the pullback has retraced more than 61.8% of the prior impulse
    leg, ADX is falling, or the trade would be taken against the bias TF. Permitted only in
    TREND_UP / TREND_DOWN (4.4), which the pipeline enforces at G3.
    """
    from beast.entry.confluence import slope_read

    cfg, bars, atr_value = ctx.cfg, ctx.setup_ind, ctx.atr
    if ctx.regime.regime is Regime.RANGE:
        return []
    idx = len(bars) - 1
    row = bars.iloc[idx]
    threshold = float(cfg.get("indicators.adx_trend_threshold"))
    if float(row["adx"]) < threshold or slope_read(bars["adx"], idx, cfg) != "rising":
        return []

    direction = Direction.LONG if ctx.regime.regime is Regime.TREND_UP else Direction.SHORT
    if direction is Direction.LONG and float(row["plus_di"]) <= float(row["minus_di"]):
        return []
    if direction is Direction.SHORT and float(row["minus_di"]) <= float(row["plus_di"]):
        return []

    swing_kind = "low" if direction is Direction.LONG else "high"
    opp_kind = "high" if direction is Direction.LONG else "low"
    leg_start = last_swing(ctx.swings, swing_kind, before_idx=idx)
    leg_end = last_swing(ctx.swings, opp_kind, before_idx=idx)
    if leg_start is None or leg_end is None or leg_end.idx <= leg_start.idx:
        return []

    impulse = abs(leg_end.price - leg_start.price)
    if impulse <= 0:
        return []
    window = bars.iloc[leg_end.idx : idx + 1]
    extreme = float(window["low"].min()) if direction is Direction.LONG else float(window["high"].max())
    retrace = abs(leg_end.price - extreme) / impulse
    if retrace > float(cfg.get("entry.setup4_max_retrace")):
        return []
    if direction is Direction.LONG and extreme < leg_start.price:
        return []  # prior structural swing broken - not a pullback, a reversal
    if direction is Direction.SHORT and extreme > leg_start.price:
        return []

    references = {
        "vwap": float(row["vwap"]),
        "bb_mid": float(row["bb_mid"]),
        "swing": leg_start.price,
    }
    tolerance = float(cfg.get("levels.sr_zone_width_atr")) * atr_value
    touched = [
        name
        for name, level in references.items()
        if np.isfinite(level) and float(row["low"]) - tolerance <= level <= float(row["high"]) + tolerance
    ]
    if not touched:
        return []

    ref_name = touched[0]
    return [
        SetupInstance(
            setup_type=4,
            direction=direction,
            ref=f"pullback:{leg_end.ts.isoformat()}:{ref_name}",
            detected_idx=idx,
            detected_ts=bars.index[idx],
            structural_price=extreme,
            stop_source=f"pullback_extreme({extreme:.2f}) + {cfg.get('exit.stop_buffer_atr')}ATR",
            pullback_ref=references[ref_name],
            pullback_extreme=extreme,
            detail=f"pullback to {ref_name} in {ctx.regime.regime.value}, {retrace:.0%} of impulse",
            meta={"reference": ref_name, "retrace": retrace},
        )
    ]


def _trigger_setup4(setup: SetupInstance, ctx) -> TriggerResult:
    """Trigger: a resumption candle on the trigger TF - a close in the trend direction that
    reclaims the pullback reference level (VWAP / BB mid / swing) it pulled back to.
    """
    tbars = ctx.trigger_ind
    close = float(tbars["close"].iloc[-1])
    level = float(setup.pullback_ref)
    if setup.direction is Direction.LONG and close > level:
        return TriggerResult(True, close, f"reclaimed {setup.meta['reference']} at {level:.2f}")
    if setup.direction is Direction.SHORT and close < level:
        return TriggerResult(True, close, f"reclaimed {setup.meta['reference']} at {level:.2f}")
    return TriggerResult(False, detail=f"{setup.meta['reference']} not reclaimed")


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

DETECTORS = {1: detect_setup1, 2: detect_setup2, 3: detect_setup3, 4: detect_setup4}
_TRIGGERS = {1: _trigger_setup1, 2: _trigger_setup2, 3: _trigger_setup3, 4: _trigger_setup4}


def detect_all(ctx) -> list[SetupInstance]:
    """Every setup instance present on this setup-TF close, freshest first (5.1)."""
    found: list[SetupInstance] = []
    for detector in DETECTORS.values():
        found.extend(detector(ctx))
    return sorted(found, key=lambda s: s.detected_idx, reverse=True)


def trigger_fired(setup: SetupInstance, ctx) -> TriggerResult:
    """Run the setup's own trigger condition on the trigger TF (G6)."""
    return _TRIGGERS[setup.setup_type](setup, ctx)
