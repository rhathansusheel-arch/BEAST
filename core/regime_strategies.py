"""Setup detection and triggering - soul file 5.2, under the 4.4 regime gate.

Beast trades four setup types. Each specifies four things: how it is *detected*
on the setup timeframe, what *triggers* the entry on the trigger timeframe, where
the *stop* sits, and what *disqualifies* it. This module implements all four,
plus the regime-driven allocation of which setups may run at all.

The detectors never place an order and never compute risk. They emit
:class:`~core.schemas.SetupInstance` objects carrying a *structural* stop price -
the raw swing, wick or zone edge, with no ATR buffer applied. The buffer,
minimum/maximum distance and R:R come later, in ``core/signal_generator.py``'s
plan builder, because those are exit-logic concerns (section 6) and belong in one
place rather than being duplicated per setup.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from core.config import Config, get_config
from core.confluence import rsi_divergence
from core.hmm_engine import HMMState, RegimeState, RuleRegimeClassifier
from core.levels import LevelEngine, find_swings, is_rejection_candle, trendline_broken
from core.schemas import (
    Direction,
    Flag,
    LevelTier,
    Regime,
    SetupInstance,
    SetupType,
    Zone,
)


# ---------------------------------------------------------------------------
# Evaluation context
# ---------------------------------------------------------------------------


@dataclass
class MarketContext:
    """Everything one evaluation cycle needs, assembled once per trigger-TF close.

    Building this once and passing it down is what keeps the analysis layer
    instrument-agnostic (soul file 3.1): nothing below this object knows whether
    the trade will be expressed as an option or a futures contract.

    Attributes:
        market: e.g. ``"NIFTY50"``.
        now: Current IST timestamp.
        bias_df / setup_df / trigger_df: Closed-candle frames with indicators,
            one per tier of the 4.3 cascade.
        atr_setup: ATR on the setup timeframe - the unit for every buffer.
        regime: Output of the 4.4 classifier.
        levels: The stateful level engine for this instrument.
        chain: Option-chain context (Nifty/Sensex only), or ``None``.
        hmm: The HMM overlay's read - context only.
        flags: Flags accumulated by the data-integrity and context layers.
        data_ok / data_detail: Result of the 4.6 integrity gate.
        news_blocked / news_detail: Result of the 5.6 blackout check.
    """

    market: str
    now: datetime
    bias_df: pd.DataFrame
    setup_df: pd.DataFrame
    trigger_df: pd.DataFrame
    atr_setup: float
    regime: RegimeState
    levels: LevelEngine
    chain: Any = None
    hmm: HMMState = field(default_factory=HMMState)
    flags: list[Flag] = field(default_factory=list)
    data_ok: bool = True
    data_detail: str = ""
    news_blocked: bool = False
    news_detail: str = ""

    @property
    def setup_index(self) -> int:
        """Positional index of the latest closed setup-TF bar."""
        return len(self.setup_df) - 1

    @property
    def last_price(self) -> float:
        """Latest closed trigger-TF close, in underlying points."""
        return float(self.trigger_df["close"].iloc[-1])


# ---------------------------------------------------------------------------
# Setup detectors
# ---------------------------------------------------------------------------


class SetupDetector:
    """Base class. Subclasses implement ``detect`` and ``trigger_fired``."""

    setup_type: SetupType

    def __init__(self, config: Config | None = None) -> None:
        self.cfg = config or get_config()

    @property
    def validity_candles(self) -> int:
        """Setup-TF candles within which the trigger must fire (soul file 5.5)."""
        return int(self.cfg.get("entry.signal_validity_candles"))

    def detect(self, ctx: MarketContext) -> list[SetupInstance]:
        """Return setup instances detected on the latest setup-TF close."""
        raise NotImplementedError

    def trigger_fired(self, setup: SetupInstance,
                      ctx: MarketContext) -> tuple[bool, float, str]:
        """Test the trigger condition on the latest trigger-TF close.

        Returns:
            ``(fired, entry_price, detail)``. ``entry_price`` is the trigger
            candle's close - soul file 10 specifies fills are simulated at the
            trigger candle's close in paper mode, and using the same reference in
            live keeps the two comparable.
        """
        raise NotImplementedError

    def _new_setup(self, ctx: MarketContext, direction: Direction, ref_id: str,
                   structural_stop: float, detail: dict[str, Any]) -> SetupInstance:
        """Construct a :class:`SetupInstance` with the validity window applied."""
        return SetupInstance(
            setup_id=f"s{int(self.setup_type)}-{uuid.uuid4().hex[:10]}",
            setup_type=self.setup_type,
            direction=direction,
            ref_id=ref_id,
            detected_index=ctx.setup_index,
            detected_at=ctx.setup_df.index[-1].to_pydatetime(),
            structural_stop=float(structural_stop),
            counter_bias=RuleRegimeClassifier.is_counter_bias(ctx.regime.regime, direction),
            expires_after_index=ctx.setup_index + self.validity_candles,
            detail=detail,
        )


class TrendlineBreakSetup(SetupDetector):
    """Setup 1 - trendline breakout/breakdown. Trend-continuation column."""

    setup_type = SetupType.TRENDLINE_BREAK

    def detect(self, ctx: MarketContext) -> list[SetupInstance]:
        """A setup-TF close beyond a valid trendline by >= 0.10 x ATR.

        The break must also show momentum: the breaking candle's range is
        ``>= 1.2 x ATR``, or the next candle continues in the break direction.
        Volatility expansion stands in for volume, which is not reliably
        available on index derivatives or spot gold.
        """
        out: list[SetupInstance] = []
        if len(ctx.setup_df) < 3 or ctx.atr_setup <= 0:
            return out

        index = ctx.setup_index
        min_touches = int(self.cfg.get("levels.trendline_min_touches"))
        momentum_atr = float(self.cfg.get("entry.setup1_momentum_atr"))
        row = ctx.setup_df.iloc[index]

        for line in ctx.levels.trendlines:
            if len(line.anchor_indices) < min_touches:
                continue
            if not trendline_broken(line, ctx.setup_df, index, ctx.atr_setup, self.cfg):
                continue

            # A close below a rising support line is a breakdown; a close above a
            # falling resistance line is a breakout.
            direction = Direction.SHORT if line.is_support else Direction.LONG

            candle_range = float(row["high"]) - float(row["low"])
            has_momentum = candle_range >= momentum_atr * ctx.atr_setup
            if not has_momentum and index >= 1:
                prior_close = float(ctx.setup_df["close"].iloc[index - 1])
                has_momentum = (
                    float(row["close"]) > prior_close
                    if direction is Direction.LONG
                    else float(row["close"]) < prior_close
                )
            if not has_momentum:
                continue

            # Stop: beyond the last opposing swing on the far side of the line.
            swings = find_swings(
                ctx.setup_df,
                int(self.cfg.get("levels.fractal_n")),
                int(self.cfg.get("levels.swing_lookback")),
            )
            opposing = [
                swing
                for swing in swings
                if swing.is_high is (direction is Direction.SHORT)
                and swing.index <= index
            ]
            if not opposing:
                continue
            structural_stop = opposing[-1].price

            out.append(
                self._new_setup(
                    ctx,
                    direction,
                    line.line_id,
                    structural_stop,
                    {
                        "line_value": line.value_at(index),
                        "anchors": len(line.anchor_indices),
                        "break_index": index,
                        "trigger_mode": str(self.cfg.get("entry.setup1_trigger_mode")),
                        "stop_source": "opposing swing beyond broken trendline",
                    },
                )
            )
        return out

    def trigger_fired(self, setup: SetupInstance,
                      ctx: MarketContext) -> tuple[bool, float, str]:
        """``break_close`` (default) or ``retest``, selected in config."""
        mode = str(setup.detail.get("trigger_mode", "break_close"))
        close = ctx.last_price
        line_value = float(setup.detail["line_value"])

        if mode == "break_close":
            beyond = (
                close > line_value
                if setup.direction is Direction.LONG
                else close < line_value
            )
            if beyond:
                return True, close, "trigger-TF close beyond the broken line"
            return False, 0.0, "no trigger-TF close beyond the line yet"

        # retest: price returns to the broken line, now flipped S<->R, and prints
        # a rejection candle there.
        row = ctx.trigger_df.iloc[-1]
        touched = float(row["low"]) <= line_value <= float(row["high"])
        if not touched:
            return False, 0.0, "line not retested yet"
        reclaimed = (
            close > line_value
            if setup.direction is Direction.LONG
            else close < line_value
        )
        if reclaimed:
            return True, close, "rejection at the retested line"
        return False, 0.0, "retest without rejection"


class SRReversalSetup(SetupDetector):
    """Setup 2 - reversal at support/resistance. Reversal column.

    The only setup permitted counter to the bias regime, and then only at
    5-of-6 confluence on a Tier A level (soul file 4.4).
    """

    setup_type = SetupType.SR_REVERSAL

    def detect(self, ctx: MarketContext) -> list[SetupInstance]:
        """Price trades into a zone and prints a rejection candle on the setup TF."""
        out: list[SetupInstance] = []
        if ctx.atr_setup <= 0 or len(ctx.setup_df) < 3:
            return out

        index = ctx.setup_index
        row = ctx.setup_df.iloc[index]
        blowthrough = float(self.cfg.get("entry.setup2_blowthrough_atr"))
        divergence_lookback = int(self.cfg.section("entry")["definitions"]["divergence_lookback"])
        fractal_n = int(self.cfg.get("levels.fractal_n"))

        # A single impulse candle wider than 2.5 x ATR into the level is a
        # momentum blowthrough risk - wait for a second test.
        arrival_range = float(row["high"]) - float(row["low"])
        blew_through = arrival_range > blowthrough * ctx.atr_setup

        touched = [
            zone
            for zone in ctx.levels.live_zones()
            if float(row["low"]) <= zone.high and float(row["high"]) >= zone.low
        ]

        for zone in touched:
            direction = Direction.LONG if zone.is_support else Direction.SHORT
            counter_bias = RuleRegimeClassifier.is_counter_bias(ctx.regime.regime, direction)

            # Disqualifier: a Tier B zone cannot carry a counter-bias trade.
            if counter_bias and zone.tier is not LevelTier.A:
                continue
            # Disqualifier: the level has already failed its allowed attempts.
            if zone.zone_id in ctx.levels.blacklisted:
                continue
            if blew_through:
                continue

            has_divergence = rsi_divergence(
                ctx.setup_df, direction, divergence_lookback, fractal_n
            )
            qualifies, reason = is_rejection_candle(
                ctx.setup_df, index, zone, direction, has_divergence, self.cfg
            )
            if not qualifies:
                continue

            # Stop: beyond the rejection wick's extreme.
            structural_stop = (
                float(row["low"]) if direction is Direction.LONG else float(row["high"])
            )
            out.append(
                self._new_setup(
                    ctx,
                    direction,
                    zone.zone_id,
                    structural_stop,
                    {
                        "zone_centre": zone.centre,
                        "zone_low": zone.low,
                        "zone_high": zone.high,
                        "zone_tier": zone.tier.value,
                        "zone_kind": zone.kind.value,
                        "zone_strength": zone.strength,
                        "rejection_extreme": structural_stop,
                        "rejection_reason": reason,
                        "stop_source": "beyond the rejection wick",
                    },
                )
            )
        return out

    def trigger_fired(self, setup: SetupInstance,
                      ctx: MarketContext) -> tuple[bool, float, str]:
        """A trigger-TF close back through the zone edge, extreme still intact."""
        row = ctx.trigger_df.iloc[-1]
        close = float(row["close"])
        extreme = float(setup.detail["rejection_extreme"])

        if setup.direction is Direction.LONG:
            if float(ctx.trigger_df["low"].iloc[-1]) < extreme:
                return False, 0.0, "rejection low was breached before the trigger"
            if close > float(setup.detail["zone_high"]):
                return True, close, "trigger-TF close back above the zone"
            return False, 0.0, "no close back above the zone edge"

        if float(ctx.trigger_df["high"].iloc[-1]) > extreme:
            return False, 0.0, "rejection high was breached before the trigger"
        if close < float(setup.detail["zone_low"]):
            return True, close, "trigger-TF close back below the zone"
        return False, 0.0, "no close back below the zone edge"


class OrderBlockRetestSetup(SetupDetector):
    """Setup 3 - order block retest. Trend-continuation column."""

    setup_type = SetupType.ORDER_BLOCK_RETEST

    def detect(self, ctx: MarketContext) -> list[SetupInstance]:
        """A fresh, unexpired order block that price has re-entered."""
        out: list[SetupInstance] = []
        if ctx.atr_setup <= 0 or ctx.setup_df.empty:
            return out

        row = ctx.setup_df.iloc[ctx.setup_index]
        low, high = float(row["low"]), float(row["high"])

        for block in ctx.levels.fresh_order_blocks():
            if not (low <= block.high and high >= block.low):
                continue
            out.append(
                self._new_setup(
                    ctx,
                    block.direction,
                    block.ob_id,
                    block.far_edge,
                    {
                        "ob_low": block.low,
                        "ob_high": block.high,
                        "far_edge": block.far_edge,
                        "stop_source": "beyond the order block far edge",
                    },
                )
            )
        return out

    def trigger_fired(self, setup: SetupInstance,
                      ctx: MarketContext) -> tuple[bool, float, str]:
        """A rejection or continuation candle closing inside the zone, in the impulse direction."""
        row = ctx.trigger_df.iloc[-1]
        close = float(row["close"])
        open_price = float(row["open"])
        low, high = float(setup.detail["ob_low"]), float(setup.detail["ob_high"])

        if not low <= close <= high:
            return False, 0.0, "trigger candle did not close inside the block"
        if setup.direction is Direction.LONG and close > open_price:
            return True, close, "bullish close inside the order block"
        if setup.direction is Direction.SHORT and close < open_price:
            return True, close, "bearish close inside the order block"
        return False, 0.0, "close inside the block but against the impulse"


class TrendContinuationSetup(SetupDetector):
    """Setup 4 - indicator confluence trend continuation. Trend-continuation column.

    Permitted only in ``TREND_UP`` / ``TREND_DOWN`` regimes.
    """

    setup_type = SetupType.TREND_CONTINUATION

    def detect(self, ctx: MarketContext) -> list[SetupInstance]:
        """ADX rising and aligned, with price in a shallow pullback to a reference."""
        if ctx.regime.regime is Regime.RANGE or ctx.atr_setup <= 0:
            return []
        if len(ctx.setup_df) < 10:
            return []

        threshold = float(self.cfg.get("indicators.adx_trend_threshold"))
        defs = self.cfg.section("entry")["definitions"]
        lookback = int(defs["rising_lookback"])
        max_retrace = float(self.cfg.get("entry.setup4_max_retrace"))
        fractal_n = int(self.cfg.get("levels.fractal_n"))

        row = ctx.setup_df.iloc[ctx.setup_index]
        if any(pd.isna(row.get(col)) for col in ("adx", "plus_di", "minus_di", "bb_mid", "vwap")):
            return []

        adx_now = float(row["adx"])
        if adx_now < threshold:
            return []
        if len(ctx.setup_df) <= lookback:
            return []
        # Disqualifier: ADX falling.
        if adx_now <= float(ctx.setup_df["adx"].iloc[-1 - lookback]):
            return []

        direction = (
            Direction.LONG if ctx.regime.regime is Regime.TREND_UP else Direction.SHORT
        )
        # Disqualifier: DI must agree with the regime.
        if direction is Direction.LONG and float(row["plus_di"]) <= float(row["minus_di"]):
            return []
        if direction is Direction.SHORT and float(row["minus_di"]) <= float(row["plus_di"]):
            return []

        swings = find_swings(
            ctx.setup_df, fractal_n, int(self.cfg.get("levels.swing_lookback"))
        )
        pullback = self._pullback_reference(ctx, row, direction, swings)
        if pullback is None:
            return []
        reference_name, reference_price, extreme = pullback

        # Disqualifier: the pullback has retraced more than 61.8% of the impulse.
        impulse = self._impulse_leg(ctx, direction, swings)
        if impulse is not None:
            leg_start, leg_end = impulse
            leg = abs(leg_end - leg_start)
            if leg > 0:
                retraced = abs(leg_end - extreme) / leg
                if retraced > max_retrace:
                    return []

        return [
            self._new_setup(
                ctx,
                direction,
                f"pullback-{ctx.setup_df.index[-1].isoformat()}",
                extreme,
                {
                    "reference": reference_name,
                    "reference_price": reference_price,
                    "pullback_extreme": extreme,
                    "adx": adx_now,
                    "stop_source": "beyond the pullback extreme",
                },
            )
        ]

    def _pullback_reference(self, ctx: MarketContext, row: pd.Series,
                            direction: Direction,
                            swings: list[Any]) -> tuple[str, float, float] | None:
        """Identify which reference price pulled back to, and the pullback extreme.

        The soul file names three valid references: session VWAP, the Bollinger
        middle band, and the most recent setup-TF swing. The nearest one that
        price actually reached is the operative reference for the trigger.
        """
        window = ctx.setup_df.iloc[-int(self.cfg.get("entry.signal_validity_candles")) - 1:]
        if window.empty:
            return None

        if direction is Direction.LONG:
            extreme = float(window["low"].min())
            reached = lambda level: extreme <= level  # noqa: E731
        else:
            extreme = float(window["high"].max())
            reached = lambda level: extreme >= level  # noqa: E731

        candidates: list[tuple[str, float]] = [
            ("VWAP", float(row["vwap"])),
            ("BB mid", float(row["bb_mid"])),
        ]
        recent = [
            swing for swing in swings if swing.is_high is (direction is Direction.SHORT)
        ]
        if recent:
            candidates.append(("recent swing", float(recent[-1].price)))

        hit = [(name, price) for name, price in candidates if reached(price)]
        if not hit:
            return None
        # The reference price closest to the current close is the one the
        # trigger must reclaim.
        close = float(row["close"])
        name, price = min(hit, key=lambda item: abs(item[1] - close))
        return name, price, extreme

    def _impulse_leg(self, ctx: MarketContext, direction: Direction,
                     swings: list[Any]) -> tuple[float, float] | None:
        """The most recent impulse leg, as ``(start_price, end_price)``."""
        if len(swings) < 2:
            return None
        lows = [swing for swing in swings if not swing.is_high]
        highs = [swing for swing in swings if swing.is_high]
        if not lows or not highs:
            return None
        if direction is Direction.LONG:
            start = lows[-1]
            later = [swing for swing in highs if swing.index > start.index]
            if not later:
                return None
            return start.price, later[-1].price
        start = highs[-1]
        later = [swing for swing in lows if swing.index > start.index]
        if not later:
            return None
        return start.price, later[-1].price

    def trigger_fired(self, setup: SetupInstance,
                      ctx: MarketContext) -> tuple[bool, float, str]:
        """A resumption candle reclaiming the pullback reference level."""
        close = ctx.last_price
        reference = float(setup.detail["reference_price"])
        name = setup.detail["reference"]

        if setup.direction is Direction.LONG and close > reference:
            return True, close, f"resumption close back above {name}"
        if setup.direction is Direction.SHORT and close < reference:
            return True, close, f"resumption close back below {name}"
        return False, 0.0, f"no resumption close through {name}"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class StrategyBook:
    """Holds the four detectors and applies the 4.4 regime permission table.

    Setup instances live across bars: once detected on a setup-TF close, a setup
    has ``entry.signal_validity_candles`` setup-TF candles for its trigger to
    fire (5.5). This class owns that lifecycle so that "one instance, one signal"
    is enforced in a single place.
    """

    def __init__(self, config: Config | None = None) -> None:
        self.cfg = config or get_config()
        self.classifier = RuleRegimeClassifier(self.cfg)
        self.detectors: dict[SetupType, SetupDetector] = {
            SetupType.TRENDLINE_BREAK: TrendlineBreakSetup(self.cfg),
            SetupType.SR_REVERSAL: SRReversalSetup(self.cfg),
            SetupType.ORDER_BLOCK_RETEST: OrderBlockRetestSetup(self.cfg),
            SetupType.TREND_CONTINUATION: TrendContinuationSetup(self.cfg),
        }
        self._active: list[SetupInstance] = []
        self._emitted_refs: set[str] = set()

    def on_setup_close(self, ctx: MarketContext) -> list[SetupInstance]:
        """Run detection on a setup-TF close and expire stale instances.

        Returns:
            The newly detected instances (the full active list is available via
            :meth:`active_setups`).
        """
        self._active = [
            setup
            for setup in self._active
            if not setup.consumed and ctx.setup_index <= setup.expires_after_index
        ]

        discovered: list[SetupInstance] = []
        for setup_type, detector in self.detectors.items():
            permitted_any = any(
                int(setup_type) in allowed
                for allowed in self.classifier.permitted_setups(ctx.regime.regime).values()
            )
            if not permitted_any:
                continue
            for setup in detector.detect(ctx):
                # One instance, one signal: a level/OB/trendline that already
                # produced a signal cannot re-arm without a fresh detection.
                if setup.ref_id in self._emitted_refs:
                    continue
                if any(
                    existing.ref_id == setup.ref_id
                    and existing.setup_type is setup.setup_type
                    and not existing.consumed
                    for existing in self._active
                ):
                    continue
                self._active.append(setup)
                discovered.append(setup)
        return discovered

    def active_setups(self, ctx: MarketContext) -> list[SetupInstance]:
        """Live setups, freshest first (soul file 5.1 pseudocode)."""
        return sorted(
            (
                setup
                for setup in self._active
                if not setup.consumed and ctx.setup_index <= setup.expires_after_index
            ),
            key=lambda setup: setup.detected_index,
            reverse=True,
        )

    def detector_for(self, setup: SetupInstance) -> SetupDetector:
        """Return the detector that owns ``setup``."""
        return self.detectors[setup.setup_type]

    def consume(self, setup: SetupInstance) -> None:
        """Mark a setup as having produced its one signal."""
        setup.consumed = True
        self._emitted_refs.add(setup.ref_id)

    def roll_session(self) -> None:
        """Clear per-session state at the start of a new session."""
        self._active.clear()
        self._emitted_refs.clear()

    def permitted(self, ctx: MarketContext, setup: SetupInstance) -> tuple[bool, str]:
        """Gate G3: is this setup permitted in this direction under this regime?"""
        if not self.classifier.is_permitted(
            ctx.regime.regime, setup.setup_type, setup.direction
        ):
            return False, (
                f"setup {int(setup.setup_type)} {setup.direction.value} not permitted "
                f"in {ctx.regime.regime.value}"
            )
        return True, f"{ctx.regime.regime.value} permits setup {int(setup.setup_type)}"
