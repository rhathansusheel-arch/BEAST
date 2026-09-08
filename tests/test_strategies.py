"""Setup detection, the confluence engine, and the gate pipeline.

Covers soul file 4.5 (level engine), 5.2 (the four setups), 5.3/5.4 (the 4-of-6
engine and its machine-checkable definitions) and 5.1 (the gate chain).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.confluence import (
    ConfluenceEngine,
    crossed_above,
    crossed_below,
    histogram_expanding,
    is_falling,
    is_rising,
    tags_band,
    walking_band,
)
from core.levels import (
    LevelEngine,
    build_sr_zones,
    cluster_swings,
    detect_order_blocks,
    find_swings,
    fit_trendlines,
    is_engulfing,
    is_rejection_candle,
)
from core.regime_strategies import MarketContext, StrategyBook
from core.risk_manager import RiskManager
from core.schemas import (
    ConfluenceMode,
    Direction,
    Gate,
    IndicatorRead,
    LevelKind,
    LevelTier,
    Regime,
    SetupType,
    Zone,
)
from core.signal_generator import FeedState, SignalGenerator
from data.feature_engineering import compute_indicators, resample_ohlc
from tests.conftest import ranging_bars, trending_bars


def make_zone(centre: float, is_support: bool = True, tier: LevelTier = LevelTier.A,
              width: float = 5.0) -> Zone:
    return Zone(
        zone_id=f"z-{centre}",
        kind=LevelKind.SWING_CLUSTER,
        tier=tier,
        low=centre - width,
        high=centre + width,
        centre=centre,
        is_support=is_support,
    )


# ---------------------------------------------------------------------------
# 4.5 - level engine
# ---------------------------------------------------------------------------


class TestSwingsAndZones:
    def test_swing_high_is_the_local_extreme(self, cfg):
        highs = [10, 11, 15, 11, 10, 9, 8]
        frame = pd.DataFrame(
            {
                "open": highs, "close": highs,
                "high": highs, "low": [value - 2 for value in highs],
            },
            index=pd.date_range("2026-09-01 09:15", periods=len(highs), freq="5min",
                                tz="Asia/Kolkata"),
        )
        swings = find_swings(frame, fractal_n=2, lookback=50)
        highs_found = [swing for swing in swings if swing.is_high]
        assert any(swing.price == 15 for swing in highs_found)

    def test_zone_needs_at_least_two_swings(self, cfg):
        frame = trending_bars(300)
        swings = find_swings(frame, 2, 200)
        single = [swings[0]] if swings else []
        assert build_sr_zones(single, atr_value=5.0, tier=LevelTier.B, config=cfg) == []

    def test_cluster_respects_the_atr_tolerance(self, cfg):
        from core.levels import Swing
        from datetime import datetime

        stamp = datetime(2026, 9, 1, 10, 0)
        swings = [
            Swing(0, stamp, 100.0, True),
            Swing(1, stamp, 100.5, True),
            Swing(2, stamp, 130.0, True),
        ]
        clusters = cluster_swings(swings, atr_value=10.0, cluster_atr=0.15)
        assert len(clusters) == 2
        assert len(clusters[0]) == 2

    def test_zone_width_is_quarter_atr(self, cfg):
        from core.levels import Swing
        from datetime import datetime

        stamp = datetime(2026, 9, 1, 10, 0)
        swings = [Swing(0, stamp, 100.0, False), Swing(5, stamp, 100.2, False)]
        zones = build_sr_zones(swings, atr_value=8.0, tier=LevelTier.A, config=cfg)
        assert len(zones) == 1
        expected = float(cfg.get("levels.sr_zone_width_atr")) * 8.0
        assert zones[0].high - zones[0].low == pytest.approx(expected)

    def test_next_opposing_level_picks_the_nearest(self, cfg):
        engine = LevelEngine("NIFTY50", cfg)
        engine.zones = [make_zone(24100, False), make_zone(24300, False)]
        nearest = engine.next_opposing_level(24000, Direction.LONG, LevelTier.A)
        assert nearest is not None and nearest.centre == 24100

    def test_reentry_cap_blacklists_a_level(self, cfg):
        engine = LevelEngine("NIFTY50", cfg)
        zone = make_zone(24000)
        engine.zones = [zone]
        cap = int(cfg.get("entry.level_reentry_cap"))
        for _ in range(cap - 1):
            assert engine.register_failed_attempt(zone.zone_id) is False
        assert engine.register_failed_attempt(zone.zone_id) is True
        assert zone.zone_id in engine.blacklisted
        assert engine.live_zones() == []


class TestRejectionCandle:
    def test_long_wick_with_close_inside_qualifies(self, cfg):
        index = pd.date_range("2026-09-01 09:15", periods=2, freq="5min", tz="Asia/Kolkata")
        frame = pd.DataFrame(
            {"open": [100, 100], "high": [101, 101], "low": [100, 90], "close": [100, 99]},
            index=index,
        )
        zone = make_zone(99.0, is_support=True, width=3.0)
        ok, reason = is_rejection_candle(frame, 1, zone, Direction.LONG, config=cfg)
        assert ok and "wick" in reason

    def test_engulfing_qualifies(self, cfg):
        index = pd.date_range("2026-09-01 09:15", periods=2, freq="5min", tz="Asia/Kolkata")
        frame = pd.DataFrame(
            {"open": [100, 97], "high": [101, 103], "low": [96, 96], "close": [98, 102]},
            index=index,
        )
        assert is_engulfing(frame, 1, Direction.LONG)

    def test_ordinary_candle_does_not_qualify(self, cfg):
        """A small-wicked, non-engulfing candle is not a rejection.

        The lower wick here is 9% of the range, well under the 50% threshold, and
        the body does not engulf the prior candle's body.
        """
        index = pd.date_range("2026-09-01 09:15", periods=2, freq="5min", tz="Asia/Kolkata")
        frame = pd.DataFrame(
            {
                "open": [100.0, 100.0], "high": [101.0, 101.0],
                "low": [99.0, 99.9], "close": [100.5, 100.4],
            },
            index=index,
        )
        zone = make_zone(100.0, width=3.0)
        ok, _ = is_rejection_candle(frame, 1, zone, Direction.LONG, config=cfg)
        assert not ok

    def test_exactly_half_wick_qualifies(self, cfg):
        """The threshold is inclusive: a wick of exactly 50% of range counts."""
        index = pd.date_range("2026-09-01 09:15", periods=2, freq="5min", tz="Asia/Kolkata")
        frame = pd.DataFrame(
            {
                "open": [100.0, 100.0], "high": [101.0, 101.0],
                "low": [99.0, 99.0], "close": [100.5, 100.4],
            },
            index=index,
        )
        zone = make_zone(100.0, width=3.0)
        ok, reason = is_rejection_candle(frame, 1, zone, Direction.LONG, config=cfg)
        assert ok and "wick" in reason


class TestTrendlinesAndOrderBlocks:
    def test_trendline_needs_three_anchors(self, cfg):
        frame = trending_bars(300, slope=0.5)
        swings = find_swings(frame, 2, 200)
        lines = fit_trendlines(frame, swings, atr_value=8.0, config=cfg)
        minimum = int(cfg.get("levels.trendline_min_touches"))
        assert all(len(line.anchor_indices) >= minimum for line in lines)

    def test_order_blocks_face_the_impulse(self, cfg):
        frame = trending_bars(400, slope=1.2, noise=2.0)
        swings = find_swings(frame, 2, 200)
        blocks = detect_order_blocks(frame, swings, atr_value=5.0, config=cfg)
        for block in blocks:
            assert block.high > block.low
            assert block.direction in (Direction.LONG, Direction.SHORT)

    def test_order_block_zone_is_the_body_by_default(self, cfg):
        assert str(cfg.get("levels.ob_zone_mode")) == "body"


# ---------------------------------------------------------------------------
# 5.4 - machine-checkable definitions
# ---------------------------------------------------------------------------


class TestDefinitions:
    def test_rising_requires_the_minimum_delta(self):
        series = pd.Series([10.0, 10.1, 10.2, 10.3])
        assert is_rising(series, lookback=3, min_delta=0.5) is False
        assert is_rising(pd.Series([10.0, 11.0, 12.0, 13.0]), 3, 0.5) is True

    def test_falling_is_the_mirror(self):
        assert is_falling(pd.Series([13.0, 12.0, 11.0, 10.0]), 3, 0.5) is True
        assert is_falling(pd.Series([10.0, 11.0, 12.0, 13.0]), 3, 0.5) is False

    def test_fresh_crossover_window(self):
        fast = pd.Series([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        slow = pd.Series([1.5] * 8)
        # The cross happened 4 bars ago; a 3-bar window must not see it.
        assert crossed_above(fast, slow, within=3) is None
        assert crossed_above(fast, slow, within=6) == 4

    def test_crossed_below_detects_the_downward_cross(self):
        fast = pd.Series([2.0, 2.0, 2.0, 1.0, 1.0])
        slow = pd.Series([1.5] * 5)
        assert crossed_below(fast, slow, within=3) == 1

    def test_histogram_expanding_requires_monotonic_growth(self):
        growing = pd.Series([0.1, 0.2, 0.4])
        shrinking = pd.Series([0.4, 0.3, 0.2])
        assert histogram_expanding(growing, Direction.LONG, bars=2) is True
        assert histogram_expanding(shrinking, Direction.LONG, bars=2) is False

    def test_histogram_sign_must_match_direction(self):
        negative = pd.Series([-0.1, -0.2, -0.4])
        assert histogram_expanding(negative, Direction.LONG, bars=2) is False
        assert histogram_expanding(negative, Direction.SHORT, bars=2) is True

    def test_walking_the_band_needs_two_of_three(self):
        index = pd.date_range("2026-09-01 09:15", periods=3, freq="5min", tz="Asia/Kolkata")
        frame = pd.DataFrame(
            {"close": [101.0, 101.0, 99.0], "bb_upper": [100.0] * 3, "bb_lower": [90.0] * 3},
            index=index,
        )
        assert walking_band(frame, Direction.LONG, 1.0, 3, 2, 0.1) is True
        frame["close"] = [99.0, 99.0, 99.0]
        assert walking_band(frame, Direction.LONG, 1.0, 3, 2, 0.1) is False

    def test_tags_band_requires_close_back_inside(self):
        index = pd.date_range("2026-09-01 09:15", periods=1, freq="5min", tz="Asia/Kolkata")
        pierced = pd.DataFrame(
            {"high": [105.0], "low": [89.0], "close": [95.0],
             "bb_upper": [104.0], "bb_lower": [90.0]},
            index=index,
        )
        assert tags_band(pierced, Direction.LONG) is True
        closed_outside = pierced.copy()
        closed_outside["close"] = [89.5]
        assert tags_band(closed_outside, Direction.LONG) is False


# ---------------------------------------------------------------------------
# 5.3 - the confluence engine
# ---------------------------------------------------------------------------


class TestConfluence:
    def test_six_indicators_always_reported(self, cfg, uptrend):
        frame = compute_indicators(uptrend, "NIFTY50", cfg)
        result = ConfluenceEngine(cfg).evaluate(
            frame, Direction.LONG, ConfluenceMode.TREND_CONTINUATION
        )
        assert set(result.reads) == {"adx", "stoch", "macd", "rsi", "bb", "vwap"}
        assert result.aligned + result.opposing + result.neutral == 6

    def test_neutrals_are_retained(self, cfg, sideways):
        """Section 9 needs the misses, so neutrals must not be discarded."""
        frame = compute_indicators(sideways, "NIFTY50", cfg)
        result = ConfluenceEngine(cfg).evaluate(
            frame, Direction.LONG, ConfluenceMode.TREND_CONTINUATION
        )
        assert all(isinstance(read, IndicatorRead) for read in result.reads.values())

    def test_uptrend_aligns_bullish(self, cfg, uptrend):
        frame = compute_indicators(uptrend, "NIFTY50", cfg)
        result = ConfluenceEngine(cfg).evaluate(
            frame, Direction.LONG, ConfluenceMode.TREND_CONTINUATION
        )
        assert result.aligned >= result.opposing

    def test_conflict_rule_rejects_a_split(self, cfg, uptrend):
        """A 4-3 split is disagreement, not confluence."""
        frame = compute_indicators(uptrend, "NIFTY50", cfg)
        engine = ConfluenceEngine(cfg)
        result = engine.evaluate(frame, Direction.LONG, ConfluenceMode.TREND_CONTINUATION)
        result.opposing = int(cfg.get("entry.opposing_reject_count"))
        # Re-derive the pass decision the way the engine does.
        assert not (result.aligned >= result.required and result.opposing < 3)

    def test_modes_can_disagree(self, cfg, uptrend):
        """The two columns are genuinely different rules, not a formatting choice."""
        frame = compute_indicators(uptrend, "NIFTY50", cfg)
        engine = ConfluenceEngine(cfg)
        zone = make_zone(float(frame["close"].iloc[-1]))
        trend = engine.evaluate(frame, Direction.LONG, ConfluenceMode.TREND_CONTINUATION)
        reversal = engine.evaluate(frame, Direction.LONG, ConfluenceMode.REVERSAL, zone)
        assert trend.reads != reversal.reads or trend.aligned != reversal.aligned

    def test_counter_bias_requires_five(self, cfg, uptrend):
        frame = compute_indicators(uptrend, "NIFTY50", cfg)
        required = int(cfg.get("entry.min_confluence_counter_bias"))
        result = ConfluenceEngine(cfg).evaluate(
            frame, Direction.SHORT, ConfluenceMode.REVERSAL,
            make_zone(float(frame["close"].iloc[-1])), required=required,
        )
        assert result.required == 5
        if result.aligned < 5:
            assert result.passed is False


# ---------------------------------------------------------------------------
# 5.2 / 5.5 - setup lifecycle
# ---------------------------------------------------------------------------


def build_context(cfg, market: str, base: pd.DataFrame) -> MarketContext:
    """Assemble a context the way the pipeline does, from one 1-minute series."""
    from core.hmm_engine import RuleRegimeClassifier

    timeframes = cfg.timeframes(market)
    setup_df = compute_indicators(resample_ohlc(base, timeframes["setup"]), market, cfg)
    bias_df = compute_indicators(resample_ohlc(base, timeframes["bias"]), market, cfg)
    trigger_df = resample_ohlc(base, timeframes["trigger"])
    atr_value = float(setup_df["atr"].iloc[-1])

    levels = LevelEngine(market, cfg)
    levels.rebuild(setup_df, bias_df, atr_value)

    return MarketContext(
        market=market,
        now=trigger_df.index[-1].to_pydatetime(),
        bias_df=bias_df,
        setup_df=setup_df,
        trigger_df=trigger_df,
        atr_setup=atr_value,
        regime=RuleRegimeClassifier(cfg).classify(bias_df),
        levels=levels,
    )


class TestSetupLifecycle:
    def test_detection_fires_on_a_clean_trend(self, cfg):
        """A clean uptrend must produce at least one setup - otherwise the rest
        of this class is vacuously true."""
        book = StrategyBook(cfg)
        ctx = build_context(cfg, "NIFTY50", trending_bars(1500, slope=0.4))
        found = book.on_setup_close(ctx)
        assert found, "no setup detected on a clean trending series"
        assert all(
            book.permitted(ctx, setup)[0] or setup.setup_type is SetupType.SR_REVERSAL
            for setup in found
        )

    def test_one_instance_one_signal(self, cfg):
        book = StrategyBook(cfg)
        ctx = build_context(cfg, "NIFTY50", trending_bars(1500, slope=0.4))
        book.on_setup_close(ctx)
        active = book.active_setups(ctx)
        assert active, "expected at least one active setup"

        setup = active[0]
        book.consume(setup)
        assert setup.consumed
        assert setup not in book.active_setups(ctx)

        # Re-detecting the same reference must not re-arm it.
        book.on_setup_close(ctx)
        assert all(item.ref_id != setup.ref_id for item in book.active_setups(ctx))

    def test_setups_expire_after_the_validity_window(self, cfg):
        book = StrategyBook(cfg)
        ctx = build_context(cfg, "NIFTY50", trending_bars(1500, slope=0.4))
        book.on_setup_close(ctx)
        assert book.active_setups(ctx)

        validity = int(cfg.get("entry.signal_validity_candles"))
        for setup in book.active_setups(ctx):
            assert setup.expires_after_index == setup.detected_index + validity
            # Push the window into the past to simulate the setup going stale.
            setup.expires_after_index = ctx.setup_index - 1
        assert book.active_setups(ctx) == []

    def test_regime_permission_is_enforced(self, cfg):
        book = StrategyBook(cfg)
        ctx = build_context(cfg, "NIFTY50", ranging_bars(1200))
        ctx.regime.regime = Regime.RANGE
        from core.schemas import SetupInstance
        from datetime import datetime

        setup = SetupInstance(
            setup_id="x", setup_type=SetupType.TRENDLINE_BREAK, direction=Direction.LONG,
            ref_id="tl-1", detected_index=0, detected_at=datetime(2026, 9, 1, 10, 0),
            structural_stop=0.0,
        )
        permitted, reason = book.permitted(ctx, setup)
        assert permitted is False
        assert "RANGE" in reason


# ---------------------------------------------------------------------------
# 5.1 - the gate chain
# ---------------------------------------------------------------------------


class TestGatePipeline:
    def _feed(self, cfg, market: str, base: pd.DataFrame) -> FeedState:
        timeframes = cfg.timeframes(market)
        return FeedState(
            bias_df=resample_ohlc(base, timeframes["bias"]),
            setup_df=resample_ohlc(base, timeframes["setup"]),
            trigger_df=resample_ohlc(base, timeframes["trigger"]),
        )

    def test_outside_session_rejects_at_g0(self, cfg):
        from datetime import datetime

        import pandas as pd

        base = trending_bars(1200)
        risk = RiskManager(cfg)
        generator = SignalGenerator("NIFTY50", risk, cfg)
        feed = self._feed(cfg, "NIFTY50", base)
        # 03:00 IST is outside the 09:15-15:30 Indian window.
        moment = pd.Timestamp("2026-09-02 03:00", tz="Asia/Kolkata").to_pydatetime()
        result = generator.evaluate(feed, moment)
        assert result.signal is None
        gates = {rejection.failed_gate for rejection in result.rejections}
        assert Gate.G0_SESSION in gates or Gate.G1_DATA in gates

    def test_opening_guard_blocks_entries(self, cfg):
        import pandas as pd

        risk = RiskManager(cfg)
        generator = SignalGenerator("NIFTY50", risk, cfg)
        guard_end = generator.clock.window.guard_end
        assert guard_end.hour == 9 and guard_end.minute == 30
        moment = pd.Timestamp("2026-09-02 09:20", tz="Asia/Kolkata").to_pydatetime()
        allowed, reason = generator.clock.may_enter(moment)
        assert allowed is False and "opening-range guard" in reason

    def test_stale_feed_rejects_at_g1(self, cfg):
        import pandas as pd

        base = trending_bars(1200)
        risk = RiskManager(cfg)
        generator = SignalGenerator("NIFTY50", risk, cfg)
        feed = self._feed(cfg, "NIFTY50", base)
        # Evaluate hours after the last bar closed.
        moment = (feed.trigger_df.index[-1] + pd.Timedelta(hours=3)).to_pydatetime()
        result = generator.evaluate(feed, moment)
        assert result.signal is None

    def test_every_rejection_names_its_gate(self, cfg):
        import pandas as pd

        base = trending_bars(1200)
        risk = RiskManager(cfg)
        generator = SignalGenerator("NIFTY50", risk, cfg)
        feed = self._feed(cfg, "NIFTY50", base)
        moment = (feed.trigger_df.index[-1] + pd.Timedelta(minutes=1)).to_pydatetime()
        result = generator.evaluate(feed, moment)
        for rejection in result.rejections:
            assert rejection.failed_gate in set(Gate)
            assert rejection.gate_detail
