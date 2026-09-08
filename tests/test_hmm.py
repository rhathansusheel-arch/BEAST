"""Regime detection - the authoritative 4.4 classifier and the HMM overlay."""

from __future__ import annotations

import numpy as np
import pytest

from core.hmm_engine import HMMRegimeEngine, HMMState, RuleRegimeClassifier
from core.schemas import Direction, Regime, SetupType
from data.feature_engineering import compute_indicators, hmm_features
from tests.conftest import ranging_bars, trending_bars


class TestRuleClassifier:
    """Soul file 4.4 - the classifier that actually gates trades."""

    def test_uptrend_classifies_as_trend_up(self, cfg, uptrend):
        frame = compute_indicators(uptrend, "NIFTY50", cfg)
        state = RuleRegimeClassifier(cfg).classify(frame)
        assert state.regime is Regime.TREND_UP
        assert state.plus_di > state.minus_di
        assert state.close_vs_basis > 0

    def test_downtrend_classifies_as_trend_down(self, cfg):
        frame = compute_indicators(trending_bars(slope=-0.6, seed=21), "NIFTY50", cfg)
        state = RuleRegimeClassifier(cfg).classify(frame)
        assert state.regime is Regime.TREND_DOWN
        assert state.minus_di > state.plus_di

    def test_chop_mostly_classifies_as_range(self, cfg, sideways):
        """Chop should sit in RANGE most of the time.

        Asserted over a window rather than on a single bar: ADX oscillates around
        the threshold in chop, so any one bar can land marginally either side.
        What matters for trade permission is that the regime is predominantly
        RANGE, which suppresses setups 1 and 4.
        """
        frame = compute_indicators(sideways, "NIFTY50", cfg)
        classifier = RuleRegimeClassifier(cfg)
        verdicts = [
            classifier.classify(frame.iloc[: end + 1]).regime
            for end in range(len(frame) - 120, len(frame))
        ]
        range_share = verdicts.count(Regime.RANGE) / len(verdicts)
        assert range_share > 0.6, f"only {range_share:.0%} of chop bars read as RANGE"

    def test_trend_has_higher_adx_than_chop(self, cfg, uptrend, sideways):
        """The chop filter only means something if the two are actually separable."""
        classifier = RuleRegimeClassifier(cfg)
        trend = classifier.classify(compute_indicators(uptrend, "NIFTY50", cfg))
        chop = classifier.classify(compute_indicators(sideways, "NIFTY50", cfg))
        assert trend.adx > chop.adx

    def test_empty_frame_falls_back_to_range(self, cfg, uptrend):
        """An unknown regime must be the most restrictive one, never a guess."""
        frame = compute_indicators(uptrend, "NIFTY50", cfg).iloc[:0]
        assert RuleRegimeClassifier(cfg).classify(frame).regime is Regime.RANGE

    def test_warming_indicators_fall_back_to_range(self, cfg, uptrend):
        frame = compute_indicators(uptrend, "NIFTY50", cfg).iloc[:5]
        assert RuleRegimeClassifier(cfg).classify(frame).regime is Regime.RANGE


class TestPermissionTable:
    """The 4.4 permission table, read literally."""

    def test_trend_up_permits_long_trend_setups(self, cfg):
        classifier = RuleRegimeClassifier(cfg)
        for setup in (SetupType.TRENDLINE_BREAK, SetupType.ORDER_BLOCK_RETEST,
                      SetupType.TREND_CONTINUATION):
            assert classifier.is_permitted(Regime.TREND_UP, setup, Direction.LONG)

    def test_trend_up_permits_only_setup_2_short(self, cfg):
        classifier = RuleRegimeClassifier(cfg)
        assert classifier.is_permitted(Regime.TREND_UP, SetupType.SR_REVERSAL, Direction.SHORT)
        for setup in (SetupType.TRENDLINE_BREAK, SetupType.ORDER_BLOCK_RETEST,
                      SetupType.TREND_CONTINUATION):
            assert not classifier.is_permitted(Regime.TREND_UP, setup, Direction.SHORT)

    def test_range_suppresses_setups_1_and_4(self, cfg):
        classifier = RuleRegimeClassifier(cfg)
        for direction in (Direction.LONG, Direction.SHORT):
            assert not classifier.is_permitted(
                Regime.RANGE, SetupType.TRENDLINE_BREAK, direction
            )
            assert not classifier.is_permitted(
                Regime.RANGE, SetupType.TREND_CONTINUATION, direction
            )
            assert classifier.is_permitted(Regime.RANGE, SetupType.SR_REVERSAL, direction)
            assert classifier.is_permitted(
                Regime.RANGE, SetupType.ORDER_BLOCK_RETEST, direction
            )

    def test_counter_bias_detection(self, cfg):
        assert RuleRegimeClassifier.is_counter_bias(Regime.TREND_UP, Direction.SHORT)
        assert RuleRegimeClassifier.is_counter_bias(Regime.TREND_DOWN, Direction.LONG)
        assert not RuleRegimeClassifier.is_counter_bias(Regime.TREND_UP, Direction.LONG)

    def test_range_has_no_counter_bias(self, cfg):
        """RANGE has no prevailing direction, so nothing taken in it is counter-bias."""
        for direction in (Direction.LONG, Direction.SHORT):
            assert not RuleRegimeClassifier.is_counter_bias(Regime.RANGE, direction)


class TestHMMOverlay:
    """The HMM is context, not a gate - that is the property under test."""

    def test_disabled_by_config_never_blocks(self, cfg):
        engine = HMMRegimeEngine(cfg)          # cfg fixture disables the overlay
        blocked, reason = engine.blocks_entry(HMMState())
        assert blocked is False
        assert "context-only" in reason or "disabled" in reason

    def test_shipped_config_leaves_the_legacy_overlay_inert(self, raw_config):
        """The shipped config disables this overlay outright.

        It is superseded by ``core/regime/``, and it is disabled rather than
        deleted only because Phase 1 of the regime build may not touch the entry
        pipeline that still imports it. Two defects make it unfit to run beside
        the new layer: it infers with ``predict_proba`` (forward-backward
        smoothing, i.e. look-ahead bias), and it labels states directionally.
        """
        engine = HMMRegimeEngine(raw_config)
        assert engine.enabled is False
        assert engine.as_gate is False
        blocked, _ = engine.blocks_entry(HMMState(usable=False, detail="cold"))
        assert blocked is False

    def test_unfitted_model_never_blocks_even_as_gate(self, raw_config):
        """A cold-start model must not silently stop Beast trading."""
        raw_config.section("hmm")["as_gate"] = True
        engine = HMMRegimeEngine(raw_config)
        assert engine.model is None
        blocked, _ = engine.blocks_entry(HMMState(usable=False))
        assert blocked is False

    def test_refuses_to_fit_below_min_train_bars(self, raw_config):
        raw_config.section("hmm")["min_train_bars"] = 5000
        engine = HMMRegimeEngine(raw_config)
        features = hmm_features(trending_bars(400), raw_config)
        assert engine.fit(features) is False

    @pytest.mark.parametrize("builder", [trending_bars, ranging_bars])
    def test_features_are_finite_and_aligned(self, cfg, builder):
        features = hmm_features(builder(400), cfg)
        assert not features.empty
        assert np.isfinite(features.to_numpy()).all()
        assert list(features.columns) == ["ret", "abs_ret", "atr_pct", "range_pct", "adx"]

    def test_fit_and_update_when_hmmlearn_present(self, raw_config):
        """A full fit/update round-trip, skipped when hmmlearn is absent."""
        pytest.importorskip("hmmlearn")
        section = raw_config.section("hmm")
        section.update({"enabled": True,          # shipped disabled; on for this test only
                        "n_candidates": [2, 3], "n_init": 2, "min_train_bars": 200,
                        "covariance_type": "diag"})
        engine = HMMRegimeEngine(raw_config)
        features = hmm_features(trending_bars(700), raw_config)

        assert engine.fit(features) is True
        state = engine.update(features)
        assert state.n_states in (2, 3)
        assert 0.0 <= state.confidence <= 1.0
        assert state.label in engine.state_labels.values()

    def test_flicker_guard_marks_unstable(self, raw_config):
        """Rapid state changes must mark the overlay unusable rather than acted on."""
        pytest.importorskip("hmmlearn")
        engine = HMMRegimeEngine(raw_config)
        engine._flicker_threshold = 1
        for state in (0, 1, 0, 1, 0):
            engine._history.push(state)
        assert engine._history.changes() > engine._flicker_threshold
