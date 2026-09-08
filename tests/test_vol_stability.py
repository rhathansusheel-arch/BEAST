"""Stability guards, the fail-safe path, and model persistence.

Not one of the three test modules the build prompt names, but the prompt also
says to report anything left uncovered rather than write a test that asserts
nothing - and shipping :class:`~core.regime.stability.StabilityTracker` with no
tests would leave the confirmation, flicker and stale-hold logic unexercised.
Those three are the difference between a volatility layer and a random number
that resizes positions.

The tracker is driven with synthetic posteriors rather than a fitted model
wherever possible, so each guard is tested in isolation and the tests stay fast.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from core.regime.contracts import CALM, TURBULENT, UNKNOWN, ModelMetadata
from core.regime.hmm_engine import ModelUnusable, VolatilityHMM
from core.regime.stability import StabilityTracker
from core.regime.vol_features import compute_features, feature_columns, feature_hash
from tests.conftest import trending_bars

IST = "Asia/Kolkata"
START = pd.Timestamp("2026-09-01 09:15", tz=IST)


class FakeEngine:
    """Just enough of :class:`VolatilityHMM` for the tracker to talk to.

    The tracker only ever asks a model two things - what a state id is called,
    and what version the model is - so nothing else needs to exist.
    """

    def __init__(self, labels: dict[int, str] | None = None) -> None:
        self._labels = labels or {0: CALM, 1: "NORMAL", 2: TURBULENT}
        self.metadata = type("Meta", (), {"model_version": "fake@2026-09-01"})()

    def label_for(self, state: int) -> tuple[str, str]:
        from core.regime.contracts import collapse

        rung = self._labels[int(state)]
        return rung, collapse(rung)


def posterior(state: int, confidence: float = 0.95, n_states: int = 3) -> np.ndarray:
    """A posterior peaked on ``state`` at ``confidence``."""
    rest = (1.0 - confidence) / (n_states - 1)
    values = np.full(n_states, rest)
    values[state] = confidence
    return values


def bar(index: int) -> datetime:
    return (START + pd.Timedelta(minutes=15 * index)).to_pydatetime()


@pytest.fixture
def tracker(cfg) -> StabilityTracker:
    return StabilityTracker("NIFTY50", cfg)


class TestConfirmation:
    """A new state must persist before it is believed."""

    def test_a_new_state_is_unconfirmed_until_it_persists(self, tracker, cfg):
        engine = FakeEngine()
        confirm_bars = int(cfg.get("regime.stability.confirm_bars"))

        for index in range(confirm_bars - 1):
            state = tracker.observe(bar(index), posterior(0), engine)
            assert state.is_confirmed is False
            assert state.consecutive_bars == index + 1
            assert "held" in state.reason

        final = tracker.observe(bar(confirm_bars - 1), posterior(0), engine)
        assert final.is_confirmed is True
        assert final.label == CALM

    def test_a_low_confidence_read_never_confirms_however_long_it_holds(
        self, tracker, cfg
    ):
        """A state the model is unsure about is not a state."""
        engine = FakeEngine()
        floor = float(cfg.get("regime.stability.min_confidence"))
        for index in range(10):
            state = tracker.observe(bar(index), posterior(0, floor - 0.05), engine)
        assert state.is_confirmed is False
        assert "confidence" in state.reason

    def test_switching_state_restarts_the_count(self, tracker):
        engine = FakeEngine()
        for index in range(5):
            tracker.observe(bar(index), posterior(0), engine)
        switched = tracker.observe(bar(5), posterior(2), engine)
        assert switched.consecutive_bars == 1
        assert switched.is_confirmed is False
        assert switched.label == TURBULENT

    def test_unconfirmed_states_are_sized_as_uncertain(self, tracker, cfg):
        engine = FakeEngine()
        state = tracker.observe(bar(0), posterior(0), engine)
        assert state.is_confirmed is False
        assert state.size_multiplier == pytest.approx(
            float(cfg.get("regime.sizing.uncertainty_size_mult"))
        )


class TestFlicker:
    """A series changing too fast to believe forces uncertainty mode."""

    def test_alternating_states_trip_the_flicker_guard(self, tracker, cfg):
        engine = FakeEngine()
        threshold = int(cfg.get("regime.stability.flicker_threshold"))
        state = None
        for index in range(threshold * 2 + 4):
            state = tracker.observe(bar(index), posterior(index % 2), engine)
        assert state.is_flickering is True
        assert state.is_confirmed is False
        assert state.flicker_rate > 0.5
        assert "changes in" in state.reason

    def test_a_steady_series_does_not_flicker(self, tracker):
        engine = FakeEngine()
        state = None
        for index in range(20):
            state = tracker.observe(bar(index), posterior(1), engine)
        assert state.is_flickering is False
        assert state.flicker_rate == pytest.approx(0.0)
        assert state.is_confirmed is True

    def test_flickering_caps_the_multiplier_at_uncertainty(self, tracker, cfg):
        engine = FakeEngine()
        state = None
        for index in range(20):
            state = tracker.observe(bar(index), posterior(index % 2), engine)
        assert state.size_multiplier <= float(
            cfg.get("regime.sizing.uncertainty_size_mult")
        )


class TestFailSafe:
    """A model bug must not halt Beast, and must not let it size at full risk."""

    def test_last_confirmed_state_is_held_for_a_bounded_number_of_bars(
        self, tracker, cfg
    ):
        engine = FakeEngine()
        for index in range(6):
            tracker.observe(bar(index), posterior(0), engine)
        assert tracker.last_confirmed is not None

        budget = int(cfg.get("regime.stability.stale_max_bars"))
        for index in range(budget):
            held = tracker.on_failure(bar(6 + index), "inference exploded")
            assert held.label == CALM
            assert held.is_stale is True
            assert held.is_usable is False
            assert held.size_multiplier == pytest.approx(
                float(cfg.get("regime.sizing.uncertainty_size_mult"))
            )

        expired = tracker.on_failure(bar(6 + budget), "inference exploded")
        assert expired.label == UNKNOWN
        assert expired.is_stale is False

    def test_failure_before_any_confirmed_state_goes_straight_to_unknown(
        self, tracker, cfg
    ):
        state = tracker.on_failure(bar(0), "model failed to load")
        assert state.label == UNKNOWN
        assert state.veto is False, "a model bug must not silently halt Beast"
        assert state.size_multiplier == pytest.approx(
            float(cfg.get("regime.sizing.uncertainty_size_mult"))
        )
        assert "model failed to load" in state.reason

    def test_unknown_vetoes_only_when_configured_to(self, cfg):
        cfg.section("regime")["veto"]["on_unknown"] = True
        tracker = StabilityTracker("NIFTY50", cfg)
        assert tracker.on_failure(bar(0), "boom").veto is True

    def test_a_successful_bar_resets_the_stale_budget(self, tracker, cfg):
        engine = FakeEngine()
        for index in range(6):
            tracker.observe(bar(index), posterior(0), engine)
        tracker.on_failure(bar(6), "transient")
        tracker.on_failure(bar(7), "transient")
        tracker.observe(bar(8), posterior(0), engine)

        budget = int(cfg.get("regime.stability.stale_max_bars"))
        for index in range(budget):
            assert tracker.on_failure(bar(9 + index), "transient").is_stale is True


class TestSessionBoundary:
    """Resetting every morning would put the first hour in uncertainty mode."""

    def test_state_and_count_carry_across_the_boundary_by_default(self, tracker, cfg):
        assert cfg.get("regime.stability.carry_across_sessions") is True
        engine = FakeEngine()
        for index in range(6):
            tracker.observe(bar(index), posterior(0), engine, session_id="2026-09-01")

        first_of_next = tracker.observe(
            bar(6), posterior(0), engine, session_id="2026-09-02"
        )
        assert first_of_next.is_confirmed is True
        assert first_of_next.consecutive_bars == 7

    def test_reset_at_the_boundary_when_carry_is_off(self, cfg):
        cfg.section("regime")["stability"]["carry_across_sessions"] = False
        tracker = StabilityTracker("NIFTY50", cfg)
        engine = FakeEngine()
        for index in range(6):
            tracker.observe(bar(index), posterior(0), engine, session_id="2026-09-01")

        first_of_next = tracker.observe(
            bar(6), posterior(0), engine, session_id="2026-09-02"
        )
        assert first_of_next.is_confirmed is False
        assert first_of_next.consecutive_bars == 1


class TestSensexDelayTag:
    """Sensex bias bars are effectively one bar stale (soul file 4.6, 12)."""

    def test_sensex_states_carry_the_delay(self, cfg):
        state = StabilityTracker("SENSEX", cfg).observe(bar(0), posterior(0), FakeEngine())
        assert state.data_delay_minutes == 15

    @pytest.mark.parametrize("market", ["NIFTY50", "XAUUSD"])
    def test_other_markets_do_not(self, cfg, market):
        state = StabilityTracker(market, cfg).observe(bar(0), posterior(0), FakeEngine())
        assert state.data_delay_minutes == 0

    def test_the_delay_survives_a_stale_hold(self, cfg):
        tracker = StabilityTracker("SENSEX", cfg)
        for index in range(6):
            tracker.observe(bar(index), posterior(0), FakeEngine())
        assert tracker.on_failure(bar(6), "feed dropped").data_delay_minutes == 15


class TestPersistence:
    """A model that outlives its feature pipeline must refuse to load."""

    @pytest.fixture
    def fitted(self, cfg, tmp_path):
        import warnings

        pytest.importorskip("hmmlearn")
        cfg.section("regime")["hmm"].update(
            {"n_candidates": [3], "n_init": 1, "covariance_type": "diag"}
        )
        features = compute_features(trending_bars(1000), "NIFTY50", cfg)
        engine = VolatilityHMM("NIFTY50", cfg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            engine.train(features, min_train_bars=200)
        engine.save(tmp_path)
        return engine, tmp_path

    def test_round_trip_preserves_the_model_and_its_metadata(self, cfg, fitted):
        engine, directory = fitted
        loaded = VolatilityHMM.load("NIFTY50", cfg, directory)
        assert loaded.metadata is not None
        assert loaded.metadata.model_version == engine.metadata.model_version
        assert loaded.metadata.feature_hash == engine.metadata.feature_hash

        observations = np.random.default_rng(0).normal(
            size=(50, len(engine.metadata.feature_list))
        )
        np.testing.assert_allclose(
            engine.predict_vol_state_filtered(observations),
            loaded.predict_vol_state_filtered(observations),
            atol=1e-12,
        )

    def test_a_readable_json_sidecar_is_written(self, fitted):
        _, directory = fitted
        sidecar = directory / "nifty50_vol_hmm.json"
        assert sidecar.exists()
        assert "feature_hash" in sidecar.read_text(encoding="utf-8")

    def test_feature_hash_mismatch_refuses_the_model(self, cfg, fitted):
        """Silent feature drift is how these systems rot."""
        import pickle

        engine, directory = fitted
        path = directory / "nifty50_vol_hmm.pkl"
        with path.open("rb") as handle:
            payload = pickle.load(handle)

        stale = payload["metadata"]
        payload["metadata"] = ModelMetadata(
            **{**stale.__dict__, "feature_hash": "deadbeefdeadbeef"}
        )
        with path.open("wb") as handle:
            pickle.dump(payload, handle)

        with pytest.raises(ModelUnusable, match="feature hash mismatch"):
            VolatilityHMM.load("NIFTY50", cfg, directory)

    def test_an_expired_model_refuses_to_load(self, cfg, fitted):
        _, directory = fitted
        limit = int(cfg.get("regime.hmm.max_model_age_days"))
        future = datetime.now(timezone.utc) + timedelta(days=limit + 2)
        with pytest.raises(ModelUnusable, match="days old"):
            VolatilityHMM.load("NIFTY50", cfg, directory, now=future)

    def test_a_missing_model_refuses_rather_than_training_silently(self, cfg, tmp_path):
        with pytest.raises(ModelUnusable, match="no persisted model"):
            VolatilityHMM.load("XAUUSD", cfg, tmp_path)

    def test_the_expected_hash_tracks_the_market_volume_policy(self, cfg):
        """Gold's hash differs from Nifty's, because its feature set does."""
        assert feature_hash(feature_columns(cfg.regime_use_volume("NIFTY50"))) != \
            feature_hash(feature_columns(cfg.regime_use_volume("XAUUSD")))


class TestTrainingRefusals:
    """Refusing to fit is a feature. Every refusal names its reason."""

    def test_too_few_bars_is_refused_with_the_bar_count(self, cfg):
        pytest.importorskip("hmmlearn")
        features = compute_features(trending_bars(900), "NIFTY50", cfg)
        engine = VolatilityHMM("NIFTY50", cfg)
        with pytest.raises(ValueError, match="below the .* minimum"):
            engine.train(features)

    def test_a_volume_policy_mismatch_is_refused(self, cfg):
        """A Gold model fitted on tick counts would train and mean nothing."""
        pytest.importorskip("hmmlearn")
        with_volume = compute_features(trending_bars(900), "NIFTY50", cfg)
        engine = VolatilityHMM("XAUUSD", cfg)
        with pytest.raises(ValueError, match="volume policy"):
            engine.train(with_volume, min_train_bars=200)
