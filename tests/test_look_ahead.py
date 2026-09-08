"""Look-ahead bias verification.

This is the test file that decides whether any backtest number is worth reading.
Three distinct forms of look-ahead are checked:

1. **Indicator look-ahead** - the value of an indicator on bar *t* must not
   change when later bars arrive. Any rolling window that peeks forward, or a
   centred moving average, shows up here immediately.
2. **Structure look-ahead** - a confirmed swing must never appear on the bars it
   needs to be confirmed by. Soul file 4.5 requires a fractal be confirmed only
   after the next N candles close, and that lag is the whole reason the level
   engine does not repaint.
3. **Frame look-ahead** - the backtest provider must never hand the engine a bar
   that had not closed by the evaluation timestamp.

If any of these fail, every performance figure in the project is fiction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.levels import find_swings
from data.feature_engineering import compute_indicators, resample_ohlc
from data.market_data import CsvHistoryProvider
from tests.conftest import trending_bars

INDICATOR_COLUMNS = [
    "atr", "adx", "plus_di", "minus_di", "rsi",
    "macd", "macd_signal", "macd_hist", "stoch_k", "stoch_d",
    "bb_mid", "bb_upper", "bb_lower", "vwap",
]


class TestIndicatorStability:
    """An indicator value, once printed on a closed bar, must never change."""

    def test_values_do_not_change_when_future_bars_arrive(self, cfg):
        full = trending_bars(500)
        truncated = full.iloc[:400]

        computed_full = compute_indicators(full, "NIFTY50", cfg).iloc[:400]
        computed_truncated = compute_indicators(truncated, "NIFTY50", cfg)

        for column in INDICATOR_COLUMNS:
            left = computed_full[column].to_numpy(dtype=float)
            right = computed_truncated[column].to_numpy(dtype=float)
            both_nan = np.isnan(left) & np.isnan(right)
            assert np.allclose(left[~both_nan], right[~both_nan], equal_nan=True, atol=1e-9), (
                f"{column} changed when future bars were appended - look-ahead bias"
            )

    @pytest.mark.parametrize("cut", [200, 300, 450])
    def test_stability_across_several_cut_points(self, cfg, cut):
        full = trending_bars(500)
        computed_full = compute_indicators(full, "NIFTY50", cfg)
        computed_cut = compute_indicators(full.iloc[:cut], "NIFTY50", cfg)
        last_full = computed_full["rsi"].iloc[cut - 1]
        last_cut = computed_cut["rsi"].iloc[-1]
        assert last_full == pytest.approx(last_cut, abs=1e-9)


class TestSwingConfirmation:
    """Soul file 4.5: a swing is confirmed only after N following candles close."""

    def test_no_swing_inside_the_confirmation_lag(self, cfg):
        frame = trending_bars(300)
        fractal_n = int(cfg.get("levels.fractal_n"))
        swings = find_swings(frame, fractal_n, lookback=200)
        assert swings, "expected at least one swing in 300 bars"
        newest = max(swing.index for swing in swings)
        assert newest <= len(frame) - 1 - fractal_n, (
            "a swing was reported before its confirmation candles had closed"
        )

    def test_confirmed_swings_are_stable(self, cfg):
        """Adding future bars must not retract an already-confirmed swing."""
        frame = trending_bars(400)
        fractal_n = int(cfg.get("levels.fractal_n"))

        early = find_swings(frame.iloc[:300], fractal_n, lookback=300)
        later = find_swings(frame, fractal_n, lookback=400)
        later_keys = {(swing.index, round(swing.price, 6), swing.is_high) for swing in later}

        for swing in early:
            if swing.index > 300 - 1 - fractal_n:
                continue
            assert (swing.index, round(swing.price, 6), swing.is_high) in later_keys, (
                "a confirmed swing disappeared when later bars arrived - repainting"
            )


class TestProviderSlicing:
    """The backtest provider must never leak an unclosed bar."""

    @pytest.fixture
    def provider(self, cfg, tmp_path):
        frame = trending_bars(1200)
        path = tmp_path / "bars.csv"
        frame.to_csv(path, index_label="datetime")
        return CsvHistoryProvider(path, "NIFTY50", cfg)

    def test_no_bar_at_or_after_the_evaluation_time(self, cfg, provider):
        moment = provider.base.index[800]
        feed = provider.slice_at(moment.to_pydatetime(), bars=500)
        assert feed is not None

        for name, frame in (
            ("bias", feed.bias_df), ("setup", feed.setup_df), ("trigger", feed.trigger_df)
        ):
            if frame.empty:
                continue
            assert frame.index[-1] < moment, f"{name} frame contains a bar at or after `now`"

    def test_last_bar_has_fully_elapsed(self, cfg, provider):
        """The final bar of each frame must have closed, not merely opened."""
        from data.feature_engineering import parse_timeframe

        moment = provider.base.index[900]
        feed = provider.slice_at(moment.to_pydatetime(), bars=500)
        timeframes = cfg.timeframes("NIFTY50")

        for role, frame in (
            ("bias", feed.bias_df), ("setup", feed.setup_df), ("trigger", feed.trigger_df)
        ):
            if frame.empty:
                continue
            interval = parse_timeframe(timeframes[role])
            assert frame.index[-1] + interval <= moment, (
                f"{role} frame's last bar had not closed by the evaluation time"
            )

    def test_slices_are_prefixes_of_each_other(self, cfg, provider):
        """An earlier slice must be a strict prefix of a later one."""
        early = provider.slice_at(provider.base.index[600].to_pydatetime(), bars=1000)
        late = provider.slice_at(provider.base.index[900].to_pydatetime(), bars=1000)
        assert early is not None and late is not None

        overlap = early.setup_df.index.intersection(late.setup_df.index)
        assert len(overlap) > 0
        pd.testing.assert_frame_equal(
            early.setup_df.loc[overlap], late.setup_df.loc[overlap]
        )


class TestResampling:
    """Aggregation must be left-closed and left-labelled, never forward-filled."""

    def test_resampled_bar_contains_only_its_own_window(self, cfg):
        frame = trending_bars(200)
        resampled = resample_ohlc(frame, "5M")
        first = resampled.iloc[0]
        window = frame.iloc[:5]
        assert first["open"] == pytest.approx(window["open"].iloc[0])
        assert first["close"] == pytest.approx(window["close"].iloc[-1])
        assert first["high"] == pytest.approx(window["high"].max())
        assert first["low"] == pytest.approx(window["low"].min())

    def test_label_is_the_window_open_not_close(self, cfg):
        """Left-labelling is what lets the caller add one interval to get the close."""
        frame = trending_bars(60)
        resampled = resample_ohlc(frame, "15M")
        assert resampled.index[0] == frame.index[0]


# ---------------------------------------------------------------------------
# Volatility layer - the mandatory look-ahead tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained():
    """A small volatility model, fitted once for the whole module.

    Deliberately tiny - three or four states, two restarts - because these tests
    are about causality, not model quality, and a full ``n_candidates`` sweep
    would dominate the suite's runtime.

    ``covariance_type`` is forced to ``diag`` here even though the shipped
    config says ``full``. On 900 synthetic bars a full covariance over 14
    features is singular: 105 free covariance parameters per state against a few
    hundred effective observations. Beast's real minimum is 12,500 bias-TF bars
    (``regime.hmm.min_train_bars``), where ``full`` is comfortably identified;
    fitting one here would be testing a degenerate model rather than the filter.
    ``core.regime.hmm_engine.EMISSION_LOG_FLOOR`` exists for the degenerate case
    and is exercised by ``TestDegenerateCovariance`` below.
    """
    import warnings

    from core.config import load_config, set_config
    from core.regime.hmm_engine import VolatilityHMM
    from core.regime.vol_features import compute_features

    pytest.importorskip("hmmlearn")
    config = load_config()
    set_config(config)
    config.section("regime")["hmm"].update(
        {"n_candidates": [3, 4], "n_init": 2, "covariance_type": "diag"}
    )

    features = compute_features(trending_bars(1400), "NIFTY50", config)
    engine = VolatilityHMM("NIFTY50", config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine.train(features, min_train_bars=200)
    scaled = engine.scaler.fit_transform(features)
    return engine, features, scaled, config


class TestVolatilityLayerLookAhead:
    """The tests that decide whether the regime layer's backtest is readable.

    A hidden Markov model is unusually easy to get wrong here, because the two
    library calls everybody reaches for - ``predict`` and ``predict_proba`` -
    both run forward-*backward* over the whole sequence and revise past states
    using future bars. The result looks excellent in backtest and behaves like a
    different system live.

    Three levels are checked, because fixing one and leaving another is the
    normal outcome:

    1. **Inference.** The filtered posterior at bar T must be identical whether
       it was computed from ``data[0:T]`` or from ``data[0:T+100]``.
    2. **Features.** The same invariant one level down - a feature at T must not
       change when later bars arrive.
    3. **Live vs batch.** The O(1) streaming path and the batch path must agree
       on the same bar, or the backtest and the live loop are different systems.
    """

    def test_no_look_ahead_in_inference(self, trained):
        """State at T is identical whether computed from data[0:T] or data[0:T+100].

        Note on the index: ``data[0:400]`` ends at positional index 399, so the
        comparison is against index 399 of the longer run, not 400. The build
        prompt wrote ``[400]``, which is off by one and would compare two
        different bars.
        """
        engine, _, scaled, _ = trained
        observations = scaled.to_numpy(dtype=float)

        short = engine.predict_vol_state_filtered(observations[0:400])
        long_ = engine.predict_vol_state_filtered(observations[0:500])

        np.testing.assert_allclose(
            short[-1], long_[399], atol=1e-12,
            err_msg="LOOK-AHEAD BIAS DETECTED in filtered inference",
        )
        assert int(np.argmax(short[-1])) == int(np.argmax(long_[399]))

    @pytest.mark.parametrize("cut", [200, 300, 450, 600])
    def test_no_look_ahead_at_several_cut_points(self, trained, cut):
        """One matching row could be luck. Every row up to the cut must match."""
        engine, _, scaled, _ = trained
        observations = scaled.to_numpy(dtype=float)

        short = engine.predict_vol_state_filtered(observations[0:cut])
        long_ = engine.predict_vol_state_filtered(observations[0:cut + 100])
        np.testing.assert_allclose(short, long_[:cut], atol=1e-12)

    def test_no_look_ahead_in_features(self, trained, cfg):
        """Features at T do not change when future bars arrive.

        Compared by timestamp rather than by position: ``compute_features``
        drops warm-up rows, so positional index 399 of the output is not bar 399
        of the input.
        """
        from core.regime.vol_features import compute_features

        bars = trending_bars(1000)
        short = compute_features(bars.iloc[:400], "NIFTY50", cfg)
        long_ = compute_features(bars.iloc[:500], "NIFTY50", cfg)

        assert not short.empty
        shared = short.index.intersection(long_.index)
        assert len(shared) > 50, "not enough overlap to be a meaningful test"
        pd.testing.assert_frame_equal(
            short.loc[shared], long_.loc[shared], atol=1e-12,
        )

    def test_filtered_is_not_the_smoothed_posterior(self, trained):
        """Prove the forward algorithm is running, not forward-backward.

        ``predict_proba`` is what a look-ahead implementation would return. If
        this assertion ever starts failing, the filter has been quietly replaced
        by the library call it exists to avoid.
        """
        engine, _, scaled, _ = trained
        observations = scaled.to_numpy(dtype=float)

        filtered = engine.predict_vol_state_filtered(observations)
        smoothed = engine.model.predict_proba(observations)

        assert not np.allclose(filtered, smoothed, atol=1e-6), (
            "the filtered posterior equals hmmlearn's smoothed posterior - "
            "inference is using future bars"
        )
        # The final bar is the one place they must agree: there is no future
        # left to smooth with, so smoothing reduces to filtering. That equality
        # is also the proof that the divergence above is smoothing and not a
        # bug in the forward recursion.
        np.testing.assert_allclose(filtered[-1], smoothed[-1], atol=1e-6)

    def test_streaming_and_batch_paths_agree(self, trained):
        """The O(1) live step must equal the batch filter, bar for bar."""
        engine, _, scaled, _ = trained
        observations = scaled.to_numpy(dtype=float)[:300]

        batch = engine.predict_vol_state_filtered(observations)
        engine.reset_stream()
        streamed = np.vstack([engine.step(row) for row in observations])

        np.testing.assert_allclose(batch, streamed, atol=1e-10)

    def test_posteriors_are_probabilities(self, trained):
        engine, _, scaled, _ = trained
        posteriors = engine.predict_vol_state_filtered(scaled.to_numpy(dtype=float))
        assert (posteriors >= 0).all() and (posteriors <= 1).all()
        np.testing.assert_allclose(posteriors.sum(axis=1), 1.0, atol=1e-9)

    def test_log_space_survives_a_long_sequence(self, trained):
        """A naive product underflows to zero and reports a uniform posterior."""
        engine, _, scaled, _ = trained
        posteriors = engine.predict_vol_state_filtered(scaled.to_numpy(dtype=float))
        assert np.isfinite(posteriors).all()
        assert posteriors[-1].max() > 1.0 / posteriors.shape[1] + 1e-6, (
            "the final posterior is uniform - the filter underflowed"
        )


class TestDegenerateCovariance:
    """The filter must degrade rather than break when a state collapses.

    EM can drive a state's covariance to near-singular, at which point
    ``multivariate_normal.logpdf`` returns ``-inf`` for observations off its
    low-variance direction, and EM can drive a transition to exactly zero, whose
    log is also ``-inf``. Either one, left alone, makes a state permanently
    unreachable: the forward recursion multiplies, so a posterior that reaches
    exactly zero can never recover no matter what later bars show.

    ``EMISSION_LOG_FLOOR`` bounds both. A floored state is still
    indistinguishable from impossible on any bar where a rival has a sane
    density - the floor changes the numerical failure mode, not the model.
    """

    def test_floor_is_applied_to_emissions_and_transitions(self, trained):
        from core.regime.hmm_engine import EMISSION_LOG_FLOOR

        engine, _, scaled, _ = trained
        emissions = engine._log_emissions(scaled.to_numpy(dtype=float))
        assert np.isfinite(emissions).all()
        assert emissions.min() >= EMISSION_LOG_FLOOR
        assert np.isfinite(engine._log_transmat).all()
        assert np.isfinite(engine._log_startprob).all()

    def test_a_collapsed_state_does_not_poison_the_sequence(self, trained):
        """Force a singular covariance and check the posterior stays a probability."""
        engine, _, scaled, _ = trained
        observations = scaled.to_numpy(dtype=float)[:200]

        # hmmlearn's covars_ setter validates against covariance_type, so a
        # diag model has to be written back in (n_states, n_features) shape even
        # though reading it expands to full matrices.
        diagonals = np.diagonal(
            np.asarray(engine.model.covars_), axis1=1, axis2=2
        ).copy()
        original = diagonals.copy()
        try:
            diagonals[0] *= 1e-12
            engine.model.covars_ = diagonals
            posteriors = engine.predict_vol_state_filtered(observations)
        finally:
            engine.model.covars_ = original

        assert np.isfinite(posteriors).all()
        np.testing.assert_allclose(posteriors.sum(axis=1), 1.0, atol=1e-9)
