"""Feature pipeline for the volatility layer.

What these tests are really guarding is that the matrix handed to the HMM says
what it claims to say. A feature bug does not raise; it trains, converges, and
produces states that mean something other than volatility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.regime.vol_features import (
    BASE_FEATURES,
    VOLUME_FEATURES,
    CausalZScoreScaler,
    assert_volume_policy,
    compute_features,
    feature_columns,
    feature_hash,
    gap_clean_log_returns,
    session_ids,
)
from tests.conftest import ranging_bars, trending_bars

IST = "Asia/Kolkata"


def multi_session_bars(sessions: int = 6, bars_per_session: int = 25,
                       start_price: float = 24000.0, gap: float = 120.0,
                       seed: int = 7) -> pd.DataFrame:
    """Bars across several Indian sessions with a deliberate overnight gap.

    Each session opens ``gap`` points away from the previous close, which is far
    larger than any intra-session move here. That makes the gap trivially
    detectable: if it leaks into a return feature, the feature explodes.
    """
    rng = np.random.default_rng(seed)
    frames = []
    price = start_price
    for session in range(sessions):
        day = pd.Timestamp("2026-09-01", tz=IST) + pd.Timedelta(days=session)
        index = pd.date_range(
            day + pd.Timedelta(hours=9, minutes=15),
            periods=bars_per_session, freq="15min", tz=IST,
        )
        price = price + (gap if session else 0.0)
        closes = price + np.cumsum(rng.normal(0.0, 4.0, bars_per_session))
        opens = np.concatenate([[closes[0]], closes[:-1]])
        frames.append(pd.DataFrame({
            "open": opens,
            "high": np.maximum(opens, closes) + 2.0,
            "low": np.minimum(opens, closes) - 2.0,
            "close": closes,
            "volume": rng.integers(1000, 5000, bars_per_session).astype(float),
        }, index=index))
        price = closes[-1]
    return pd.concat(frames)


class TestSessionTagging:
    """The overnight gap is not an intraday return (soul file 4.6)."""

    def test_every_bar_gets_a_session_id(self, cfg):
        frame = multi_session_bars(sessions=4)
        sessions = session_ids(frame, "09:15", IST)
        assert len(sessions) == len(frame)
        assert sessions.nunique() == 4

    def test_bars_before_the_open_belong_to_the_previous_session(self, cfg):
        """Gold opens at 05:00 IST, so 04:30 is yesterday's tail, not today's head."""
        index = pd.DatetimeIndex([
            pd.Timestamp("2026-09-02 04:30", tz=IST),
            pd.Timestamp("2026-09-02 05:30", tz=IST),
        ])
        frame = pd.DataFrame(
            {"open": [2500.0, 2501.0], "high": [2502.0, 2503.0],
             "low": [2499.0, 2500.0], "close": [2501.0, 2502.0]},
            index=index,
        )
        sessions = session_ids(frame, "05:00", IST)
        assert sessions.iloc[0] != sessions.iloc[1]
        assert sessions.iloc[0] == pd.Timestamp("2026-09-01", tz=IST)

    def test_first_bar_of_each_session_has_no_one_bar_return(self, cfg):
        frame = multi_session_bars(sessions=5)
        sessions = session_ids(frame, "09:15", IST)
        returns = gap_clean_log_returns(frame["close"], sessions)
        first_of_session = sessions.ne(sessions.shift(1))
        assert returns[first_of_session].isna().all()
        assert returns[~first_of_session].notna().all()

    def test_the_gap_never_reaches_a_return_feature(self, cfg):
        """A 120-point gap would dwarf every intra-session move if it leaked.

        Sixteen sessions rather than six: ``sma200_distance_pct`` needs 200
        bias-TF bars of warm-up, and an Indian 15M session is only 25.
        """
        frame = multi_session_bars(sessions=16, gap=120.0)
        features = compute_features(frame, "NIFTY50", cfg, use_volume=False)
        # An intra-session 15M move here is a few points on 24,000, so ~1e-4.
        # The gap is 120/24000 = 5e-3, fifty times larger.
        assert features["ret_1"].abs().max() < 1e-3, (
            "an overnight gap leaked into the one-bar return"
        )

    def test_multi_bar_windows_may_span_a_session_boundary(self, cfg):
        """A 20-bar window that reset each morning would be useless on 25 bars."""
        frame = multi_session_bars(sessions=16)
        features = compute_features(frame, "NIFTY50", cfg, use_volume=False)
        assert features["ret_20"].notna().all()
        assert len(features) > 25, "features collapsed to a single session"


class TestFeatureContract:
    """Column set, ordering and hashing are part of the model contract."""

    def test_columns_match_the_declared_order(self, cfg):
        features = compute_features(trending_bars(900), "NIFTY50", cfg)
        assert tuple(features.columns) == feature_columns(True)

    def test_volume_disabled_market_has_no_volume_columns(self, cfg):
        """XAUUSD spot is OTC; its "volume" is the feed's own tick count."""
        features = compute_features(trending_bars(900), "XAUUSD", cfg)
        assert tuple(features.columns) == BASE_FEATURES
        for column in VOLUME_FEATURES:
            assert column not in features.columns

    def test_volume_policy_assertion_catches_a_mismatch(self, cfg):
        gold = compute_features(trending_bars(900), "XAUUSD", cfg)
        assert_volume_policy(gold, "XAUUSD", cfg)          # passes
        with pytest.raises(ValueError, match="volume policy"):
            assert_volume_policy(gold, "NIFTY50", cfg)     # nifty expects volume

    def test_volume_requested_but_absent_raises(self, cfg):
        frame = trending_bars(600).drop(columns=["volume"])
        with pytest.raises(ValueError, match="no volume column"):
            compute_features(frame, "NIFTY50", cfg, use_volume=True)

    def test_feature_hash_changes_with_the_column_set(self, cfg):
        assert feature_hash(feature_columns(True)) != feature_hash(feature_columns(False))

    def test_feature_hash_is_stable_for_the_same_columns(self, cfg):
        assert feature_hash(feature_columns(True)) == feature_hash(feature_columns(True))

    def test_matrix_is_finite_and_has_no_gaps(self, cfg):
        for builder in (trending_bars, ranging_bars):
            features = compute_features(builder(900), "NIFTY50", cfg)
            assert not features.empty
            assert np.isfinite(features.to_numpy(dtype=float)).all()
            assert features.index.is_monotonic_increasing


class TestCausalScaler:
    """A causal model fed by a non-causal scaler is still look-ahead."""

    def test_transform_last_matches_a_manual_trailing_window(self, cfg):
        features = compute_features(trending_bars(900), "NIFTY50", cfg)
        scaler = CausalZScoreScaler(lookback=100)
        scaled = scaler.transform_last(features)

        window = features.tail(100)
        expected = (window.iloc[-1] - window.mean()) / window.std()
        np.testing.assert_allclose(scaled, expected.to_numpy(dtype=float), atol=1e-9)

    def test_live_and_batch_scaling_agree_on_the_same_bar(self, cfg):
        """The live path and the batch path must scale a bar identically.

        ``transform_last`` is what the live loop calls on the newest bar;
        ``fit_transform`` is what training and backtests call over a whole
        frame. If they disagreed, a backtest and a live session would hand the
        model different numbers for the same bar - and the backtest, having
        seen 600 rows, would be the one that was cheating.
        """
        features = compute_features(trending_bars(1000), "NIFTY50", cfg)
        scaler = CausalZScoreScaler(lookback=100)

        cutoff = features.index[400]
        live = scaler.transform_last(features.loc[:cutoff])
        batch = scaler.fit_transform(features).loc[cutoff].to_numpy(dtype=float)
        np.testing.assert_allclose(live, batch, atol=1e-9)

    def test_fit_transform_row_is_stable_when_history_grows(self, cfg):
        features = compute_features(trending_bars(1000), "NIFTY50", cfg)
        scaler = CausalZScoreScaler(lookback=100)

        early = scaler.fit_transform(features.iloc[:500])
        late = scaler.fit_transform(features.iloc[:700])
        shared = early.index.intersection(late.index)
        assert len(shared) > 100
        pd.testing.assert_frame_equal(
            early.loc[shared], late.loc[shared], atol=1e-12
        )

    def test_short_history_is_refused_rather_than_guessed(self, cfg):
        features = compute_features(trending_bars(900), "NIFTY50", cfg)
        scaler = CausalZScoreScaler(lookback=100, min_periods=50)
        with pytest.raises(ValueError, match="at least 50 bars"):
            scaler.transform_last(features.iloc[:10])

    def test_constant_feature_does_not_produce_infinities(self, cfg):
        """A feature with zero spread carries no information; it must go inert."""
        frame = pd.DataFrame(
            {"a": np.ones(200), "b": np.arange(200, dtype=float)},
            index=pd.date_range("2026-01-01", periods=200, freq="15min", tz=IST),
        )
        scaler = CausalZScoreScaler(lookback=50)
        scaled = scaler.fit_transform(frame)
        assert np.isfinite(scaled.to_numpy(dtype=float)).all()
        assert (scaled["a"] == 0.0).all()
