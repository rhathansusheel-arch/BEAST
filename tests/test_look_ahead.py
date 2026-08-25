"""Verify no look-ahead bias in regime inference.

`HMMEngine.predict_regime_filtered` must depend only on the observations up
to and including the row being scored. This is what distinguishes it from
`GaussianHMM.predict()` (Viterbi), which is globally optimal over a whole
sequence and can silently revise earlier states using later data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.hmm_engine import HMMEngine
from data.feature_engineering import compute_features


def _make_synthetic_bars(n_bars: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=n_bars, freq="B")
    returns = rng.normal(loc=0.0003, scale=0.01, size=n_bars)
    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + rng.uniform(0, 0.01, n_bars))
    low = close * (1 - rng.uniform(0, 0.01, n_bars))
    open_ = close * (1 + rng.uniform(-0.005, 0.005, n_bars))
    volume = rng.integers(1_000_000, 5_000_000, n_bars).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


@pytest.fixture(scope="module")
def trained_engine() -> tuple[HMMEngine, pd.DataFrame]:
    bars = _make_synthetic_bars(1000)
    features = compute_features(bars)
    config = {
        "n_candidates": [3],
        "n_init": 2,
        "covariance_type": "diag",
        "min_train_bars": 252,
        "stability_bars": 3,
        "flicker_window": 20,
        "flicker_threshold": 4,
        "min_confidence": 0.55,
    }
    engine = HMMEngine(config).fit(features)
    return engine, features


def test_no_look_ahead_bias(trained_engine):
    """Regime at T must be identical with data[0:T] vs data[0:T+100]."""
    engine, features = trained_engine

    regime_short = engine.predict_regime_filtered(features.iloc[:400]).iloc[-1]
    regime_long = engine.predict_regime_filtered(features.iloc[:500]).iloc[:400].iloc[-1]

    assert regime_short == regime_long, "LOOK-AHEAD BIAS DETECTED"


def test_filtered_probabilities_are_valid_distributions(trained_engine):
    """Sanity check that we're exercising the forward algorithm (probabilities
    sum to 1 at every row), not a stub that silently returns garbage."""
    engine, features = trained_engine
    filtered = engine.predict_regime_proba(features.iloc[:400])

    assert filtered.shape[0] == 400
    assert np.allclose(filtered.sum(axis=1), 1.0, atol=1e-6)
    assert (filtered.to_numpy() >= 0).all()
