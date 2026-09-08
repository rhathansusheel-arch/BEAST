"""Shared fixtures.

Two things every test needs:

* A **config with the blockers filled in.** The shipped ``beast_config.yaml`` leaves
  lot sizes, gold contract specs and option liquidity floors as ``null`` on
  purpose - Beast must refuse to trade until the operator supplies them. Tests
  that want to exercise the *trading* path therefore need a config where those
  are populated, and the one test that checks the refusal uses the raw config.

* **Deterministic synthetic bars.** Every generator here is seeded, so a failing
  assertion reproduces exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import Config, load_config, set_config  # noqa: E402

IST = "Asia/Kolkata"


def _fill_blockers(cfg: Config) -> Config:
    """Populate the deliberately-unset values with plausible live numbers."""
    instruments = cfg.section("instruments")
    instruments["nifty"].update({"lot_size": 75, "strike_interval": 50})
    instruments["sensex"].update({"lot_size": 20, "strike_interval": 100})
    instruments["gold"].update(
        {
            "venue": "MCX_GOLDM",
            "contract_multiplier": 10.0,
            "tick_size": 1.0,
            "tick_value": 10.0,
        }
    )
    options = cfg.section("options")
    options.update({"min_oi": 100000, "min_volume": 50000, "min_premium": 5.0})
    cfg.section("data")["gold_spread_max"] = 0.60
    cfg.section("hmm")["enabled"] = False       # keep unit tests fast and offline
    cfg.section("ai")["enabled"] = False        # never call the API from tests
    return cfg


@pytest.fixture
def raw_config() -> Config:
    """The shipped config, blockers left unset."""
    cfg = load_config()
    set_config(cfg)
    return cfg


@pytest.fixture
def cfg() -> Config:
    """Config with blockers populated - the default for behavioural tests."""
    config = _fill_blockers(load_config())
    set_config(config)
    return config


# ---------------------------------------------------------------------------
# Synthetic bar generators
# ---------------------------------------------------------------------------


def _frame(closes: np.ndarray, index: pd.DatetimeIndex,
           noise: float, seed: int) -> pd.DataFrame:
    """Wrap a close series into a coherent OHLCV frame."""
    rng = np.random.default_rng(seed)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    spread = rng.uniform(noise * 0.3, noise, len(closes))
    highs = np.maximum(opens, closes) + spread
    lows = np.minimum(opens, closes) - spread
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": rng.integers(500, 5000, len(closes)).astype(float),
        },
        index=index,
    )


def session_index(bars: int, start: str = "2026-09-01 09:15",
                  freq: str = "1min") -> pd.DatetimeIndex:
    """A tz-aware IST index of ``bars`` bars."""
    return pd.date_range(start, periods=bars, freq=freq, tz=IST)


def trending_bars(bars: int = 600, slope: float = 0.6, start_price: float = 24000.0,
                  noise: float = 3.0, seed: int = 11) -> pd.DataFrame:
    """A clean uptrend - should classify as ``TREND_UP`` on the bias timeframe."""
    rng = np.random.default_rng(seed)
    closes = start_price + np.cumsum(rng.normal(slope, noise * 0.4, bars))
    return _frame(closes, session_index(bars), noise, seed)


def ranging_bars(bars: int = 600, centre: float = 24000.0, amplitude: float = 25.0,
                 noise: float = 3.0, seed: int = 12) -> pd.DataFrame:
    """A sideways market - should classify as ``RANGE``.

    A mean-reverting (Ornstein-Uhlenbeck) walk rather than a sine wave. A clean
    sine has long, smooth directional runs and reads as a *strong trend* on ADX,
    which is the opposite of the chop this fixture is meant to represent.
    """
    rng = np.random.default_rng(seed)
    reversion = 0.85
    deviations = np.zeros(bars)
    for index in range(1, bars):
        deviations[index] = reversion * deviations[index - 1] + rng.normal(0, noise)
    scale = amplitude / max(1e-9, np.abs(deviations).max())
    closes = centre + deviations * scale
    return _frame(closes, session_index(bars), noise, seed)


def crash_bars(bars: int = 600, start_price: float = 24000.0, drop: float = 400.0,
               noise: float = 4.0, seed: int = 13) -> pd.DataFrame:
    """A sharp decline - used for the stop and loss-limit tests."""
    rng = np.random.default_rng(seed)
    ramp = np.linspace(0, -drop, bars)
    closes = start_price + ramp + rng.normal(0, noise * 0.5, bars)
    return _frame(closes, session_index(bars), noise, seed)


@pytest.fixture
def uptrend() -> pd.DataFrame:
    return trending_bars()


@pytest.fixture
def sideways() -> pd.DataFrame:
    return ranging_bars()


@pytest.fixture
def selloff() -> pd.DataFrame:
    return crash_bars()
