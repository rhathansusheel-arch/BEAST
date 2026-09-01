"""Shared fixtures: a populated config and synthetic underlying data.

Appendix A ships with its blockers unset, which is correct - Beast must refuse to trade
until the operator supplies them. Tests that need Beast to actually reach G8/G9 fill those
in explicitly, so it stays obvious which values are assumptions of the test rather than of
the Soul File.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from beast.analysis.option_chain import ChainSnapshot, StrikeQuote
from beast.config import Config
from beast.constants import Market


@pytest.fixture
def cfg():
    return Config.load()


@pytest.fixture
def ready_cfg(cfg):
    """Appendix A with the Open Item 17/18/19 blockers populated."""
    return cfg.with_overrides(
        **{
            "capital": 500000.0,
            "options.min_oi": 100000,
            "options.min_volume": 10000,
            "options.min_premium": 20.0,
            "instruments.nifty.lot_size": 75,
            "instruments.nifty.strike_interval": 50,
            "instruments.sensex.lot_size": 20,
            "instruments.sensex.strike_interval": 100,
            "instruments.gold.venue": "MCX",
            "instruments.gold.contract_multiplier": 100.0,
            "instruments.gold.tick_size": 1.0,
            "instruments.gold.tick_value": 100.0,
            "data.gold_spread_max": 0.5,
        }
    )


def make_bars(
    n: int = 400,
    start: datetime | None = None,
    freq_min: int = 1,
    trend: float = 0.0,
    noise: float = 5.0,
    base: float = 24000.0,
    seed: int = 7,
) -> pd.DataFrame:
    """Synthetic 1-minute underlying OHLCV on an Indian session clock."""
    rng = np.random.default_rng(seed)
    start = start or datetime(2026, 9, 1, 9, 15)
    index = pd.date_range(start, periods=n, freq=f"{freq_min}min")
    steps = rng.normal(trend, noise, n).cumsum()
    close = base + steps
    high = close + np.abs(rng.normal(0, noise / 2, n))
    low = close - np.abs(rng.normal(0, noise / 2, n))
    open_ = np.concatenate([[base], close[:-1]])
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum.reduce([high, open_, close]),
            "low": np.minimum.reduce([low, open_, close]),
            "close": close,
            "volume": rng.integers(1000, 5000, n).astype(float),
        },
        index=index,
    )


def make_chain(spot: float = 24180.0, when: datetime | None = None, dte: int = 2) -> ChainSnapshot:
    """A Nifty-shaped chain: ATM +/- 10 strikes on a 50-point ladder."""
    when = when or datetime(2026, 9, 1, 10, 42)
    atm = round(spot / 50) * 50
    quotes: list[StrikeQuote] = []
    for step in range(-10, 11):
        strike = atm + step * 50
        moneyness = (spot - strike) / 50.0
        call_delta = float(np.clip(0.5 + moneyness * 0.06, 0.02, 0.98))
        quotes.append(
            StrikeQuote(strike, "CE", 179.5 - step * 4, 180.5 - step * 4, 180 - step * 4,
                        oi=1_400_000 - abs(step) * 50_000, oi_change=20_000 - step * 1_000,
                        volume=320_000, iv=13.4, delta=call_delta)
        )
        quotes.append(
            StrikeQuote(strike, "PE", 149.5 + step * 4, 150.5 + step * 4, 150 + step * 4,
                        oi=1_300_000 - abs(step + 1) * 50_000, oi_change=15_000 + step * 1_000,
                        volume=280_000, iv=13.9, delta=-(1 - call_delta))
        )
    return ChainSnapshot(
        underlying="NIFTY50",
        spot=spot,
        expiry=(when + timedelta(days=dte)).date(),
        taken_at=when,
        strike_interval=50.0,
        quotes=quotes,
        iv_percentile=42.0,
    )
