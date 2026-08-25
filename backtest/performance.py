"""Sharpe, drawdown, regime breakdown, and benchmark comparisons."""

from __future__ import annotations

import pandas as pd


def compute_sharpe(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Compute the annualized Sharpe ratio of a returns series."""
    raise NotImplementedError


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    """Compute the maximum peak-to-trough drawdown of an equity curve."""
    raise NotImplementedError


def regime_breakdown(returns: pd.Series, regimes: pd.Series) -> pd.DataFrame:
    """Compute per-regime performance statistics (return, vol, Sharpe, time-in-regime)."""
    raise NotImplementedError


def compare_to_benchmark(returns: pd.Series, benchmark_returns: pd.Series) -> dict[str, float]:
    """Compute alpha/beta and relative performance vs. a benchmark series."""
    raise NotImplementedError
