"""Crash injection and gap simulation for stress testing."""

from __future__ import annotations

import pandas as pd

from backtest.backtester import Backtester


class StressTest:
    """Applies synthetic shocks to historical data and re-runs the backtester."""

    def __init__(self, backtester: Backtester) -> None:
        self.backtester = backtester

    def inject_crash(
        self, data: pd.DataFrame, drop_pct: float, duration_days: int
    ) -> pd.DataFrame:
        """Return a copy of `data` with a synthetic multi-day crash of `drop_pct` injected."""
        raise NotImplementedError

    def inject_gap(self, data: pd.DataFrame, gap_pct: float) -> pd.DataFrame:
        """Return a copy of `data` with a single-bar gap of `gap_pct` injected."""
        raise NotImplementedError
