"""Walk-forward allocation backtester."""

from __future__ import annotations

from typing import Any

import pandas as pd

from core.signal_generator import SignalGenerator


class Backtester:
    """Runs a walk-forward backtest using rolling train/test windows."""

    def __init__(self, signal_generator: SignalGenerator, config: dict[str, Any]) -> None:
        """
        Args:
            signal_generator: Produces trade signals from features.
            config: The `backtest` section of settings.yaml (slippage_pct,
                initial_capital, train_window, test_window, step_size,
                risk_free_rate).
        """
        self.signal_generator = signal_generator
        self.config = config

    def run(self, historical_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Run the walk-forward backtest and return the resulting equity curve/trade log."""
        raise NotImplementedError

    def _walk_forward_windows(
        self, data: pd.DataFrame
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Split `data` into rolling (train, test) window pairs per train/test/step config."""
        raise NotImplementedError
