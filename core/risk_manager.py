"""Position sizing, leverage, and drawdown limits.

Enforces per-trade risk, exposure caps, position/trade count limits, and
daily/weekly drawdown circuit breakers per the `risk` section of
settings.yaml.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


class RiskManager:
    """Applies portfolio-level risk constraints to proposed trades."""

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Args:
            config: The `risk` section of settings.yaml (max_risk_per_trade,
                max_exposure, max_leverage, max_single_position,
                max_concurrent, max_daily_trades, daily/weekly drawdown
                thresholds, max_dd_from_peak).
        """
        self.config = config

    def size_position(self, target_allocation: float, equity: float, price: float) -> int:
        """Convert a target allocation into a share quantity, capped by risk limits."""
        raise NotImplementedError

    def check_exposure_limits(self, positions: dict[str, float], equity: float) -> bool:
        """Verify aggregate exposure/leverage stay within configured caps."""
        raise NotImplementedError

    def check_trade_limits(self, trades_today: int, open_positions: int) -> bool:
        """Verify daily trade count and concurrent position limits."""
        raise NotImplementedError

    def check_drawdown_limit(self, equity_curve: pd.Series) -> str:
        """Evaluate daily/weekly drawdown against reduce/halt thresholds.

        Returns:
            One of "normal", "reduce", or "halt".
        """
        raise NotImplementedError
