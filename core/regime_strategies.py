"""Vol-based allocation strategies per detected regime.

Maps a (regime, trend, confidence) tuple to a target portfolio allocation
and leverage, per the thresholds defined in the `strategy` section of
settings.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AllocationDecision:
    """Target allocation output for a single rebalance decision."""

    target_allocation: float
    target_leverage: float
    regime: int
    confidence: float


class RegimeStrategy:
    """Translates regime + trend state into target allocation/leverage."""

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Args:
            config: The `strategy` section of settings.yaml (low/mid/high vol
                allocations, low_vol_leverage, rebalance_threshold,
                uncertainty_size_mult).
        """
        self.config = config

    def allocate(
        self,
        regime: int,
        confidence: float,
        trending: bool,
        vol_bucket: str,
    ) -> AllocationDecision:
        """Compute the target allocation/leverage for the current regime state."""
        raise NotImplementedError

    def needs_rebalance(self, current_allocation: float, target_allocation: float) -> bool:
        """Check whether drift exceeds `rebalance_threshold`."""
        raise NotImplementedError
