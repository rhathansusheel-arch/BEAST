"""Combines HMM regime detection and strategy allocation into trade signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from core.hmm_engine import HMMEngine
from core.regime_strategies import RegimeStrategy
from core.risk_manager import RiskManager


@dataclass
class TradeSignal:
    """A single actionable trade signal for one symbol."""

    symbol: str
    side: str
    quantity: int
    target_allocation: float
    regime: int
    confidence: float


class SignalGenerator:
    """Pipes market features through the HMM, strategy, and risk manager to produce signals."""

    def __init__(
        self,
        hmm_engine: HMMEngine,
        strategy: RegimeStrategy,
        risk_manager: RiskManager,
    ) -> None:
        self.hmm_engine = hmm_engine
        self.strategy = strategy
        self.risk_manager = risk_manager

    def generate(
        self, symbol: str, features: pd.DataFrame, equity: float
    ) -> Optional[TradeSignal]:
        """Produce a trade signal for `symbol`, or None if no action is warranted."""
        raise NotImplementedError

    def generate_batch(
        self, features_by_symbol: dict[str, pd.DataFrame], equity: float
    ) -> list[TradeSignal]:
        """Generate signals across all configured symbols."""
        raise NotImplementedError
