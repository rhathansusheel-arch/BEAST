"""Alpaca API wrapper (built on alpaca-py)."""

from __future__ import annotations

from typing import Any, Optional


class AlpacaClient:
    """Thin wrapper around alpaca-py's trading and data clients."""

    def __init__(self, api_key: str, api_secret: str, paper: bool = True) -> None:
        """
        Args:
            api_key: Alpaca API key ID.
            api_secret: Alpaca API secret key.
            paper: Whether to use the paper-trading endpoint.
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper = paper
        self.trading_client: Optional[Any] = None
        self.data_client: Optional[Any] = None

    def connect(self) -> None:
        """Initialize the underlying alpaca-py TradingClient/StockHistoricalDataClient."""
        raise NotImplementedError

    def get_account(self) -> Any:
        """Fetch account info (equity, buying power, status)."""
        raise NotImplementedError

    def is_market_open(self) -> bool:
        """Check whether the market is currently open."""
        raise NotImplementedError
