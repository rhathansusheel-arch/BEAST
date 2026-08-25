"""Track open positions and P&L."""

from __future__ import annotations

from typing import Any

from broker.alpaca_client import AlpacaClient


class PositionTracker:
    """Reads and caches current positions and P&L from Alpaca."""

    def __init__(self, client: AlpacaClient) -> None:
        self.client = client

    def get_positions(self) -> dict[str, Any]:
        """Return current open positions keyed by symbol."""
        raise NotImplementedError

    def get_pnl(self) -> dict[str, float]:
        """Return realized/unrealized P&L, overall and per-symbol."""
        raise NotImplementedError

    def get_equity_curve(self) -> Any:
        """Return the account's historical equity curve."""
        raise NotImplementedError
