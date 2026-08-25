"""Order placement, modification, and cancellation."""

from __future__ import annotations

from typing import Any, Optional

from broker.alpaca_client import AlpacaClient


class OrderExecutor:
    """Submits and manages orders through an AlpacaClient."""

    def __init__(self, client: AlpacaClient) -> None:
        self.client = client

    def place_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        order_type: str = "market",
        limit_price: Optional[float] = None,
    ) -> Any:
        """Submit an order and return the resulting order object."""
        raise NotImplementedError

    def cancel_order(self, order_id: str) -> None:
        """Cancel an open order by ID."""
        raise NotImplementedError

    def cancel_all_orders(self) -> None:
        """Cancel all open orders."""
        raise NotImplementedError

    def get_order_status(self, order_id: str) -> str:
        """Return the current status of an order."""
        raise NotImplementedError
