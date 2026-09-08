"""Broker adapters and the interface they all implement.

The soul file separates analysis from execution (3.1). This package is the
execution side of that line: nothing here knows what a setup or a confluence
count is, and nothing in ``core/`` knows which broker is connected.

Two adapters ship:

* :mod:`broker.zerodha_client` - Nifty/Sensex index spot and the weekly option
  chain, via Kite Connect.
* :mod:`broker.paper_broker` - the simulated venue. XAUUSD routes here, and
  so does everything else while ``mode: paper``. It replaced an Alpaca adapter
  that could reach none of Beast's three markets.

Both satisfy :class:`BrokerClient`, so ``main.py`` routes by market using
``broker.routing`` in config and never branches on broker identity.

Paper mode is enforced here as well as in the runner. Every adapter checks
``config.is_paper`` inside ``place_order`` and refuses to transmit. Two
independent switches - ``mode`` and ``broker.paper_trading`` - must both be off
before a real order can leave the process.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any

import pandas as pd


class OrderSide(str, Enum):
    """Direction of an order at the broker."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order types Beast uses. Hard-flat exits always use ``MARKET``."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"


@dataclass
class Quote:
    """Top-of-book for one instrument."""

    symbol: str
    bid: float
    ask: float
    last: float
    timestamp: datetime

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return self.last

    @property
    def spread(self) -> float:
        if self.bid <= 0 or self.ask <= 0:
            return float("inf")
        return self.ask - self.bid


@dataclass
class OrderRequest:
    """An order Beast wants placed."""

    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    tag: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderResult:
    """What the broker said.

    Attributes:
        accepted: True when the broker acknowledged the order.
        order_id: Broker order id, or a synthetic ``paper-...`` id.
        filled_quantity: Quantity filled, when known.
        average_price: Fill price, when known.
        paper: True when this was simulated rather than transmitted.
        message: Broker message or the reason for refusal.
    """

    accepted: bool
    order_id: str = ""
    filled_quantity: int = 0
    average_price: float = 0.0
    paper: bool = True
    message: str = ""


class BrokerClient(ABC):
    """The interface every adapter implements."""

    name: str = "broker"

    @abstractmethod
    def connect(self) -> bool:
        """Establish the session. Returns True on success."""

    @abstractmethod
    def is_connected(self) -> bool:
        """Whether the adapter currently has a usable session."""

    @abstractmethod
    def history(self, symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
        """Return the last ``bars`` closed candles as a tz-aware OHLC frame."""

    @abstractmethod
    def quote(self, symbol: str) -> Quote | None:
        """Return top-of-book for ``symbol``, or ``None`` when unavailable."""

    @abstractmethod
    def place_order(self, request: OrderRequest) -> OrderResult:
        """Place an order. Must refuse to transmit while in paper mode."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a resting order."""

    def option_chain(self, underlying: str, expiry: date) -> Any:
        """Return a chain snapshot. Only option-capable adapters implement this."""
        raise NotImplementedError(f"{self.name} does not provide option chains")

    def expiries(self, underlying: str) -> list[date]:
        """Available option expiries, nearest first."""
        raise NotImplementedError(f"{self.name} does not provide option expiries")

    def futures_contracts(self, underlying: str) -> list[tuple[str, date]]:
        """Available futures contracts as ``(symbol, expiry)``."""
        raise NotImplementedError(f"{self.name} does not provide futures contracts")

    def capital(self) -> float | None:
        """Account equity, for live sizing. ``None`` when unavailable."""
        return None
