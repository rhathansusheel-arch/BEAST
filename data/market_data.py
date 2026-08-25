"""Real-time and historical data fetching."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Iterable

import pandas as pd

from broker.alpaca_client import AlpacaClient


class MarketData:
    """Fetches historical bars and streams live bars via AlpacaClient."""

    def __init__(self, client: AlpacaClient) -> None:
        self.client = client

    def get_historical_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV bars for a symbol."""
        raise NotImplementedError

    def get_historical_bars_batch(
        self,
        symbols: Iterable[str],
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> dict[str, pd.DataFrame]:
        """Fetch historical OHLCV bars for multiple symbols."""
        raise NotImplementedError

    def stream_live_bars(self, symbols: Iterable[str], on_bar: Callable[[Any], None]) -> None:
        """Subscribe to a live bar stream, invoking `on_bar` for each update."""
        raise NotImplementedError
