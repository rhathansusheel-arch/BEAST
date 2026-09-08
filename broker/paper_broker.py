"""Simulated broker - the only execution venue Beast has today.

This adapter replaces the Alpaca client that previously served XAUUSD. Alpaca
was never a viable venue for Beast: it trades US equities and crypto, and can
reach neither XAUUSD nor Indian index options, which are the only three
instruments in the soul file's scope. An adapter that cannot trade any of
Beast's markets is worse than no adapter, because it looks like coverage.

What this class is, and is not
------------------------------
It **is** the execution half of soul file v3.1 instruction item 6: paper trading
with *complete* simulated order execution. Entries, stops, targets, trails and
exits are all placed and tracked here exactly as they will be in live mode, and
every order carries through to a recorded state. The only difference between
paper and live is that no bytes leave the process.

It is **not** a market-data source. It serves bars and quotes only from a series
the caller feeds it (:meth:`prime`), which is what the backtester and the test
suite do. In a live paper session the real data comes from the market-data
adapter for that instrument, and this class only simulates the order side.

Fill model (soul file 6.8)
--------------------------
The conservative assumptions, stated plainly because a paper P&L built on
optimistic fills is a lie the operator will only discover with real money:

* A market order fills at the reference price plus slippage **against** Beast -
  buys pay up, sells receive less.
* When a bar's range covers both the stop and the target, the **stop** fills.
  ``exit.stop_fills_first_in_bar`` governs this and defaults to true. Assuming
  the favourable fill is the single most common way a backtest flatters itself.
* A limit order fills only if price traded strictly through it, never merely
  touched it.

Immutable Rule enforcement
--------------------------
Paper mode is enforced here as well as in the runner, and this adapter refuses
to transmit under any configuration, because it has nothing to transmit to.
"""

from __future__ import annotations

import itertools
from datetime import date, datetime
from typing import Any

import pandas as pd

from broker import (
    BrokerClient,
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderType,
    Quote,
)
from core.config import Config, get_config


class PaperBroker(BrokerClient):
    """A deterministic simulated venue.

    Args:
        config: Injected for tests; defaults to the singleton.
        seed_frames: Optional ``{symbol: OHLC frame}`` primed at construction,
            equivalent to calling :meth:`prime` for each entry.

    Attributes:
        orders: Every order this session, keyed by order id. Retained rather
            than discarded so the journal and the reconcile path can both ask
            what was sent, which is the question that matters after a crash.
    """

    name = "paper"

    def __init__(self, config: Config | None = None,
                 seed_frames: dict[str, pd.DataFrame] | None = None) -> None:
        self.cfg = config or get_config()
        self._connected = False
        self._frames: dict[str, pd.DataFrame] = {}
        self._counter = itertools.count(1)
        self.orders: dict[str, dict[str, Any]] = {}
        for symbol, frame in (seed_frames or {}).items():
            self.prime(symbol, frame)

    # -- session -------------------------------------------------------------

    def connect(self) -> bool:
        """Always succeeds. There is nothing to authenticate against."""
        self._connected = True
        return True

    def is_connected(self) -> bool:
        return self._connected

    # -- data ----------------------------------------------------------------

    def prime(self, symbol: str, frame: pd.DataFrame) -> None:
        """Install the OHLC series this adapter will serve for ``symbol``.

        Args:
            frame: Tz-aware OHLC frame. Stored as a copy, so a caller mutating
                its own frame afterwards cannot retroactively change what the
                simulated broker already reported.
        """
        self._frames[symbol.upper()] = frame.copy()

    def history(self, symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
        """Return the last ``bars`` closed candles of the primed series.

        Returns an empty frame when the symbol was never primed, rather than
        raising: an absent series is a data problem for the caller's staleness
        gate (4.6) to report, not an execution error.
        """
        frame = self._frames.get(symbol.upper())
        if frame is None or frame.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        return frame.tail(int(bars)).copy()

    def quote(self, symbol: str) -> Quote | None:
        """Synthesise top-of-book from the last primed bar.

        The spread is taken from ``broker.paper.slippage_pct`` around the close,
        so the 4.6 spread check exercises the same code path it will in live.
        """
        frame = self._frames.get(symbol.upper())
        if frame is None or frame.empty:
            return None
        row = frame.iloc[-1]
        last = float(row["close"])
        half = last * float(self.cfg.get("broker.paper.slippage_pct", 0.0005)) / 2.0
        stamp = frame.index[-1]
        return Quote(
            symbol=symbol.upper(),
            bid=last - half,
            ask=last + half,
            last=last,
            timestamp=stamp.to_pydatetime() if hasattr(stamp, "to_pydatetime") else datetime.now(),
        )

    # -- execution -----------------------------------------------------------

    def place_order(self, request: OrderRequest) -> OrderResult:
        """Accept and simulate an order.

        A market order fills immediately at the reference price adjusted by
        slippage against Beast. A limit order is accepted as working and is
        filled later by :meth:`settle_bar`, never at submission - a limit that
        fills the instant it is placed is the optimistic assumption 6.8 exists
        to forbid.
        """
        order_id = f"paper-{next(self._counter):06d}"
        reference = self._reference_price(request.symbol, request.limit_price)

        if reference is None:
            self.orders[order_id] = {"request": request, "status": "REJECTED"}
            return OrderResult(
                accepted=False,
                order_id=order_id,
                paper=True,
                message=f"no primed price series for {request.symbol}",
            )

        if request.order_type is OrderType.MARKET:
            fill = self._slipped(reference, request.side)
            self.orders[order_id] = {
                "request": request, "status": "FILLED", "fill_price": fill,
            }
            return OrderResult(
                accepted=True,
                order_id=order_id,
                filled_quantity=request.quantity,
                average_price=fill,
                paper=True,
                message="simulated market fill",
            )

        self.orders[order_id] = {"request": request, "status": "WORKING"}
        return OrderResult(
            accepted=True,
            order_id=order_id,
            filled_quantity=0,
            average_price=0.0,
            paper=True,
            message="simulated limit order working",
        )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a working order. Cancelling a filled order is a no-op, not an error."""
        record = self.orders.get(order_id)
        if record is None:
            return False
        if record["status"] == "WORKING":
            record["status"] = "CANCELLED"
            return True
        return False

    def settle_bar(self, symbol: str, bar: pd.Series) -> list[str]:
        """Fill every working limit order that ``bar`` traded through.

        Args:
            bar: One OHLC row, normally the just-closed candle.

        Returns:
            The ids filled by this bar, so the caller can reconcile.

        The comparison is strict (``low < limit`` for a buy, not ``<=``): a bar
        that merely touched the limit is not evidence of a fill, because Beast
        was not necessarily at the front of the queue at that price.
        """
        filled: list[str] = []
        low, high = float(bar["low"]), float(bar["high"])
        for order_id, record in self.orders.items():
            if record["status"] != "WORKING":
                continue
            request: OrderRequest = record["request"]
            if request.symbol.upper() != symbol.upper() or request.limit_price is None:
                continue
            limit = float(request.limit_price)
            through = low < limit if request.side is OrderSide.BUY else high > limit
            if through:
                record.update({"status": "FILLED", "fill_price": limit})
                filled.append(order_id)
        return filled

    def resolve_bar_exit(self, stop_price: float, target_price: float,
                         bar: pd.Series, is_long: bool) -> tuple[str, float] | None:
        """Decide which of a stop and a target a single bar hit (soul file 6.8).

        Returns:
            ``("STOP" | "TARGET", price)``, or ``None`` when the bar touched
            neither.

        When the bar's range covers both, the stop wins while
        ``exit.stop_fills_first_in_bar`` is true. Intrabar sequence is unknown
        from OHLC alone, and resolving the ambiguity in Beast's favour would
        inflate every backtest win rate by exactly the trades that were most
        marginal.
        """
        low, high = float(bar["low"]), float(bar["high"])
        stop_hit = low <= stop_price if is_long else high >= stop_price
        target_hit = high >= target_price if is_long else low <= target_price

        if stop_hit and target_hit:
            if bool(self.cfg.get("exit.stop_fills_first_in_bar", True)):
                return "STOP", stop_price
            return "TARGET", target_price
        if stop_hit:
            return "STOP", stop_price
        if target_hit:
            return "TARGET", target_price
        return None

    # -- capabilities the soul file needs but no venue is configured for ------

    def futures_contracts(self, underlying: str) -> list[tuple[str, date]]:
        """No contract master exists in simulation.

        Raises:
            NotImplementedError: Always. Gold contract specs are BLOCKER 17 -
                venue, multiplier, tick size and tick value are all ``null`` in
                config, so there is nothing to synthesise a contract from and
                guessing one would defeat the refusal that blocker exists for.
        """
        raise NotImplementedError(
            "PaperBroker has no futures contract master. Gold specs are unset "
            "(soul file open item 17: instruments.gold.venue and friends)."
        )

    def capital(self) -> float | None:
        """Simulated equity comes from config, not from an account."""
        return float(self.cfg.get("risk.capital"))

    # -- internals -----------------------------------------------------------

    def _reference_price(self, symbol: str, limit_price: float | None) -> float | None:
        """The price a market order transacts against."""
        frame = self._frames.get(symbol.upper())
        if frame is not None and not frame.empty:
            return float(frame.iloc[-1]["close"])
        return limit_price

    def _slipped(self, price: float, side: OrderSide) -> float:
        """Apply slippage against Beast, never in its favour."""
        pct = float(self.cfg.get("broker.paper.slippage_pct", 0.0005))
        return price * (1.0 + pct) if side is OrderSide.BUY else price * (1.0 - pct)
