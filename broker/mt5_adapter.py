"""MetaTrader 5 execution adapter for XAUUSD.

Implements :class:`~broker.BrokerClient` - the interface the migration brief
calls ``ExecutionAdapter`` - and holds no MT5 client of its own. Every call goes
through :class:`~broker.mt5_connection.MT5Connection`, which serialises, bounds
and materialises it. That split is the point: this module knows about orders and
lots, and nothing about RPyC, Wine or netrefs.

The order path, and why it is this long
---------------------------------------
A market order here is not one call. It is: a fresh tick, a stop-distance check
against the broker's own stops level, a volume rounded **down** to the symbol's
step, an ``order_check`` dry run, and only then ``order_send``. Each step exists
because skipping it produces a rejection that looks like a bridge fault - and an
operator debugging the wrong layer at 03:00 is the failure this design is built
to avoid.

Rounding is always down. Soul file 7.1 sizes a position from a risk budget, so
rounding up spends more risk than the rule allowed. A size that rounds below the
broker's minimum is a **skipped trade**, logged with its reason, never a trade at
the minimum.

Idempotency
-----------
``broker/retry.py`` already refuses to retry a non-idempotent call, and says why:
a submit that timed out may already have reached the exchange. It also says the
fix "belongs to the ops layer, which is not built yet". :meth:`_already_executed`
is that layer for MT5. After a timeout or a disconnect, the account is asked what
happened - positions, working orders, then deal history, all filtered by magic
number and the order's tag - and a resend happens only when all three say the
order is absent.

Stops
-----
The stop loss is attached to the entry request so it rests at the broker. If the
VPS, the bridge or Beast dies between entry and exit, the position is still
protected. Beast continues to manage the stop in its own loop; the server-side
copy is a floor, not a replacement, and it is only ever tightened by
``core/exit_manager.py``'s existing ratchet.

``OrderRequest`` has no stop field, so it travels in ``metadata["sl"]``. Adding
fields to the shared dataclass would touch the Zerodha and paper adapters and the
order executor for a value only this venue can use.
"""

from __future__ import annotations

import logging
import math
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
from broker.mt5_connection import (
    BridgeError,
    BridgeTimeout,
    ConnectionState,
    MT5Connection,
    SymbolSpec,
)
from core.config import Config, get_config

logger = logging.getLogger("beast.broker.mt5")

TIMEFRAME_NAMES = {
    "1M": "TIMEFRAME_M1",
    "3M": "TIMEFRAME_M3",
    "5M": "TIMEFRAME_M5",
    "15M": "TIMEFRAME_M15",
    "30M": "TIMEFRAME_M30",
    "60M": "TIMEFRAME_H1",
    "1H": "TIMEFRAME_H1",
    "4H": "TIMEFRAME_H4",
    "1D": "TIMEFRAME_D1",
}

RATE_FIELDS = ["time", "open", "high", "low", "close",
               "tick_volume", "spread", "real_volume"]

RETCODE_DONE = 10009
RETCODE_DONE_PARTIAL = 10010
#: Only a price that moved is worth re-sending, and only with a fresh tick.
RETCODE_RETRYABLE = {10004, 10020, 10021}
RETCODE_TEXT = {
    10004: "requote",
    10006: "request rejected",
    10012: "request timed out at the broker",
    10013: "invalid request",
    10014: "invalid volume",
    10015: "invalid price",
    10016: "invalid stops - check stops_level",
    10017: "trading disabled",
    10018: "market closed",
    10019: "insufficient funds",
    10020: "price changed",
    10021: "no quotes to process",
    10027: "Algo Trading disabled in the terminal",
    10030: "unsupported filling mode",
    10031: "no connection to the trade server",
}


class MT5Adapter(BrokerClient):
    """XAUUSD spot data and order execution through a MetaTrader 5 terminal.

    Args:
        config: Injected for tests.
        connection: An :class:`MT5Connection`. Injected in tests with a fake
            client; built from the ``gold`` profile otherwise.
    """

    name = "mt5"

    def __init__(self, config: Config | None = None,
                 connection: MT5Connection | None = None) -> None:
        self.cfg = config or get_config()
        self.timezone = str(self.cfg.get("sessions.timezone"))
        self.link = connection or MT5Connection(
            profile=str(self.cfg.get("broker.mt5.active_profile", "gold")),
            config=self.cfg,
        )
        self._bars: dict[tuple[str, str], pd.DataFrame] = {}

    # -- session -------------------------------------------------------------

    def connect(self) -> bool:
        return self.link.connect()

    def is_connected(self) -> bool:
        return self.link.state in (ConnectionState.READY, ConnectionState.DEGRADED)

    def disconnect(self) -> None:
        self.link.disconnect()

    def capital(self) -> float | None:
        """Account equity, for sizing. None when the bridge cannot answer."""
        try:
            account = self.link.call("account_info")
        except BridgeError as error:
            logger.error("MT5 equity unavailable: %s", error)
            return None
        return float(getattr(account, "equity", 0.0) or 0.0)

    # -- data ----------------------------------------------------------------

    def history(self, symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
        """Return the last ``bars`` **closed** candles, tz-aware.

        Position 0 is the bar still forming, so every fetch starts at position 1.
        Beast drops an in-progress candle downstream too (soul file 4.2), but an
        adapter that hands one over is already wrong at its own boundary.

        After the first call only the newest few bars are fetched and merged by
        timestamp. Re-pulling the full window every cycle is the redundant round
        trip this layer exists to remove; the merge keeps the semantics
        identical because bars are keyed by open time and a closed bar never
        changes.
        """
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        key = (symbol.upper(), timeframe.upper())
        cached = self._bars.get(key)
        wanted = int(bars)

        if cached is not None and len(cached) >= wanted:
            count = int(self.cfg.get("broker.mt5.refresh_bars", 5))
        else:
            count = wanted

        fresh = self._fetch_bars(symbol, timeframe, count)
        if fresh is None:
            # Serve the cache rather than an empty frame: one failed refresh is
            # not a reason to blind the engine, and the health check is what
            # decides the bridge is broken.
            return cached.tail(wanted).copy() if cached is not None else empty

        merged = fresh if cached is None else pd.concat([cached, fresh])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        self._bars[key] = merged.tail(max(wanted, count) * 2)
        return merged.tail(wanted).copy()

    def bulk_history(self, symbol: str, timeframe: str,
                     start: datetime, end: datetime) -> pd.DataFrame:
        """Chunked ``copy_rates_range`` for the regime layer and backtests.

        Warns when fewer bars arrive than the range implies: the terminal's "Max
        bars in chart" setting silently caps history, and a model fitted on a
        truncated window is worse than one that failed to fit.
        """
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        name = self.link.resolve_symbol(symbol)
        granularity = self._granularity(timeframe)
        if name is None or granularity is None:
            return empty

        try:
            rates = self.link.call("copy_rates_range", name, granularity, start, end,
                                   call_class="history")
        except BridgeError as error:
            logger.error("MT5 bulk history failed for %s %s: %s", name, timeframe, error)
            return empty

        frame = self._to_frame(rates)
        expected = self._expected_bars(timeframe, start, end)
        if expected and len(frame) < expected * 0.5:
            logger.warning(
                "MT5 returned %d %s bars for %s where the range implies roughly %d. "
                "The terminal's 'Max bars in chart' setting caps history.",
                len(frame), timeframe, name, expected,
            )
        return frame

    def quote(self, symbol: str) -> Quote | None:
        """Top-of-book from a fresh tick."""
        tick = self._tick(symbol)
        if tick is None:
            return None
        bid = float(getattr(tick, "bid", 0.0) or 0.0)
        ask = float(getattr(tick, "ask", 0.0) or 0.0)
        last = float(getattr(tick, "last", 0.0) or 0.0)
        if last <= 0.0:
            # Spot metals quote no last-traded price; the mid is the honest
            # stand-in and Quote.spread still reads the real book.
            last = (bid + ask) / 2.0 if bid > 0 and ask > 0 else max(bid, ask)
        return Quote(
            symbol=symbol.upper(), bid=bid, ask=ask, last=last,
            timestamp=self.server_time(getattr(tick, "time", 0)),
        )

    def _tick(self, symbol: str):
        name = self.link.resolve_symbol(symbol)
        if name is None:
            return None
        try:
            return self.link.call("symbol_info_tick", name)
        except BridgeError as error:
            logger.error("MT5 tick failed for %s: %s", name, error)
            return None

    def _fetch_bars(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame | None:
        name = self.link.resolve_symbol(symbol)
        granularity = self._granularity(timeframe)
        if name is None or granularity is None:
            return None
        try:
            rates = self.link.call("copy_rates_from_pos", name, granularity, 1, int(count),
                                   call_class="history")
        except BridgeError as error:
            logger.error("MT5 history failed for %s %s: %s", name, timeframe, error)
            return None
        frame = self._to_frame(rates)
        return frame if not frame.empty else None

    def _granularity(self, timeframe: str):
        constant = TIMEFRAME_NAMES.get(timeframe.upper())
        if constant is None:
            logger.error("No MT5 timeframe for the label %r", timeframe)
            return None
        value = getattr(self.link._client, constant, None)
        if value is None:
            logger.error("The MT5 client has no %s", constant)
        return value

    def _to_frame(self, rates) -> pd.DataFrame:
        """Convert an MT5 rate array into the frame the indicator engine expects."""
        if rates is None or len(rates) == 0:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        frame = pd.DataFrame(rates)
        if "time" not in frame.columns:
            # A bridge can hand back a sequence with the dtype names stripped.
            frame = pd.DataFrame(list(rates), columns=RATE_FIELDS)

        stamps = pd.to_datetime(frame["time"], unit="s", utc=True)
        stamps = stamps - pd.Timedelta(hours=self.link.facts.server_utc_offset_hours)
        frame.index = stamps.dt.tz_convert(self.timezone)

        volume = frame.get("real_volume")
        if volume is None or float(pd.Series(volume).abs().sum()) == 0.0:
            volume = frame.get("tick_volume", 0)
        frame["volume"] = volume

        return frame[["open", "high", "low", "close", "volume"]].astype(float).sort_index()

    def server_time(self, epoch_seconds) -> datetime:
        """Convert an MT5 server-clock epoch into a tz-aware Beast timestamp."""
        stamp = pd.Timestamp(int(epoch_seconds or 0), unit="s", tz="UTC")
        stamp = stamp - pd.Timedelta(hours=self.link.facts.server_utc_offset_hours)
        return stamp.tz_convert(self.timezone).to_pydatetime()

    @staticmethod
    def _expected_bars(timeframe: str, start: datetime, end: datetime) -> int:
        minutes = {"1M": 1, "3M": 3, "5M": 5, "15M": 15, "30M": 30,
                   "60M": 60, "1H": 60, "4H": 240, "1D": 1440}.get(timeframe.upper())
        if not minutes:
            return 0
        span = (end - start).total_seconds() / 60.0
        return int(span / minutes)

    # -- execution -----------------------------------------------------------

    def place_order(self, request: OrderRequest) -> OrderResult:
        """Validate, dry-run, then transmit a market order.

        ``metadata`` carries what :class:`OrderRequest` has no field for:
        ``sl`` and ``tp`` as underlying prices, ``volume_lots`` when the sizing
        layer already knows the lot size, and ``trade_id`` for reconciliation.
        """
        if not self.is_connected():
            return OrderResult(False, paper=True, message="MT5 bridge is not connected")

        if request.order_type is not OrderType.MARKET:
            return OrderResult(
                False, paper=False,
                message=f"{request.order_type.value} orders are not implemented on MT5",
            )

        spec = self.link.spec(request.symbol)
        if spec is None:
            return OrderResult(False, paper=False,
                               message=f"no MT5 symbol for {request.symbol}")

        volume = self._volume_for(request, spec)
        if volume is None:
            requested = request.metadata.get("volume_lots", request.quantity)
            logger.warning(
                "Skipping %s: size %s rounds below the %s minimum of %s lots. "
                "Rounding up would spend more than the risk budget allows.",
                request.symbol, requested, spec.name, spec.volume_min,
            )
            return OrderResult(
                False, paper=False,
                message=f"size {requested} rounds below the minimum {spec.volume_min} lots",
            )

        return self._send_with_retries(request, spec, volume)

    def _send_with_retries(self, request: OrderRequest, spec: SymbolSpec,
                           volume: float) -> OrderResult:
        """Send, and re-price at most ``retry.max_attempts`` times on a requote."""
        attempts = int(self.cfg.get("broker.mt5.retry.max_attempts", 3))
        comment = self._comment(request)
        last: OrderResult | None = None

        for attempt in range(1, attempts + 1):
            tick = self._tick(request.symbol)
            if tick is None:
                return OrderResult(False, paper=False,
                                   message="no fresh tick - refusing to price the order")

            is_buy = request.side is OrderSide.BUY
            price = float(tick.ask if is_buy else tick.bid)
            stops = self._stops_for(request, spec, price, is_buy)
            if isinstance(stops, str):
                return OrderResult(False, paper=False, message=stops)

            payload = {
                "action": self.link._client.TRADE_ACTION_DEAL,
                "symbol": spec.name,
                "volume": volume,
                "type": (self.link._client.ORDER_TYPE_BUY if is_buy
                         else self.link._client.ORDER_TYPE_SELL),
                "price": price,
                "deviation": int(self.cfg.get("broker.mt5.deviation_points", 20)),
                "magic": self._magic(),
                "comment": comment,
                "type_time": self.link._client.ORDER_TIME_GTC,
                "type_filling": self._filling_mode(spec),
            }
            payload.update(stops)

            checked = self._order_check(payload)
            if isinstance(checked, str):
                return OrderResult(False, paper=False, message=checked)

            try:
                result = self.link.call("order_send", payload, call_class="fast")
            except BridgeTimeout as error:
                # The order may have reached the broker anyway. Ask the account
                # before ever considering a resend.
                executed = self._already_executed(spec.name, comment)
                if executed is not None:
                    logger.warning("order_send timed out but the order is live: %s", comment)
                    return executed
                return OrderResult(False, paper=False,
                                   message=f"order_send timed out and no matching order "
                                           f"exists: {error}")
            except BridgeError as error:
                return OrderResult(False, paper=False, message=str(error))

            retcode = int(getattr(result, "retcode", -1))
            if retcode in (RETCODE_DONE, RETCODE_DONE_PARTIAL):
                return OrderResult(
                    accepted=True,
                    order_id=str(getattr(result, "order", "")),
                    filled_quantity=request.quantity,
                    average_price=float(getattr(result, "price", 0.0) or 0.0),
                    paper=False,
                    message=f"{'DEMO' if self.link.facts.is_demo else 'LIVE'} fill: "
                            f"{getattr(result, 'volume', volume)} lots of {spec.name} "
                            f"at {getattr(result, 'price', 0.0)}",
                )

            last = OrderResult(
                False, paper=False,
                message=f"retcode {retcode} ({RETCODE_TEXT.get(retcode, 'unknown')}): "
                        f"{getattr(result, 'comment', '')}",
            )
            if retcode not in RETCODE_RETRYABLE:
                logger.error("MT5 rejected %s: %s", spec.name, last.message)
                return last
            logger.warning("MT5 %s on attempt %d/%d - re-pricing",
                           RETCODE_TEXT.get(retcode, retcode), attempt, attempts)

        return last or OrderResult(False, paper=False, message="order was not sent")

    def _order_check(self, payload: dict) -> dict | str:
        """Dry-run the order. Returns the payload, or a refusal message."""
        try:
            checked = self.link.call("order_check", payload)
        except BridgeError as error:
            logger.warning("order_check unavailable (%s) - sending unchecked", error)
            return payload
        retcode = int(getattr(checked, "retcode", 0))
        if retcode not in (0, RETCODE_DONE):
            return (f"order_check refused: retcode {retcode} "
                    f"({RETCODE_TEXT.get(retcode, 'unknown')}) "
                    f"{getattr(checked, 'comment', '')}")
        return payload

    def _already_executed(self, name: str, comment: str) -> OrderResult | None:
        """Ask the account whether an order actually landed (spec step 4).

        Checked in the order a timed-out submit can appear: a filled position, a
        working order, then the deal history for something that filled and
        closed. Anything found means the order exists and must not be resent.
        """
        magic = self._magic()

        for method in ("positions_get", "orders_get"):
            try:
                rows = self.link.call(method, symbol=name, allow_none=True) or ()
            except BridgeError:
                continue
            for row in rows:
                if int(getattr(row, "magic", -1)) == magic and \
                        str(getattr(row, "comment", "")) == comment:
                    return OrderResult(
                        accepted=True,
                        order_id=str(getattr(row, "ticket", "")),
                        average_price=float(getattr(row, "price_open", 0.0) or 0.0),
                        paper=False,
                        message=f"recovered from {method}: the order had executed",
                    )

        try:
            deals = self.link.call("history_deals_get", allow_none=True,
                                   call_class="history") or ()
        except BridgeError:
            deals = ()
        for deal in deals:
            if int(getattr(deal, "magic", -1)) == magic and \
                    str(getattr(deal, "comment", "")) == comment:
                return OrderResult(
                    accepted=True,
                    order_id=str(getattr(deal, "order", "")),
                    average_price=float(getattr(deal, "price", 0.0) or 0.0),
                    paper=False,
                    message="recovered from deal history: the order had executed",
                )
        return None

    def close_position(self, ticket: int) -> OrderResult:
        """Close one position by ticket, on netting and hedging accounts alike.

        Passing ``position`` pins the deal to that ticket, which is what a
        hedging account needs to close the intended leg rather than opening an
        offsetting one. On a netting account there is only ever one position per
        symbol and the same request closes it, so one code path serves both.
        """
        try:
            positions = self.link.call("positions_get", allow_none=True) or ()
        except BridgeError as error:
            return OrderResult(False, paper=False, message=f"positions_get failed: {error}")

        position = next((row for row in positions
                         if int(getattr(row, "ticket", -1)) == int(ticket)), None)
        if position is None:
            return OrderResult(False, paper=False, message=f"position {ticket} is not open")

        name = str(position.symbol)
        spec = self.link.spec(name) or self.link.spec("XAUUSD")
        tick = self.link.call("symbol_info_tick", name)
        was_buy = int(position.type) == int(self.link._client.ORDER_TYPE_BUY)

        payload = {
            "action": self.link._client.TRADE_ACTION_DEAL,
            "symbol": name,
            "volume": float(position.volume),
            "type": (self.link._client.ORDER_TYPE_SELL if was_buy
                     else self.link._client.ORDER_TYPE_BUY),
            "position": int(ticket),
            "price": float(tick.bid if was_buy else tick.ask),
            "deviation": int(self.cfg.get("broker.mt5.deviation_points", 20)),
            "magic": self._magic(),
            "comment": "beast close",
            "type_time": self.link._client.ORDER_TIME_GTC,
            "type_filling": self._filling_mode(spec) if spec else 1,
        }

        try:
            result = self.link.call("order_send", payload)
        except BridgeError as error:
            return OrderResult(False, paper=False, message=f"close failed: {error}")

        retcode = int(getattr(result, "retcode", -1))
        if retcode in (RETCODE_DONE, RETCODE_DONE_PARTIAL):
            return OrderResult(True, str(getattr(result, "order", "")), paper=False,
                               message=f"closed {ticket} at {getattr(result, 'price', 0.0)}")
        return OrderResult(False, paper=False,
                           message=f"close retcode {retcode} "
                                   f"({RETCODE_TEXT.get(retcode, 'unknown')})")

    def cancel_order(self, order_id: str) -> bool:
        """Remove a working order by ticket. A filled deal cannot be cancelled."""
        try:
            ticket = int(order_id)
        except (TypeError, ValueError):
            logger.error("MT5 order ids are numeric tickets, got %r", order_id)
            return False
        try:
            result = self.link.call("order_send", {
                "action": self.link._client.TRADE_ACTION_REMOVE,
                "order": ticket,
            })
        except BridgeError as error:
            logger.error("MT5 cancel failed for %s: %s", ticket, error)
            return False
        return int(getattr(result, "retcode", -1)) == RETCODE_DONE

    def futures_contracts(self, underlying: str) -> list[tuple[str, date]]:
        """MT5 serves XAUUSD as a spot CFD - no contract, no expiry, no roll.

        Raises:
            NotImplementedError: Always. ``data/market_data.py`` catches this and
                records no contracts, which is the truth. Synthesising an expiry
                would feed the rollover checks a date that does not exist.
        """
        raise NotImplementedError(
            "MT5 serves XAUUSD as a spot CFD: no futures contract, no expiry, no roll."
        )

    # -- internals -----------------------------------------------------------

    def _magic(self) -> int:
        return int(self.cfg.get("broker.mt5.magic", 20260910))

    def _comment(self, request: OrderRequest) -> str:
        """Beast's trade id, trimmed to the broker's comment limit.

        The comment is half of the reconciliation key, so it has to survive the
        round trip intact - a comment the broker truncates cannot be matched
        against afterwards.
        """
        limit = int(self.cfg.get("broker.mt5.comment_limit", 31))
        text = str(request.metadata.get("trade_id") or request.tag or "beast")
        return text[:limit]

    def _volume_for(self, request: OrderRequest, spec: SymbolSpec) -> float | None:
        """Round a requested size **down** onto the symbol's lot grid.

        Returns:
            The lot size, or None when it falls below the broker's minimum -
            which is a skipped trade, not a trade at the minimum. Soul file 7.1
            rejects a sub-minimum size rather than inflating it, and that rule
            does not stop being true because the venue changed.
        """
        step = spec.volume_step or 0.01
        minimum = spec.volume_min or step

        override = request.metadata.get("volume_lots")
        requested = float(override) if override is not None else float(request.quantity) * minimum

        volume = round(math.floor(requested / step) * step, 8)
        if volume < minimum:
            return None
        if spec.volume_max and volume > spec.volume_max:
            volume = round(math.floor(spec.volume_max / step) * step, 8)
        return volume

    def _stops_for(self, request: OrderRequest, spec: SymbolSpec,
                   price: float, is_buy: bool) -> dict | str:
        """Build the server-side SL/TP, or explain why they are unusable.

        The broker enforces a minimum distance (``stops_level``) and a band
        around the market where stops cannot be touched at all
        (``freeze_level``). A stop inside either is rejected with retcode 10016,
        which reads like a bridge fault unless it is caught here.
        """
        stops: dict[str, float] = {}
        minimum_distance = max(spec.stops_level, spec.freeze_level) * spec.point

        for field_name, key in (("sl", "sl"), ("tp", "tp")):
            raw = request.metadata.get(key)
            if raw is None:
                continue
            level = float(raw)

            if key == "sl":
                wrong_side = level >= price if is_buy else level <= price
            else:
                wrong_side = level <= price if is_buy else level >= price
            if wrong_side:
                return (f"{key} {level} is on the wrong side of {price} for a "
                        f"{'buy' if is_buy else 'sell'}")

            if minimum_distance and abs(price - level) < minimum_distance:
                return (f"{key} {level} is {abs(price - level):.5f} from {price}, "
                        f"inside the broker's minimum of {minimum_distance:.5f} "
                        f"(stops_level {spec.stops_level}, freeze_level {spec.freeze_level})")

            stops[field_name] = round(level, spec.digits or 2)

        return stops

    def _filling_mode(self, spec: SymbolSpec) -> int:
        """Pick a filling mode the symbol actually allows.

        The symbol's mask uses FOK=1, IOC=2, BOC=4 while the request's
        ``type_filling`` is an enum FOK=0, IOC=1, RETURN=2. Passing the mask
        straight through is the classic cause of retcode 10030 on an order that
        is otherwise valid.
        """
        client = self.link._client
        if spec.filling_mask & 1:
            return int(getattr(client, "ORDER_FILLING_FOK", 0))
        if spec.filling_mask & 2:
            return int(getattr(client, "ORDER_FILLING_IOC", 1))
        return int(getattr(client, "ORDER_FILLING_RETURN", 2))
