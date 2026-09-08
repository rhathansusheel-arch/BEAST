"""Zerodha Kite Connect adapter - Nifty/Sensex spot and the weekly option chain.

This is the adapter that matters most for the option path, because three of the
soul file's blockers live at this boundary: lot size, strike interval and the
liquidity floors. Open item 18 recommends reading lot sizes and strike ladders
from the broker rather than config, since the exchange changes them by
notification - :meth:`ZerodhaClient.contract_specs` does exactly that, and
``main.py`` uses it to fill the config gaps at startup when
``prefer_broker_contract_master`` is set.

Greeks: Kite's instrument feed does not publish delta. Beast needs delta for the
5.7.2 band and for the 7.1 sizing formula, so :func:`black_scholes_delta`
computes it from the quoted IV. That is an approximation and it is labelled as
one; where the broker supplies a delta, that value wins.
"""

from __future__ import annotations

import logging
import math
from datetime import date, datetime, timedelta

import pandas as pd

from broker import BrokerClient, OrderRequest, OrderResult, OrderSide, OrderType, Quote
from core.config import Config, get_config
from core.option_chain import ChainSnapshot, OptionQuote, slice_around_atm

logger = logging.getLogger("beast.broker.zerodha")

# Kite interval names keyed by the soul file's timeframe labels.
INTERVAL_MAP = {
    "1M": "minute",
    "3M": "3minute",
    "5M": "5minute",
    "15M": "15minute",
    "30M": "30minute",
    "60M": "60minute",
    "1D": "day",
}

SPOT_SYMBOLS = {
    "NIFTY50": ("NSE", "NIFTY 50"),
    "SENSEX": ("BSE", "SENSEX"),
}

CHAIN_NAMES = {"NIFTY50": "NIFTY", "SENSEX": "SENSEX"}


def black_scholes_delta(spot: float, strike: float, iv_pct: float, dte_days: float,
                        option_type: str, rate: float = 0.065) -> float:
    """Black-Scholes delta, used when the feed does not publish one.

    Args:
        spot: Underlying price.
        strike: Strike price.
        iv_pct: Implied volatility in percent, as the chain reports it.
        dte_days: Calendar days to expiry. Intraday fractions matter enormously
            on expiry day, so callers should pass a fractional value there.
        option_type: ``"CE"`` or ``"PE"``.
        rate: Risk-free rate.

    Returns:
        Delta as a positive magnitude for calls and a negative value for puts.
        Returns the intrinsic-value delta (1/0) when time or volatility is zero,
        which is the correct limit and keeps expiry-day selection sane.
    """
    if spot <= 0 or strike <= 0:
        return 0.0
    sigma = max(iv_pct, 0.0) / 100.0
    tau = max(dte_days, 0.0) / 365.0

    if sigma <= 0 or tau <= 0:
        if option_type == "CE":
            return 1.0 if spot > strike else 0.0
        return -1.0 if spot < strike else 0.0

    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma**2) * tau) / (sigma * math.sqrt(tau))
    cdf = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
    return cdf if option_type == "CE" else cdf - 1.0


class ZerodhaClient(BrokerClient):
    """Kite Connect adapter.

    Args:
        config: Injected for tests.
        kite: Injected ``KiteConnect`` instance - tests pass a stub, and this is
            also how a shared session is reused across adapters.
    """

    name = "zerodha"

    def __init__(self, config: Config | None = None, kite: object | None = None) -> None:
        self.cfg = config or get_config()
        self.timezone = str(self.cfg.get("sessions.timezone"))
        self._kite = kite
        self._instruments: pd.DataFrame | None = None
        self._token_cache: dict[tuple[str, str], int] = {}

    # -- session -------------------------------------------------------------

    def connect(self) -> bool:
        """Construct the Kite client from credentials and load the contract master."""
        if self._kite is not None:
            return True
        try:
            from kiteconnect import KiteConnect
        except ImportError:
            logger.error("kiteconnect is not installed; Nifty/Sensex data unavailable")
            return False

        api_key = self.cfg.credential("zerodha.api_key", "ZERODHA_API_KEY")
        access_token = self.cfg.credential("zerodha.access_token", "ZERODHA_ACCESS_TOKEN")
        if not api_key or not access_token:
            logger.error(
                "Zerodha credentials missing. Set ZERODHA_API_KEY and ZERODHA_ACCESS_TOKEN "
                "(the access token is refreshed daily by the Kite login flow)."
            )
            return False

        self._kite = KiteConnect(api_key=api_key)
        self._kite.set_access_token(access_token)
        return self._load_instruments()

    def is_connected(self) -> bool:
        return self._kite is not None

    def _load_instruments(self) -> bool:
        """Cache the NFO/BFO contract master for the session."""
        try:
            segment = str(self.cfg.get("broker.zerodha.exchange_index"))
            self._instruments = pd.DataFrame(self._kite.instruments(segment))
            return not self._instruments.empty
        except Exception as error:
            logger.error("Failed to load the Zerodha contract master: %s", error)
            return False

    # -- reference data ------------------------------------------------------

    def contract_specs(self, market: str) -> dict[str, float] | None:
        """Read lot size and strike interval from the contract master.

        This is what open item 18 recommends over hardcoding them in config,
        since both change by exchange notification.

        Returns:
            ``{"lot_size": ..., "strike_interval": ...}``, or ``None`` when the
            master is unavailable.
        """
        if self._instruments is None or self._instruments.empty:
            return None
        name = CHAIN_NAMES.get(market.upper())
        if name is None:
            return None

        rows = self._instruments[
            (self._instruments["name"] == name)
            & (self._instruments["instrument_type"].isin(["CE", "PE"]))
        ]
        if rows.empty:
            return None

        lot_size = int(rows["lot_size"].mode().iloc[0])
        strikes = sorted({float(value) for value in rows["strike"].unique() if value > 0})
        if len(strikes) < 2:
            return {"lot_size": lot_size, "strike_interval": 0.0}
        gaps = [round(b - a, 4) for a, b in zip(strikes, strikes[1:])]
        interval = float(pd.Series(gaps).mode().iloc[0])
        return {"lot_size": lot_size, "strike_interval": interval}

    def expiries(self, underlying: str) -> list[date]:
        """Available option expiries for ``underlying``, nearest first."""
        if self._instruments is None:
            return []
        name = CHAIN_NAMES.get(underlying.upper())
        rows = self._instruments[
            (self._instruments["name"] == name)
            & (self._instruments["instrument_type"].isin(["CE", "PE"]))
        ]
        if rows.empty:
            return []
        values = sorted({pd.Timestamp(value).date() for value in rows["expiry"].unique()})
        return [value for value in values if value >= datetime.now().date()]

    # -- market data ---------------------------------------------------------

    def _token(self, exchange: str, symbol: str) -> int | None:
        """Resolve an instrument token, caching the lookup."""
        key = (exchange, symbol)
        if key in self._token_cache:
            return self._token_cache[key]
        try:
            data = self._kite.ltp([f"{exchange}:{symbol}"])
            token = int(data[f"{exchange}:{symbol}"]["instrument_token"])
        except Exception as error:
            logger.error("Token lookup failed for %s:%s - %s", exchange, symbol, error)
            return None
        self._token_cache[key] = token
        return token

    def history(self, symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
        """Fetch historical candles for an index spot symbol.

        Returns:
            A tz-aware OHLC frame in IST. Empty when the fetch fails - callers
            treat an empty frame as "not enough history", which suppresses
            entries rather than guessing.
        """
        exchange, tradingsymbol = SPOT_SYMBOLS.get(symbol.upper(), ("NSE", symbol))
        token = self._token(exchange, tradingsymbol)
        interval = INTERVAL_MAP.get(timeframe.upper())
        if token is None or interval is None:
            return pd.DataFrame()

        # Request generously: Kite counts calendar days, not bars, and weekends
        # and holidays would otherwise short the window.
        minutes = _timeframe_minutes(timeframe)
        days = max(2, int((bars * minutes) / (6.25 * 60)) + 3)
        end = datetime.now()
        start = end - timedelta(days=days)

        try:
            raw = self._kite.historical_data(token, start, end, interval)
        except Exception as error:
            logger.error("Historical fetch failed for %s: %s", symbol, error)
            return pd.DataFrame()

        return _to_frame(raw, self.timezone).tail(bars)

    def quote(self, symbol: str) -> Quote | None:
        """Top-of-book for an index spot or option symbol."""
        exchange, tradingsymbol = SPOT_SYMBOLS.get(symbol.upper(), ("NFO", symbol))
        key = f"{exchange}:{tradingsymbol}"
        try:
            data = self._kite.quote([key])[key]
        except Exception as error:
            logger.error("Quote failed for %s: %s", key, error)
            return None

        depth = data.get("depth", {})
        bids, asks = depth.get("buy", []), depth.get("sell", [])
        return Quote(
            symbol=symbol,
            bid=float(bids[0]["price"]) if bids else 0.0,
            ask=float(asks[0]["price"]) if asks else 0.0,
            last=float(data.get("last_price", 0.0)),
            timestamp=datetime.now(),
        )

    def option_chain(self, underlying: str, expiry: date) -> ChainSnapshot | None:
        """Build an ATM +/- N strike snapshot for ``expiry`` (soul file 4.7).

        Delta is taken from the feed when present and computed from IV otherwise.
        OI change is measured against the session-open OI the caller has cached;
        Kite reports today's OI and the previous day's close, so the intraday
        addition is derived from the two.
        """
        if self._instruments is None:
            return None
        name = CHAIN_NAMES.get(underlying.upper())
        if name is None:
            return None

        spot_quote = self.quote(underlying)
        if spot_quote is None or spot_quote.last <= 0:
            return None
        spot = spot_quote.last

        rows = self._instruments[
            (self._instruments["name"] == name)
            & (self._instruments["instrument_type"].isin(["CE", "PE"]))
            & (pd.to_datetime(self._instruments["expiry"]).dt.date == expiry)
        ]
        if rows.empty:
            return None

        instrument = self.cfg.instrument_key(underlying)
        interval = self.cfg.get(f"instruments.{instrument}.strike_interval", None)
        if interval in (None, 0):
            specs = self.contract_specs(underlying) or {}
            interval = specs.get("strike_interval") or 0.0
        n_strikes = int(self.cfg.get("options.chain_snapshot_strikes"))

        wanted = rows[abs(rows["strike"] - spot) <= n_strikes * float(interval) + 1e-9]
        if wanted.empty:
            return None

        segment = str(self.cfg.get("broker.zerodha.exchange_index"))
        keys = [f"{segment}:{symbol}" for symbol in wanted["tradingsymbol"]]
        try:
            quotes = self._kite.quote(keys)
        except Exception as error:
            logger.error("Chain quote failed: %s", error)
            return None

        dte_days = max(0.0, (expiry - datetime.now().date()).days)
        parsed: list[OptionQuote] = []
        for _, row in wanted.iterrows():
            key = f"{segment}:{row['tradingsymbol']}"
            data = quotes.get(key)
            if not data:
                continue
            depth = data.get("depth", {})
            bids, asks = depth.get("buy", []), depth.get("sell", [])
            iv = float(data.get("implied_volatility", 0.0) or 0.0)
            option_type = str(row["instrument_type"])
            delta = float(data.get("delta", 0.0) or 0.0)
            if delta == 0.0:
                delta = black_scholes_delta(
                    spot, float(row["strike"]), iv, dte_days, option_type
                )
            oi = int(data.get("oi", 0) or 0)
            parsed.append(
                OptionQuote(
                    strike=float(row["strike"]),
                    option_type=option_type,
                    bid=float(bids[0]["price"]) if bids else 0.0,
                    ask=float(asks[0]["price"]) if asks else 0.0,
                    oi=oi,
                    oi_change=oi - int(data.get("oi_day_low", oi) or oi),
                    volume=int(data.get("volume", 0) or 0),
                    iv=iv,
                    delta=delta,
                    tradingsymbol=str(row["tradingsymbol"]),
                )
            )

        return ChainSnapshot(
            underlying=underlying,
            spot=spot,
            expiry=expiry,
            taken_at=datetime.now(),
            quotes=slice_around_atm(parsed, spot, float(interval), n_strikes),
        )

    # -- orders --------------------------------------------------------------

    def place_order(self, request: OrderRequest) -> OrderResult:
        """Place an order, or simulate it in paper mode.

        Paper mode is checked here as well as in the runner. Defence in depth is
        appropriate for the one function in the system that can lose money.
        """
        if self.cfg.is_paper or bool(self.cfg.get("broker.zerodha.paper", True)):
            return OrderResult(
                accepted=True,
                order_id=f"paper-{request.tag or request.symbol}",
                filled_quantity=request.quantity,
                average_price=request.limit_price or 0.0,
                paper=True,
                message="paper mode - no order transmitted",
            )
        if self.cfg.get("instruments.nifty.option_side") != "long_only":
            return OrderResult(False, paper=False, message="option_side must be long_only")

        try:
            order_id = self._kite.place_order(
                variety=self._kite.VARIETY_REGULAR,
                exchange=str(self.cfg.get("broker.zerodha.exchange_index")),
                tradingsymbol=request.symbol,
                transaction_type=(
                    self._kite.TRANSACTION_TYPE_BUY
                    if request.side is OrderSide.BUY
                    else self._kite.TRANSACTION_TYPE_SELL
                ),
                quantity=request.quantity,
                product=self._kite.PRODUCT_MIS,
                order_type=(
                    self._kite.ORDER_TYPE_MARKET
                    if request.order_type is OrderType.MARKET
                    else self._kite.ORDER_TYPE_LIMIT
                ),
                price=request.limit_price,
                tag=request.tag[:20] if request.tag else None,
            )
            return OrderResult(True, str(order_id), paper=False, message="submitted")
        except Exception as error:
            logger.error("Order placement failed: %s", error)
            return OrderResult(False, paper=False, message=str(error))

    def cancel_order(self, order_id: str) -> bool:
        if self.cfg.is_paper:
            return True
        try:
            self._kite.cancel_order(variety=self._kite.VARIETY_REGULAR, order_id=order_id)
            return True
        except Exception as error:
            logger.error("Cancel failed for %s: %s", order_id, error)
            return False

    def capital(self) -> float | None:
        """Available equity margin, for live sizing."""
        if self.cfg.is_paper or self._kite is None:
            return None
        try:
            margins = self._kite.margins("equity")
            return float(margins["available"]["live_balance"])
        except Exception as error:
            logger.error("Margin fetch failed: %s", error)
            return None


def _timeframe_minutes(timeframe: str) -> int:
    """Minutes in a timeframe label such as ``"15M"``."""
    text = timeframe.upper()
    if text.endswith("M"):
        return int(text[:-1])
    if text.endswith("H"):
        return int(text[:-1]) * 60
    return 375  # one Indian session


def _to_frame(rows: list[dict], timezone: str) -> pd.DataFrame:
    """Convert Kite candles into the OHLC frame the indicator engine expects."""
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.set_index("date").sort_index()
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize(timezone)
    else:
        frame.index = frame.index.tz_convert(timezone)
    columns = ["open", "high", "low", "close"]
    if "volume" in frame.columns:
        columns.append("volume")
    return frame[columns].astype(float)
