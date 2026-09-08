"""Market data assembly - turns broker feeds into the pipeline's ``FeedState``.

Responsibilities:

* Pull the three cascade timeframes for each instrument (soul file 4.3) and hold
  them in memory, trimmed to ``data.history_bars``.
* Enforce the closed-candle rule at the boundary (4.2): a partially formed bar
  never reaches the indicator engine.
* Pull the option chain on each setup-TF close for Nifty/Sensex, and keep the
  trailing ATM IV history the percentile in 4.7.4 needs.
* Track the median ATR the volatility-adjusted sizing in section 7 requires.

The provider is injected, so the same assembly code serves live trading, the
backtester (which replays a stored 1-minute series) and the tests (which use a
synthetic provider).
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from broker import BrokerClient
from core.config import Config, get_config
from core.option_chain import ChainSnapshot
from core.signal_generator import FeedState
from data.feature_engineering import atr, parse_timeframe, resample_ohlc, validate_ohlc

logger = logging.getLogger("beast.data")


@dataclass
class InstrumentFeed:
    """In-memory state for one instrument.

    Attributes:
        frames: OHLC frames keyed by ``"bias"``, ``"setup"``, ``"trigger"``.
        chain: Latest chain snapshot, refreshed on setup-TF closes.
        iv_history: Trailing daily ATM IV, for the percentile.
        atr_history: Trailing daily ATR, for the volatility factor.
    """

    market: str
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    chain: ChainSnapshot | None = None
    expiries: list[date] = field(default_factory=list)
    contracts: list[tuple[str, date]] = field(default_factory=list)
    iv_history: deque = field(default_factory=lambda: deque(maxlen=30))
    atr_history: deque = field(default_factory=lambda: deque(maxlen=20))
    last_setup_stamp: pd.Timestamp | None = None
    spread: float | None = None


class MarketDataService:
    """Assembles :class:`~core.signal_generator.FeedState` for each instrument.

    Args:
        brokers: Broker adapters keyed by the names in ``broker.routing``.
        config: Injected for tests.
    """

    def __init__(self, brokers: dict[str, BrokerClient], config: Config | None = None) -> None:
        self.cfg = config or get_config()
        self.brokers = brokers
        self.feeds: dict[str, InstrumentFeed] = {}
        self.history_bars = int(self.cfg.get("data.history_bars"))
        self.timezone = str(self.cfg.get("sessions.timezone"))

    # -- routing -------------------------------------------------------------

    def broker_for(self, market: str) -> BrokerClient | None:
        name = self.cfg.get("broker.routing").get(market.upper())
        return self.brokers.get(name) if name else None

    def feed(self, market: str) -> InstrumentFeed:
        """Return (creating if needed) the in-memory feed for ``market``."""
        if market not in self.feeds:
            self.feeds[market] = InstrumentFeed(market=market)
        return self.feeds[market]

    # -- refresh -------------------------------------------------------------

    def refresh(self, market: str, now: datetime) -> FeedState | None:
        """Pull fresh data and build a :class:`FeedState`.

        Returns:
            ``None`` when the trigger timeframe has no data at all - there is
            nothing to evaluate and nothing to report beyond the log line.
        """
        broker = self.broker_for(market)
        if broker is None:
            logger.error("No broker configured for %s", market)
            return None

        state = self.feed(market)
        timeframes = self.cfg.timeframes(market)

        for role in ("bias", "setup", "trigger"):
            label = timeframes[role]
            frame = broker.history(market, label, self.history_bars)
            if frame is None or frame.empty:
                continue
            frame = self._trim_open_bar(frame, label, now)
            if not frame.empty:
                state.frames[role] = frame

        if "trigger" not in state.frames or state.frames["trigger"].empty:
            return None

        setup_df = state.frames.get("setup", pd.DataFrame())
        if not setup_df.empty:
            self._track_atr(state, setup_df)
            stamp = setup_df.index[-1]
            if state.last_setup_stamp != stamp:
                state.last_setup_stamp = stamp
                self._refresh_chain(state, broker, market, now)

        quote = broker.quote(market)
        state.spread = quote.spread if quote else None

        if self.cfg.market_family(market) == "gold" and not state.contracts:
            try:
                state.contracts = broker.futures_contracts(market)
            except NotImplementedError:
                state.contracts = []

        return FeedState(
            bias_df=state.frames.get("bias", pd.DataFrame()),
            setup_df=setup_df,
            trigger_df=state.frames["trigger"],
            spread=state.spread,
            chain=state.chain,
            expiries=state.expiries,
            contracts=state.contracts,
            iv_history=list(state.iv_history),
            atr_median=self._median_atr(state),
        )

    def _trim_open_bar(self, frame: pd.DataFrame, timeframe: str,
                       now: datetime) -> pd.DataFrame:
        """Drop a trailing in-progress candle (soul file 4.2).

        Entry decisions are never made from an in-progress candle. Doing this at
        the data boundary means no downstream component has to remember to.
        """
        if frame.empty:
            return frame
        interval = parse_timeframe(timeframe)
        stamp = pd.Timestamp(now)
        if stamp.tz is None:
            stamp = stamp.tz_localize(self.timezone)
        else:
            stamp = stamp.tz_convert(self.timezone)
        last_open = frame.index[-1]
        if stamp < last_open + interval:
            return frame.iloc[:-1]
        return frame

    def _refresh_chain(self, state: InstrumentFeed, broker: BrokerClient,
                       market: str, now: datetime) -> None:
        """Pull a chain snapshot for the nearest weekly expiry (4.7)."""
        if self.cfg.market_family(market) != "indian":
            return
        try:
            state.expiries = broker.expiries(market)
        except NotImplementedError:
            return
        if not state.expiries:
            return

        expiry = next((value for value in state.expiries if value >= now.date()), None)
        if expiry is None:
            return
        try:
            snapshot = broker.option_chain(market, expiry)
        except NotImplementedError:
            return
        except Exception as error:
            logger.error("Chain fetch failed for %s: %s", market, error)
            return

        if snapshot is None:
            return
        state.chain = snapshot

        # One ATM IV sample per session feeds the 30-session percentile.
        from core.option_chain import ChainAnalyzer

        atm_iv = ChainAnalyzer(self.cfg).atm_iv(snapshot)
        if atm_iv:
            if not state.iv_history or state.iv_history[-1] != atm_iv:
                state.iv_history.append(atm_iv)

    def _track_atr(self, state: InstrumentFeed, setup_df: pd.DataFrame) -> None:
        """Maintain the trailing ATR sample the vol factor is computed against."""
        period = int(self.cfg.get("indicators.atr_period"))
        if len(setup_df) < period + 1:
            return
        value = float(atr(setup_df, period).iloc[-1])
        if value > 0:
            state.atr_history.append(value)

    def _median_atr(self, state: InstrumentFeed) -> float | None:
        """Median ATR over the trailing sample.

        Returns ``None`` when the sample is too small - the risk manager then
        applies a vol factor of 1.0, which is no adjustment rather than a guess.
        """
        if len(state.atr_history) < 5:
            return None
        return float(pd.Series(list(state.atr_history)).median())


# ---------------------------------------------------------------------------
# Offline provider - CSV replay for the backtester and tests
# ---------------------------------------------------------------------------


class CsvHistoryProvider:
    """Serves stored 1-minute bars and derives the cascade from them.

    Deriving all three timeframes from a single 1-minute series guarantees the
    bias, setup and trigger frames come from exactly the same ticks. Loading
    three separately-sourced files is the classic way a backtest quietly gains
    look-ahead at the timeframe boundaries.
    """

    def __init__(self, path: str | Path, market: str, config: Config | None = None) -> None:
        self.cfg = config or get_config()
        self.market = market
        self.base = self._load(Path(path))

        # Resample once, up front. Doing it per bar makes a replay quadratic in
        # the number of bars, which turns a twelve-session backtest into a
        # coffee break. Slicing a precomputed frame by timestamp is equivalent
        # because resampling is a pure function of the bars it sees, and
        # `slice_at` still only ever exposes bars that had closed.
        self._resampled: dict[str, pd.DataFrame] = {}
        self._closes: dict[str, pd.DatetimeIndex] = {}
        for role, label in self.cfg.timeframes(market).items():
            frame = resample_ohlc(self.base, label)
            self._resampled[role] = frame
            self._closes[role] = frame.index + parse_timeframe(label)

    def _load(self, path: Path) -> pd.DataFrame:
        """Read a CSV with a datetime column plus OHLC(V)."""
        frame = pd.read_csv(path)
        stamp_col = next(
            (col for col in ("datetime", "date", "timestamp", "time") if col in frame.columns),
            frame.columns[0],
        )
        frame[stamp_col] = pd.to_datetime(frame[stamp_col])
        frame = frame.set_index(stamp_col).sort_index()
        frame.columns = [str(col).lower() for col in frame.columns]
        if frame.index.tz is None:
            frame.index = frame.index.tz_localize(str(self.cfg.get("sessions.timezone")))
        else:
            frame.index = frame.index.tz_convert(str(self.cfg.get("sessions.timezone")))
        columns = [col for col in ("open", "high", "low", "close", "volume") if col in frame.columns]
        frame = frame[columns].astype(float)
        validate_ohlc(frame)
        return frame

    def slice_at(self, moment: datetime, bars: int) -> FeedState | None:
        """Build a :class:`FeedState` as it would have looked at ``moment``.

        Only bars that had *closed* by ``moment`` are included. This is the
        single most important line in the backtester: everything after it is
        arithmetic, and everything before it is where look-ahead bias creeps in.
        """
        stamp = pd.Timestamp(moment)
        if stamp.tz is None:
            stamp = stamp.tz_localize(str(self.cfg.get("sessions.timezone")))

        frames = {}
        for role in ("bias", "setup", "trigger"):
            # A bar is visible only once its close time is at or before `now`.
            # searchsorted on the precomputed close times is the whole guard.
            visible = int(self._closes[role].searchsorted(stamp, side="right"))
            frames[role] = self._resampled[role].iloc[max(0, visible - bars): visible]

        if frames["trigger"].empty:
            return None

        setup_df = frames["setup"]
        period = int(self.cfg.get("indicators.atr_period"))
        median = None
        if len(setup_df) > period * 3:
            series = atr(setup_df, period).dropna()
            if len(series) >= 5:
                median = float(series.tail(200).median())

        return FeedState(
            bias_df=frames["bias"],
            setup_df=setup_df,
            trigger_df=frames["trigger"],
            spread=None,
            chain=None,
            atr_median=median,
        )

    def timestamps(self, start: datetime | None = None,
                   end: datetime | None = None) -> list[pd.Timestamp]:
        """Trigger-TF close times to iterate over in a backtest."""
        closes = self._closes["trigger"]
        if start is not None:
            closes = closes[closes >= pd.Timestamp(start)]
        if end is not None:
            closes = closes[closes <= pd.Timestamp(end)]
        return list(closes)
