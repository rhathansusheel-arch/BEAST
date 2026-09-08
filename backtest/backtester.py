"""Walk-forward backtester.

Two properties matter more than anything else here:

1. **No look-ahead.** Every bar the engine sees at time *t* had closed strictly
   before *t*. That is enforced in one place -
   :meth:`data.market_data.CsvHistoryProvider.slice_at` - and verified by
   ``tests/test_look_ahead.py``. The three cascade timeframes are all derived
   from one 1-minute series so they cannot disagree about what had happened.

2. **The same code path as live.** The backtester drives the real
   :class:`~core.signal_generator.SignalGenerator` and the real
   :class:`~broker.position_tracker.PositionTracker` through a simulated broker.
   Nothing about the rules is reimplemented for backtesting - if it were, the two
   would drift and the results would stop meaning anything. That is also what
   soul file section 10 asks for: paper-mode tracking runs identically to live.

Walk-forward matters for the HMM overlay, which is the only fitted component in
the system. Each fold fits the model on its training slice and evaluates on the
out-of-sample slice that follows, so a state assignment never benefits from data
it could not have seen.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from broker import BrokerClient, OrderRequest, OrderResult, Quote
from broker.order_executor import OrderExecutor
from broker.position_tracker import PositionTracker
from core.config import Config, get_config
from core.risk_manager import RiskManager
from core.schemas import Rejection, Signal, TradeRecord
from core.session import SessionClock
from core.signal_generator import FeedState, SignalGenerator
from data.market_data import CsvHistoryProvider
from data.news_calendar import NewsCalendar

logger = logging.getLogger("beast.backtest")


class SimulatedBroker(BrokerClient):
    """A broker that fills at the price it is told and transmits nothing.

    Fills are simulated at the trigger candle's close with the conservative
    assumptions in 6.8 - the slippage model lives in
    :class:`~broker.order_executor.OrderExecutor` and is shared with live, so
    backtest and paper produce identical accounting.
    """

    name = "simulated"

    def __init__(self, config: Config | None = None) -> None:
        self.cfg = config or get_config()
        self.orders: list[OrderRequest] = []
        self._last_price: float = 0.0

    def set_price(self, price: float) -> None:
        """Tell the broker the current mark, used for quotes and fills."""
        self._last_price = price

    def connect(self) -> bool:
        return True

    def is_connected(self) -> bool:
        return True

    def history(self, symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
        # The backtester feeds frames directly; nothing pulls history here.
        return pd.DataFrame()

    def quote(self, symbol: str) -> Quote | None:
        return Quote(symbol, self._last_price, self._last_price, self._last_price, datetime.now())

    def place_order(self, request: OrderRequest) -> OrderResult:
        self.orders.append(request)
        return OrderResult(
            accepted=True,
            order_id=f"sim-{len(self.orders)}",
            filled_quantity=request.quantity,
            average_price=request.limit_price or self._last_price,
            paper=True,
            message="simulated fill",
        )

    def cancel_order(self, order_id: str) -> bool:
        return True

    def expiries(self, underlying: str) -> list[date]:
        return []

    def futures_contracts(self, underlying: str) -> list[tuple[str, date]]:
        return []


@dataclass
class Fold:
    """One walk-forward fold."""

    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    trades: list[TradeRecord] = field(default_factory=list)
    rejections: list[Rejection] = field(default_factory=list)
    signals: list[Signal] = field(default_factory=list)


@dataclass
class BacktestResult:
    """Everything a run produced."""

    market: str
    folds: list[Fold] = field(default_factory=list)
    equity_curve: pd.Series | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @property
    def trades(self) -> list[TradeRecord]:
        """Every out-of-sample trade, in order."""
        return [trade for fold in self.folds for trade in fold.trades]

    @property
    def rejections(self) -> list[Rejection]:
        return [item for fold in self.folds for item in fold.rejections]

    def gate_histogram(self) -> dict[str, int]:
        """Rejections by gate - the section 9 diagnostic."""
        counts: dict[str, int] = {}
        for rejection in self.rejections:
            counts[rejection.failed_gate.value] = counts.get(rejection.failed_gate.value, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))


class Backtester:
    """Replays stored bars through the live decision path.

    Args:
        market: Which instrument to test.
        data_path: CSV of 1-minute bars.
        config: Injected for tests.
    """

    def __init__(self, market: str, data_path: str | Path,
                 config: Config | None = None) -> None:
        self.cfg = config or get_config()
        self.market = market.upper()
        self.provider = CsvHistoryProvider(data_path, self.market, self.cfg)
        self.clock = SessionClock(self.market, self.cfg)

    # -- walk-forward --------------------------------------------------------

    def folds(self, timestamps: list[pd.Timestamp]) -> list[Fold]:
        """Split the timeline into train/test folds.

        Windows are configured in bars (``backtest.train_window``,
        ``test_window``, ``step_size``) and interpreted here on the **bias**
        timeframe, because the only fitted component - the HMM overlay - is
        fitted on bias-TF features.
        """
        train = int(self.cfg.get("backtest.train_window"))
        test = int(self.cfg.get("backtest.test_window"))
        step = int(self.cfg.get("backtest.step_size"))

        bias_tf = self.cfg.timeframes(self.market)["bias"]
        from data.feature_engineering import resample_ohlc

        bias = resample_ohlc(self.provider.base, bias_tf)
        if len(bias) < train + test:
            logger.warning(
                "Only %d bias bars available; need %d for one fold. Running a single "
                "in-sample pass instead.", len(bias), train + test,
            )
            if not timestamps:
                return []
            return [Fold(0, timestamps[0], timestamps[0], timestamps[0], timestamps[-1])]

        folds: list[Fold] = []
        start = 0
        while start + train + test <= len(bias):
            folds.append(
                Fold(
                    index=len(folds),
                    train_start=bias.index[start],
                    train_end=bias.index[start + train - 1],
                    test_start=bias.index[start + train],
                    test_end=bias.index[min(start + train + test - 1, len(bias) - 1)],
                )
            )
            start += step
        return folds

    # -- the run -------------------------------------------------------------

    def run(self, start: datetime | None = None,
            end: datetime | None = None) -> BacktestResult:
        """Replay the series and return the result.

        Each fold gets a fresh :class:`SignalGenerator`, risk manager and
        position tracker, so state from one out-of-sample window cannot leak into
        the next - including the level blacklist, the consecutive-loss counter
        and the HMM's fitted parameters.
        """
        result = BacktestResult(market=self.market, started_at=datetime.now())
        timestamps = self.provider.timestamps(start, end)
        if not timestamps:
            logger.error("No timestamps to replay for %s", self.market)
            result.finished_at = datetime.now()
            return result

        for fold in self.folds(timestamps):
            window = [
                stamp for stamp in timestamps if fold.test_start <= stamp <= fold.test_end
            ]
            if not window:
                continue
            self._run_fold(fold, window)
            result.folds.append(fold)
            logger.info(
                "fold %d: %d trades, %d rejections (%s to %s)",
                fold.index, len(fold.trades), len(fold.rejections),
                fold.test_start.date(), fold.test_end.date(),
            )

        result.equity_curve = self._equity_curve(result.trades)
        result.finished_at = datetime.now()
        return result

    def _run_fold(self, fold: Fold, window: list[pd.Timestamp]) -> None:
        """Replay one out-of-sample window bar by bar."""
        broker = SimulatedBroker(self.cfg)
        risk = RiskManager(self.cfg, capital=float(self.cfg.get("backtest.initial_capital")))
        executor = OrderExecutor({name: broker for name in ("zerodha", "paper", "simulated")}, self.cfg)
        tracker = PositionTracker(executor, risk, self.cfg)
        calendar = NewsCalendar(self.cfg)
        generator = SignalGenerator(self.market, risk, self.cfg, calendar)

        # Only as much history as the rules actually read: indicator warmup plus
        # the swing lookback the level engine scans. Handing the engine the full
        # `data.history_bars` would recompute thousands of rows of indicators on
        # every bar for values no rule ever looks at.
        from data.feature_engineering import warmup_bars

        bars = min(
            int(self.cfg.get("data.history_bars")),
            max(
                int(self.cfg.get("levels.swing_lookback")) * 2,
                warmup_bars(self.cfg) * 4,
            ),
        )

        for stamp in window:
            moment = stamp.to_pydatetime()
            feed = self.provider.slice_at(moment, bars)
            if feed is None or feed.trigger_df.empty:
                continue

            last_bar = feed.trigger_df.iloc[-1]
            price = float(last_bar["close"])
            broker.set_price(price)

            # Manage the open position first: exits evaluate on live price and
            # must not wait behind entry evaluation.
            if tracker.open_markets():
                tracker.on_trigger_close(
                    self.market, feed.trigger_df,
                    self._atr(feed), moment, self.clock,
                )
                update = tracker.on_price(
                    self.market, price, None, moment, self.clock,
                    high=float(last_bar["high"]), low=float(last_bar["low"]),
                    bar_open=float(last_bar["open"]),
                )
                if update.trade is not None:
                    fold.trades.append(update.trade)

            if self.clock.must_flatten(moment):
                for trade in tracker.flatten_all(
                    moment, {self.market: self.clock}, {self.market: price}
                ):
                    fold.trades.append(trade)
                continue

            outcome = generator.evaluate(feed, moment)
            fold.rejections.extend(outcome.rejections)
            if outcome.signal is None:
                continue

            fold.signals.append(outcome.signal)
            if self.market not in tracker.open_markets():
                tracker.open(outcome.signal, price, moment)

    def _atr(self, feed: FeedState) -> float:
        """Setup-TF ATR for the trailing computation."""
        from data.feature_engineering import atr

        period = int(self.cfg.get("indicators.atr_period"))
        if len(feed.setup_df) < period + 1:
            return 0.0
        value = atr(feed.setup_df, period).iloc[-1]
        return float(value) if pd.notna(value) else 0.0

    # -- output --------------------------------------------------------------

    def _equity_curve(self, trades: list[TradeRecord]) -> pd.Series:
        """Cumulative R, indexed by exit time.

        R rather than currency, because R is the unit every rule in the soul file
        is expressed in and it is comparable across the two markets despite their
        different risk-per-trade percentages.
        """
        if not trades:
            return pd.Series(dtype=float)
        points = [
            (trade.exit_time, trade.r_multiple or 0.0)
            for trade in trades
            if trade.exit_time is not None
        ]
        if not points:
            return pd.Series(dtype=float)
        points.sort(key=lambda item: item[0])
        index = pd.to_datetime([item[0] for item in points])
        return pd.Series([item[1] for item in points], index=index).cumsum()


def run_backtest(market: str, data_path: str | Path, start: datetime | None = None,
                 end: datetime | None = None,
                 config: Config | None = None) -> BacktestResult:
    """Convenience wrapper used by ``main.py --backtest``."""
    return Backtester(market, data_path, config).run(start, end)
