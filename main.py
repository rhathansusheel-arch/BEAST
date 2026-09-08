"""Beast - entry point.

Usage::

    python main.py                          # run the loop (paper mode by default)
    python main.py --check                  # validate config, list blockers, exit
    python main.py --dry-run                # full pipeline, no positions opened
    python main.py --train-only             # fit the volatility models and exit
    python main.py --backtest data/nifty_1m.csv --market NIFTY50
    python main.py --stress-test data/nifty_1m.csv --market NIFTY50
    python main.py --compare data/nifty_1m.csv --market NIFTY50
    python main.py --dashboard              # read-only view of the persisted state

Operational mode is soul file v3.1 section 10 as amended by instruction item 6:
**paper trading with complete order execution**. Every qualifying signal is
executed as a complete simulated trade and carried through to a recorded exit
against a 6.6 code. Going live requires two deliberate changes - ``mode: live``
and ``broker.paper_trading: false`` - and both are checked at the order boundary
as well as here.

Two clocks, not one
-------------------
Section 4.2: *"Beast evaluates only on closed candles. Exception: stop-loss,
target and trailing-stop triggers evaluate on live price, not candle close. A
stop is a stop."*

So the loop runs two paths with different cadences and different preconditions:

* The **live-price path** runs every iteration and manages open positions -
  stops, targets, trails, the premium backstop, and the session-close flatten
  sequence. It keeps running when the entry path is paused. Sections 4.6 and 5.6
  both block *entries only*; exits continue.
* The **closed-bar path** runs on bar closes and looks for entries, through the
  G0-G9 gate chain. Its tiers are the section 4.3 cascade - bias 15M Indian /
  30M Gold, setup 5M/15M, trigger 1M - not one flat bar size.
"""

from __future__ import annotations

import argparse
import logging
import signal as os_signal
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from broker import BrokerClient
from broker.order_executor import OrderExecutor
from broker.paper_broker import PaperBroker
from broker.position_tracker import PositionTracker
from broker.retry import RetryPolicy, try_call
from broker.zerodha_client import ZerodhaClient
from core.ai_analyst import AIAnalyst
from core.config import Config, describe_unset, get_config
from core.learning import PerformanceTracker
from core.override import OverrideGuard
from core.regime import StabilityTracker, VolatilityHMM
from core.regime.hmm_engine import HMM_AVAILABLE, ModelUnusable
from core.regime.vol_features import compute_features, session_ids
from core.risk_manager import RiskManager
from core.session import SessionClock
from core.session_state import (
    build_snapshot,
    compare_positions,
    load_snapshot,
    restore_counters,
    save_snapshot,
    snapshot_path,
)
from core.signal_generator import SignalGenerator
from data.feature_engineering import parse_timeframe
from data.market_data import MarketDataService
from data.news_calendar import NewsCalendar
from monitoring.alerts import AlertKind, AlertManager
from monitoring.dashboard import Dashboard
from monitoring.journal import Journal
from monitoring.logger import (
    log_model_event,
    log_regime_change,
    log_rejection,
    log_signal,
    log_trade,
    log_vol_state,
    set_runtime_context,
    setup_logging,
)

#: Consecutive feed-refresh failures before entries are paused for a market.
#: One miss is a hiccup; three in a row is a feed that is not there. Exits keep
#: running throughout - this only ever gates entries.
FEED_FAILURES_BEFORE_PAUSE = 3

#: Consecutive unhandled per-market errors before the whole runner stops. A loop
#: that throws every cycle is not trading, it is generating alerts, and an
#: operator woken by the tenth identical page is worse off than one woken by the
#: first and told Beast stopped.
ERROR_STREAK_BEFORE_HALT = 10


@dataclass
class SessionStats:
    """Counters for the shutdown summary."""

    started_at: datetime = field(default_factory=datetime.now)
    cycles: int = 0
    signals: int = 0
    trades_closed: int = 0
    rejections: int = 0
    feed_failures: int = 0
    errors: int = 0

    def summary(self, now: datetime, risk) -> dict[str, object]:
        return {
            "started_at": self.started_at.isoformat(),
            "ended_at": now.isoformat(),
            "duration_minutes": round(
                (now - self.started_at).total_seconds() / 60.0, 1
            ),
            "cycles": self.cycles,
            "signals": self.signals,
            "trades_closed": self.trades_closed,
            "rejections": self.rejections,
            "feed_failures": self.feed_failures,
            "errors": self.errors,
            "realised_pnl": {
                family: round(state.realised_pnl, 2)
                for family, state in risk.state.items()
            },
        }


class BeastRunner:
    """Wires every component together and runs the evaluation loop.

    Args:
        config: Injected for tests.
        use_dashboard: Whether the rich dashboard owns the terminal.
        dry_run: Run the full pipeline but never open a position. Signals are
            generated, journalled and narrated exactly as usual; only the call
            that would create a managed position is skipped, and what it would
            have done is logged instead.
    """

    def __init__(self, config: Config | None = None, use_dashboard: bool = True,
                 dry_run: bool = False) -> None:
        self.cfg = config or get_config()
        self.use_dashboard = use_dashboard
        self.dry_run = dry_run
        self.logger = setup_logging(self.cfg, quiet_console=use_dashboard)
        self.running = False

        self.markets: list[str] = [
            market.upper() for market in self.cfg.get("broker.symbols")
        ]
        self.brokers: dict[str, BrokerClient] = {
            "zerodha": ZerodhaClient(self.cfg),
            "paper": PaperBroker(self.cfg),
        }

        self.journal = Journal(self.cfg)
        self.alerts = AlertManager(self.cfg)
        self.calendar = NewsCalendar(self.cfg)
        self.risk = RiskManager(self.cfg)
        self.executor = OrderExecutor(self.brokers, self.cfg)
        self.positions = PositionTracker(self.executor, self.risk, self.cfg)
        self.learning = PerformanceTracker(self.journal, self.cfg)
        self.overrides = OverrideGuard(self.cfg)
        self.ai = AIAnalyst(self.cfg)
        self.data = MarketDataService(self.brokers, self.cfg)
        self.dashboard = Dashboard(self.cfg) if use_dashboard else None

        self.clocks = {market: SessionClock(market, self.cfg) for market in self.markets}
        self.generators = {
            market: SignalGenerator(
                market, self.risk, self.cfg, self.calendar,
                confluence_override=self.learning.confluence_requirement,
            )
            for market in self.markets
        }

        # -- volatility layer, one model and one tracker per market ----------
        # Frozen at startup and never hot-swapped mid-session: vol_state would
        # jump for reasons that are not in the market, and positions before and
        # after would have been sized under different models.
        self.vol_engines: dict[str, VolatilityHMM | None] = {m: None for m in self.markets}
        self.vol_trackers: dict[str, StabilityTracker] = {
            market: StabilityTracker(market, self.cfg) for market in self.markets
        }
        self.vol_states: dict[str, object | None] = {m: None for m in self.markets}
        self._last_bias_stamp: dict[str, pd.Timestamp | None] = {m: None for m in self.markets}
        self._models_trained_at: dict[str, datetime | None] = {m: None for m in self.markets}

        self._broker_positions: list[str] = []
        # Previous readings, so a transition can be alerted rather than a level
        # re-announced every cycle.
        self._previous_vol_label: dict[str, str] = {}
        self._previous_regime: dict[str, str] = {}
        self._flicker_announced: dict[str, bool] = {}
        self._breaker_announced: dict[str, str] = {}
        self._broker_connected: dict[str, bool] = {}
        self._pnl_band: dict[str, int] = {}
        self._feed_failures: dict[str, int] = {m: 0 for m in self.markets}
        self._entry_pause_reason: dict[str, str] = {}
        self._error_streak = 0
        self._last_review: datetime | None = None
        self._recent_signal_lines: list[str] = []
        self._recent_alert_lines: list[str] = []
        self.stats = SessionStats()
        self._shutdown_done = False

    # =====================================================================
    # STARTUP
    # =====================================================================

    def startup(self, wait_for_open: bool = True) -> bool:
        """Bring Beast up. Returns False when the loop must not start.

        The order matters and is not arbitrary:

        1. Config - nothing else is meaningful if a threshold is wrong.
        2. Brokers - and the account behind them.
        3. Market hours.
        4. Volatility models, loaded or trained, then **frozen** for the session.
        5. Risk manager, with capital from the broker in live mode.
        6. Positions synced **from the broker**, which is the authority.
        7. ``state_snapshot.json``, used to restore the day's risk counters and
           to *detect* position disagreements - never to declare a position.
        8. Feeds, then the system state and "System online".

        Positions are synced before the snapshot is read on purpose. Reading the
        snapshot first would invite treating it as the position list, and it is
        not: it records what Beast believed at its last shutdown, which is wrong
        by construction if anything filled afterwards and absent entirely after
        a hard kill.
        """
        if not self._step_config():
            return False
        self._step_brokers()
        if not self._step_market_hours(wait_for_open):
            return False
        self._step_models()
        self._step_risk()
        self._step_sync_positions()
        self._step_recover_snapshot()
        self._step_start_feeds()
        self._step_report_state()
        return True

    # -- 1. config -----------------------------------------------------------

    def _step_config(self) -> bool:
        """Validate the config. A consistency error stops startup; a blocker does not."""
        problems = self.cfg.validate()
        if problems:
            for problem in problems:
                self.logger.error("CONFIG: %s", problem)
            return False

        blockers = self.cfg.unset_blockers()
        if blockers:
            self.logger.warning(describe_unset(blockers))
            self.alerts.config_blocker(blockers)
        return True

    # -- 2. brokers ----------------------------------------------------------

    def _step_brokers(self) -> None:
        """Connect every adapter and verify the account behind it.

        Connection is retried with exponential backoff: a broker that is briefly
        unreachable at 09:14 should not cost the session. A broker that stays
        unreachable does not stop startup either - Beast runs, reports the
        adapter as down, and refuses the trades that need it. Refusing to start
        would also refuse to *manage* whatever is already open.
        """
        policy = RetryPolicy(attempts=3, base_delay=1.0)
        for name, broker in self.brokers.items():
            connected = try_call(
                broker.connect, f"connect {name}", default=False, policy=policy
            )
            if connected:
                equity = try_call(broker.capital, f"{name} account", default=None,
                                  policy=policy)
                detail = f", equity {equity:,.0f}" if equity else ""
                self.logger.info("connected: %s%s", name, detail)
            else:
                self.logger.warning(
                    "could not connect: %s - trades routed to it will be refused", name
                )
        self._fill_contract_specs()

    def _fill_contract_specs(self) -> None:
        """Read lot size and strike interval from the broker (open item 18).

        Both change by exchange notification, so the broker's contract master is
        the better source. Config remains the override: a value already set is
        never replaced.
        """
        zerodha = self.brokers.get("zerodha")
        if zerodha is None or not zerodha.is_connected():
            return
        for market in self.markets:
            if self.cfg.market_family(market) != "indian":
                continue
            key = self.cfg.instrument_key(market)
            if not self.cfg.get(f"instruments.{key}.prefer_broker_contract_master", True):
                continue
            specs = zerodha.contract_specs(market) if hasattr(zerodha, "contract_specs") else None
            if not specs:
                continue
            section = self.cfg.section("instruments")[key]
            for field_name in ("lot_size", "strike_interval"):
                if section.get(field_name) is None and specs.get(field_name):
                    section[field_name] = specs[field_name]
                    self.logger.info(
                        "%s.%s read from the broker contract master: %s",
                        key, field_name, specs[field_name],
                    )

    # -- 3. market hours -----------------------------------------------------

    def _step_market_hours(self, wait_for_open: bool) -> bool:
        """Wait for a session to open, or decline to start.

        Returns False only when ``wait_for_open`` is false and nothing is open.
        Waiting is the default because starting early and idling is harmless,
        while exiting at 09:10 means nobody is watching the 09:15 open.

        Beast keeps running through the *close* rather than exiting, because the
        flatten sequence (6.7) and the exit path both live inside the loop.
        """
        now = datetime.now()
        open_markets = [m for m in self.markets if self.clocks[m].is_open(now)]
        if open_markets:
            self.logger.info("open now: %s", ", ".join(open_markets))
            return True

        next_open = self._next_open(now)
        if not wait_for_open:
            self.logger.warning(
                "no market open (next: %s) and --exit-if-closed was given",
                next_open or "unknown",
            )
            return False

        self.logger.info(
            "no market open yet - waiting. Next session: %s", next_open or "unknown"
        )
        return True

    def _next_open(self, now: datetime) -> str | None:
        """Nearest session open across the configured markets, as text."""
        candidates = []
        for market in self.markets:
            window = self.cfg.session(market)
            candidates.append(f"{market} {window['open']}")
        return ", ".join(candidates) if candidates else None

    # -- 4. volatility models ------------------------------------------------

    def _step_models(self) -> None:
        """Load each market's model, retraining a missing or stale one.

        Two ages, and they are different things:

        * ``regime.hmm.retrain_interval_days`` (7) - when a *retrain is due*.
          A model this old still works; it is simply time to refresh it.
        * ``regime.hmm.max_model_age_days`` (10) - when a model is *refused*.
          Past this it will not load at all.

        The model is then **frozen for the session**. Never hot-swapped
        mid-session: ``vol_state`` would jump for reasons that are not in the
        market, and positions opened before and after would have been sized
        under different models.

        A market with no usable model is not a reason to stop. The layer
        degrades to ``UNKNOWN`` at the uncertainty multiplier, which is the
        documented fail-safe - a model fault must reduce size, never halt
        trading.
        """
        if not self.cfg.regime_enabled():
            self.logger.info("regime layer disabled - sizing uses vol_factor alone")
            return
        if not HMM_AVAILABLE:
            self.logger.warning(
                "hmmlearn is not installed - vol_state will be UNKNOWN for every "
                "market. Install it, or set regime.enabled: false."
            )
            return

        retrain_days = int(self.cfg.get("regime.hmm.retrain_interval_days"))
        for market in self.markets:
            engine = self._load_model(market)
            if engine is not None and self._retrain_due(engine, retrain_days):
                self.logger.info(
                    "%s: model is older than the %d-day retrain interval", market, retrain_days
                )
                engine = None
            if engine is None:
                engine = self._train_model(market)
            self.vol_engines[market] = engine
            if engine is not None and engine.metadata is not None:
                engine.reset_stream()
                self._models_trained_at[market] = engine.metadata.trained_at
                self.logger.info(
                    "%s: model frozen for the session - %s (%d states, %s)",
                    market, engine.metadata.model_version, engine.metadata.n_states,
                    "BIC margin is noise" if engine.metadata.bic_margin_is_noise
                    else f"BIC margin {engine.metadata.bic_margin:.1f}",
                )

    def _load_model(self, market: str) -> VolatilityHMM | None:
        """Load a persisted model, or ``None`` with the reason logged."""
        try:
            return VolatilityHMM.load(market, self.cfg)
        except ModelUnusable as error:
            # The exception text already names the market.
            self.logger.info("%s", error)
            return None

    def _retrain_due(self, engine: VolatilityHMM, retrain_days: int) -> bool:
        if engine.metadata is None:
            return True
        age = datetime.now(engine.metadata.trained_at.tzinfo) - engine.metadata.trained_at
        return age > timedelta(days=retrain_days)

    def _train_model(self, market: str) -> VolatilityHMM | None:
        """Fit a model from bias-TF history. Returns ``None`` when it cannot.

        Training refuses below ``regime.hmm.min_train_bars`` - 12,500 bias-TF
        bars for the Indian markets, 16,000 for Gold, roughly two years each.
        That refusal is deliberate and is not worked around here: a volatility
        model fitted to three months of history has not seen a volatile quarter.
        """
        features = self._bias_features(market)
        if features is None or features.empty:
            self.logger.warning("%s: no bias-TF history to train on", market)
            return None

        engine = VolatilityHMM(market, self.cfg)
        try:
            engine.train(features)
        except (ValueError, RuntimeError) as error:
            self.logger.warning("%s: not training - %s", market, error)
            return None

        path = engine.save()
        self.logger.info("%s: trained and saved to %s", market, path)
        meta = engine.metadata
        if meta is not None:
            log_model_event(market, "trained", meta.to_dict())
            self.alerts.model_retrained(
                market, meta.model_version, meta.n_states, meta.n_samples,
                meta.bic_margin_is_noise,
            )
        return engine

    def _bias_features(self, market: str) -> pd.DataFrame | None:
        """Pull bias-TF history and compute the raw feature matrix."""
        broker = self.data.broker_for(market)
        if broker is None:
            return None
        timeframe = self.cfg.timeframes(market)["bias"]
        bars = try_call(
            lambda: broker.history(market, timeframe, int(self.cfg.get("data.history_bars"))),
            f"{market} bias history", default=pd.DataFrame(),
        )
        if bars is None or bars.empty:
            return None
        try:
            return compute_features(bars, market, self.cfg)
        except ValueError as error:
            self.logger.warning("%s: feature computation failed - %s", market, error)
            return None

    # -- 5. risk -------------------------------------------------------------

    def _step_risk(self) -> None:
        """Set capital, then roll each family into today's session."""
        capital = self._live_capital()
        if capital:
            self.risk.capital = capital
            self.logger.info("capital from broker: %s", f"{capital:,.0f}")
        else:
            self.logger.info("capital from config: %s", f"{self.risk.capital:,.0f}")

        now = datetime.now()
        for market in self.markets:
            family = self.cfg.market_family(market)
            self.risk.roll_session(family, self.clocks[market].session_day(now))

    def _live_capital(self) -> float | None:
        """Account equity, when running live."""
        if self.cfg.is_paper:
            return None
        for name, broker in self.brokers.items():
            value = try_call(broker.capital, f"{name} equity", default=None)
            if value:
                return value
        return None

    # -- 6. positions --------------------------------------------------------

    def _step_sync_positions(self) -> None:
        """Reconcile against the broker. The broker is the authority, always.

        What is implemented here is the *detection* half: which markets the
        broker says hold a position, so a disagreement with the snapshot can be
        alerted at step 7.

        What is **not** implemented here, and is the ops layer's job, is the
        repair half - verifying that every broker position has a resting stop
        covering its full quantity and placing one immediately if it does not,
        restoring the target from the trade plan, rehydrating trail state under
        the 6.3 ratchet rule, cancelling orphan orders, and forcing SAFE mode on
        any position with no plan in the database. Until that exists, a position
        found here is reported loudly and left alone rather than half-managed.
        """
        held: list[str] = []
        for market in self.markets:
            broker = self.data.broker_for(market)
            if broker is None or not broker.is_connected():
                continue
            if self.positions.get(market) is not None:
                held.append(market)
        self._broker_positions = held

        tracked = self.positions.open_markets()
        if tracked:
            self.logger.info("positions carried in memory: %s", ", ".join(tracked))
        else:
            self.logger.info("no open positions tracked at startup")

    # -- 7. snapshot recovery ------------------------------------------------

    def _step_recover_snapshot(self) -> None:
        """Read ``state_snapshot.json`` and restore the day's risk counters."""
        path = snapshot_path(self.cfg)
        snapshot = load_snapshot(self.cfg)
        if snapshot is None:
            self.logger.info("no usable state snapshot at %s - starting fresh", path)
            return

        if not snapshot.clean_exit:
            message = (
                f"previous run did not shut down cleanly (snapshot written "
                f"{snapshot.written_at}). Verify open positions and their resting "
                f"stops at the broker before trusting anything below."
            )
            self.logger.error("%s", message)
            self._alert(AlertKind.ERROR, "SYSTEM", message, datetime.now())
        else:
            self.logger.info("recovering from clean shutdown at %s", snapshot.written_at)

        now = datetime.now()
        session_days = {
            self.cfg.market_family(market): self.clocks[market].session_day(now)
            for market in self.markets
        }
        for note in restore_counters(snapshot, self.risk, session_days, self.cfg):
            self.logger.info("recovery: %s", note)

        for line in compare_positions(snapshot, self._broker_positions):
            self.logger.error("RECONCILE: %s", line)
            self._alert(AlertKind.ERROR, "SYSTEM", line, now)

        for market, stored in (snapshot.model_versions or {}).items():
            engine = self.vol_engines.get(market)
            current = engine.metadata.model_version if engine and engine.metadata else "none"
            if stored != current:
                self.logger.info(
                    "%s: model changed across the restart (%s -> %s) - vol_state is "
                    "not comparable with the previous session's",
                    market, stored, current,
                )

    # -- 8. feeds and report -------------------------------------------------

    def _step_start_feeds(self) -> None:
        """Prime each market's feed so the first cycle is not a cold start."""
        now = datetime.now()
        for market in self.markets:
            state = try_call(
                lambda m=market: self.data.refresh(m, now),
                f"{market} initial feed", default=None,
            )
            if state is None:
                self._feed_failures[market] += 1
                self.logger.warning("%s: no data at startup", market)
            else:
                self.logger.info(
                    "%s: feed primed - %d bias / %d setup / %d trigger bars",
                    market, len(state.bias_df), len(state.setup_df), len(state.trigger_df),
                )

    def _step_report_state(self) -> None:
        """Print what Beast is about to do, then say it is online."""
        self.logger.info("=" * 68)
        self.logger.info(
            "BEAST %s | %s | %s",
            self.cfg.mode.upper(),
            "PAPER - no order is transmitted" if self.cfg.is_paper else "LIVE",
            "DRY RUN - no position will be opened" if self.dry_run else "full execution",
        )
        self.logger.info("markets: %s", ", ".join(self.markets))
        self.logger.info("capital: %s", f"{self.risk.capital:,.0f}")
        for market in self.markets:
            timeframes = self.cfg.timeframes(market)
            window = self.cfg.session(market)
            engine = self.vol_engines.get(market)
            version = engine.metadata.model_version if engine and engine.metadata else "none"
            self.logger.info(
                "  %-8s cascade %s/%s/%s | %s-%s | last entry %s | risk %.0f%% | "
                "cap %.0f%% | model %s",
                market, timeframes["bias"], timeframes["setup"], timeframes["trigger"],
                window["open"], window["hard_flat"], window["last_entry"],
                self.cfg.risk_per_trade(market) * 100,
                self.cfg.daily_loss_cap(market) * 100, version,
            )
        blockers = self.cfg.unset_blockers()
        if blockers:
            self.logger.warning("refusing affected trades - unset: %s", ", ".join(blockers))
        self.logger.info("=" * 68)
        self.logger.info("System online")

    # =====================================================================
    # MAIN LOOP
    # =====================================================================

    def run(self, once: bool = False, wait_for_open: bool = True) -> int:
        """Run until interrupted. Returns a process exit code."""
        if not self.startup(wait_for_open=wait_for_open):
            self.logger.error("startup failed; not starting the loop")
            return 1

        self.running = True
        os_signal.signal(os_signal.SIGINT, self._handle_signal)
        try:
            os_signal.signal(os_signal.SIGTERM, self._handle_signal)
        except (AttributeError, ValueError):  # not available on every platform
            pass

        if self.dashboard:
            self.dashboard.start()

        interval = self._loop_interval()
        exit_code = 0
        try:
            while self.running:
                started = time.monotonic()
                try:
                    self.tick(datetime.now())
                    self._error_streak = 0
                except Exception as error:
                    exit_code = self._handle_unhandled(error)
                    if exit_code:
                        break
                if once:
                    break
                elapsed = time.monotonic() - started
                time.sleep(max(1.0, interval - elapsed))
        finally:
            self.shutdown(clean=exit_code == 0)
        return exit_code

    def _loop_interval(self) -> float:
        """Seconds between cycles.

        Paced by the shortest trigger timeframe, but capped by the dashboard
        refresh so the live-price exit path is evaluated far more often than
        once per trigger bar. A stop that is only checked every trigger close is
        not a stop.
        """
        intervals = [
            parse_timeframe(self.cfg.timeframes(market)["trigger"]).total_seconds()
            for market in self.markets
        ]
        shortest = min(intervals) if intervals else 60.0
        return min(shortest, float(self.cfg.get("monitoring.dashboard_refresh_seconds")) * 4)

    def tick(self, now: datetime) -> None:
        """One evaluation cycle across every market."""
        self.stats.cycles += 1
        dashboard_state: dict[str, dict] = {}
        risk_state: dict[str, dict] = {}

        for market in self.markets:
            if not self.clocks[market].is_open(now):
                continue
            try:
                self._tick_market(market, now, dashboard_state)
            except Exception as error:  # one bad market must not stop the others
                self.stats.errors += 1
                self.logger.exception("error while evaluating %s: %s", market, error)
                self._alert(AlertKind.ERROR, market, str(error), now)
            risk_state[market] = self.risk.headroom(market)

        self._check_circuit_breakers(now)
        self._check_broker_sessions(now)
        self._maybe_retrain(now)
        self._maybe_weekly_review(now)
        self._refresh_dashboard(now, dashboard_state, risk_state)

    def _tick_market(self, market: str, now: datetime, dashboard_state: dict) -> None:
        """Both clocks for one market: exits always, entries only when allowed."""
        clock = self.clocks[market]
        self.risk.roll_session(self.cfg.market_family(market), clock.session_day(now))

        feed = self.data.refresh(market, now)
        if feed is None:
            self._on_feed_failure(market, now)
            # The bar feed being down must not stop an open position being
            # managed. 4.6 blocks entries on a stale feed and leaves exits
            # running, so fall back to a direct quote before giving up.
            self._manage_position_without_feed(market, now, clock)
            return
        self._on_feed_success(market, now)

        price = float(feed.trigger_df["close"].iloc[-1])

        # ---- (a) live-price path: exits, trails, flatten -------------------
        # Runs first and unconditionally. Sections 4.6 and 5.6 block entries
        # only; a stale feed or a news blackout never suspends an exit.
        self._manage_position(market, feed, price, now, clock)

        # ---- (b) closed-bar path: volatility state, then entries -----------
        self._update_vol_state(market, feed, now)

        paused = self._entry_pause_reason.get(market)
        if paused:
            self._record_context(market, dashboard_state, None, clock, now, paused)
            return

        outcome = self.generators[market].evaluate(feed, now)

        for alert in outcome.alerts:
            self._alert(AlertKind.DATA_STALE, market, alert, now)
        for rejection in outcome.rejections:
            self.stats.rejections += 1
            self.journal.record_rejection(rejection)
            log_rejection(rejection.to_dict())

        self._announce_regime_change(market, outcome.context)
        self._record_context(market, dashboard_state, outcome.context, clock, now, None)

        if outcome.signal is None:
            return
        self._on_signal(market, outcome.signal, price, now)

    def _manage_position(self, market: str, feed, price: float, now: datetime,
                         clock: SessionClock) -> None:
        """The live-price exit path (sections 6.1-6.3, 6.7, 6.10)."""
        position = self.positions.get(market)
        if position is None:
            return

        premium = self._current_premium(market, position)
        atr_value = self._setup_atr(feed)

        note = self.positions.on_trigger_close(
            market, feed.trigger_df, atr_value, now, clock
        )
        if note:
            self.logger.info("%s: %s", market, note)

        update = self.positions.on_price(market, price, premium, now, clock)
        for alert in update.alerts:
            self._alert(AlertKind.LOSS_LIMIT_PAUSE, market, alert, now)
        if update.trade is not None:
            self.stats.trades_closed += 1
            self.journal.record_trade(update.trade)
            self._alert(
                AlertKind.TRADE_CLOSED, market,
                f"closed {update.trade.exit_reason.value} at "
                f"{update.trade.exit_price:.2f}, {update.trade.r_multiple:+.2f}R",
                now,
            )

    def _manage_position_without_feed(self, market: str, now: datetime,
                                      clock: SessionClock) -> None:
        """Exit path fallback when the bar feed is down but a position is open.

        A quote endpoint and a historical-bar endpoint are different calls and
        fail independently, so the bar feed being unavailable does not mean the
        price is. When a quote can be had, stops, targets and trails are
        evaluated against it exactly as they would be normally - only the
        trigger-bar work (which needs a frame) is skipped.

        When nothing can be had at all, that is the worst state in this module:
        a live position whose stop cannot be evaluated. It is escalated rather
        than logged quietly, because the resting stop at the broker is now the
        only thing protecting the position and the operator needs to know that
        is all that is left.
        """
        position = self.positions.get(market)
        if position is None:
            return

        broker = self.data.broker_for(market)
        quote = None
        if broker is not None:
            quote = try_call(
                lambda: broker.quote(market), f"{market} fallback quote",
                default=None, policy=RetryPolicy(attempts=2),
            )
        if quote is None:
            self._alert(
                AlertKind.DATA_STALE, market,
                "OPEN POSITION AND NO PRICE: neither bars nor a quote are "
                "available, so stops and targets cannot be evaluated in-process. "
                "The resting stop at the broker is the only protection right now.",
                now,
            )
            return

        price = float(quote.mid)
        premium = self._current_premium(market, position)
        update = self.positions.on_price(market, price, premium, now, clock)
        self.logger.warning(
            "%s: managing the open position from a quote (%.2f) - bar feed is down",
            market, price,
        )
        for alert in update.alerts:
            self._alert(AlertKind.LOSS_LIMIT_PAUSE, market, alert, now)
        if update.trade is not None:
            self.stats.trades_closed += 1
            self.journal.record_trade(update.trade)
            self._alert(
                AlertKind.TRADE_CLOSED, market,
                f"closed {update.trade.exit_reason.value} at "
                f"{update.trade.exit_price:.2f}, {update.trade.r_multiple:+.2f}R "
                f"(from a quote; bar feed was down)",
                now,
            )

    def _on_signal(self, market: str, signal, price: float, now: datetime) -> None:
        """Journal, narrate and act on an emitted signal."""
        self.stats.signals += 1
        self.journal.record_signal(signal)
        log_signal(signal.to_dict(), signal.reason_line)
        self._recent_signal_lines.append(signal.reason_line)

        narration = self.ai.narrate_signal(signal)
        if narration.used_model:
            self.logger.info("%s", narration.text)

        if self.dry_run:
            self.logger.info(
                "DRY RUN - not opening: %s %s %s qty %s, stop %.2f target %.2f",
                market, signal.direction.value, signal.setup_type,
                getattr(signal, "quantity", "?"), signal.stop_price, signal.target_price,
            )
            return

        # Paper mode still opens a tracked position: section 10 as amended
        # requires performance tracking to run identically to live so the data
        # is comparable later.
        self.positions.open(signal, price, now)

    # -- volatility state ----------------------------------------------------

    def _update_vol_state(self, market: str, feed, now: datetime) -> None:
        """Recompute ``vol_state`` on each new **bias**-TF close.

        Steps 2-5 of the loop, in one place: features on a rolling window with no
        future data, filtered inference (forward algorithm only), the persistence
        check, and the flicker check that forces uncertainty mode.

        It runs once per bias bar, not once per cycle. The model is a bias-TF
        model; scoring it on every trigger tick would report the same bar over
        and over and inflate the consecutive-bar count into a confirmation that
        never happened.

        **This changes nothing yet.** The state is computed, logged and shown on
        the dashboard, and is passed to neither sizing nor the gate chain. Gate
        G10 and the ``min(vol_factor, size_multiplier)`` combination are section
        5.1 and section 7 behaviour, and the soul file is the authority on both -
        they land when the v3.2 amendments in ``docs/soul-v3.2-proposed-diff.md``
        are approved, and not before.
        """
        engine = self.vol_engines.get(market)
        tracker = self.vol_trackers[market]

        if feed.bias_df.empty:
            return
        stamp = feed.bias_df.index[-1]
        if self._last_bias_stamp.get(market) == stamp:
            return
        self._last_bias_stamp[market] = stamp

        if engine is None or engine.metadata is None:
            self.vol_states[market] = tracker.on_failure(
                stamp.to_pydatetime(), "no usable model for this market"
            )
            return

        try:
            features = compute_features(feed.bias_df, market, self.cfg)
            if features.empty:
                raise ValueError("feature matrix is empty after warm-up")
            observation = engine.scaler.transform_last(features)
            posterior = engine.step(observation)
            sessions = session_ids(
                feed.bias_df,
                str(self.cfg.session(market)["open"]),
                str(self.cfg.get("sessions.timezone")),
            )
            state = tracker.observe(
                stamp.to_pydatetime(), posterior, engine,
                session_id=sessions.loc[features.index[-1]],
            )
        except Exception as error:
            # Hold the last confirmed state for a bounded number of bars, then
            # UNKNOWN. A model fault reduces size; it never halts trading.
            self.logger.warning("%s: vol_state inference failed - %s", market, error)
            engine.reset_stream()
            state = tracker.on_failure(stamp.to_pydatetime(), str(error))

        self.vol_states[market] = state
        log_vol_state(market, state)
        self._announce_vol_change(market, state)

    def _announce_vol_change(self, market: str, state) -> None:
        """Alert on a **confirmed** transition, and on flicker starting or stopping.

        Only confirmed transitions. An unconfirmed state moves with the
        posterior and would page several times an hour for a market that never
        changed regime - which is exactly the flicker the stability layer exists
        to absorb, and re-emitting it as an alert would undo that work.
        """
        stability = self.cfg.section("regime")["stability"]
        window = int(stability["flicker_window"])
        threshold = int(stability["flicker_threshold"])

        if state.is_confirmed:
            previous = self._previous_vol_label.get(market)
            if previous is not None and previous != state.label:
                self.alerts.vol_state_change(
                    market, previous, state.label,
                    state.probability, state.consecutive_bars,
                )
            self._previous_vol_label[market] = state.label

        was_flickering = self._flicker_announced.get(market, False)
        if state.is_flickering and not was_flickering:
            self.alerts.flicker_exceeded(
                market, int(round(state.flicker_rate * window)), window, threshold
            )
        self._flicker_announced[market] = bool(state.is_flickering)

    def _announce_regime_change(self, market: str, context) -> None:
        """Alert on a section 4.4 regime transition - this one changes setups."""
        if context is None or getattr(context, "regime", None) is None:
            return
        current = context.regime.regime.value
        previous = self._previous_regime.get(market)
        self._previous_regime[market] = current
        if previous is None or previous == current:
            return
        log_regime_change(market, previous, current, context.regime.detail)
        self.alerts.regime_change(market, previous, current, context.regime.adx)

    # -- feed health ---------------------------------------------------------

    def _on_feed_failure(self, market: str, now: datetime) -> None:
        """Count a missing refresh and pause entries once it is a pattern.

        Entries only. The exit path is unaffected by design: 4.6 and 5.6 both
        block new entries on a stale feed, and both leave exits running, because
        a position with a live stop needs that stop evaluated more than it needs
        the feed to be healthy.
        """
        self.stats.feed_failures += 1
        self._feed_failures[market] += 1
        if self._feed_failures[market] < FEED_FAILURES_BEFORE_PAUSE:
            self.logger.warning(
                "%s: no data this cycle (%d/%d)",
                market, self._feed_failures[market], FEED_FAILURES_BEFORE_PAUSE,
            )
            return
        if market not in self._entry_pause_reason:
            reason = (
                f"feed unavailable for {self._feed_failures[market]} cycles - "
                f"entries paused, exits still running"
            )
            self._entry_pause_reason[market] = reason
            self.logger.error("%s: %s", market, reason)
            self.alerts.feed_down(
                market, self._feed_failures[market],
                has_position=self.positions.get(market) is not None,
            )

    def _on_feed_success(self, market: str, now: datetime) -> None:
        """Clear a feed pause once data is flowing again."""
        self._feed_failures[market] = 0
        if market in self._entry_pause_reason:
            del self._entry_pause_reason[market]
            self.logger.info("%s: feed recovered - entries resume", market)
            self.alerts.circuit_breaker(market, "data feed recovered", tripped=False)

    # -- circuit breakers ----------------------------------------------------

    def _check_circuit_breakers(self, now: datetime) -> None:
        """Alert on every condition that starts or stops blocking new entries.

        Three independent breakers, and none of them touches an open position:

        * The section 7 session pause - daily loss cap or three consecutive
          losses. Cleared only by a new session, never by Beast.
        * A feed pause, from :meth:`_on_feed_failure`.
        * The runner's own error streak, handled in :meth:`_handle_unhandled`.

        Edge-triggered in both directions. An operator told entries stopped and
        never told they resumed assumes Beast is still halted, and will either
        intervene unnecessarily or stop reading the alerts.
        """
        for market in self.markets:
            paused, reason = self.risk.is_paused(market)
            previous = self._breaker_announced.get(market)
            if paused and previous != reason:
                self.alerts.circuit_breaker(market, reason, tripped=True)
                self._breaker_announced[market] = reason
            elif not paused and previous is not None:
                self.alerts.circuit_breaker(market, previous, tripped=False)
                del self._breaker_announced[market]
            self._check_pnl_band(market, now)

    def _check_pnl_band(self, market: str, now: datetime) -> None:
        """Alert as the day's loss crosses each quarter of the section 7 cap.

        Banded rather than thresholded on an absolute number, so the alert fires
        once per band crossed instead of every cycle once past a line, and so it
        means the same thing whatever the market's cap happens to be.
        """
        headroom = self.risk.headroom(market)
        cap = float(headroom.get("daily_cap", 0.0))
        realised = float(headroom.get("realised_pnl", 0.0))
        if cap <= 0 or realised >= 0:
            return
        fraction = abs(realised) / cap
        band = int(fraction * 4)          # 0, 1, 2, 3, 4 -> quarters of the cap
        if band <= self._pnl_band.get(market, 0):
            return
        self._pnl_band[market] = band
        self.alerts.large_pnl(
            self.cfg.market_family(market), realised, cap, fraction
        )

    def _check_broker_sessions(self, now: datetime) -> None:
        """Alert when a broker session drops or comes back."""
        for name, broker in self.brokers.items():
            connected = bool(broker.is_connected())
            previous = self._broker_connected.get(name)
            self._broker_connected[name] = connected
            if previous is None or previous == connected:
                continue
            if connected:
                self.alerts.api_restored(name, self._broker_latency(name))
            else:
                self.alerts.api_lost(name)

    def _broker_latency(self, name: str) -> float | None:
        """Round-trip time of a trivial call, in milliseconds, or ``None``."""
        broker = self.brokers.get(name)
        if broker is None:
            return None
        started = time.monotonic()
        ok = try_call(broker.is_connected, f"{name} ping", default=None,
                      policy=RetryPolicy(attempts=1))
        if ok is None:
            return None
        return (time.monotonic() - started) * 1000.0

    def _handle_unhandled(self, error: Exception) -> int:
        """Log a traceback, save state, alert - and halt on a repeating fault.

        Returns a non-zero exit code when the loop must stop. One unhandled
        error is worth surviving; ten in a row means Beast is not trading, it is
        generating alerts, and stopping is more useful to the operator than
        continuing.
        """
        self._error_streak += 1
        self.stats.errors += 1
        self.logger.error("unhandled error in the loop:\n%s", traceback.format_exc())
        self._alert(AlertKind.ERROR, "SYSTEM", str(error), datetime.now())
        self._save_snapshot(clean_exit=False)

        if self._error_streak >= ERROR_STREAK_BEFORE_HALT:
            self.logger.critical(
                "halting after %d consecutive unhandled errors. Open positions are "
                "NOT closed - their stops are resting at the broker. Square them off "
                "manually or restart Beast to resume managing them.",
                self._error_streak,
            )
            self.running = False
            return 1
        return 0

    # -- periodic work -------------------------------------------------------

    def _maybe_retrain(self, now: datetime) -> None:
        """Retrain when due - but never inside a session (section 3 rule 5).

        The model is frozen for the session, so a refit found to be due here is
        deferred until every market is closed. Swapping a model mid-session
        would move ``vol_state`` for reasons that are not in the market.
        """
        if not self.cfg.regime_enabled() or not HMM_AVAILABLE:
            return
        if any(self.clocks[market].is_open(now) for market in self.markets):
            return

        retrain_days = int(self.cfg.get("regime.hmm.retrain_interval_days"))
        for market in self.markets:
            trained_at = self._models_trained_at.get(market)
            if trained_at is not None:
                age = datetime.now(trained_at.tzinfo) - trained_at
                if age <= timedelta(days=retrain_days):
                    continue
            self.logger.info("%s: retraining outside session hours", market)
            engine = self._train_model(market)
            if engine is not None and engine.metadata is not None:
                self.vol_engines[market] = engine
                self.vol_trackers[market] = StabilityTracker(market, self.cfg)
                self._models_trained_at[market] = engine.metadata.trained_at
                self._last_bias_stamp[market] = None

    def _maybe_weekly_review(self, now: datetime) -> None:
        """Emit the section 9 weekly summary once a week."""
        if self._last_review and now - self._last_review < timedelta(days=7):
            return
        if self._last_review is None:
            self._last_review = now
            return
        self._last_review = now
        self.print_review(now)

    def print_review(self, now: datetime | None = None) -> str:
        """Build, log and return the weekly review."""
        now = now or datetime.now()
        summary = self.learning.weekly_summary(self.markets, now)
        deterministic = (
            self.learning.format_weekly_summary(summary)
            + "\n"
            + self.overrides.format_summary()
        )
        response = self.ai.weekly_review(
            summary, self.overrides.format_summary(), deterministic
        )
        self.logger.info("%s", response.text)
        return response.text

    # -- view ----------------------------------------------------------------

    def _record_context(self, market: str, dashboard_state: dict, context,
                        clock: SessionClock, now: datetime,
                        paused: str | None) -> None:
        """Assemble one market's dashboard row."""
        state = self.vol_states.get(market)
        stability = self.cfg.section("regime")["stability"]
        window = int(stability["flicker_window"])
        row: dict[str, object] = {
            "phase": clock.phase(now).value,
            "entries": paused or "allowed",
            "vol_label": state.label if state is not None else "UNKNOWN",
            "vol_probability": state.probability if state is not None else 0.0,
            "vol_confirmed": bool(state.is_confirmed) if state is not None else False,
            "vol_bars": state.consecutive_bars if state is not None else 0,
            "vol_flickering": bool(state.is_flickering) if state is not None else False,
            "vol_flicker_changes": (
                int(round(state.flicker_rate * window)) if state is not None else 0
            ),
            "vol_flicker_window": window,
            "size_multiplier": state.size_multiplier if state is not None else 1.0,
            "data_delay_minutes": state.data_delay_minutes if state is not None else 0,
        }
        if context is not None:
            row.update({
                "regime": context.regime.regime.value,
                "adx": context.regime.adx,
                "atr": context.atr_setup,
                "levels": len(context.levels.live_zones()),
                "chain": context.chain.to_dict() if context.chain else None,
            })
        dashboard_state[market] = row

    def _refresh_dashboard(self, now: datetime, dashboard_state: dict,
                           risk_state: dict) -> None:
        """Publish this cycle's state to the log context and the dashboard.

        The runtime context is refreshed even when no dashboard is attached:
        every log line carries it, and a headless run is exactly the one whose
        log somebody reads afterwards.
        """
        positions = self.positions.snapshot()
        self._publish_runtime_context(dashboard_state, risk_state, positions)
        if not self.dashboard:
            return
        self.dashboard.update(
            now=now,
            capital=self.risk.capital,
            markets=dashboard_state,
            positions=positions,
            gates=self.journal.gate_histogram(
                since=now.replace(hour=0, minute=0, second=0, microsecond=0)
            ),
            signals=self._recent_signal_lines[-5:],
            alerts=self._recent_alert_lines[-6:],
            blockers=self.cfg.unset_blockers(),
            risk=risk_state,
            system=self._system_state(),
        )

    def _publish_runtime_context(self, dashboard_state: dict, risk_state: dict,
                                 positions: list) -> None:
        """Refresh the context stamped onto every log record."""
        capital = float(self.risk.capital) or 1.0
        set_runtime_context(
            vol_state={
                market: str(row.get("vol_label", "UNKNOWN"))
                for market, row in dashboard_state.items()
            },
            vol_probability={
                market: float(row.get("vol_probability") or 0.0)
                for market, row in dashboard_state.items()
            },
            regime={
                market: str(row.get("regime", "-"))
                for market, row in dashboard_state.items()
            },
            equity=capital,
            open_positions=[str(row.get("market")) for row in positions],
            daily_pnl={
                family: round(state.realised_pnl, 2)
                for family, state in self.risk.state.items()
            },
            daily_pnl_pct={
                family: round(state.realised_pnl / capital, 6)
                for family, state in self.risk.state.items()
            },
            mode=self.cfg.mode,
        )

    def _system_state(self) -> dict:
        """The SYSTEM panel: feed health, broker sessions, model ages, counts."""
        models: dict[str, str] = {}
        for market, engine in self.vol_engines.items():
            if engine is None or engine.metadata is None:
                models[market] = "none"
                continue
            age = datetime.now(engine.metadata.trained_at.tzinfo) - engine.metadata.trained_at
            models[market] = f"{age.days}d"

        counts: dict[str, tuple[int, int]] = {}
        caps = self.cfg.get("risk.max_concurrent")
        for family in ("indian", "gold"):
            used = sum(
                1 for market in self.positions.open_markets()
                if self.cfg.market_family(market) == family
            )
            counts[family] = (used, int(caps.get(family, 0)))

        return {
            "feeds_ok": not self._entry_pause_reason,
            "brokers": {
                name: {
                    "connected": bool(broker.is_connected()),
                    "latency_ms": self._broker_latency(name)
                    if broker.is_connected() else None,
                }
                for name, broker in self.brokers.items()
            },
            "models": models,
            "position_counts": counts,
        }

    def _alert(self, kind: AlertKind, market: str, message: str, now: datetime) -> None:
        if self.alerts.send(kind, market, message, now):
            self._recent_alert_lines.append(f"{kind.value} {market}: {message}")

    def _setup_atr(self, feed) -> float:
        from data.feature_engineering import atr

        period = int(self.cfg.get("indicators.atr_period"))
        if len(feed.setup_df) < period + 1:
            return 0.0
        value = atr(feed.setup_df, period).iloc[-1]
        return float(value) if pd.notna(value) else 0.0

    def _current_premium(self, market: str, position) -> float | None:
        """Latest premium for an option position, for the 6.10 backstop."""
        leg = position.signal.option_leg
        if leg is None:
            return None
        broker = self.data.broker_for(market)
        if broker is None:
            return position.last_premium
        quote = try_call(
            lambda: broker.quote(leg.tradingsymbol),
            f"{market} premium quote", default=None,
        )
        return quote.mid if quote else position.last_premium

    # =====================================================================
    # SHUTDOWN
    # =====================================================================

    def _handle_signal(self, signum, _frame) -> None:
        """SIGINT/SIGTERM: finish this cycle, then stop."""
        name = os_signal.Signals(signum).name if hasattr(os_signal, "Signals") else signum
        self.logger.warning("%s received - shutting down after this cycle", name)
        self.running = False

    def shutdown(self, clean: bool = True) -> None:
        """Stop cleanly.

        In order: release the terminal, close the feeds, **leave positions
        open**, write the snapshot, print the session summary.

        Positions are deliberately not closed. Their stops are resting at the
        broker, and a shutdown that flattened everything would turn a routine
        restart - a deploy, a config change, a machine reboot - into a realised
        loss on every open trade. What shutdown owes the operator instead is to
        say loudly which positions it is leaving behind.
        """
        if self._shutdown_done:
            return
        self._shutdown_done = True
        self.running = False
        now = datetime.now()

        if self.dashboard:
            self.dashboard.stop()

        for name, broker in self.brokers.items():
            closer = getattr(broker, "disconnect", None) or getattr(broker, "close", None)
            if callable(closer):
                try_call(closer, f"disconnect {name}", default=None,
                         policy=RetryPolicy(attempts=1))

        open_markets = self.positions.open_markets()
        if open_markets:
            self.logger.warning(
                "shutting down with open position(s): %s. These are NOT closed by "
                "shutdown - their stops are resting. Square them off at the broker "
                "or restart Beast to keep managing them.",
                ", ".join(open_markets),
            )

        self._save_snapshot(clean_exit=clean)
        self._print_session_summary(now)
        self.journal.close()
        self.logger.info("stopped")

    def _save_snapshot(self, clean_exit: bool) -> None:
        """Write ``state_snapshot.json``. Never raises."""
        snapshot = build_snapshot(
            risk=self.risk,
            positions=self.positions,
            vol_states=self.vol_states,
            model_versions={
                market: (
                    engine.metadata.model_version
                    if engine and engine.metadata else "none"
                )
                for market, engine in self.vol_engines.items()
            },
            session_summary=self.stats.summary(datetime.now(), self.risk),
            clean_exit=clean_exit,
            config=self.cfg,
        )
        path = save_snapshot(snapshot, self.cfg)
        self.logger.info(
            "state snapshot written to %s (clean_exit=%s)", path, clean_exit
        )

    def _print_session_summary(self, now: datetime) -> None:
        """The end-of-run report."""
        summary = self.stats.summary(now, self.risk)
        self.logger.info("=" * 68)
        self.logger.info("SESSION SUMMARY")
        self.logger.info(
            "  ran %s minutes over %d cycles",
            summary["duration_minutes"], summary["cycles"],
        )
        self.logger.info(
            "  %d signals | %d trades closed | %d gate rejections",
            summary["signals"], summary["trades_closed"], summary["rejections"],
        )
        self.logger.info(
            "  %d feed failures | %d errors",
            summary["feed_failures"], summary["errors"],
        )
        for family, pnl in summary["realised_pnl"].items():
            state = self.risk.state[family]
            # Thousands separators are an f-string feature; %-formatting has no
            # equivalent, and logging uses %-formatting.
            self.logger.info(
                "  %-7s realised %s | %d consecutive losses%s",
                family, f"{pnl:+,.0f}", state.consecutive_losses,
                f" | PAUSED: {state.pause_reason}" if state.paused else "",
            )
        for market in self.markets:
            state = self.vol_states.get(market)
            if state is not None:
                self.logger.info("  %-8s last vol_state %s", market, state.label)
        open_markets = self.positions.open_markets()
        self.logger.info(
            "  open positions left running: %s",
            ", ".join(open_markets) if open_markets else "none",
        )
        self.logger.info("=" * 68)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_check(cfg: Config) -> int:
    """Validate config and report blockers."""
    print(f"settings: {cfg.path}")
    print(f"mode: {cfg.mode} ({'paper' if cfg.is_paper else 'LIVE'})")
    print(f"markets: {', '.join(str(item) for item in cfg.get('broker.symbols'))}")

    problems = cfg.validate()
    if problems:
        print("\nCONFIG ERRORS - Beast will not start:")
        for problem in problems:
            print(f"  - {problem}")
    else:
        print("\nconfig consistency: OK")

    blockers = cfg.unset_blockers()
    print()
    print(describe_unset(blockers))
    if blockers:
        print(
            "\nThese are not bugs. The soul file treats an unset threshold as FAILING, "
            "not passing, so the affected trades are refused at G8/G9 with a logged "
            "reason until real numbers are supplied (open items 15, 17, 18, 19)."
        )
    return 1 if problems else 0


def cmd_train_only(cfg: Config, markets: list[str] | None = None) -> int:
    """Fit each market's volatility model and exit.

    Reports what was trained and what was refused, and why. A refusal is the
    expected outcome below ``regime.hmm.min_train_bars`` and is not an error:
    roughly two years of bias-TF history is the bar, and a model fitted to three
    months has never seen a volatile quarter.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not HMM_AVAILABLE:
        print("hmmlearn is not installed - nothing to train.")
        return 1

    runner = BeastRunner(cfg, use_dashboard=False)
    runner._step_brokers()
    targets = markets or runner.markets

    trained, refused = 0, 0
    for market in targets:
        engine = runner._train_model(market)
        if engine is None or engine.metadata is None:
            refused += 1
            continue
        trained += 1
        meta = engine.metadata
        print(f"\n{market}: {meta.model_version}")
        print(f"  states     {meta.n_states} (BIC {meta.bic:,.1f})")
        print(
            f"  margin     {meta.bic_margin:,.1f}"
            + ("  <- noise, the state count was not really chosen by the data"
               if meta.bic_margin_is_noise else "")
        )
        print(f"  samples    {meta.n_samples:,} bias-TF bars")
        print(f"  window     {meta.train_start} -> {meta.train_end}")
        print(f"  converged  {meta.converged} in {meta.n_iter} iterations")
        print(f"  labels     {meta.label_map}")
        print(f"  features   {len(meta.feature_list)} ({meta.feature_hash})")

    runner.journal.close()
    print(f"\ntrained {trained}, refused {refused}")
    return 0 if trained else 1


def cmd_backtest(args: argparse.Namespace, cfg: Config) -> int:
    """Run a walk-forward backtest and print the report."""
    from backtest.backtester import run_backtest
    from backtest.performance import analyse, format_report

    path = Path(args.backtest)
    if not path.exists():
        print(f"data file not found: {path}")
        return 1

    result = run_backtest(args.market, path, config=cfg)
    print(format_report(analyse(result.trades, cfg), f"BACKTEST {args.market}"))
    print()
    histogram = result.gate_histogram()
    if histogram:
        print("rejections by gate:")
        for gate, count in histogram.items():
            print(f"  {gate}: {count}")
    print(f"\nfolds: {len(result.folds)} | signals: {sum(len(f.signals) for f in result.folds)}")
    return 0


def cmd_stress(args: argparse.Namespace, cfg: Config) -> int:
    """Run the stress suite against a data file."""
    from backtest.stress_test import StressTester, format_results
    from data.market_data import CsvHistoryProvider

    path = Path(args.stress_test)
    if not path.exists():
        print(f"data file not found: {path}")
        return 1

    provider = CsvHistoryProvider(path, args.market, cfg)
    tester = StressTester(cfg)
    results = []
    for scenario in tester.default_suite():
        _, outcome = tester.apply(scenario, provider.base)
        results.append(outcome)
    print(format_results(results))
    return 0


def cmd_compare(args: argparse.Namespace, cfg: Config) -> int:
    """Benchmark the strategy against buy-and-hold on the same bars.

    The comparison is in **R**, not currency, because that is the only unit in
    which Beast and buy-and-hold are commensurable: Beast risks a fixed fraction
    per trade and holds nothing overnight, so its currency P&L depends on
    capital and its R depends only on the rules.

    Read it with the obvious caveat attached. Buy-and-hold on an intraday index
    series carries overnight gap risk that Beast never takes, and Beast pays
    spread and slippage on every entry and exit that buy-and-hold pays twice in
    total. Neither number is the other's fair opponent; the comparison is a
    sanity check on whether the rules add anything, not a verdict.
    """
    from backtest.backtester import run_backtest
    from backtest.performance import analyse, buy_and_hold_benchmark, format_report
    from data.market_data import CsvHistoryProvider

    path = Path(args.compare)
    if not path.exists():
        print(f"data file not found: {path}")
        return 1

    result = run_backtest(args.market, path, config=cfg)
    report = analyse(result.trades, cfg)
    print(format_report(report, f"BEAST {args.market}"))

    provider = CsvHistoryProvider(path, args.market, cfg)
    bars = provider.base
    risk_points = report.average_risk_points if hasattr(report, "average_risk_points") else 0.0
    if not risk_points:
        # Fall back to the median stop distance actually used, so the benchmark
        # is expressed in the same R unit the strategy traded in.
        distances = [
            abs(trade.entry_price - trade.stop_price)
            for trade in result.trades
            if getattr(trade, "stop_price", None) and getattr(trade, "entry_price", None)
        ]
        risk_points = (
            sorted(distances)[len(distances) // 2] if distances else 0.0
        )

    print("\nBENCHMARK")
    if not risk_points:
        print("  no completed trades - cannot express buy-and-hold in R")
        return 0

    benchmark_r = buy_and_hold_benchmark(bars, risk_points)
    strategy_r = sum(getattr(trade, "r_multiple", 0.0) or 0.0 for trade in result.trades)
    print(f"  1R for this run      {risk_points:,.2f} points")
    print(f"  buy and hold         {benchmark_r:+.2f}R")
    print(f"  Beast                {strategy_r:+.2f}R over {len(result.trades)} trades")
    print(f"  difference           {strategy_r - benchmark_r:+.2f}R")
    print(
        "\n  Buy-and-hold here holds through every overnight gap, which Beast never\n"
        "  does, and pays spread twice against Beast's twice-per-trade. Treat this\n"
        "  as a sanity check on whether the rules add anything, not as a verdict."
    )
    return 0


def cmd_dashboard(cfg: Config) -> int:
    """Print a read-only view of the persisted state.

    This does **not** attach to a running process. Beast exposes no IPC or
    control socket, and inventing one to serve a status pane would put a network
    listener on the trading host for the sake of a display. What this renders
    instead is the durable state a running instance leaves behind - the latest
    ``state_snapshot.json`` plus today's journal - which is the same information
    one cycle out of date.

    For a live view, run Beast in the foreground; the dashboard is on by default.
    """
    snapshot = load_snapshot(cfg)
    if snapshot is None:
        print(f"no state snapshot at {snapshot_path(cfg)}")
        print("Beast has not run yet, or ran without writing one.")
        return 1

    journal = Journal(cfg)
    dashboard = Dashboard(cfg)
    try:
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        dashboard.update(
            now=datetime.fromisoformat(snapshot.written_at) if snapshot.written_at
            else datetime.now(),
            capital=snapshot.capital,
            markets={
                market: {"vol_state": state.get("vol_state", "-"),
                         "phase": "-",
                         "regime": "-"}
                for market, state in snapshot.vol_states.items()
            },
            positions=snapshot.believed_positions,
            gates=journal.gate_histogram(since=today),
            signals=[],
            alerts=[],
            blockers=cfg.unset_blockers(),
            risk={},
        )
        dashboard.print_once()
    finally:
        journal.close()

    print()
    print(f"snapshot written {snapshot.written_at} | clean_exit={snapshot.clean_exit}")
    if not snapshot.clean_exit:
        print("  the run that wrote this did NOT shut down cleanly")
    if snapshot.session_summary:
        print("last session:")
        for key, value in snapshot.session_summary.items():
            print(f"  {key}: {value}")
    print(
        "\nThis is the persisted state, not a live feed. Beast has no control "
        "socket to attach to."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Beast trading agent")
    parser.add_argument("--check", action="store_true", help="validate config and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="full pipeline, no positions opened")
    parser.add_argument("--train-only", action="store_true",
                        help="fit the volatility models and exit")
    parser.add_argument("--backtest", metavar="CSV", help="walk-forward backtester")
    parser.add_argument("--stress-test", metavar="CSV", dest="stress_test",
                        help="run the stress suite")
    parser.add_argument("--compare", metavar="CSV",
                        help="benchmark against buy-and-hold on the same bars")
    parser.add_argument("--dashboard", action="store_true",
                        help="print the persisted state and exit")
    parser.add_argument("--once", action="store_true", help="run a single evaluation cycle")
    parser.add_argument("--no-dashboard", action="store_true", help="plain logging output")
    parser.add_argument("--exit-if-closed", action="store_true",
                        help="exit instead of waiting when no market is open")
    parser.add_argument("--review", action="store_true", help="print the weekly review and exit")
    parser.add_argument("--market", default="NIFTY50", help="market for backtest/stress/compare")
    parser.add_argument("--config", help="path to an alternative beast_config.yaml")
    args = parser.parse_args(argv)

    if args.config:
        from core.config import load_config, set_config

        set_config(load_config(args.config))
    cfg = get_config()

    if args.check:
        return cmd_check(cfg)
    if args.dashboard:
        return cmd_dashboard(cfg)
    if args.train_only:
        return cmd_train_only(cfg)
    for flag, handler in (
        (args.backtest, cmd_backtest),
        (args.stress_test, cmd_stress),
        (args.compare, cmd_compare),
    ):
        if flag:
            logging.basicConfig(level=logging.INFO, format="%(message)s")
            return handler(args, cfg)

    runner = BeastRunner(
        cfg,
        use_dashboard=not args.no_dashboard and not args.review,
        dry_run=args.dry_run,
    )
    if args.review:
        print(runner.print_review())
        runner.journal.close()
        return 0

    return runner.run(once=args.once, wait_for_open=not args.exit_if_closed)


if __name__ == "__main__":
    sys.exit(main())
