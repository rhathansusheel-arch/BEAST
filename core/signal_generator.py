"""The entry pipeline - soul file 5.1, gates G0 through G9.

    Gates run in order and short-circuit. The first failing gate rejects the
    candidate; the rejection is recorded with the gate ID so Section 9 can learn
    where signals die.

This module is the orchestrator. It owns no market rules of its own: every gate
delegates to the component that owns that rule - the session clock (G0), the
integrity checks in this file (G1), the news calendar and risk manager (G2), the
regime classifier (G3), the strategy book (G4, G6), the confluence engine (G5),
the plan builder (G7), the instrument selector (G8) and the risk manager (G9).

Reading the ``evaluate`` method top to bottom should read like the pseudocode in
5.1, because that is what it is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Callable

import pandas as pd

from core.config import Config, ConfigBlockerError, get_config
from core.confluence import ConfluenceEngine
from core.exit_manager import TradePlanBuilder
from core.hmm_engine import HMMRegimeEngine, HMMState, RuleRegimeClassifier
from core.instrument_selector import InstrumentSelector
from core.levels import LevelEngine
from core.option_chain import ChainAnalyzer, ChainSnapshot
from core.regime_strategies import MarketContext, StrategyBook
from core.risk_manager import RiskManager
from core.schemas import (
    ChainContext,
    ConfluenceResult,
    Direction,
    Flag,
    Gate,
    LevelTier,
    Rejection,
    SetupInstance,
    Signal,
    TradePlan,
)
from core.session import SessionClock
from data.feature_engineering import compute_indicators, parse_timeframe, warmup_bars
from data.news_calendar import NewsCalendar

UNDERLYING_NAMES = {
    "NIFTY50": "NIFTY50",
    "NIFTY": "NIFTY50",
    "SENSEX": "SENSEX",
    "GOLD": "XAUUSD",
    "XAUUSD": "XAUUSD",
}


@dataclass
class FeedState:
    """What the data layer hands the pipeline each cycle.

    Attributes:
        bias_df / setup_df / trigger_df: Raw OHLC frames per cascade tier,
            already trimmed to closed candles.
        spread: Current bid-ask spread in underlying points (XAUUSD).
        chain: Latest option-chain snapshot (Nifty/Sensex), or ``None``.
        expiries: Available option expiries.
        contracts: ``(symbol, expiry)`` futures contracts (Gold).
        iv_history: Trailing ATM IV, for the percentile.
        atr_median: Median ATR over ``risk.vol_atr_median_days`` sessions.
    """

    bias_df: pd.DataFrame
    setup_df: pd.DataFrame
    trigger_df: pd.DataFrame
    spread: float | None = None
    chain: ChainSnapshot | None = None
    expiries: list[date] = field(default_factory=list)
    contracts: list[tuple[str, date]] = field(default_factory=list)
    iv_history: list[float] = field(default_factory=list)
    atr_median: float | None = None


@dataclass
class EvaluationResult:
    """Outcome of one trigger-TF close.

    Exactly one of ``signal`` and ``rejections`` is interesting: a signal means
    every gate passed; otherwise ``rejections`` says where each candidate died.
    """

    signal: Signal | None = None
    rejections: list[Rejection] = field(default_factory=list)
    context: MarketContext | None = None
    alerts: list[str] = field(default_factory=list)


class SignalGenerator:
    """Runs the 5.1 gate chain for one instrument.

    Args:
        market: ``"NIFTY50"``, ``"SENSEX"`` or ``"XAUUSD"``.
        risk: Shared risk manager - shared, because the concurrent-position cap
            and the Nifty/Sensex correlation rule span instruments.
        config: Injected for tests.
        calendar: Shared news calendar.
        confluence_override: Optional callable returning a per-setup confluence
            requirement, supplied by ``core/learning.py``. Section 9's only
            permitted adjustment is raising a setup's requirement from 4 to 5;
            nothing else about the rules may change.
    """

    def __init__(self, market: str, risk: RiskManager, config: Config | None = None,
                 calendar: NewsCalendar | None = None,
                 confluence_override: Callable[[int, str], int | None] | None = None) -> None:
        self.cfg = config or get_config()
        self.market = market.upper()
        self.underlying = UNDERLYING_NAMES.get(self.market, self.market)
        self.family = self.cfg.market_family(self.market)
        self.instrument_key = self.cfg.instrument_key(self.market)

        self.clock = SessionClock(self.market, self.cfg)
        self.levels = LevelEngine(self.market, self.cfg)
        self.strategies = StrategyBook(self.cfg)
        self.confluence = ConfluenceEngine(self.cfg)
        self.classifier = RuleRegimeClassifier(self.cfg)
        self.planner = TradePlanBuilder(self.cfg)
        self.selector = InstrumentSelector(self.cfg)
        self.chain_analyzer = ChainAnalyzer(self.cfg)
        self.hmm = HMMRegimeEngine(self.cfg)
        self.calendar = calendar or NewsCalendar(self.cfg)
        self.risk = risk
        self.confluence_override = confluence_override

        self.timeframes = self.cfg.timeframes(self.market)
        self._last_setup_stamp: pd.Timestamp | None = None
        self._last_bias_stamp: pd.Timestamp | None = None
        self._session_day: date | None = None
        self._recent_signals: list[tuple[datetime, int, Direction, float]] = []
        self._chain_context: ChainContext | None = None
        # Indicators and confluence are pure functions of a closed frame, and the
        # setup/bias frames only change on their own timeframe's close. Caching
        # them per frame identity keeps a 1M trigger cadence from recomputing a
        # 15M indicator set five times for the same bar.
        self._indicator_cache: dict[str, tuple[object, int, pd.DataFrame]] = {}
        self._confluence_cache: dict[tuple, ConfluenceResult] = {}

    # -- main entry point ----------------------------------------------------

    def evaluate(self, feed: FeedState, now: datetime) -> EvaluationResult:
        """Run one evaluation cycle on a trigger-TF close.

        Mirrors the 5.1 pseudocode: build context, run G0-G2 once, then walk the
        active setups freshest-first through G3-G9, returning the first that
        passes every gate.

        Returns:
            An :class:`EvaluationResult`. At most one signal is emitted per bar
            (5.1, "One signal per bar").
        """
        result = EvaluationResult()
        now = self.clock.localise(now)

        ctx, alerts = self.build_context(feed, now)
        result.context = ctx
        result.alerts.extend(alerts)
        if ctx is None:
            result.rejections.append(
                self._reject(Gate.G1_DATA, "insufficient history to build a context", now)
            )
            return result

        # G0 - session.
        expiry_cutoff = self._expiry_day_cutoff(feed, now)
        allowed, reason = self.clock.may_enter(now, expiry_cutoff)
        if not allowed:
            result.rejections.append(self._reject(Gate.G0_SESSION, reason, now))
            return result

        # G1 - data integrity.
        if not ctx.data_ok:
            result.rejections.append(self._reject(Gate.G1_DATA, ctx.data_detail, now))
            return result

        # G2 - no-trade conditions.
        if ctx.news_blocked:
            result.rejections.append(self._reject(Gate.G2_NO_TRADE, ctx.news_detail, now))
            return result
        paused, pause_reason = self.risk.is_paused(self.market)
        if paused:
            result.rejections.append(
                self._reject(Gate.G2_NO_TRADE, f"loss-limit pause: {pause_reason}", now)
            )
            return result
        blocked, hmm_reason = self.hmm.blocks_entry(ctx.hmm)
        if blocked:
            result.rejections.append(self._reject(Gate.G2_NO_TRADE, hmm_reason, now))
            return result

        # Walk the active setups, freshest first.
        candidates = self.strategies.active_setups(ctx)
        qualified: list[tuple[SetupInstance, TradePlan, ConfluenceResult, object, object]] = []

        for setup in candidates:
            outcome = self._evaluate_setup(setup, ctx, feed, now)
            if isinstance(outcome, Rejection):
                result.rejections.append(outcome)
                continue
            qualified.append(outcome)

        if not qualified:
            return result

        setup, plan, confluence, leg, sized = self._choose(qualified)
        signal = self._emit(setup, plan, confluence, leg, sized, ctx, now)
        self.strategies.consume(setup)
        result.signal = signal
        return result

    # -- per-setup gate chain -------------------------------------------------

    def _evaluate_setup(self, setup: SetupInstance, ctx: MarketContext, feed: FeedState,
                        now: datetime):
        """Run G3-G9 for one setup. Returns a tuple on success, else a Rejection."""
        # G3 - regime / bias.
        permitted, reason = self.strategies.permitted(ctx, setup)
        if not permitted:
            return self._reject(Gate.G3_REGIME, reason, now, setup)

        # G4 - setup detection (already satisfied) plus expiry and reuse checks.
        if setup.consumed:
            return self._reject(Gate.G4_SETUP, "setup instance already traded", now, setup)
        if ctx.setup_index > setup.expires_after_index:
            return self._reject(
                Gate.G4_SETUP,
                f"setup expired after {self.cfg.get('entry.signal_validity_candles')} setup-TF candles",
                now, setup,
            )

        # G5 - confluence. Cached per setup-TF bar: the six reads are a function
        # of the closed setup frame, so re-deriving them on every trigger bar
        # would produce the same answer at five times the cost.
        zone = self._zone_for(setup)
        required = self._required_confluence(setup)
        cache_key = (
            ctx.setup_df.index[-1], len(ctx.setup_df), setup.direction,
            setup.confluence_mode, zone.zone_id if zone else None, required,
        )
        confluence = self._confluence_cache.get(cache_key)
        if confluence is None:
            confluence = self.confluence.evaluate(
                ctx.setup_df, setup.direction, setup.confluence_mode, zone, required
            )
            if len(self._confluence_cache) > 256:
                self._confluence_cache.clear()
            self._confluence_cache[cache_key] = confluence
        if not confluence.passed:
            detail = (
                f"{confluence.aligned}/{required} aligned, {confluence.opposing} opposing"
                + (" - conflict rule" if confluence.rejected_by_conflict else "")
            )
            return self._reject(Gate.G5_CONFLUENCE, detail, now, setup, confluence)

        # G6 - trigger.
        detector = self.strategies.detector_for(setup)
        fired, entry_price, trigger_detail = detector.trigger_fired(setup, ctx)
        if not fired:
            return self._reject(Gate.G6_TRIGGER, trigger_detail, now, setup, confluence)

        if self._is_duplicate(setup, entry_price, ctx, now):
            return self._reject(
                Gate.G6_TRIGGER, "duplicate of a recent signal, suppressed", now, setup, confluence
            )

        # G7 - trade viability.
        is_expiry_day = self._is_expiry_day(feed, now)
        plan = self.planner.build(setup, entry_price, ctx.atr_setup, self.levels, is_expiry_day)
        if not plan.viable:
            return self._reject(
                Gate.G7_VIABILITY, plan.reject_reason or "plan not viable", now, setup, confluence
            )

        # G8 - instrument selection.
        try:
            selection = self.selector.select(
                plan, self.market, now, snapshot=feed.chain, contracts=feed.contracts
            )
        except ConfigBlockerError as error:
            return self._reject(Gate.G8_INSTRUMENT, str(error), now, setup, confluence)
        if not selection.ok:
            return self._reject(Gate.G8_INSTRUMENT, selection.reason, now, setup, confluence)

        # G9 - risk and portfolio.
        vol_factor = self.risk.vol_factor(ctx.atr_setup, feed.atr_median or ctx.atr_setup)
        sized = self.risk.size(
            plan, self.market, vol_factor,
            option_leg=selection.option_leg, futures_leg=selection.futures_leg,
        )
        if not sized.permitted:
            return self._reject(Gate.G9_RISK, sized.reason, now, setup, confluence)

        allowed, portfolio_reason = self.risk.check_portfolio(
            self.market, setup.direction, sized.risk_amount, now
        )
        if not allowed:
            return self._reject(Gate.G9_RISK, portfolio_reason, now, setup, confluence)

        if selection.futures_leg is not None:
            selection.futures_leg.contracts = sized.quantity

        return setup, plan, confluence, selection, sized

    # -- context construction (4.2 - 4.7) ------------------------------------

    def build_context(self, feed: FeedState,
                      now: datetime) -> tuple[MarketContext | None, list[str]]:
        """Assemble the evaluation context and run the 4.6 integrity gate."""
        alerts: list[str] = []
        minimum = warmup_bars(self.cfg)
        if len(feed.setup_df) < minimum or len(feed.bias_df) < minimum or feed.trigger_df.empty:
            return None, alerts

        setup_df = self._indicators("setup", feed.setup_df)
        bias_df = self._indicators("bias", feed.bias_df)
        trigger_df = feed.trigger_df

        atr_setup = float(setup_df["atr"].iloc[-1])
        regime = self.classifier.classify(bias_df)

        session_day = self.clock.session_day(now)
        if self._session_day != session_day:
            self._session_day = session_day
            self.levels.roll_session(session_day)
            self.strategies.roll_session()
            self.risk.roll_session(self.family, session_day)
            self._recent_signals.clear()

        flags: list[Flag] = []
        data_ok, data_detail = self._data_integrity(feed, setup_df, now, flags, alerts)

        # Rebuild levels on each setup-TF close (4.2 cadence).
        setup_stamp = setup_df.index[-1]
        is_new_setup_bar = self._last_setup_stamp != setup_stamp
        if is_new_setup_bar:
            self._last_setup_stamp = setup_stamp
            session_end = self._session_end(now)
            self.levels.rebuild(setup_df, bias_df, atr_setup, session_end)

        chain_context = self._apply_chain(feed, now, atr_setup, flags)

        hmm_state = self._update_hmm(bias_df)
        if hmm_state.flickering:
            flags.append(Flag.HMM_UNSTABLE)

        news_blocked, news_detail = self.calendar.blackout(self.market, now)
        if news_blocked:
            flags.append(Flag.NEWS_NEAR)

        ctx = MarketContext(
            market=self.market,
            now=now,
            bias_df=bias_df,
            setup_df=setup_df,
            trigger_df=trigger_df,
            atr_setup=atr_setup,
            regime=regime,
            levels=self.levels,
            chain=chain_context,
            hmm=hmm_state,
            flags=flags,
            data_ok=data_ok,
            data_detail=data_detail,
            news_blocked=news_blocked,
            news_detail=news_detail,
        )

        # Detection runs on setup-TF closes only (4.2 cadence). Running it on
        # every trigger close would re-detect the same setup repeatedly and
        # quietly defeat the "one instance, one signal" rule in 5.5.
        if is_new_setup_bar:
            self.strategies.on_setup_close(ctx)
        return ctx, alerts

    def _indicators(self, role: str, frame: pd.DataFrame) -> pd.DataFrame:
        """Compute indicators for ``frame``, reusing the last result when unchanged.

        The cache key is the frame's last bar and its length, which together
        identify a closed frame exactly. Nothing stale can be served: a new bar
        changes the timestamp, and a longer history changes the length.
        """
        stamp = frame.index[-1]
        cached = self._indicator_cache.get(role)
        if cached is not None and cached[0] == stamp and cached[1] == len(frame):
            return cached[2]
        computed = compute_indicators(frame, self.market, self.cfg)
        self._indicator_cache[role] = (stamp, len(frame), computed)
        return computed

    def _data_integrity(self, feed: FeedState, setup_df: pd.DataFrame, now: datetime,
                        flags: list[Flag], alerts: list[str]) -> tuple[bool, str]:
        """Gate G1 - the 4.6 checks."""
        trigger_interval = parse_timeframe(self.timeframes["trigger"])
        multiplier = float(self.cfg.get("data.stale_feed_multiplier"))
        last_bar = pd.Timestamp(feed.trigger_df.index[-1])
        if last_bar.tz is None:
            last_bar = last_bar.tz_localize(self.clock.timezone)
        age = pd.Timestamp(now) - last_bar
        if age > multiplier * trigger_interval:
            alerts.append(
                f"{self.market}: FEED STALE - last close {age} old. New entries suppressed."
            )
            return False, f"feed stale ({age} since the last trigger-TF close)"

        # Sensex carries a data-delay tag on every emission and is excluded from
        # trigger timeframes tighter than the configured minimum.
        if self.market.startswith("SENSEX"):
            flags.append(Flag.SENSEX_DELAY)
            floor = parse_timeframe(str(self.cfg.get("data.sensex_min_trigger_tf")))
            if trigger_interval < floor:
                return False, (
                    f"Sensex trigger timeframe {self.timeframes['trigger']} is tighter than the "
                    f"{self.cfg.get('data.sensex_min_trigger_tf')} floor for a delayed feed"
                )

        # XAUUSD spread ceiling.
        if self.family == "gold" and feed.spread is not None:
            ceiling = self.cfg.get("data.gold_spread_max", None)
            if ceiling is None:
                alerts.append(
                    "GOLD: data.gold_spread_max is unset - the high-spread rule cannot be "
                    "enforced. Entries are suppressed until a ceiling is supplied."
                )
                return False, "gold_spread_max unset; high-spread rule unenforceable"
            if str(self.cfg.get("data.gold_spread_max_mode")) == "atr_multiple":
                ceiling = float(ceiling) * float(setup_df["atr"].iloc[-1])
            if feed.spread > float(ceiling):
                flags.append(Flag.HIGH_SPREAD)
                alerts.append(
                    f"GOLD: HIGH SPREAD ALERT - {feed.spread:.2f} above the "
                    f"{float(ceiling):.2f} ceiling. No entry; exits unaffected."
                )
                return False, f"spread {feed.spread:.2f} above the {float(ceiling):.2f} ceiling"

        # Gap detection: recompute levels before the first entry is permitted.
        gapped, gap_detail = self._detect_gap(setup_df)
        if gapped:
            flags.append(Flag.GAP_SESSION)
            alerts.append(f"{self.market}: {gap_detail} - levels recomputed before entry.")

        return True, "feed fresh, spread within limit"

    def _detect_gap(self, setup_df: pd.DataFrame) -> tuple[bool, str]:
        """True when the session opened more than ``gap_recompute_atr`` from the prior close."""
        import numpy as np

        threshold = float(self.cfg.get("data.gap_recompute_atr"))
        local = setup_df.index.tz_convert(self.clock.timezone)
        open_time = self.clock.window.open_time
        minutes = local.hour * 60 + local.minute
        open_minutes = open_time.hour * 60 + open_time.minute
        # Bars before the session open belong to the previous session.
        session_day = pd.Series(
            np.where(minutes >= open_minutes, local.date, (local - pd.Timedelta(days=1)).date),
            index=setup_df.index,
        )

        days = list(dict.fromkeys(session_day.tolist()))
        if len(days) < 2:
            return False, ""
        today = setup_df[session_day == days[-1]]
        prior = setup_df[session_day == days[-2]]
        if today.empty or prior.empty:
            return False, ""

        atr_value = float(setup_df["atr"].iloc[-1])
        gap = abs(float(today["open"].iloc[0]) - float(prior["close"].iloc[-1]))
        if atr_value > 0 and gap > threshold * atr_value:
            return True, f"session gapped {gap:.2f} ({gap / atr_value:.2f}ATR) from the prior close"
        return False, ""

    def _apply_chain(self, feed: FeedState, now: datetime, atr_value: float,
                     flags: list[Flag]) -> ChainContext | None:
        """Analyse the option chain and merge its Tier A levels (4.7)."""
        if self.family != "indian" or feed.chain is None:
            return None
        try:
            context, zones, chain_flags = self.chain_analyzer.analyse(
                feed.chain, now, atr_value, self.market, feed.iv_history
            )
        except ConfigBlockerError:
            # strike_interval unset: chain levels cannot be built. Price-structure
            # levels continue to work, so this degrades context rather than
            # halting the cycle. G8 will refuse the trade with the same message.
            return None

        flags.extend(chain_flags)
        if zones:
            self.levels.merge_external_levels(zones, atr_value)
            context.level_convergence = any(zone.converged_with for zone in zones)

        event = self.calendar.event_in_window(
            self.market, now, int(self.cfg.get("options.theta_guard_minutes"))
        )
        if event is not None:
            flags.append(Flag.IV_CRUSH_RISK)

        if feed.chain.dte(now.date()) == 0:
            flags.append(Flag.EXPIRY_DAY)

        self._chain_context = context
        return context

    def _update_hmm(self, bias_df: pd.DataFrame) -> HMMState:
        """Fit or update the HMM overlay on bias-TF closes."""
        if not self.hmm.enabled:
            return HMMState(detail="hmm disabled in config")

        stamp = bias_df.index[-1]
        if self._last_bias_stamp == stamp:
            return self.hmm.update(self._hmm_features(bias_df))
        self._last_bias_stamp = stamp

        features = self._hmm_features(bias_df)
        if self.hmm.needs_refit():
            try:
                self.hmm.fit(features)
            except RuntimeError as error:
                return HMMState(detail=str(error))
        return self.hmm.update(features)

    def _hmm_features(self, bias_df: pd.DataFrame) -> pd.DataFrame:
        from data.feature_engineering import hmm_features

        return hmm_features(bias_df, self.cfg)

    # -- helpers -------------------------------------------------------------

    def _zone_for(self, setup: SetupInstance):
        """The S/R zone a reversal setup is being evaluated at, if any."""
        for zone in self.levels.live_zones():
            if zone.zone_id == setup.ref_id:
                return zone
        return None

    def _required_confluence(self, setup: SetupInstance) -> int:
        """Confluence threshold for this setup (4.4, plus section 9's adjustment)."""
        base = int(self.cfg.get("entry.min_confluence"))
        if setup.counter_bias:
            base = int(self.cfg.get("entry.min_confluence_counter_bias"))
        if self.confluence_override is not None:
            adjusted = self.confluence_override(int(setup.setup_type), self.market)
            if adjusted is not None:
                base = max(base, int(adjusted))
        return base

    def _is_duplicate(self, setup: SetupInstance, entry_price: float,
                      ctx: MarketContext, now: datetime) -> bool:
        """Suppress identical signals inside the validity window (5.5)."""
        tolerance = float(self.cfg.get("entry.duplicate_suppression_atr")) * ctx.atr_setup
        window = parse_timeframe(self.timeframes["setup"]) * int(
            self.cfg.get("entry.signal_validity_candles")
        )
        cutoff = pd.Timestamp(now) - window
        self._recent_signals = [
            item for item in self._recent_signals if pd.Timestamp(item[0]) >= cutoff
        ]
        return any(
            stamp_type == int(setup.setup_type)
            and stamp_direction is setup.direction
            and abs(price - entry_price) <= tolerance
            for _, stamp_type, stamp_direction, price in self._recent_signals
        )

    def _choose(self, qualified: list):
        """Pick one signal per bar (soul file 5.1).

        Beast takes the setup with the **tighter structural stop** - better R per
        unit risk. Ties break toward the higher confluence count, then toward the
        setup type with the better trailing expectancy from section 9 (supplied
        through ``confluence_override``'s owner; absent that, the lower setup id
        keeps the choice deterministic).
        """
        def key(item):
            setup, plan, confluence, _leg, _sized = item
            return (plan.risk_points, -confluence.aligned, int(setup.setup_type))

        return min(qualified, key=key)

    def _is_expiry_day(self, feed: FeedState, now: datetime) -> bool:
        return feed.chain is not None and feed.chain.dte(now.date()) == 0

    def _expiry_day_cutoff(self, feed: FeedState, now: datetime) -> time | None:
        """The earlier expiry-day entry cutoff, when it applies (5.7.4)."""
        if not self._is_expiry_day(feed, now):
            return None
        expiry_cfg = self.cfg.get("options.expiry_day")
        if not bool(expiry_cfg.get("enabled", True)):
            return None
        hour, minute = (int(part) for part in str(expiry_cfg["last_entry"]).split(":"))
        return time(hour=hour, minute=minute)

    def _session_end(self, now: datetime) -> datetime:
        """The close time of the current session, for order-block expiry."""
        local = self.clock.localise(now)
        return local.replace(
            hour=self.clock.window.close_time.hour,
            minute=self.clock.window.close_time.minute,
            second=0,
            microsecond=0,
        )

    def _reject(self, gate: Gate, detail: str, now: datetime,
                setup: SetupInstance | None = None,
                confluence: ConfluenceResult | None = None) -> Rejection:
        """Build a rejection-log row (Appendix C)."""
        return Rejection(
            timestamp=now,
            instrument=self.market,
            failed_gate=gate,
            gate_detail=detail,
            setup_type=setup.setup_type if setup else None,
            direction=setup.direction if setup else None,
            confluence_count=(
                {
                    "aligned": confluence.aligned,
                    "opposing": confluence.opposing,
                    "neutral": confluence.neutral,
                }
                if confluence
                else None
            ),
        )

    # -- emission -------------------------------------------------------------

    def _emit(self, setup: SetupInstance, plan: TradePlan, confluence: ConfluenceResult,
              selection, sized, ctx: MarketContext, now: datetime) -> Signal:
        """Build the Appendix B signal object."""
        leg_type = "OPTION" if selection.option_leg is not None else "FUTURES"
        signal = Signal(
            market=self.market,
            underlying=self.underlying,
            direction=setup.direction,
            setup_type=setup.setup_type,
            setup_ref=setup.ref_id,
            regime=ctx.regime.regime,
            counter_bias=setup.counter_bias,
            confluence_mode=setup.confluence_mode,
            confluence_count={
                "aligned": confluence.aligned,
                "opposing": confluence.opposing,
                "neutral": confluence.neutral,
            },
            indicator_reads=confluence.reads_as_str(),
            timeframes=dict(self.timeframes),
            entry_price=round(plan.entry_price, 2),
            stop_price=round(plan.stop_price, 2),
            stop_source=plan.stop_source,
            target_price=round(plan.target_price, 2),
            target_r=plan.target_r,
            trail=plan.trail,
            risk_pct=self.cfg.risk_per_trade(self.market),
            vol_factor=round(sized.vol_factor, 4),
            atr_setup_tf=round(ctx.atr_setup, 4),
            timestamp_ist=now,
            leg_type=leg_type,
            option_leg=selection.option_leg,
            futures_leg=selection.futures_leg,
            chain_context=ctx.chain,
            hmm_context=ctx.hmm.to_dict(),
            flags=list(ctx.flags),
            mode=self.cfg.mode,
        )
        signal.reason_line = format_reason_line(signal, confluence, ctx)
        self._recent_signals.append(
            (now, int(setup.setup_type), setup.direction, plan.entry_price)
        )
        return signal


# ---------------------------------------------------------------------------
# Section 11 - the operator-facing one-liner
# ---------------------------------------------------------------------------


def format_reason_line(signal: Signal, confluence: ConfluenceResult,
                       ctx: MarketContext) -> str:
    """Render the section 11 signal line.

    Every options signal states the **underlying plan first** and the leg second,
    so the operator can always see what Beast thinks price will do separately
    from what it bought to express that.
    """
    aligned = [name for name, read in confluence.reads.items()
               if read.value == ("bull" if signal.direction is Direction.LONG else "bear")]
    neutral = [name for name, read in confluence.reads.items() if read.value == "neutral"]
    opposing = [name for name, read in confluence.reads.items()
                if name not in aligned and name not in neutral]

    mode_word = "bullish" if signal.direction is Direction.LONG else "bearish"
    tally = (
        f"{confluence.aligned}/6 {mode_word}"
        f"{' reversal-mode' if confluence.mode.value == 'REVERSAL' else ''}: "
        f"{', '.join(aligned) if aligned else 'none'}"
    )
    asides = []
    if neutral:
        asides.append(f"{', '.join(neutral)} neutral")
    if opposing:
        asides.append(f"{', '.join(opposing)} opposing")
    if asides:
        tally += f" ({'; '.join(asides)})"

    head = (
        f"{signal.market} {signal.direction.value}"
        f"{' (futures)' if signal.leg_type == 'FUTURES' else ''} | "
        f"Setup {int(signal.setup_type)} {signal.setup_type.label} "
        f"{signal.timeframes['setup']} | "
    )

    if signal.leg_type == "OPTION" and signal.option_leg is not None:
        leg = signal.option_leg
        body = (
            f"underlying entry {signal.entry_price:g} | SL {signal.stop_price:g} | "
            f"TP {signal.target_price:g} ({signal.target_r}R) | "
            f"BUY {leg.strike:g} {leg.option_type} {leg.expiry.strftime('%d-%b')} "
            f"@ {leg.mid_premium:.2f} | D{leg.delta:.2f} | "
            f"{leg.lots} lot(s) ({leg.binding_cap} cap binding) | "
            f"premium stop {leg.premium_stop:.2f} | {tally}"
        )
    else:
        contracts = signal.futures_leg.contracts if signal.futures_leg else 0
        body = (
            f"entry {signal.entry_price:g} | SL {signal.stop_price:g} "
            f"({signal.stop_source}) | TP {signal.target_price:g} ({signal.target_r}R) | "
            f"trail arms at {signal.trail.activate_at:g} | "
            f"{contracts} contract(s) | {tally}"
        )

    line = head + body

    if signal.chain_context is not None:
        chain = signal.chain_context
        if signal.direction is Direction.LONG and chain.max_put_oi_strike:
            line += f" | max put OI {chain.max_put_oi_strike:g} supports"
        elif signal.direction is Direction.SHORT and chain.max_call_oi_strike:
            line += f" | max call OI {chain.max_call_oi_strike:g} caps"

    if Flag.SENSEX_DELAY in signal.flags:
        line += "\n[!] Sensex feed delay up to 15 min - price may be stale."
    for flag in (Flag.EXPIRY_DAY, Flag.IV_ELEVATED, Flag.PCR_EXTREME, Flag.GAP_SESSION):
        if flag in signal.flags:
            line += f" | {flag.value}"
    return line
