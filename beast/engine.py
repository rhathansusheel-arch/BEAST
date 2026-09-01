"""Section 10 - operational mode, and the orchestration around the gate chain.

**Current phase: paper trading, alert-only.** Beast identifies and logs every qualifying
trade as if live but places no real orders. All performance tracking (Section 9) runs
identically to live mode so the data is comparable later. Fills are simulated at the
trigger candle's close, with the conservative assumptions in 6.8.

The order of work in a cycle matters and is fixed: **exits first, then entries.** A no-trade
condition blocks entries only; Section 6 exits keep running regardless (5.6, 6.8).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from beast.analysis.context import Context, LevelState, MarketFeed, build_context
from beast.constants import Direction, ExitReason, Market
from beast.entry import pipeline
from beast.entry.lifecycle import Lifecycle
from beast.exit import manager as exit_manager
from beast.exit.manager import Position, SessionPhase
from beast.ops.learning import Learning
from beast.ops.override import OverrideGuard, OverrideRequest
from beast.ops.precedence import integrity_report, soul_sha256
from beast.ops.reporting import alert, exit_line, flag_alerts
from beast.ops.store import Store
from beast.risk.limits import RiskState
from beast.schemas import Signal, TradeRecord


@dataclass
class CycleResult:
    """What one evaluation cycle produced, ready to be printed or alerted on."""

    market: Market
    now: datetime
    signal: Optional[Signal] = None
    closed: list[TradeRecord] = field(default_factory=list)
    rejections: list = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        out = list(self.alerts)
        out += [exit_line(r) for r in self.closed]
        if self.signal is not None:
            out.append(self.signal.reason_line)
        return out


class Beast:
    """The agent. One instance owns session state across all markets it trades."""

    def __init__(self, cfg, calendar=None) -> None:
        _assert_supported_config(cfg)
        self.cfg = cfg
        self.calendar = calendar
        self.store = Store(cfg)
        self.risk = RiskState(cfg)
        self.learning = Learning(cfg)
        self.overrides = OverrideGuard(cfg)
        self.lifecycles: dict[Market, Lifecycle] = {}
        self.level_states: dict[Market, LevelState] = {}
        self.positions: dict[Market, Position] = {}
        self.soul_sha256 = soul_sha256()

    # -- introspection ---------------------------------------------------------

    def integrity(self) -> dict:
        """What Beast is currently governed by, and what is blocking it from trading."""
        return integrity_report(self.cfg)

    # -- the cycle -------------------------------------------------------------

    def on_trigger_close(self, feed: MarketFeed, now: datetime) -> CycleResult:
        """Run one evaluation cycle for one instrument on a closed trigger-TF candle."""
        market = feed.market
        lifecycle = self.lifecycles.setdefault(market, Lifecycle())
        level_state = self.level_states.setdefault(market, LevelState())

        ctx = build_context(feed, self.cfg, now, level_state, self.calendar)
        result = CycleResult(market=market, now=now)
        result.alerts.extend(flag_alerts(market, ctx.flags))

        # 1. Exits always run first and are never blocked by a no-trade condition.
        closed = self._manage_position(ctx, feed)
        if closed is not None:
            result.closed.append(closed)

        # 2. Entries.
        outcome = pipeline.evaluate(ctx, self.risk, lifecycle, self.calendar, self.learning)
        for rejection in outcome.rejections:
            self.learning.record_rejection(rejection)
            self.store.record_rejection(rejection)
        result.rejections = outcome.rejections

        if outcome.signal is not None:
            self.store.record_signal(outcome.signal)
            result.signal = outcome.signal
            self._open_position(outcome.signal, ctx)
        return result

    # -- position handling -----------------------------------------------------

    def _open_position(self, signal: Signal, ctx: Context) -> None:
        """Open the position the signal describes.

        In paper mode (Section 10) the fill is simulated at the trigger candle's close and
        no order is sent. In live mode this is where execution would be called - the
        bookkeeping below is identical either way, deliberately.
        """
        record = TradeRecord(
            signal=signal,
            entry_time=ctx.now,
            entry_fill_price=signal.entry_price,
            entry_underlying=signal.entry_price,
        )
        option = signal.leg.get("_option_only")
        premium = float(option["mid_premium"]) if option else None
        if option:
            record.entry_premium = premium
            record.delta_at_entry = float(option["delta"])
            record.iv_at_entry = float(option["iv"])
            record.dte = int(option["dte"])

        position = Position(
            signal=signal,
            record=record,
            direction=signal.direction,
            entry_price=signal.entry_price,
            stop_price=signal.stop_price,
            target_price=signal.target_price,
            r_points=abs(signal.entry_price - signal.stop_price),
            trail_method=signal.trail.method,
            trail_mult=signal.trail.mult,
            trail_activate_at=signal.trail.activate_at,
            entry_time=ctx.now,
            entry_premium=premium,
            premium_stop=float(option["premium_stop"]) if option else None,
            flags=list(signal.flags),
        )
        position.high_since_entry = signal.entry_price
        position.low_since_entry = signal.entry_price
        self.positions[ctx.market] = position
        self.risk.open_position(ctx.market, signal.direction)

    def _manage_position(self, ctx: Context, feed: MarketFeed) -> Optional[TradeRecord]:
        """Run Sections 6.3 - 6.10 against the live bar for an open position."""
        position = self.positions.get(ctx.market)
        if position is None:
            return None

        bar = ctx.trigger_ind.iloc[-1]
        high, low, close = float(bar["high"]), float(bar["low"]), float(bar["close"])
        premium = _premium_estimate(position, close)
        position.observe(high, low, premium)

        phase = exit_manager.session_phase(ctx.market, ctx.now, self.cfg)
        recent_swing = _recent_swing(ctx, position.direction)
        if phase is SessionPhase.FLATTEN and recent_swing is not None:
            position.tighten_for_flatten(recent_swing, ctx.atr, self.cfg)
        else:
            position.update_trail(ctx.atr, self.cfg, recent_swing)

        if ctx.market.is_option_market:
            position.check_theta_guard(ctx.now, self.cfg)

        decision = exit_manager.evaluate(
            position, high, low, close, ctx.now, self.cfg, ctx.market, premium=premium
        )
        if decision is None:
            return None
        return self._close_position(ctx, position, decision)

    def _close_position(self, ctx: Context, position: Position, decision) -> TradeRecord:
        record = position.record
        record.exit_time = ctx.now
        record.exit_price = decision.price
        record.exit_reason = decision.reason
        record.bars_held = position.bars_held
        record.trail_activated = position.trail_activated
        record.mae_r = position.mae_r
        record.mfe_r = position.mfe_r
        record.exit_underlying = decision.price
        record.exit_premium = decision.premium
        record.mae_premium = position.mae_premium
        record.mfe_premium = position.mfe_premium
        record.signal.flags = sorted(set(record.signal.flags) | set(position.flags))

        for key, value in exit_manager.r_multiples(position, decision.price, decision.premium).items():
            setattr(record, key, value)

        self.positions.pop(ctx.market, None)
        self.risk.close_position(ctx.market)

        pnl = _pnl(record, position)
        self.risk.record_trade_result(ctx.market, ctx.now, pnl)
        if pnl < 0 and record.signal.setup_ref:
            self.risk.note_level_attempt(ctx.market, ctx.now, record.signal.setup_ref)

        self.learning.record_trade(record)
        self.store.record_trade(record)

        paused, why = self.risk.is_paused(ctx.market, ctx.now)
        if paused:
            # Section 10 - on hitting the limit Beast auto-pauses and says why.
            record.signal.flags.append("PAUSED")
            self.store.record_signal(record.signal)
        return record

    # -- Section 8 -------------------------------------------------------------

    def request_override(
        self,
        market: Market,
        request: OverrideRequest,
        now: datetime,
        mark_price: Optional[float] = None,
        mark_premium: Optional[float] = None,
        hypothetical_r: Optional[float] = None,
    ) -> TradeRecord:
        """Route an operator override through the Section 8 friction step.

        Raises :class:`~beast.ops.override.OverrideRefused` unless the exact confirmation
        phrase was typed. Every attempt is logged either way, with the trade context and
        ``hypothetical_r_if_held_to_target`` - the number that makes the weekly override
        report in Section 8.4 possible.
        """
        position = self.positions.get(market)
        if position is None:
            raise ValueError(f"no open {market.value} position to override")

        record = self.overrides.authorise(request, position, now, hypothetical_r)
        self.store.record_override(record)

        price = mark_price if mark_price is not None else position.entry_price
        decision = exit_manager.ExitDecision(
            ExitReason.OVERRIDE,
            price,
            "operator override - logged as a rule deviation (Section 8)",
            premium=mark_premium,
        )
        ctx = _MarketContext(self.cfg, market, now)
        trade = self._close_position(ctx, position, decision)
        trade.override = record
        trade.hypothetical_r_if_held_to_target = hypothetical_r
        self.store.record_trade(trade)
        return trade

    # -- Section 9 / 11 --------------------------------------------------------

    def weekly_report(self) -> dict:
        report = self.learning.weekly_summary(override_count=self.overrides.count)
        report["overrides"] = self.overrides.weekly_report()
        report["soul_sha256"] = self.soul_sha256
        return report


class UnsupportedConfig(Exception):
    """A config toggle is on that this build does not implement."""


def _assert_supported_config(cfg) -> None:
    """Refuse to start on a toggle Beast would otherwise silently ignore.

    The Soul File ships three switches whose *enabled* behaviour it deliberately leaves
    unbuilt - partial exits (6.4), the OI tag as a gate (4.7.2) and live execution
    (Section 10). Turning one on and having Beast quietly carry on as before would be the
    worst of both worlds: the operator would believe a rule is active that is not. So this
    fails loudly instead.
    """
    problems = []
    if cfg.get("exit.partial_exit_enabled", False):
        problems.append(
            "exit.partial_exit_enabled is true, but 6.4 leaves the shape unconfirmed "
            "(50% at +1R, remainder trailed) - build it before switching it on"
        )
    if cfg.get("options.oi_tag_as_gate", False):
        problems.append(
            "options.oi_tag_as_gate is true, but 4.7.2 defines no gating rule for the OI tag - "
            "it is context until trade history says otherwise"
        )
    if not cfg.is_paper:
        problems.append(
            "mode is 'live', but Section 10's current phase is paper/alert-only and no broker "
            "execution path exists - Beast will not pretend to place orders"
        )
    if problems:
        raise UnsupportedConfig("; ".join(problems))


@dataclass
class _MarketContext:
    """The minimum a close needs when there is no evaluation cycle behind it (Section 8)."""

    cfg: object
    market: Market
    now: datetime


def _premium_estimate(position: Position, underlying_close: float) -> Optional[float]:
    """A delta-linear premium mark for paper mode.

    Real premium comes from the feed in live mode. This estimate exists only so paper-mode
    bookkeeping (6.10's premium stop, Appendix C's premium fields) has something honest to
    work with; it deliberately ignores theta and vega, which is precisely why 6.10's
    premium hard stop exists as a real-money backstop rather than a modelled one.
    """
    option = position.signal.leg.get("_option_only")
    if not option or position.entry_premium is None:
        return None
    delta = float(option["delta"])
    move = (underlying_close - position.entry_price) * position.direction.sign
    return max(0.0, position.entry_premium + move * delta)


def _recent_swing(ctx: Context, direction: Direction) -> Optional[float]:
    """The most recent confirmed swing in the trade's favour, for structure trailing (6.3)."""
    from beast.analysis.levels import find_swings, last_swing

    kind = "low" if direction is Direction.LONG else "high"
    swings = find_swings(
        ctx.trigger_ind, int(ctx.cfg.get("levels.fractal_n")), int(ctx.cfg.get("levels.swing_lookback"))
    )
    swing = last_swing(swings, kind)
    return swing.price if swing else None


def _pnl(record: TradeRecord, position: Position) -> float:
    """Realised P&L in account currency, for the Section 7 daily loss state."""
    option = record.signal.leg.get("_option_only")
    if option and record.exit_premium is not None and record.entry_premium is not None:
        return (record.exit_premium - record.entry_premium) * int(option["lots"]) * int(option["lot_size"])
    futures = record.signal.leg.get("_futures_only")
    if futures and record.exit_price is not None:
        move = (record.exit_price - position.entry_price) * position.direction.sign
        return move * float(futures["contract_multiplier"]) * int(futures["contracts"])
    return 0.0
