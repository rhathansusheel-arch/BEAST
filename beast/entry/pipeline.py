"""Section 5.1 - the entry pipeline, as an ordered short-circuiting gate chain.

"Gates run in order and short-circuit. The first failing gate rejects the candidate; the
rejection is recorded with the gate ID so Section 9 can learn where signals die."

G0 session -> G1 data integrity -> G2 no-trade -> G3 regime -> G4 setup -> G5 confluence
-> G6 trigger -> G7 viability -> G8 instrument -> G9 risk -> emit.

Every rejection carries its gate ID and a human-readable detail, because the operator needs
to see whether Beast is missing trades at G5 (confluence too strict) or G7 (structure too
wide) rather than only seeing what it took (Section 9).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from beast.analysis.regime import is_counter_bias, permitted_setups, required_confluence
from beast.constants import Direction, Gate, Market, SETUP_MODES, Tier
from beast.entry import confluence as confluence_engine
from beast.entry import instruments, lifecycle as lifecycle_mod, setups as setups_mod
from beast.exit import manager as exit_manager
from beast.exit import plan as plan_mod
from beast.ops.immutable import assert_exit_plan_complete
from beast.ops.precedence import soul_sha256
from beast.ops.reporting import reason_line
from beast.risk import sizing
from beast.schemas import Rejection, Signal, TrailSpec


@dataclass
class PipelineResult:
    """What one trigger-TF close produced: at most one signal, plus every rejection."""

    signal: Optional[Signal] = None
    rejections: list[Rejection] = field(default_factory=list)

    def reject(self, ctx, gate: Gate, detail: str, setup=None, counts=None) -> "PipelineResult":
        self.rejections.append(
            Rejection(
                timestamp=ctx.now,
                instrument=ctx.market.value,
                setup_type=getattr(setup, "setup_type", None),
                direction=getattr(getattr(setup, "direction", None), "value", None),
                failed_gate=gate,
                gate_detail=detail,
                confluence_count=counts,
            )
        )
        return self


def evaluate(ctx, risk_state, lifecycle, calendar=None, learning=None) -> PipelineResult:
    """Run the G0-G9 chain for one instrument on one trigger-TF close."""
    cfg = ctx.cfg
    result = PipelineResult()

    # G0 - Session (Section 3, 6.7, 5.7.4)
    allowed, why = exit_manager.entries_allowed(ctx.market, ctx.now, cfg, expiry_day=ctx.expiry_day)
    if not allowed:
        return result.reject(ctx, Gate.G0, why)

    # G1 - Data integrity (4.6)
    if not ctx.integrity.ok:
        return result.reject(ctx, Gate.G1, ctx.integrity.detail())

    # G2 - No-trade conditions (5.5, 5.6, 7)
    paused, pause_reason = risk_state.is_paused(ctx.market, ctx.now)
    if paused:
        return result.reject(ctx, Gate.G2, f"market paused - {pause_reason}")
    cooling, cool_reason = risk_state.in_cooldown(ctx.market, ctx.now)
    if cooling:
        return result.reject(ctx, Gate.G2, cool_reason)
    if calendar is not None:
        blackout, news_reason = calendar.in_blackout(ctx.market, ctx.now, cfg)
        if blackout:
            return result.reject(ctx, Gate.G2, news_reason)

    # G4 - Setup detection (5.2, 5.5). New instances are armed, and every instance still
    # inside its 5-candle validity window is re-evaluated - the trigger is allowed to fire
    # on a later trigger-TF close than the one that detected the setup, which is the whole
    # point of the validity window. Detection runs before G3 so the regime gate can be
    # applied per candidate direction, as 4.4's per-direction table requires.
    lifecycle.register(setups_mod.detect_all(ctx), ctx)
    candidates = lifecycle.active(ctx, cfg)
    if not candidates:
        return result.reject(ctx, Gate.G4, "no armed Setup 1-4 instance for this instrument")

    qualified: list[tuple] = []
    setup_tf_minutes = _tf_minutes(ctx.timeframes["setup"])

    for setup in candidates:
        setup.counter_bias = is_counter_bias(ctx.regime.regime, setup.direction)

        # G3 - Regime / bias (4.4)
        permitted = permitted_setups(ctx.regime.regime, setup.direction, cfg)
        if setup.setup_type not in permitted:
            result.reject(
                ctx,
                Gate.G3,
                f"Setup {setup.setup_type} {setup.direction.value} not permitted in {ctx.regime.regime.value}",
                setup,
            )
            continue
        if setup.counter_bias:
            if setup.setup_type != 2:
                result.reject(ctx, Gate.G3, "only Setup 2 may be taken counter-bias", setup)
                continue
            if setup.zone is None or setup.zone.tier is not Tier.A:
                result.reject(
                    ctx, Gate.G3, "counter-bias reversal must be at a Tier A level (4.4)", setup
                )
                continue

        # 5.5 - one instance one signal, level blacklist. (Expiry is applied by
        # ``lifecycle.active`` before a setup ever reaches this loop.)
        if lifecycle.already_traded(setup.ref):
            result.reject(ctx, Gate.G4, "setup instance has already produced a signal (5.5)", setup)
            continue
        if risk_state.level_blacklisted(ctx.market, ctx.now, setup.ref):
            result.reject(
                ctx, Gate.G4, "level blacklisted after repeated failed attempts this session (5.5)", setup
            )
            continue

        # G5 - Confluence (5.3, 5.4)
        mode = SETUP_MODES[setup.setup_type]
        idx = len(ctx.setup_ind) - 1
        reads = confluence_engine.evaluate(ctx.setup_ind, idx, mode, cfg, ctx.atr, zone=setup.zone)
        tally = confluence_engine.tally(reads, setup.direction, mode)
        tightened = bool(learning and learning.is_tightened(ctx.market, setup.setup_type))
        required = required_confluence(setup.counter_bias, cfg, tightened)
        ok, detail = confluence_engine.passes(tally, required, cfg)
        if not ok:
            result.reject(ctx, Gate.G5, detail, setup, tally.as_counts())
            continue

        # G6 - Trigger (5.2, 5.5)
        trigger = setups_mod.trigger_fired(setup, ctx)
        if not trigger.fired:
            result.reject(ctx, Gate.G6, trigger.detail, setup, tally.as_counts())
            continue

        if lifecycle.is_duplicate(
            ctx.market,
            setup.direction,
            setup.setup_type,
            trigger.entry_price,
            ctx.atr,
            ctx.now,
            cfg,
            setup_tf_minutes,
        ):
            result.reject(ctx, Gate.G6, "duplicate signal suppressed (5.5)", setup, tally.as_counts())
            continue

        # G7 - Trade viability (6.1, 6.2, 4.7.1)
        trade_plan = plan_mod.build(setup, ctx, trigger.entry_price)
        if not trade_plan.viable:
            result.reject(ctx, Gate.G7, trade_plan.reject_reason, setup, tally.as_counts())
            continue
        assert_exit_plan_complete(trade_plan)
        qualified.append((setup, trade_plan, tally, trigger))

    if not qualified:
        return result

    # 5.1 - one signal per bar: tighter structural stop wins, then confluence, then the
    # trailing 30-trade expectancy for that setup type (Section 9).
    expectancy = (lambda st: learning.expectancy(ctx.market, st)) if learning else None
    chosen = lifecycle_mod.choose_one([(s, p, t) for s, p, t, _ in qualified], expectancy)
    setup, trade_plan, tally = chosen
    trigger = next(tr for s, _, _, tr in qualified if s is setup)

    # G8 - Instrument selection (5.7)
    ctx.pending_direction = setup.direction
    leg, leg_detail = instruments.select(trade_plan, ctx)
    if leg is None:
        return result.reject(ctx, Gate.G8, leg_detail, setup, tally.as_counts())

    # G9 - Risk & portfolio (7, 7.1)
    can_open, portfolio_detail = risk_state.can_open(ctx.market, setup.direction)
    if not can_open:
        return result.reject(ctx, Gate.G9, portfolio_detail, setup, tally.as_counts())

    if ctx.market is Market.GOLD:
        sized = sizing.size_futures(trade_plan, cfg, ctx.market, ctx.atr, ctx.atr_median)
    else:
        sized = sizing.size_option(trade_plan, leg, cfg, ctx.market, ctx.atr, ctx.atr_median)
    if not sized.permitted:
        return result.reject(ctx, Gate.G9, sized.reason, setup, tally.as_counts())

    budget = risk_state.correlated_risk_budget(ctx.market, setup.direction)
    if budget < 1.0:
        scaled = int(sized.quantity * budget)
        if scaled < 1:
            return result.reject(
                ctx,
                Gate.G9,
                "correlated Nifty/Sensex exposure leaves less than one lot of risk budget (7)",
                setup,
                tally.as_counts(),
            )
        sized.quantity = scaled
        sized.reason += f"; halved for correlated {ctx.market.value} exposure"

    if ctx.market is Market.GOLD:
        leg.contracts = sized.quantity
    else:
        leg.lots = sized.quantity
        leg.lot_size = sized.lot_size
        leg.total_premium_outlay = sized.total_premium_outlay
        leg.binding_cap = sized.binding_cap

    signal = _emit(ctx, setup, trade_plan, tally, leg, sized, leg_detail)
    lifecycle.record_emission(
        ctx.market, setup.direction, setup.setup_type, trade_plan.entry_price, setup.ref, ctx.now
    )
    result.signal = signal
    return result


def _emit(ctx, setup, trade_plan, tally, leg, sized, leg_detail: str) -> Signal:
    """Build the Appendix B Signal object. In paper mode no order follows it (Section 10)."""
    cfg = ctx.cfg
    signal = Signal(
        market=ctx.market,
        direction=setup.direction,
        setup_type=setup.setup_type,
        setup_ref=setup.ref,
        regime=ctx.regime.regime,
        counter_bias=setup.counter_bias,
        confluence_mode=tally.mode,
        confluence_count=tally.as_counts(),
        indicator_reads=tally.read_strings(),
        timeframes=dict(ctx.timeframes),
        entry_price=round(trade_plan.entry_price, 2),
        stop_price=round(trade_plan.stop_price, 2),
        stop_source=trade_plan.stop_source,
        target_price=round(trade_plan.target_price, 2),
        target_r=round(trade_plan.target_r, 2),
        trail=TrailSpec(
            activate_at=round(trade_plan.trail.activate_at, 2),
            method=trade_plan.trail.method,
            mult=trade_plan.trail.mult,
        ),
        risk_pct=sized.risk_pct,
        vol_factor=round(sized.vol_factor, 3),
        atr_setup_tf=round(ctx.atr, 3),
        leg=leg.to_dict(),
        chain_context=ctx.chain_context,
        flags=list(ctx.flags),
        mode=cfg.mode,
        timestamp_ist=ctx.now,
        soul_sha256=soul_sha256(),
    )
    signal.reason_line = reason_line(signal, tally, setup, leg, leg_detail, ctx)
    return signal


def _tf_minutes(tf: str) -> int:
    from beast.analysis.indicators import tf_minutes

    return tf_minutes(tf)
