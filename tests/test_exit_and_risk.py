"""Sections 6 and 7 - the exit plan, the trade management rules, and position sizing."""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from beast.analysis.levels import Zone
from beast.constants import Direction, ExitReason, Market, Tier, ZoneKind
from beast.entry.setups import SetupInstance
from beast.exit import manager as exit_manager
from beast.exit import plan as plan_mod
from beast.exit.manager import Position, SessionPhase
from beast.risk import sizing
from beast.risk.limits import RiskState
from beast.schemas import OptionLeg, Signal, TradeRecord, TrailSpec
from conftest import make_chain

NOW = datetime(2026, 9, 1, 10, 42)


def context(cfg, atr=20.0, zones=None, market=Market.NIFTY, chain=None):
    return SimpleNamespace(
        cfg=cfg, market=market, now=NOW, atr=atr, atr_median=atr,
        zones=zones or [], chain=chain, next_chain=None, futures_contract=None,
        pending_direction=Direction.LONG,
    )


def setup_at(stop_structural, direction=Direction.LONG, setup_type=2, zone=None):
    return SetupInstance(
        setup_type=setup_type, direction=direction, ref="zone:test", detected_idx=10,
        detected_ts=NOW, structural_price=stop_structural, stop_source="rejection_wick",
        zone=zone, meta={},
    )


# -- 6.1 / 6.2 ---------------------------------------------------------------


def test_plan_reproduces_the_soul_file_worked_example(ready_cfg):
    """7.1's worked example: entry 24180, stop 24130, 2R target, trail arming at +1R."""
    plan = plan_mod.build(setup_at(24135.0), context(ready_cfg), 24180.0)
    assert plan.viable
    assert plan.stop_price == pytest.approx(24130.0)
    assert plan.r_points == pytest.approx(50.0)
    assert plan.target_price == pytest.approx(24280.0)
    assert plan.trail.activate_at == pytest.approx(24230.0)


def test_stop_tighter_than_the_minimum_is_widened(ready_cfg):
    plan = plan_mod.build(setup_at(24178.0), context(ready_cfg), 24180.0)
    assert plan.viable
    assert plan.r_points == pytest.approx(0.5 * 20.0)
    assert "widened" in plan.stop_source


def test_stop_wider_than_the_maximum_is_rejected_not_sized_down(ready_cfg):
    plan = plan_mod.build(setup_at(24100.0), context(ready_cfg), 24180.0)
    assert not plan.viable
    assert "never sized down" in plan.reject_reason


def test_blocking_tier_a_level_rejects_the_trade(ready_cfg):
    wall = Zone(kind=ZoneKind.RESISTANCE, low=24220.0, high=24230.0, tier=Tier.A, source="oi:max_call_oi")
    plan = plan_mod.build(setup_at(24135.0), context(ready_cfg, zones=[wall]), 24180.0)
    assert not plan.viable
    assert "blocks the 2.0R target" in plan.reject_reason


def test_target_level_policy_targets_the_level_with_a_floor(ready_cfg):
    cfg = ready_cfg.with_overrides(**{"exit.target_infeasible_policy": "target_level_min_1.5R"})
    wall = Zone(kind=ZoneKind.RESISTANCE, low=24270.0, high=24280.0, tier=Tier.A, source="oi:max_call_oi")
    plan = plan_mod.build(setup_at(24135.0), context(cfg, zones=[wall]), 24180.0)
    assert plan.viable
    assert plan.target_price == pytest.approx(24270.0)
    assert plan.target_r == pytest.approx(1.8)


def test_expiry_day_arms_the_trail_earlier(ready_cfg):
    chain = make_chain(spot=24180.0, when=NOW, dte=0)
    plan = plan_mod.build(setup_at(24135.0), context(ready_cfg, chain=chain), 24180.0)
    assert plan.trail.activate_at == pytest.approx(24180.0 + 0.7 * 50.0)


# -- 6.3 / 6.6 / 6.7 / 6.8 / 6.10 -------------------------------------------


def position(cfg, entry=24180.0, stop=24130.0, target=24280.0, premium=180.0, premium_stop=117.0):
    leg = OptionLeg("2026-09-03", 2, 24200, "CE", 0.52, 13.4, premium, premium - 0.5, premium + 0.5,
                    1_450_000, 320_000, premium_stop, lots=3, lot_size=75)
    signal = Signal(
        market=Market.NIFTY, direction=Direction.LONG, setup_type=2, setup_ref="zone:test",
        regime=__import__("beast.constants", fromlist=["Regime"]).Regime.RANGE, counter_bias=False,
        confluence_mode=__import__("beast.constants", fromlist=["ConfluenceMode"]).ConfluenceMode.REVERSAL,
        confluence_count={"aligned": 4, "opposing": 1, "neutral": 1}, indicator_reads={},
        timeframes={"bias": "15M", "setup": "5M", "trigger": "1M"}, entry_price=entry,
        stop_price=stop, stop_source="rejection_wick", target_price=target, target_r=2.0,
        trail=TrailSpec(entry + 50.0, "atr_chandelier", 1.5), risk_pct=0.03, vol_factor=1.0,
        atr_setup_tf=20.0, leg=leg.to_dict(),
    )
    return Position(
        signal=signal, record=TradeRecord(signal=signal), direction=Direction.LONG,
        entry_price=entry, stop_price=stop, target_price=target, r_points=entry - stop,
        trail_method="atr_chandelier", trail_mult=1.5, trail_activate_at=entry + 50.0,
        entry_time=NOW, entry_premium=premium, premium_stop=premium_stop,
        high_since_entry=entry, low_since_entry=entry,
    )


def test_trail_activates_to_breakeven_then_ratchets_only_tighter(ready_cfg):
    pos = position(ready_cfg)
    pos.observe(high=24232.0, low=24180.0)
    pos.update_trail(20.0, ready_cfg)
    assert pos.trail_activated
    assert pos.stop_price == pytest.approx(24202.0)  # 24232 - 1.5 * 20

    pos.observe(high=24210.0, low=24200.0)  # a pullback must not loosen the stop
    pos.update_trail(20.0, ready_cfg)
    assert pos.stop_price == pytest.approx(24202.0)


def test_stop_is_assumed_filled_first_when_both_hit_in_one_candle(ready_cfg):
    pos = position(ready_cfg)
    decision = exit_manager.evaluate(
        pos, high=24300.0, low=24120.0, close=24290.0, now=NOW, cfg=ready_cfg, market=Market.NIFTY
    )
    assert decision.reason is ExitReason.SL
    assert "never the favourable fill" in decision.detail


def test_gap_through_the_stop_records_the_real_fill(ready_cfg):
    pos = position(ready_cfg)
    decision = exit_manager.evaluate(
        pos, high=24100.0, low=24050.0, close=24060.0, now=NOW, cfg=ready_cfg,
        market=Market.NIFTY, gap_price=24080.0,
    )
    assert decision.price == 24080.0
    assert pos.r_at(decision.price) < -1.0, "worse than -1R is recorded, not clamped"


def test_premium_hard_stop_can_only_cut_risk_short(ready_cfg):
    pos = position(ready_cfg)
    decision = exit_manager.evaluate(
        pos, high=24185.0, low=24160.0, close=24170.0, now=NOW, cfg=ready_cfg,
        market=Market.NIFTY, premium=110.0,
    )
    assert decision.reason is ExitReason.PREMIUM_STOP


def test_session_hard_flat_closes_everything(ready_cfg):
    pos = position(ready_cfg)
    decision = exit_manager.evaluate(
        pos, high=24185.0, low=24175.0, close=24180.0, now=NOW.replace(hour=15, minute=26),
        cfg=ready_cfg, market=Market.NIFTY,
    )
    assert decision.reason is ExitReason.SESSION


def test_theta_guard_flags_but_does_not_act(ready_cfg):
    pos = position(ready_cfg)
    later = NOW + timedelta(minutes=50)
    assert pos.check_theta_guard(later, ready_cfg)
    assert "THETA_DRAG" in pos.flags
    decision = exit_manager.evaluate(
        pos, high=24185.0, low=24175.0, close=24180.0, now=later, cfg=ready_cfg,
        market=Market.NIFTY, premium=175.0,
    )
    assert decision is None, "the theta guard is logged only, never a forced exit"


def test_session_phases(ready_cfg):
    def phase(h, m):
        return exit_manager.session_phase(Market.NIFTY, NOW.replace(hour=h, minute=m), ready_cfg)

    assert phase(10, 0) == SessionPhase.NORMAL
    assert phase(15, 5) == SessionPhase.NO_NEW_ENTRIES
    assert phase(15, 21) == SessionPhase.FLATTEN
    assert phase(15, 26) == SessionPhase.HARD_FLAT


def test_opening_range_guard_and_last_entry_cutoff(ready_cfg):
    ok, why = exit_manager.entries_allowed(Market.NIFTY, NOW.replace(hour=9, minute=20), ready_cfg)
    assert not ok and "opening-range guard" in why
    ok, _ = exit_manager.entries_allowed(Market.NIFTY, NOW.replace(hour=9, minute=31), ready_cfg)
    assert ok
    ok, why = exit_manager.entries_allowed(Market.NIFTY, NOW.replace(hour=15, minute=5), ready_cfg)
    assert not ok and "last-entry cutoff" in why


def test_expiry_day_moves_the_cutoff_earlier(ready_cfg):
    ok, why = exit_manager.entries_allowed(
        Market.NIFTY, NOW.replace(hour=14, minute=0), ready_cfg, expiry_day=True
    )
    assert not ok and "13:30" in why


# -- Section 7 ---------------------------------------------------------------


def test_option_sizing_worked_example(ready_cfg):
    plan = plan_mod.build(setup_at(24135.0), context(ready_cfg), 24180.0)
    leg = OptionLeg("2026-09-03", 2, 24200, "CE", 0.52, 13.4, 180.0, 179.5, 180.5,
                    1_450_000, 320_000, 117.0)
    sized = sizing.size_option(plan, leg, ready_cfg, Market.NIFTY, 20.0, 20.0)
    assert sized.permitted
    assert sized.quantity == 3, "the premium outlay cap binds before the risk cap"
    assert sized.binding_cap == "premium_outlay"
    assert sized.total_premium_outlay == pytest.approx(3 * 75 * 180.0)


def test_risk_cap_binds_on_a_cheap_leg(ready_cfg):
    plan = plan_mod.build(setup_at(24135.0), context(ready_cfg), 24180.0)
    cheap = OptionLeg("2026-09-03", 2, 24200, "CE", 0.52, 13.4, 40.0, 39.5, 40.5,
                      1_450_000, 320_000, 26.0)
    sized = sizing.size_option(plan, cheap, ready_cfg, Market.NIFTY, 20.0, 20.0)
    assert sized.binding_cap == "risk"
    assert sized.quantity == 7  # 15000 / (50 * 0.52 * 75) = 7.69 -> 7


def test_sub_one_lot_is_rejected_at_g9(ready_cfg):
    small = ready_cfg.with_overrides(capital=20000.0)
    plan = plan_mod.build(setup_at(24135.0), context(small), 24180.0)
    leg = OptionLeg("2026-09-03", 2, 24200, "CE", 0.52, 13.4, 180.0, 179.5, 180.5,
                    1_450_000, 320_000, 117.0)
    sized = sizing.size_option(plan, leg, small, Market.NIFTY, 20.0, 20.0)
    assert not sized.permitted
    assert "below one lot" in sized.reason


def test_vol_factor_only_ever_shrinks(ready_cfg):
    assert sizing.vol_factor(atr_current=10.0, atr_median=20.0, cfg=ready_cfg) == 1.0
    assert sizing.vol_factor(atr_current=40.0, atr_median=20.0, cfg=ready_cfg) == 0.5
    assert sizing.vol_factor(atr_current=100.0, atr_median=20.0, cfg=ready_cfg) == 0.5


def test_gold_refuses_to_size_until_contract_specs_exist(cfg):
    plan = plan_mod.build(setup_at(2414.0), context(cfg, atr=4.0, market=Market.GOLD), 2418.4)
    sized = sizing.size_futures(plan, cfg, Market.GOLD, 4.0, 4.0)
    assert not sized.permitted
    assert "contract_multiplier" in sized.reason


def test_daily_loss_cap_and_consecutive_losses_pause_the_market(ready_cfg):
    state = RiskState(ready_cfg)
    for _ in range(3):
        state.record_trade_result(Market.NIFTY, NOW, -1000.0)
    paused, why = state.is_paused(Market.NIFTY, NOW)
    assert paused and "consecutive" in why
    assert not state.is_paused(Market.GOLD, NOW)[0], "the pause is per-market"


def test_correlated_nifty_sensex_rules(ready_cfg):
    state = RiskState(ready_cfg)
    state.open_position(Market.NIFTY, Direction.LONG)
    ok, why = state.can_open(Market.SENSEX, Direction.SHORT)
    assert not ok and "opposite-direction" in why
    ok, _ = state.can_open(Market.SENSEX, Direction.LONG)
    assert ok
    assert state.correlated_risk_budget(Market.SENSEX, Direction.LONG) == 0.5


def test_post_loss_cooldown(ready_cfg):
    state = RiskState(ready_cfg)
    state.record_trade_result(Market.NIFTY, NOW, -500.0)
    cooling, _ = state.in_cooldown(Market.NIFTY, NOW + timedelta(minutes=5))
    assert cooling
    cooling, _ = state.in_cooldown(Market.NIFTY, NOW + timedelta(minutes=16))
    assert not cooling


def test_level_blacklist_after_two_failed_attempts(ready_cfg):
    state = RiskState(ready_cfg)
    state.note_level_attempt(Market.NIFTY, NOW, "zone:x")
    assert not state.level_blacklisted(Market.NIFTY, NOW, "zone:x")
    state.note_level_attempt(Market.NIFTY, NOW, "zone:x")
    assert state.level_blacklisted(Market.NIFTY, NOW, "zone:x")
