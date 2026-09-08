"""Risk management - section 7, 7.1, and the exit rules in section 6.

These are the tests that matter most, because every rule they cover appears in
the immutable list in section 13. A regression here is not a bug in a feature,
it is a violation of the document.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from core.config import ConfigBlockerError
from core.exit_manager import ExitManager, ManagedPosition, TradePlanBuilder
from core.levels import LevelEngine
from core.risk_manager import RiskManager
from core.schemas import (
    Direction,
    ExitReason,
    FuturesLeg,
    LevelKind,
    LevelTier,
    OptionLeg,
    SetupInstance,
    SetupType,
    TradePlan,
    TrailPlan,
    Zone,
)
from core.session import SessionClock

IST_NOW = datetime(2026, 9, 2, 11, 0)


def make_plan(entry: float = 24180.0, stop: float = 24130.0, target: float = 24280.0,
              direction: Direction = Direction.LONG) -> TradePlan:
    risk = abs(entry - stop)
    return TradePlan(
        direction=direction,
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        stop_source="test",
        target_r=2.0,
        trail=TrailPlan(entry + direction.sign * risk, 1.0, "atr_chandelier", 1.5),
        atr=20.0,
        risk_points=risk,
    )


def make_option_leg(delta: float = 0.52, premium: float = 180.0,
                    lot_size: int = 75) -> OptionLeg:
    return OptionLeg(
        expiry=date(2026, 9, 3), dte=1, strike=24200.0, option_type="CE",
        delta=delta, iv=13.4, mid_premium=premium, bid=premium - 0.5,
        ask=premium + 0.5, oi=1_450_000, volume=320_000,
        premium_stop=round(premium * 0.65, 2), lot_size=lot_size,
    )


# ---------------------------------------------------------------------------
# 7.1 - delta-based option sizing
# ---------------------------------------------------------------------------


class TestOptionSizing:
    def test_worked_example_from_the_soul_file(self, cfg):
        """Soul file 7.1's worked example, reproduced exactly.

        Capital 5,00,000, Nifty risk 5% (v3.1) = 25,000. Entry 24,180, stop
        24,130 (50 pts). Leg 24,200 CE, delta 0.52, lot size 75. Loss per lot =
        50 x 0.52 x 75 = 1,950, so risk allows 12 lots. Premium 180 means 12
        lots would cost 1,62,000 against the 50,000 outlay cap, so the outlay
        cap binds at 3 lots. Raising Nifty risk to 5% did not raise the number
        of lots at all - the outlay cap absorbed the whole increase, which is
        soul file open question D.
        """
        risk = RiskManager(cfg, capital=500_000)
        plan = make_plan()
        leg = make_option_leg()

        sized = risk.size_option(plan, leg, "NIFTY50", vol_factor=1.0)

        assert sized.permitted is True
        assert sized.quantity == 3
        assert sized.binding_cap == "premium_outlay"
        assert leg.lots == 3
        assert leg.total_premium_outlay == pytest.approx(3 * 75 * 180)

    def test_risk_cap_binds_on_a_cheap_leg(self, cfg):
        """With a low premium the risk cap, not the outlay cap, is the limit."""
        risk = RiskManager(cfg, capital=500_000)
        sized = risk.size_option(make_plan(), make_option_leg(premium=20.0), "NIFTY50", 1.0)
        assert sized.permitted is True
        assert sized.binding_cap == "risk"
        assert sized.quantity == 12         # v3.1: floor(25000 / 1950), risk 5%

    def test_sub_one_lot_is_rejected_never_rounded_up(self, cfg):
        """Rule 13.11: never round a sub-1-lot position up to force a trade."""
        risk = RiskManager(cfg, capital=20_000)
        sized = risk.size_option(make_plan(), make_option_leg(), "NIFTY50", 1.0)
        assert sized.permitted is False
        assert sized.quantity == 0
        assert "13.11" in sized.reason or "Rejected" in sized.reason

    def test_vol_factor_only_shrinks(self, cfg):
        risk = RiskManager(cfg, capital=500_000)
        full = risk.size_option(make_plan(), make_option_leg(premium=20.0), "NIFTY50", 1.0)
        shrunk = risk.size_option(make_plan(), make_option_leg(premium=20.0), "NIFTY50", 0.5)
        assert shrunk.quantity <= full.quantity

    def test_vol_factor_is_capped_at_one(self, cfg):
        risk = RiskManager(cfg)
        # Current ATR far below the median would otherwise scale exposure up.
        assert risk.vol_factor(atr_current=1.0, atr_median=100.0) == 1.0

    def test_vol_factor_respects_the_floor(self, cfg):
        risk = RiskManager(cfg)
        floor = float(cfg.get("risk.vol_factor_floor"))
        assert risk.vol_factor(atr_current=1000.0, atr_median=1.0) == floor

    def test_missing_atr_means_no_adjustment(self, cfg):
        risk = RiskManager(cfg)
        assert risk.vol_factor(0.0, 0.0) == 1.0


class TestFuturesSizing:
    def test_linear_sizing(self, cfg):
        risk = RiskManager(cfg, capital=500_000)
        plan = make_plan(entry=2418.4, stop=2414.1, target=2427.0)
        leg = FuturesLeg("GOLDM26OCT", contract_multiplier=10.0, tick_size=1.0, tick_value=10.0)
        sized = risk.size_futures(plan, leg, "XAUUSD", vol_factor=1.0)

        # 2% of 500,000 = 10,000; loss per contract = 4.3 x 10 = 43.
        assert sized.permitted is True
        assert sized.quantity == int(10_000 // 43)

    def test_sub_one_contract_is_rejected(self, cfg):
        risk = RiskManager(cfg, capital=1_000)
        plan = make_plan(entry=2418.4, stop=2400.0, target=2450.0)
        leg = FuturesLeg("GOLDM26OCT", 100.0, 1.0, 10.0)
        sized = risk.size_futures(plan, leg, "XAUUSD", 1.0)
        assert sized.permitted is False


class TestConfigBlockers:
    def test_unset_gold_specs_refuse_to_size(self, raw_config):
        """Soul file 3.1: Beast refuses to size a Gold trade until specs exist."""
        from core.instrument_selector import FuturesContractSelector

        selector = FuturesContractSelector(raw_config)
        result = selector.select([("GOLDM26OCT", date(2026, 10, 28))], IST_NOW, "XAUUSD")
        assert result.ok is False
        assert "contract_multiplier" in result.reason or "unset" in result.reason

    def test_unset_liquidity_floors_fail_closed(self, raw_config):
        """5.7.3: an unset filter is treated as failing, not passing."""
        from core.instrument_selector import OptionLegSelector
        from core.option_chain import OptionQuote

        selector = OptionLegSelector(raw_config)
        quote = OptionQuote(24200, "CE", bid=179.5, ask=180.5, oi=10**7,
                            volume=10**6, iv=13.4, delta=0.52)
        ok, reason = selector.passes_liquidity(quote)
        assert ok is False
        assert "min_oi" in reason or "unset" in reason


# ---------------------------------------------------------------------------
# 7 - loss limits and exposure
# ---------------------------------------------------------------------------


class TestLossLimits:
    def test_three_consecutive_losses_pause_the_market(self, cfg):
        risk = RiskManager(cfg, capital=500_000)
        risk.roll_session("indian", date(2026, 9, 2))
        for _ in range(3):
            risk.register_close("NIFTY50", -1000.0, IST_NOW)
        paused, reason = risk.is_paused("NIFTY50")
        assert paused is True
        assert "consecutive" in reason

    def test_a_win_resets_the_consecutive_counter(self, cfg):
        risk = RiskManager(cfg, capital=500_000)
        risk.roll_session("indian", date(2026, 9, 2))
        risk.register_close("NIFTY50", -1000.0, IST_NOW)
        risk.register_close("NIFTY50", -1000.0, IST_NOW)
        risk.register_close("NIFTY50", +500.0, IST_NOW)
        risk.register_close("NIFTY50", -1000.0, IST_NOW)
        assert risk.is_paused("NIFTY50")[0] is False

    def test_daily_percentage_cap_pauses(self, cfg):
        risk = RiskManager(cfg, capital=100_000)
        risk.roll_session("indian", date(2026, 9, 2))
        # v3.1 Nifty cap is 15% of 100,000 = 15,000. Alternate wins so the
        # 3-loss trigger is not what fires - the percentage cap must be able to
        # pause on its own.
        risk.register_close("NIFTY50", -8000.0, IST_NOW)
        risk.register_close("NIFTY50", +100.0, IST_NOW)
        risk.register_close("NIFTY50", -8000.0, IST_NOW)
        paused, reason = risk.is_paused("NIFTY50")
        assert paused is True
        assert "daily loss cap" in reason

    def test_pause_is_per_market(self, cfg):
        risk = RiskManager(cfg, capital=500_000)
        risk.roll_session("indian", date(2026, 9, 2))
        risk.roll_session("gold", date(2026, 9, 2))
        for _ in range(3):
            risk.register_close("NIFTY50", -1000.0, IST_NOW)
        assert risk.is_paused("NIFTY50")[0] is True
        assert risk.is_paused("XAUUSD")[0] is False

    def test_new_session_clears_the_pause(self, cfg):
        risk = RiskManager(cfg, capital=500_000)
        risk.roll_session("indian", date(2026, 9, 2))
        for _ in range(3):
            risk.register_close("NIFTY50", -1000.0, IST_NOW)
        assert risk.is_paused("NIFTY50")[0] is True
        risk.roll_session("indian", date(2026, 9, 3))
        assert risk.is_paused("NIFTY50")[0] is False

    def test_post_loss_cooldown_blocks_entry(self, cfg):
        risk = RiskManager(cfg, capital=500_000)
        risk.roll_session("indian", date(2026, 9, 2))
        risk.register_close("NIFTY50", -500.0, IST_NOW)
        allowed, reason = risk.check_portfolio(
            "NIFTY50", Direction.LONG, 15_000, IST_NOW + timedelta(minutes=5)
        )
        assert allowed is False and "cooldown" in reason

        later = IST_NOW + timedelta(minutes=int(cfg.get("entry.post_loss_cooldown_min")) + 1)
        allowed, _ = risk.check_portfolio("NIFTY50", Direction.LONG, 15_000, later)
        assert allowed is True


class TestCorrelationAndExposure:
    def test_opposite_direction_nifty_sensex_is_refused(self, cfg):
        risk = RiskManager(cfg, capital=500_000)
        risk.register_open("NIFTY50", Direction.LONG, 15_000, IST_NOW)
        allowed, reason = risk.check_portfolio("SENSEX", Direction.SHORT, 15_000, IST_NOW)
        assert allowed is False
        assert "opposite-direction" in reason

    def test_same_direction_combined_risk_is_capped(self, cfg):
        risk = RiskManager(cfg, capital=500_000)
        risk.register_open("NIFTY50", Direction.LONG, 15_000, IST_NOW)
        # A second full-size long would double the Indian exposure.
        allowed, reason = risk.check_portfolio("SENSEX", Direction.LONG, 15_000, IST_NOW)
        assert allowed is False
        assert "single trade" in reason

    def test_same_direction_within_allowance_is_permitted(self, cfg):
        risk = RiskManager(cfg, capital=500_000)
        risk.register_open("NIFTY50", Direction.LONG, 7_000, IST_NOW)
        allowed, _ = risk.check_portfolio("SENSEX", Direction.LONG, 7_000, IST_NOW)
        assert allowed is True

    def test_gold_concurrent_cap_is_one(self, cfg):
        risk = RiskManager(cfg, capital=500_000)
        risk.register_open("XAUUSD", Direction.LONG, 10_000, IST_NOW)
        allowed, reason = risk.check_portfolio("XAUUSD", Direction.LONG, 10_000, IST_NOW)
        assert allowed is False
        assert "concurrent-position cap" in reason


# ---------------------------------------------------------------------------
# 6 - exit plan construction
# ---------------------------------------------------------------------------


def make_setup(structural_stop: float, direction: Direction = Direction.LONG) -> SetupInstance:
    return SetupInstance(
        setup_id="s", setup_type=SetupType.SR_REVERSAL, direction=direction,
        ref_id="z", detected_index=10, detected_at=IST_NOW,
        structural_stop=structural_stop, detail={"stop_source": "rejection wick"},
    )


# A 50-point structural stop on Nifty needs an ATR wide enough for the 6.1
# bounds to admit it: 50 + 0.25xATR buffer must stay inside 2.5xATR.
SETUP_ATR = 25.0


class TestStopPlacement:
    def test_buffer_is_applied_beyond_structure(self, cfg):
        builder = TradePlanBuilder(cfg)
        levels = LevelEngine("NIFTY50", cfg)
        plan = builder.build(make_setup(24130.0), entry_price=24180.0,
                             atr_value=SETUP_ATR, levels=levels)
        assert plan.viable
        buffer_points = float(cfg.get("exit.stop_buffer_atr")) * SETUP_ATR
        assert plan.stop_price == pytest.approx(24130.0 - buffer_points)

    def test_tight_structure_is_widened_to_the_minimum(self, cfg):
        builder = TradePlanBuilder(cfg)
        levels = LevelEngine("NIFTY50", cfg)
        # Structure only 2 points away; the minimum is 0.5 x ATR = 10 points.
        plan = builder.build(make_setup(24178.0), entry_price=24180.0,
                             atr_value=SETUP_ATR, levels=levels)
        assert plan.viable
        assert plan.risk_points == pytest.approx(0.5 * SETUP_ATR)
        assert "widened" in plan.stop_source

    def test_too_wide_structure_is_rejected_not_sized_down(self, cfg):
        """6.1: the maximum is a hard trade-rejection, not a size-down."""
        builder = TradePlanBuilder(cfg)
        levels = LevelEngine("NIFTY50", cfg)
        plan = builder.build(make_setup(24000.0), entry_price=24180.0,
                             atr_value=SETUP_ATR, levels=levels)
        assert plan.viable is False
        assert "maximum" in (plan.reject_reason or "")

    def test_target_is_two_r_by_default(self, cfg):
        builder = TradePlanBuilder(cfg)
        levels = LevelEngine("NIFTY50", cfg)
        plan = builder.build(make_setup(24130.0), 24180.0, SETUP_ATR, levels)
        assert plan.target_r == pytest.approx(2.0)
        assert plan.target_price == pytest.approx(
            plan.entry_price + 2.0 * plan.risk_points
        )

    def test_opposing_tier_a_level_rejects_the_trade(self, cfg):
        """G7: Beast does not shrink its R:R to make a marginal setup fit."""
        builder = TradePlanBuilder(cfg)
        levels = LevelEngine("NIFTY50", cfg)
        levels.zones = [
            Zone(zone_id="wall", kind=LevelKind.MAX_CALL_OI, tier=LevelTier.A,
                 low=24230.0, high=24240.0, centre=24235.0, is_support=False)
        ]
        plan = builder.build(make_setup(24130.0), 24180.0, SETUP_ATR, levels)
        assert plan.viable is False
        assert "opposing Tier A level" in (plan.reject_reason or "")

    def test_alternative_policy_targets_the_level(self, cfg):
        cfg.section("exit")["target_infeasible_policy"] = "target_level_min_1.5R"
        builder = TradePlanBuilder(cfg)
        levels = LevelEngine("NIFTY50", cfg)
        levels.zones = [
            Zone(zone_id="wall", kind=LevelKind.MAX_CALL_OI, tier=LevelTier.A,
                 low=24280.0, high=24290.0, centre=24285.0, is_support=False)
        ]
        plan = builder.build(make_setup(24130.0), 24180.0, SETUP_ATR, levels)
        assert plan.viable is True
        assert plan.target_r >= float(cfg.get("exit.target_level_min_r"))


# ---------------------------------------------------------------------------
# 6.3 / 6.8 / 6.10 - live exit behaviour
# ---------------------------------------------------------------------------


def make_position(cfg, signal_factory, direction: Direction = Direction.LONG,
                  option: bool = False) -> ManagedPosition:
    plan = make_plan(direction=direction)
    signal = signal_factory(plan, option)
    return ManagedPosition(
        signal=signal, plan=plan, entry_time=IST_NOW,
        entry_underlying=plan.entry_price,
        entry_premium=180.0 if option else None,
    )


@pytest.fixture
def signal_factory(cfg):
    from core.schemas import ConfluenceMode, Regime, Signal

    def build(plan: TradePlan, option: bool) -> Signal:
        return Signal(
            market="NIFTY50" if option else "XAUUSD",
            underlying="NIFTY50" if option else "XAUUSD",
            direction=plan.direction,
            setup_type=SetupType.SR_REVERSAL,
            setup_ref="z",
            regime=Regime.RANGE,
            counter_bias=False,
            confluence_mode=ConfluenceMode.REVERSAL,
            confluence_count={"aligned": 4, "opposing": 1, "neutral": 1},
            indicator_reads={},
            timeframes={"bias": "15M", "setup": "5M", "trigger": "1M"},
            entry_price=plan.entry_price,
            stop_price=plan.stop_price,
            stop_source="test",
            target_price=plan.target_price,
            target_r=plan.target_r,
            trail=plan.trail,
            risk_pct=0.03,
            vol_factor=1.0,
            atr_setup_tf=20.0,
            timestamp_ist=IST_NOW,
            leg_type="OPTION" if option else "FUTURES",
            option_leg=make_option_leg() if option else None,
            futures_leg=None if option else FuturesLeg("GOLDM", 10.0, 1.0, 10.0, contracts=1),
        )

    return build


class TestExitBehaviour:
    def test_stop_fills_first_when_both_hit_in_one_bar(self, cfg, signal_factory):
        """6.8: never assume the favourable fill."""
        manager = ExitManager(cfg)
        position = make_position(cfg, signal_factory)
        clock = SessionClock("XAUUSD", cfg)
        decision = manager.on_price(
            position, 24180.0, None, IST_NOW, clock,
            high=24300.0, low=24100.0, bar_open=24180.0,
        )
        assert decision.should_exit
        assert decision.reason is ExitReason.SL

    def test_gap_through_the_stop_records_the_gap_price(self, cfg, signal_factory):
        """6.8: the R-multiple is recorded as actual, worse than -1R."""
        manager = ExitManager(cfg)
        position = make_position(cfg, signal_factory)
        clock = SessionClock("XAUUSD", cfg)
        decision = manager.on_price(
            position, 24050.0, None, IST_NOW, clock,
            high=24070.0, low=24040.0, bar_open=24060.0,
        )
        assert decision.should_exit
        assert decision.exit_price == pytest.approx(24060.0)
        assert position.r_at(decision.exit_price) < -1.0

    def test_stop_never_widens(self, cfg, signal_factory):
        manager = ExitManager(cfg)
        position = make_position(cfg, signal_factory)
        original = position.current_stop
        assert manager._tighten(position, original - 100.0) == original
        assert manager._tighten(position, original + 10.0) == original + 10.0

    def test_premium_stop_fires_before_the_underlying_stop(self, cfg, signal_factory):
        """6.10: the backstop for when delta/IV break the mapping."""
        manager = ExitManager(cfg)
        position = make_position(cfg, signal_factory, option=True)
        clock = SessionClock("NIFTY50", cfg)
        # Underlying is still comfortably above the stop, but premium collapsed.
        decision = manager.on_price(position, 24175.0, 100.0, IST_NOW, clock)
        assert decision.should_exit
        assert decision.reason is ExitReason.PREMIUM_STOP

    def test_premium_stop_is_65_percent_of_entry(self, cfg):
        leg = make_option_leg(premium=200.0)
        expected = 200.0 * (1.0 - float(cfg.get("options.premium_stop_pct")))
        assert leg.premium_stop == pytest.approx(round(expected, 2))

    def test_time_stop_is_off_by_default(self, cfg, signal_factory):
        assert bool(cfg.get("exit.time_stop_enabled")) is False
        manager = ExitManager(cfg)
        position = make_position(cfg, signal_factory)
        position.bars_held = 500
        clock = SessionClock("XAUUSD", cfg)
        decision = manager.on_price(position, 24181.0, None, IST_NOW, clock)
        assert decision.should_exit is False

    def test_theta_guard_flags_but_does_not_exit(self, cfg, signal_factory):
        """6.10: logged only, no forced action."""
        from core.schemas import Flag

        manager = ExitManager(cfg)
        position = make_position(cfg, signal_factory, option=True)
        clock = SessionClock("NIFTY50", cfg)
        later = IST_NOW + timedelta(minutes=int(cfg.get("options.theta_guard_minutes")) + 1)
        decision = manager.on_price(position, 24185.0, 175.0, later, clock)
        assert decision.should_exit is False
        assert Flag.THETA_DRAG in position.flags

    def test_hard_flat_closes_everything(self, cfg, signal_factory):
        manager = ExitManager(cfg)
        position = make_position(cfg, signal_factory, option=True)
        clock = SessionClock("NIFTY50", cfg)
        at_flat = datetime(2026, 9, 2, 15, 26)
        decision = manager.on_price(position, 24185.0, 190.0, at_flat, clock)
        assert decision.should_exit
        assert decision.reason is ExitReason.SESSION

    def test_mae_and_mfe_are_tracked_in_r(self, cfg, signal_factory):
        manager = ExitManager(cfg)
        position = make_position(cfg, signal_factory)
        clock = SessionClock("XAUUSD", cfg)
        manager.on_price(position, 24200.0, None, IST_NOW, clock,
                         high=24210.0, low=24170.0, bar_open=24180.0)
        assert position.mfe_r > 0
        assert position.mae_r < 0
