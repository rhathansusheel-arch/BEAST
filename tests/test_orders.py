"""Execution - order placement, position tracking, the journal, and overrides.

Covers section 10 (paper mode places no orders but tracks identically), 5.7 (G8
strike selection), 6.9 (what every exit records), section 8 (the override
friction step) and Appendix C (persistence).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from broker import BrokerClient, OrderRequest, OrderResult, OrderSide, Quote
from broker.order_executor import OrderExecutor
from broker.position_tracker import PositionTracker
from core.exit_manager import ExitDecision, ExitManager, ManagedPosition
from core.instrument_selector import InstrumentSelector, OptionLegSelector
from core.option_chain import ChainAnalyzer, ChainSnapshot, OptionQuote
from core.override import OverrideAction, OverrideGuard, OverrideRequest
from core.risk_manager import RiskManager
from core.schemas import (
    ConfluenceMode,
    Direction,
    ExitReason,
    FuturesLeg,
    OptionLeg,
    Regime,
    SetupType,
    Signal,
    TradePlan,
    TrailPlan,
)
from core.session import SessionClock
from monitoring.journal import Journal

IST_NOW = datetime(2026, 9, 2, 11, 0)


class RecordingBroker(BrokerClient):
    """Captures every order without transmitting anything."""

    name = "recording"

    def __init__(self, price: float = 24180.0) -> None:
        self.orders: list[OrderRequest] = []
        self.price = price

    def connect(self) -> bool:
        return True

    def is_connected(self) -> bool:
        return True

    def history(self, symbol, timeframe, bars):
        import pandas as pd
        return pd.DataFrame()

    def quote(self, symbol: str) -> Quote:
        return Quote(symbol, self.price, self.price, self.price, IST_NOW)

    def place_order(self, request: OrderRequest) -> OrderResult:
        self.orders.append(request)
        return OrderResult(
            accepted=True, order_id=f"rec-{len(self.orders)}",
            filled_quantity=request.quantity,
            average_price=request.limit_price or self.price,
            paper=True, message="recorded",
        )

    def cancel_order(self, order_id: str) -> bool:
        return True


def make_signal(option: bool = True, direction: Direction = Direction.LONG) -> Signal:
    entry, stop = 24180.0, 24130.0
    risk = abs(entry - stop)
    leg = (
        OptionLeg(
            expiry=date(2026, 9, 3), dte=1, strike=24200.0, option_type="CE",
            delta=0.52, iv=13.4, mid_premium=180.0, bid=179.5, ask=180.5,
            oi=1_450_000, volume=320_000, premium_stop=117.0, lot_size=75,
            tradingsymbol="NIFTY26903C24200", lots=3, total_premium_outlay=40_500,
            binding_cap="premium_outlay",
        )
        if option
        else None
    )
    return Signal(
        market="NIFTY50" if option else "XAUUSD",
        underlying="NIFTY50" if option else "XAUUSD",
        direction=direction,
        setup_type=SetupType.SR_REVERSAL,
        setup_ref="zone-1",
        regime=Regime.RANGE,
        counter_bias=False,
        confluence_mode=ConfluenceMode.REVERSAL,
        confluence_count={"aligned": 4, "opposing": 1, "neutral": 1},
        indicator_reads={"adx": "neutral", "stoch": "bull", "macd": "bull",
                         "rsi": "bull", "bb": "bull", "vwap": "bear"},
        timeframes={"bias": "15M", "setup": "5M", "trigger": "1M"},
        entry_price=entry,
        stop_price=stop,
        stop_source="rejection wick + 0.25ATR",
        target_price=entry + 2 * risk,
        target_r=2.0,
        trail=TrailPlan(entry + risk, 1.0, "atr_chandelier", 1.5),
        risk_pct=0.03,
        vol_factor=1.0,
        atr_setup_tf=25.0,
        timestamp_ist=IST_NOW,
        leg_type="OPTION" if option else "FUTURES",
        option_leg=leg,
        futures_leg=None if option else FuturesLeg("GOLDM26OCT", 10.0, 1.0, 10.0, contracts=2),
        reason_line="test signal",
    )


@pytest.fixture
def wired(cfg):
    """Executor, risk manager and tracker wired to a recording broker."""
    broker = RecordingBroker()
    executor = OrderExecutor({"zerodha": broker, "paper": broker}, cfg)
    risk = RiskManager(cfg, capital=500_000)
    tracker = PositionTracker(executor, risk, cfg)
    return broker, executor, risk, tracker


# ---------------------------------------------------------------------------
# Section 10 - paper mode
# ---------------------------------------------------------------------------


class TestPaperMode:
    def test_shipped_config_is_paper(self, raw_config):
        assert raw_config.is_paper is True

    def test_live_needs_both_switches_off(self, cfg):
        cfg.data["mode"] = "live"
        assert cfg.is_paper is True, "broker.paper_trading still forces paper"
        cfg.section("broker")["paper_trading"] = False
        assert cfg.is_paper is False

    def test_option_entry_is_always_a_buy(self, cfg, wired):
        """Rule 13.10: long CE / long PE only. Never sold, never spread."""
        broker, executor, _risk, _tracker = wired
        for direction in (Direction.LONG, Direction.SHORT):
            executor.enter(make_signal(option=True, direction=direction), 24180.0, IST_NOW)
        assert all(order.side is OrderSide.BUY for order in broker.orders)

    def test_futures_short_sells(self, cfg, wired):
        broker, executor, _risk, _tracker = wired
        executor.enter(make_signal(option=False, direction=Direction.SHORT), 2418.0, IST_NOW)
        assert broker.orders[-1].side is OrderSide.SELL

    def test_quantity_is_lots_times_lot_size(self, cfg, wired):
        broker, executor, _risk, _tracker = wired
        signal = make_signal()
        executor.enter(signal, 24180.0, IST_NOW)
        assert broker.orders[-1].quantity == signal.option_leg.lots * signal.option_leg.lot_size

    def test_slippage_is_applied_against_the_position(self, cfg, wired):
        """6.8: never assume the favourable fill."""
        _broker, executor, _risk, _tracker = wired
        fill = executor.enter(make_signal(), 24180.0, IST_NOW)
        assert fill is not None
        assert fill.price > 180.0          # a buy fills above mid
        assert fill.slippage > 0

    def test_exit_is_always_a_market_order(self, cfg, wired):
        """A flatten must complete; it is never a limit order."""
        from broker import OrderType

        broker, executor, _risk, _tracker = wired
        signal = make_signal()
        executor.exit(signal, 225, ExitReason.SESSION, 24160.0, 170.0, IST_NOW)
        assert broker.orders[-1].order_type is OrderType.MARKET


# ---------------------------------------------------------------------------
# 5.7 - G8 strike selection
# ---------------------------------------------------------------------------


def make_chain(spot: float = 24180.0, expiry: date = date(2026, 9, 10),
               taken_at: datetime = IST_NOW) -> ChainSnapshot:
    quotes: list[OptionQuote] = []
    for offset in range(-5, 6):
        strike = 24200.0 + offset * 50
        moneyness = (spot - strike) / 100.0
        call_delta = min(0.95, max(0.05, 0.5 + moneyness * 0.35))
        for option_type, delta in (("CE", call_delta), ("PE", -(1 - call_delta))):
            premium = max(6.0, 180.0 - abs(offset) * 25)
            quotes.append(
                OptionQuote(
                    strike=strike, option_type=option_type,
                    bid=premium - 0.4, ask=premium + 0.4,
                    oi=500_000 - abs(offset) * 20_000,
                    oi_change=10_000 - offset * 500,
                    volume=200_000 - abs(offset) * 10_000,
                    iv=13.0 + abs(offset) * 0.4, delta=delta,
                    tradingsymbol=f"NIFTY{int(strike)}{option_type}",
                )
            )
    return ChainSnapshot("NIFTY50", spot, expiry, taken_at, quotes, session_open_spot=24100.0)


def make_plan() -> TradePlan:
    entry, stop = 24180.0, 24130.0
    return TradePlan(
        direction=Direction.LONG, entry_price=entry, stop_price=stop,
        target_price=24280.0, stop_source="test", target_r=2.0,
        trail=TrailPlan(24230.0, 1.0, "atr_chandelier", 1.5),
        atr=25.0, risk_points=abs(entry - stop),
    )


class TestStrikeSelection:
    def test_selects_a_strike_inside_the_delta_band(self, cfg):
        selector = OptionLegSelector(cfg)
        result = selector.select(make_plan(), make_chain(), "NIFTY50", IST_NOW)
        assert result.ok, result.reason
        low, high = cfg.get("options.delta_band")
        assert low <= result.option_leg.delta <= high

    def test_bullish_signal_buys_a_call(self, cfg):
        selector = OptionLegSelector(cfg)
        result = selector.select(make_plan(), make_chain(), "NIFTY50", IST_NOW)
        assert result.option_leg.option_type == "CE"

    def test_premium_stop_is_attached(self, cfg):
        selector = OptionLegSelector(cfg)
        leg = selector.select(make_plan(), make_chain(), "NIFTY50", IST_NOW).option_leg
        expected = leg.mid_premium * (1 - float(cfg.get("options.premium_stop_pct")))
        assert leg.premium_stop == pytest.approx(round(expected, 2))

    def test_illiquid_chain_is_rejected(self, cfg):
        """G8 refuses rather than buying a worse strike to force the trade."""
        chain = make_chain()
        for quote in chain.quotes:
            quote.oi = 10
            quote.volume = 5
        result = OptionLegSelector(cfg).select(make_plan(), chain, "NIFTY50", IST_NOW)
        assert result.ok is False
        assert "liquidity" in result.reason

    def test_one_sided_book_fails(self, cfg):
        selector = OptionLegSelector(cfg)
        quote = OptionQuote(24200, "CE", bid=0.0, ask=180.0, oi=10**6,
                            volume=10**6, iv=13.0, delta=0.52)
        ok, reason = selector.passes_liquidity(quote)
        assert ok is False and "one-sided" in reason

    def test_wide_spread_fails(self, cfg):
        selector = OptionLegSelector(cfg)
        quote = OptionQuote(24200, "CE", bid=170.0, ask=190.0, oi=10**6,
                            volume=10**6, iv=13.0, delta=0.52)
        ok, reason = selector.passes_liquidity(quote)
        assert ok is False and "spread" in reason

    def test_expiry_day_excludes_otm(self, cfg):
        """5.7.4: ATM or ITM only on expiry day."""
        today = IST_NOW.date()
        chain = make_chain(spot=24180.0, expiry=today)
        result = OptionLegSelector(cfg).select(make_plan(), chain, "NIFTY50", IST_NOW)
        if result.ok:
            assert result.option_leg.strike <= chain.spot, "an OTM call was selected on expiry day"

    def test_expiry_day_cutoff_rolls_to_the_next_weekly(self, cfg):
        selector = OptionLegSelector(cfg)
        today = IST_NOW.date()
        later = datetime.combine(today, datetime.min.time()).replace(hour=14, minute=0)
        chosen, reason = selector.choose_expiry([today, date(2026, 9, 10)], later)
        assert chosen == date(2026, 9, 10)
        assert "cutoff" in reason

    def test_nearest_weekly_is_the_default(self, cfg):
        selector = OptionLegSelector(cfg)
        chosen, reason = selector.choose_expiry(
            [date(2026, 9, 10), date(2026, 9, 17)], IST_NOW
        )
        assert chosen == date(2026, 9, 10)
        assert "nearest weekly" in reason

    def test_routes_futures_for_gold(self, cfg):
        selector = InstrumentSelector(cfg)
        result = selector.select(
            make_plan(), "XAUUSD", IST_NOW,
            contracts=[("GOLDM26DEC", date(2026, 12, 5))],
        )
        assert result.ok, result.reason
        assert result.futures_leg is not None

    def test_rollover_window_uses_the_next_month(self, cfg):
        selector = InstrumentSelector(cfg)
        soon = (IST_NOW + timedelta(days=1)).date()
        result = selector.select(
            make_plan(), "XAUUSD", IST_NOW,
            contracts=[("GOLDM26SEP", soon), ("GOLDM26OCT", date(2026, 10, 28))],
        )
        assert result.ok
        assert result.futures_leg.contract == "GOLDM26OCT"


# ---------------------------------------------------------------------------
# 4.7 - chain context
# ---------------------------------------------------------------------------


class TestChainContext:
    def test_oi_levels_and_context_are_derived(self, cfg):
        analyzer = ChainAnalyzer(cfg)
        context, zones, flags = analyzer.analyse(
            make_chain(), IST_NOW, atr_value=25.0, market="NIFTY50"
        )
        assert context.max_call_oi_strike is not None
        assert context.max_put_oi_strike is not None
        assert context.pcr is not None
        assert zones, "expected OI-derived Tier A zones"

    def test_stale_chain_drops_levels_but_not_context(self, cfg):
        """4.6: a stale chain degrades context, it does not halt trading."""
        from core.schemas import Flag

        old = IST_NOW - timedelta(seconds=int(cfg.get("options.chain_max_age_sec")) + 60)
        context, zones, flags = ChainAnalyzer(cfg).analyse(
            make_chain(taken_at=old), IST_NOW, 25.0, "NIFTY50"
        )
        assert context.stale is True
        assert zones == []
        assert Flag.STALE_CHAIN in flags

    def test_oi_tag_classification(self, cfg):
        from core.schemas import OITag

        analyzer = ChainAnalyzer(cfg)
        chain = make_chain()
        assert analyzer.oi_tag(chain, price_change=50.0) is OITag.LONG_BUILDUP
        assert analyzer.oi_tag(chain, price_change=-50.0) is OITag.SHORT_BUILDUP

    def test_max_pain_is_a_strike_in_the_chain(self, cfg):
        chain = make_chain()
        pain = ChainAnalyzer(cfg).max_pain(chain)
        assert pain in {quote.strike for quote in chain.quotes}

    def test_iv_percentile_is_none_without_history(self, cfg):
        """An unknown percentile must not read as a low one."""
        assert ChainAnalyzer(cfg).iv_percentile(14.0, []) is None


# ---------------------------------------------------------------------------
# 6.9 / Appendix C - what every exit records
# ---------------------------------------------------------------------------


class TestPositionLifecycle:
    def test_open_then_close_produces_a_full_record(self, cfg, wired):
        _broker, _executor, _risk, tracker = wired
        signal = make_signal()
        position = tracker.open(signal, 24180.0, IST_NOW)
        assert position is not None

        decision = ExitDecision(True, ExitReason.TP, 24280.0, 240.0, "target")
        trade, _alerts = tracker.close("NIFTY50", decision, IST_NOW + timedelta(minutes=20))

        assert trade is not None
        assert trade.exit_reason is ExitReason.TP
        assert trade.entry_premium is not None and trade.exit_premium is not None
        assert trade.entry_underlying is not None and trade.exit_underlying is not None
        assert trade.underlying_r_multiple is not None
        assert trade.premium_r_multiple is not None
        assert trade.hypothetical_r_if_held_to_target is not None
        assert trade.delta_at_entry == pytest.approx(0.52)

    def test_r_of_record_is_the_premium_r_for_options(self, cfg, wired):
        """6.9: the R of record is the premium-based one - that is the actual money."""
        _broker, _executor, _risk, tracker = wired
        tracker.open(make_signal(), 24180.0, IST_NOW)
        decision = ExitDecision(True, ExitReason.TP, 24280.0, 240.0, "target")
        trade, _ = tracker.close("NIFTY50", decision, IST_NOW + timedelta(minutes=20))
        assert trade.r_multiple == pytest.approx(trade.premium_r_multiple)
        assert trade.r_multiple != pytest.approx(trade.underlying_r_multiple)

    def test_closing_registers_the_loss_with_risk(self, cfg, wired):
        _broker, _executor, risk, tracker = wired
        risk.roll_session("indian", IST_NOW.date())
        tracker.open(make_signal(), 24180.0, IST_NOW)
        decision = ExitDecision(True, ExitReason.SL, 24130.0, 110.0, "stop")
        tracker.close("NIFTY50", decision, IST_NOW + timedelta(minutes=10))
        assert risk.state["indian"].consecutive_losses == 1
        assert risk.state["indian"].realised_pnl < 0

    def test_second_position_on_the_same_market_is_refused(self, cfg, wired):
        _broker, _executor, _risk, tracker = wired
        assert tracker.open(make_signal(), 24180.0, IST_NOW) is not None
        assert tracker.open(make_signal(), 24180.0, IST_NOW) is None

    def test_snapshot_reports_live_state(self, cfg, wired):
        _broker, _executor, _risk, tracker = wired
        tracker.open(make_signal(), 24180.0, IST_NOW)
        rows = tracker.snapshot()
        assert len(rows) == 1
        assert rows[0]["market"] == "NIFTY50"
        assert "unrealised_r" in rows[0]


# ---------------------------------------------------------------------------
# Appendix C - persistence
# ---------------------------------------------------------------------------


class TestJournal:
    def test_signal_and_trade_round_trip(self, cfg, wired):
        _broker, _executor, _risk, tracker = wired
        journal = Journal(cfg, path=":memory:")
        signal = make_signal()
        journal.record_signal(signal)

        tracker.open(signal, 24180.0, IST_NOW)
        decision = ExitDecision(True, ExitReason.TP, 24280.0, 240.0, "target")
        trade, _ = tracker.close("NIFTY50", decision, IST_NOW + timedelta(minutes=20))
        journal.record_trade(trade)

        rows = journal.recent_trades(market="NIFTY50")
        assert len(rows) == 1
        assert rows[0]["exit_reason"] == "TP"
        assert rows[0]["setup_type"] == int(SetupType.SR_REVERSAL)
        journal.close()

    def test_rejections_are_keyed_by_gate(self, cfg):
        from core.schemas import Gate, Rejection

        journal = Journal(cfg, path=":memory:")
        for gate in (Gate.G5_CONFLUENCE, Gate.G5_CONFLUENCE, Gate.G7_VIABILITY):
            journal.record_rejection(
                Rejection(IST_NOW, "NIFTY50", gate, "detail", SetupType.SR_REVERSAL,
                          Direction.LONG, {"aligned": 3})
            )
        histogram = journal.gate_histogram()
        assert histogram["G5"] == 2
        assert histogram["G7"] == 1
        journal.close()

    def test_signal_dict_matches_appendix_b(self, cfg):
        payload = make_signal().to_dict()
        for field in (
            "signal_id", "timestamp_ist", "market", "underlying", "direction",
            "setup_type", "setup_ref", "regime", "counter_bias", "confluence_mode",
            "confluence_count", "indicator_reads", "timeframes", "entry_price",
            "stop_price", "stop_source", "target_price", "target_r", "trail",
            "risk_pct", "vol_factor", "atr_setup_tf", "leg", "chain_context",
            "flags", "mode", "reason_line",
        ):
            assert field in payload, f"Appendix B field missing: {field}"
        assert "_option_only" in payload["leg"]


# ---------------------------------------------------------------------------
# Section 8 - override friction
# ---------------------------------------------------------------------------


class TestOverrides:
    @pytest.fixture
    def position(self, cfg) -> ManagedPosition:
        signal = make_signal()
        from broker.position_tracker import _plan_from_signal

        return ManagedPosition(
            signal=signal, plan=_plan_from_signal(signal), entry_time=IST_NOW,
            entry_underlying=24180.0, entry_premium=180.0,
        )

    def test_close_without_confirmation_is_refused(self, cfg, position):
        guard = OverrideGuard(cfg)
        verdict = guard.request(
            OverrideRequest(OverrideAction.CLOSE_EARLY, "NIFTY50", IST_NOW),
            position, ExitManager(cfg),
        )
        assert verdict.accepted is False
        assert verdict.requires_confirmation is True
        assert verdict.expected_phrase == "CONFIRM OVERRIDE: closing against plan"

    def test_near_miss_confirmation_is_refused(self, cfg, position):
        """Friction is the feature - a partial match must not unlock it."""
        guard = OverrideGuard(cfg)
        for text in ("confirm override: closing against plan", "CONFIRM OVERRIDE", "y"):
            verdict = guard.request(
                OverrideRequest(OverrideAction.CLOSE_EARLY, "NIFTY50", IST_NOW, text),
                position, ExitManager(cfg),
            )
            assert verdict.accepted is False

    def test_exact_confirmation_is_accepted_and_logged(self, cfg, position):
        guard = OverrideGuard(cfg)
        verdict = guard.request(
            OverrideRequest(
                OverrideAction.CLOSE_EARLY, "NIFTY50", IST_NOW,
                "CONFIRM OVERRIDE: closing against plan",
            ),
            position, ExitManager(cfg),
        )
        assert verdict.accepted is True
        assert verdict.record is not None
        assert verdict.record.hypothetical_r_if_held_to_target == pytest.approx(2.0)
        assert verdict.record.context["signal_id"] == position.signal.signal_id

    def test_widening_a_stop_is_never_unlockable(self, cfg, position):
        """Rule 13.7: never widen a stop after entry - no confirmation exists for it."""
        guard = OverrideGuard(cfg)
        verdict = guard.request(
            OverrideRequest(
                OverrideAction.TIGHTEN_STOP, "NIFTY50", IST_NOW,
                "CONFIRM OVERRIDE: tightening stop against plan",
                new_stop=position.current_stop - 100,
            ),
            position, ExitManager(cfg),
        )
        assert verdict.accepted is False
        assert verdict.requires_confirmation is False
        assert "13.7" in verdict.message

    def test_adding_to_a_loser_is_never_unlockable(self, cfg, position):
        guard = OverrideGuard(cfg)
        verdict = guard.request(
            OverrideRequest(
                OverrideAction.ADD_TO_LOSER, "NIFTY50", IST_NOW,
                "CONFIRM OVERRIDE: adding to a loser",
            ),
            position, ExitManager(cfg),
        )
        assert verdict.accepted is False
        assert "never adds" in verdict.message

    def test_every_attempt_is_counted(self, cfg, position):
        """Section 8, point 4: override frequency becomes visible data."""
        guard = OverrideGuard(cfg)
        for _ in range(3):
            guard.request(
                OverrideRequest(OverrideAction.CLOSE_EARLY, "NIFTY50", IST_NOW),
                position, ExitManager(cfg),
            )
        summary = guard.summary()
        assert summary["attempts"] == 3
        assert summary["accepted"] == 0
        assert summary["refused"] == 3

    def test_permitted_exits_need_no_override(self, cfg):
        guard = OverrideGuard(cfg)
        for reason in (ExitReason.SL, ExitReason.TP, ExitReason.TRAIL,
                       ExitReason.SESSION, ExitReason.PREMIUM_STOP):
            assert guard.permitted_exit(reason) is True
        assert guard.permitted_exit(ExitReason.OVERRIDE) is False
