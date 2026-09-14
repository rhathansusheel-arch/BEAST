"""Gold as a spot CFD on MT5 - DECISIONS D-56 to D-62.

Everything runs against a frozen ``symbol_info`` fixture. The numbers are the
SHAPE of a MetaQuotes-Demo XAUUSD symbol (100 oz/lot, 0.01 lots, 100-point
stops level) so the arithmetic is realistic, but they are a fixture, not a
reading - the live values go in config from ``scripts/mt5_symbol_specs.py``.

The guarantee that matters most is the first test: a ``null`` spec still
refuses the trade after this change.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

from core.config import ConfigBlockerError
from core.instrument_selector import CfdSelector, InstrumentSelector
from core.risk_manager import RiskManager
from core.schemas import Direction, FuturesLeg, TradePlan, TrailPlan

IST_NOW = datetime(2026, 9, 14, 12, 0)

#: Frozen fixture in the shape MT5 returns for XAUUSD on MetaQuotes-Demo.
XAUUSD_SPEC = SimpleNamespace(
    name="XAUUSD",
    trade_contract_size=100.0,
    digits=2,
    point=0.01,
    trade_tick_size=0.01,
    trade_tick_value=1.0,
    volume_min=0.01,
    volume_step=0.01,
    volume_max=100.0,
    trade_stops_level=100,
    trade_freeze_level=0,
    filling_mode=1,
    currency_profit="USD",
    trade_mode=4,
)

ACCOUNT_USD = SimpleNamespace(company="MetaQuotes Ltd.", server="MetaQuotes-Demo",
                              currency="USD", trade_mode=0, login=1)


def plan(entry: float = 2500.00, stop: float = 2495.00, target: float = 2510.00) -> TradePlan:
    risk = abs(entry - stop)
    return TradePlan(
        direction=Direction.LONG, entry_price=entry, stop_price=stop, target_price=target,
        stop_source="test", target_r=2.0,
        trail=TrailPlan(entry + risk, 1.0, "atr_chandelier", 1.5), atr=3.0, risk_points=risk,
    )


def cfd_config(cfg, **overrides):
    """A gold section filled the way _fill_gold_specs would fill it from the fixture."""
    gold = cfg.section("instruments")["gold"]
    gold.update({
        "trade": "cfd",
        "venue": "MetaQuotes Ltd. / MetaQuotes-Demo",
        "symbol": XAUUSD_SPEC.name,
        "contract_multiplier": XAUUSD_SPEC.trade_contract_size,
        "tick_size": XAUUSD_SPEC.trade_tick_size,
        "tick_value": XAUUSD_SPEC.trade_tick_value,
        "point": XAUUSD_SPEC.point,
        "volume_min": XAUUSD_SPEC.volume_min,
        "volume_step": XAUUSD_SPEC.volume_step,
        "volume_max": XAUUSD_SPEC.volume_max,
        "stops_level_points": XAUUSD_SPEC.trade_stops_level,
        "filling_mode": XAUUSD_SPEC.filling_mode,
        "account_currency": "USD",
    })
    gold.update(overrides)
    return cfg


# -- 1. fail-closed survives ---------------------------------------------------

def test_null_specs_still_reject_at_g8(raw_config):
    raw_config.section("instruments")["gold"]["trade"] = "cfd"
    result = CfdSelector(raw_config).select(plan(), "XAUUSD")
    assert result.ok is False
    assert "symbol" in result.reason or "unset" in result.reason


def test_null_volume_rules_still_reject_at_g9(cfg):
    cfd_config(cfg, volume_step=None)
    risk = RiskManager(cfg, capital=10_000)
    leg = FuturesLeg("XAUUSD", 100.0, 0.01, 1.0, expiry=None)
    with pytest.raises(ConfigBlockerError):
        risk.size_cfd(plan(), leg, "XAUUSD", 1.0)
    # and through the dispatcher the blocker becomes a refusal, not a crash
    sized = risk.size(plan(), "XAUUSD", 1.0, futures_leg=leg)
    assert sized.permitted is False
    assert "volume_step" in sized.reason


def test_unreachable_broker_leaves_config_null(raw_config):
    """The startup fill must not invent values when nothing was read."""
    gold = raw_config.section("instruments")["gold"]
    assert gold["contract_multiplier"] is None
    assert "instruments.gold.symbol" in raw_config.unset_blockers()
    assert "instruments.gold.account_currency" in raw_config.unset_blockers()


# -- 2. volume rounds down, never up -------------------------------------------

def test_worked_example_rounds_down_to_the_step(cfg):
    """$10,000 x 2% = $200 risk; $5 stop = 500 ticks x $1 = $500/lot; 0.4 lots."""
    cfd_config(cfg)
    risk = RiskManager(cfg, capital=10_000)
    risk.broker_currency["gold"] = "USD"
    leg = FuturesLeg("XAUUSD", 100.0, 0.01, 1.0, expiry=None)

    sized = risk.size_cfd(plan(2500.00, 2495.00), leg, "XAUUSD", vol_factor=1.0)

    assert sized.permitted is True
    assert sized.quantity == 1
    assert sized.volume_lots == pytest.approx(0.40)
    assert sized.risk_amount == pytest.approx(200.0)


def test_volume_never_rounds_up(cfg):
    cfd_config(cfg)
    risk = RiskManager(cfg, capital=10_000)
    risk.broker_currency["gold"] = "USD"
    leg = FuturesLeg("XAUUSD", 100.0, 0.01, 1.0, expiry=None)
    # $200 / ($5.13 stop = 513 ticks = $513/lot) = 0.3898 -> 0.38, not 0.39
    sized = risk.size_cfd(plan(2500.00, 2494.87), leg, "XAUUSD", 1.0)
    assert sized.volume_lots == pytest.approx(0.38)


def test_below_minimum_volume_is_rejected_not_rounded_up(cfg):
    cfd_config(cfg)
    risk = RiskManager(cfg, capital=100)          # $2 risk budget
    risk.broker_currency["gold"] = "USD"
    leg = FuturesLeg("XAUUSD", 100.0, 0.01, 1.0, expiry=None)
    sized = risk.size_cfd(plan(2500.00, 2495.00), leg, "XAUUSD", 1.0)   # 0.004 lots
    assert sized.permitted is False
    assert sized.volume_lots == 0.0
    assert "minimum" in sized.reason


def test_volume_is_capped_at_max(cfg):
    cfd_config(cfg, volume_max=0.5)
    risk = RiskManager(cfg, capital=1_000_000)
    risk.broker_currency["gold"] = "USD"
    leg = FuturesLeg("XAUUSD", 100.0, 0.01, 1.0, expiry=None)
    sized = risk.size_cfd(plan(2500.00, 2495.00), leg, "XAUUSD", 1.0)
    assert sized.volume_lots == pytest.approx(0.5)


# -- 3. stops level at the gate -------------------------------------------------

def test_stop_inside_stops_level_is_rejected_at_g8(cfg):
    cfd_config(cfg)                                  # 100 points x 0.01 = $1.00 minimum
    result = CfdSelector(cfg).select(plan(2500.00, 2499.50), "XAUUSD")
    assert result.ok is False
    assert "minimum" in result.reason and "widening" in result.reason


def test_stop_at_or_beyond_stops_level_passes_g8(cfg):
    cfd_config(cfg)
    result = CfdSelector(cfg).select(plan(2500.00, 2499.00), "XAUUSD")
    assert result.ok is True
    assert result.futures_leg.contract == "XAUUSD"
    assert result.futures_leg.expiry is None


def test_facade_routes_cfd(cfg):
    cfd_config(cfg)
    result = InstrumentSelector(cfg).select(plan(), "XAUUSD", IST_NOW, contracts=[])
    assert result.ok is True, result.reason


# -- 4. broker never overwrites config -----------------------------------------

def test_broker_read_never_overwrites_a_set_value(cfg):
    from main import BeastRunner
    cfd_config(cfg, tick_value=0.5)                  # operator override
    section = cfg.section("instruments")["gold"]
    section["volume_max"] = None                     # one null to fill

    spec = SimpleNamespace(name="XAUUSD", contract_size=100.0, tick_size=0.01, tick_value=1.0,
                           point=0.01, volume_min=0.01, volume_step=0.01, volume_max=100.0,
                           stops_level=100, filling_mask=1)
    link = SimpleNamespace(spec=lambda market: spec,
                           facts=SimpleNamespace(company="MetaQuotes Ltd.", server="MetaQuotes-Demo",
                                                 currency="USD"))
    mt5 = SimpleNamespace(is_connected=lambda: True, link=link)

    runner = BeastRunner.__new__(BeastRunner)
    runner.cfg = cfg
    runner.brokers = {"mt5": mt5}
    runner.markets = ["XAUUSD"]
    runner.risk = RiskManager(cfg, capital=10_000)
    runner.alerts = SimpleNamespace(circuit_breaker=lambda *a, **k: None)
    import logging
    runner.logger = logging.getLogger("test")

    runner._fill_gold_specs()

    assert section["tick_value"] == 0.5, "config must win over the broker"
    assert section["volume_max"] == 100.0, "a null must be filled from the broker"
    assert runner.risk.broker_currency["gold"] == "USD"


def test_changed_broker_spec_is_not_applied(cfg):
    from main import BeastRunner
    import logging
    cfd_config(cfg)
    section = cfg.section("instruments")["gold"]
    first = SimpleNamespace(name="XAUUSD", contract_size=100.0, tick_size=0.01, tick_value=1.0,
                            point=0.01, volume_min=0.01, volume_step=0.01, volume_max=100.0,
                            stops_level=100, filling_mask=1)
    second = SimpleNamespace(**{**vars(first), "volume_min": 0.1})
    state = {"spec": first}
    link = SimpleNamespace(spec=lambda market: state["spec"],
                           facts=SimpleNamespace(company="X", server="Y", currency="USD"))
    tripped = []
    runner = BeastRunner.__new__(BeastRunner)
    runner.cfg = cfg; runner.brokers = {"mt5": SimpleNamespace(is_connected=lambda: True, link=link)}
    runner.markets = ["XAUUSD"]; runner.risk = RiskManager(cfg, capital=10_000)
    runner.alerts = SimpleNamespace(circuit_breaker=lambda *a, **k: tripped.append(a))
    runner.logger = logging.getLogger("test")

    runner._fill_gold_specs()
    state["spec"] = second
    runner._fill_gold_specs()

    assert section["volume_min"] == 0.01, "a changed broker value must not be applied"
    assert tripped, "and the operator must be alerted"


# -- 5. currency mismatch blocks gold ------------------------------------------

def test_currency_mismatch_refuses_gold(cfg):
    cfd_config(cfg, account_currency="INR")
    risk = RiskManager(cfg, capital=500_000)
    risk.broker_currency["gold"] = "USD"
    leg = FuturesLeg("XAUUSD", 100.0, 0.01, 1.0, expiry=None)
    sized = risk.size_cfd(plan(), leg, "XAUUSD", 1.0)
    assert sized.permitted is False
    assert "INR" in sized.reason and "USD" in sized.reason
    assert "no conversion rate" in sized.reason


def test_undeclared_currency_is_a_blocker(cfg):
    cfd_config(cfg, account_currency=None)
    risk = RiskManager(cfg, capital=10_000)
    leg = FuturesLeg("XAUUSD", 100.0, 0.01, 1.0, expiry=None)
    sized = risk.size(plan(), "XAUUSD", 1.0, futures_leg=leg)
    assert sized.permitted is False
    assert "account_currency" in sized.reason


# -- 6. golden: fixture -> config -> gate -> size -> order metadata ------------

def test_golden_path_from_fixture_to_order(cfg):
    """The whole chain on the frozen fixture, ending in what the adapter receives."""
    from broker.order_executor import OrderExecutor
    from core.schemas import Signal
    cfd_config(cfg)
    risk = RiskManager(cfg, capital=10_000)
    risk.broker_currency["gold"] = "USD"

    selection = InstrumentSelector(cfg).select(plan(2500.00, 2495.00), "XAUUSD", IST_NOW)
    assert selection.ok, selection.reason
    sized = risk.size(plan(2500.00, 2495.00), "XAUUSD", 1.0, futures_leg=selection.futures_leg)
    assert sized.permitted, sized.reason
    selection.futures_leg.contracts = sized.quantity
    selection.futures_leg.volume_lots = sized.volume_lots
    assert selection.futures_leg.is_cfd
    assert selection.futures_leg.size_multiplier == pytest.approx(100.0 * 0.40)

    sent = []
    class Recording:
        name = "mt5"
        def connect(self): return True
        def is_connected(self): return True
        def history(self, *a): raise NotImplementedError
        def quote(self, *a): return None
        def cancel_order(self, *a): return True
        def place_order(self, request):
            sent.append(request)
            from broker import OrderResult
            return OrderResult(True, "1", request.quantity, 2500.0, paper=True)

    cfg.section("broker")["routing"]["XAUUSD"] = "mt5"
    executor = OrderExecutor({"mt5": Recording()}, cfg)
    signal = Signal.__new__(Signal)
    signal.market = "XAUUSD"; signal.leg_type = "FUTURES"; signal.signal_id = "sig-golden"
    signal.direction = Direction.LONG; signal.futures_leg = selection.futures_leg
    signal.option_leg = None; signal.stop_price = 2495.00
    executor.enter(signal, 2500.00, IST_NOW)

    request = sent[0]
    assert request.quantity == 1
    assert request.metadata["volume_lots"] == pytest.approx(0.40)
    assert request.metadata["sl"] == 2495.00
    assert request.metadata["trade_id"] == "sig-golden"
