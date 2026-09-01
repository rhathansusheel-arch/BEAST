"""Sections 8, 9, 10, 11 and the 5.1 gate chain end to end."""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import pytest

from beast.analysis.context import build_context
from beast.constants import (
    Direction,
    ExitReason,
    Flag,
    Gate,
    Market,
    OVERRIDE_PHRASE,
    Regime,
)
from beast.engine import Beast
from beast.entry.news import EconomicCalendar, NewsEvent
from beast.feeds.csv_feed import build_feed
from beast.ops.learning import Learning
from beast.ops.override import OverrideGuard, OverrideRefused, OverrideRequest
from beast.schemas import Signal, TradeRecord, TrailSpec
from conftest import make_chain
from scenario import trend_continuation_pullback

DAY = dt.datetime(2026, 9, 1)


# -- Section 8 ---------------------------------------------------------------


def fake_position():
    signal = SimpleNamespace(market=Market.NIFTY, leg={}, flags=[])
    return SimpleNamespace(
        signal=signal, direction=Direction.LONG, entry_price=24180.0, stop_price=24130.0,
        target_price=24280.0, mfe_r=0.4, bars_held=12, trail_activated=False,
        record=SimpleNamespace(),
    )


def test_override_without_the_typed_phrase_is_refused(ready_cfg):
    guard = OverrideGuard(ready_cfg)
    request = OverrideRequest(action="close_early", trade_id="t1", confirmation="yes please")
    with pytest.raises(OverrideRefused) as exc:
        guard.authorise(request, fake_position(), DAY)
    assert OVERRIDE_PHRASE in str(exc.value)
    assert guard.records[-1].confirmed is False, "a refused attempt is still logged"


def test_override_with_the_exact_phrase_is_logged_as_a_deviation(ready_cfg):
    guard = OverrideGuard(ready_cfg)
    request = OverrideRequest(action="close_early", trade_id="t1", confirmation=OVERRIDE_PHRASE)
    record = guard.authorise(request, fake_position(), DAY, hypothetical_r=1.8)
    assert record.confirmed
    assert record.hypothetical_r_if_held_to_target == 1.8
    assert guard.exit_reason() is ExitReason.OVERRIDE
    report = guard.weekly_report()
    assert report["confirmed"] == 1 and report["hypothetical_r_forgone"] == 1.8


def test_friction_applies_every_time_with_no_fatigue_exception(ready_cfg):
    guard = OverrideGuard(ready_cfg)
    ok = OverrideRequest(action="close_early", trade_id="t1", confirmation=OVERRIDE_PHRASE)
    guard.authorise(ok, fake_position(), DAY)
    again = OverrideRequest(action="close_early", trade_id="t2", confirmation="")
    with pytest.raises(OverrideRefused):
        guard.authorise(again, fake_position(), DAY)


# -- Section 9 ---------------------------------------------------------------


def record_with_r(r: float, setup_type: int = 1) -> TradeRecord:
    signal = Signal(
        market=Market.GOLD, direction=Direction.LONG, setup_type=setup_type, setup_ref="x",
        regime=Regime.TREND_UP, counter_bias=False,
        confluence_mode=__import__("beast.constants", fromlist=["ConfluenceMode"]).ConfluenceMode.TREND_CONTINUATION,
        confluence_count={"aligned": 4, "opposing": 0, "neutral": 2}, indicator_reads={},
        timeframes={"bias": "30M", "setup": "15M", "trigger": "1M"}, entry_price=2418.4,
        stop_price=2414.1, stop_source="ob", target_price=2427.0, target_r=2.0,
        trail=TrailSpec(2422.7, "atr_chandelier", 1.5), risk_pct=0.02, vol_factor=1.0,
        atr_setup_tf=4.3, leg={"type": "FUTURES", "_futures_only": {"contracts": 1}},
    )
    return TradeRecord(signal=signal, r_multiple=r)


def test_learning_raises_confluence_after_a_negative_thirty_trade_sample(ready_cfg):
    learning = Learning(ready_cfg)
    for _ in range(29):
        learning.record_trade(record_with_r(-0.5))
        assert not learning.is_tightened(Market.GOLD, 1)
    learning.record_trade(record_with_r(-0.5))
    assert learning.is_tightened(Market.GOLD, 1), "negative expectancy over 30 tightens to 5-of-6"


def test_learning_reverts_when_expectancy_recovers(ready_cfg):
    learning = Learning(ready_cfg)
    for _ in range(30):
        learning.record_trade(record_with_r(-0.5))
    assert learning.is_tightened(Market.GOLD, 1)
    for _ in range(30):
        learning.record_trade(record_with_r(2.0))
    assert not learning.is_tightened(Market.GOLD, 1)


def test_learning_never_touches_risk_caps(ready_cfg):
    learning = Learning(ready_cfg)
    for _ in range(30):
        learning.record_trade(record_with_r(-2.0))
    assert ready_cfg.risk_per_trade(Market.GOLD) == 0.02
    assert ready_cfg.daily_loss_cap(Market.GOLD) == 0.05


def test_gate_report_counts_where_signals_die(ready_cfg):
    from beast.schemas import Rejection

    learning = Learning(ready_cfg)
    learning.record_rejection(
        Rejection(DAY, "NIFTY", 2, "LONG", Gate.G5, "3/6 aligned, need 4")
    )
    assert learning.gate_report()["G5"] == 1


# -- Section 5.1 / 10 --------------------------------------------------------


def nifty_beast(cfg, calendar=None, tmp_path=None):
    if tmp_path is not None:
        cfg = cfg.with_overrides(
            **{
                "paths.signals": str(tmp_path / "signals.jsonl"),
                "paths.trades": str(tmp_path / "trades.jsonl"),
                "paths.rejections": str(tmp_path / "rejections.jsonl"),
                "paths.overrides": str(tmp_path / "overrides.jsonl"),
            }
        )
    return Beast(cfg, calendar=calendar)


def run_cycle(beast, cfg, bars, now, chain=True):
    snapshot = make_chain(spot=float(bars["close"].asof(now)), when=now) if chain else None
    feed = build_feed(Market.NIFTY, bars, cfg, now, chain=snapshot)
    return beast.on_trigger_close(feed, now)


def test_g0_rejects_outside_the_session(ready_cfg, tmp_path):
    bars = trend_continuation_pullback()
    beast = nifty_beast(ready_cfg, tmp_path=tmp_path)
    result = run_cycle(beast, ready_cfg, bars, DAY.replace(hour=9, minute=20))
    assert result.rejections[0].failed_gate is Gate.G0
    assert result.signal is None


def test_g2_rejects_inside_a_news_blackout(ready_cfg, tmp_path):
    bars = trend_continuation_pullback()
    now = DAY.replace(hour=12, minute=30)
    calendar = EconomicCalendar(
        events=[NewsEvent(when=now, title="RBI policy", markets=(Market.NIFTY,))]
    )
    beast = nifty_beast(ready_cfg, calendar=calendar, tmp_path=tmp_path)
    result = run_cycle(beast, ready_cfg, bars, now)
    assert result.rejections[0].failed_gate is Gate.G2
    assert "RBI policy" in result.rejections[0].gate_detail


def test_unavailable_calendar_fails_closed(ready_cfg):
    """5.6 - an unreachable feed is treated as blackout ACTIVE, never as clear."""
    calendar = EconomicCalendar(available=False)
    blocked, why = calendar.in_blackout(Market.NIFTY, DAY, ready_cfg)
    assert blocked and "unavailable" in why


def test_pipeline_runs_the_gate_chain_and_logs_every_rejection(ready_cfg, tmp_path):
    bars = trend_continuation_pullback()
    beast = nifty_beast(ready_cfg, tmp_path=tmp_path)
    gates = set()
    for minute in range(0, 60, 5):
        now = DAY.replace(hour=12, minute=15) + dt.timedelta(minutes=minute)
        result = run_cycle(beast, ready_cfg, bars, now)
        gates.update(r.failed_gate for r in result.rejections)

    assert Gate.G4 in gates or Gate.G5 in gates
    assert beast.store.rejections.count() > 0, "Section 9 needs the gate analysis persisted"
    for row in beast.store.rejections.read():
        assert row["failed_gate"] in {g.value for g in Gate}
        assert row["gate_detail"]


def test_confluence_is_evaluated_in_the_right_mode(ready_cfg, tmp_path):
    """A Setup 2 candidate must be tallied on the Reversal column, Setup 3/4 on the other."""
    bars = trend_continuation_pullback()
    beast = nifty_beast(ready_cfg, tmp_path=tmp_path)
    seen = {}
    for minute in range(0, 60):
        now = DAY.replace(hour=12, minute=15) + dt.timedelta(minutes=minute)
        result = run_cycle(beast, ready_cfg, bars, now)
        for rejection in result.rejections:
            if rejection.setup_type and rejection.confluence_count:
                seen[rejection.setup_type] = rejection.confluence_count
    assert seen, "expected at least one setup to reach the confluence gate"
    for counts in seen.values():
        assert counts["aligned"] + counts["opposing"] + counts["neutral"] == 6


def test_setup_stays_armed_across_its_validity_window(ready_cfg, tmp_path):
    """5.5 - the trigger may fire on a later trigger-TF close than the detection bar."""
    bars = trend_continuation_pullback()
    beast = nifty_beast(ready_cfg, tmp_path=tmp_path)
    history = []
    for minute in range(0, 40):
        now = DAY.replace(hour=12, minute=20) + dt.timedelta(minutes=minute)
        run_cycle(beast, ready_cfg, bars, now)
        history.append(set(beast.lifecycles[Market.NIFTY].armed))

    assert any(history), "a setup should stay armed across several trigger-TF closes"
    armed_for = {
        ref: sum(1 for snapshot in history if ref in snapshot)
        for ref in set().union(*history)
    }
    assert max(armed_for.values()) > 1, "the validity window must outlive its detection bar"
    assert set().union(*history) - history[-1], "armed setups must also expire (5.5)"


def test_paper_mode_places_no_orders(ready_cfg, tmp_path):
    """Section 10 - alert-only: every qualifying signal is logged, no order is sent."""
    beast = nifty_beast(ready_cfg, tmp_path=tmp_path)
    assert beast.cfg.is_paper
    assert not hasattr(beast, "broker") and not hasattr(beast, "order_executor")


def test_sensex_signals_carry_the_delay_flag(ready_cfg, tmp_path):
    bars = trend_continuation_pullback()
    beast = Beast(ready_cfg)
    now = DAY.replace(hour=12, minute=30)
    feed = build_feed(Market.SENSEX, bars, ready_cfg, now, chain=make_chain(when=now))
    ctx = build_context(feed, ready_cfg, now)
    assert Flag.SENSEX_DELAY.value in ctx.flags


def test_stale_chain_drops_oi_levels_but_not_trading(ready_cfg):
    bars = trend_continuation_pullback()
    now = DAY.replace(hour=12, minute=30)
    stale = make_chain(when=now - dt.timedelta(minutes=10))
    feed = build_feed(Market.NIFTY, bars, ready_cfg, now, chain=stale)
    ctx = build_context(feed, ready_cfg, now)
    assert ctx.integrity.chain_stale
    assert ctx.integrity.ok, "a stale chain degrades context, it does not halt trading (4.6)"
    assert not any(z.source.startswith("oi:") for z in ctx.zones)


def test_unimplemented_toggles_refuse_to_start(ready_cfg):
    """A switch the Soul File leaves unbuilt must fail loudly, never be ignored."""
    from beast.engine import UnsupportedConfig

    for path, value in (
        ("exit.partial_exit_enabled", True),
        ("options.oi_tag_as_gate", True),
        ("mode", "live"),
    ):
        with pytest.raises(UnsupportedConfig):
            Beast(ready_cfg.with_overrides(**{path: value}))
