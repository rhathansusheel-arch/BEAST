"""Appendix B emission, Section 11 reporting, and the paper-mode trade lifecycle."""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import pytest

from beast.analysis.levels import Zone
from beast.constants import (
    OVERRIDE_PHRASE,
    ConfluenceMode,
    Direction,
    ExitReason,
    Market,
    Read,
    Regime,
    Tier,
    ZoneKind,
)
from beast.engine import Beast
from beast.entry import pipeline
from beast.entry.confluence import tally
from beast.entry.setups import SetupInstance
from beast.exit import plan as plan_mod
from beast.exit.manager import ExitDecision
from beast.ops.precedence import soul_sha256
from beast.risk import sizing
from beast.schemas import ChainContext, OptionLeg
from conftest import make_chain

NOW = dt.datetime(2026, 9, 1, 10, 42)

BULL_READS = {
    "adx": Read.NEUTRAL,
    "stoch": Read.BULL,
    "macd": Read.BULL,
    "rsi": Read.BULL,
    "bb": Read.BULL,
    "vwap": Read.BEAR,
}


def build_signal(cfg, market=Market.NIFTY):
    zone = Zone(kind=ZoneKind.SUPPORT, low=24115.0, high=24125.0, tier=Tier.A, source="price_structure")
    ctx = SimpleNamespace(
        cfg=cfg, market=market, now=NOW, atr=20.0, atr_median=20.0, zones=[],
        chain=make_chain(spot=24180.0, when=NOW), next_chain=None, futures_contract=None,
        pending_direction=Direction.LONG, regime=SimpleNamespace(regime=Regime.RANGE),
        timeframes=cfg.timeframes(market),
        chain_context=ChainContext(max_put_oi_strike=24100.0, pcr=0.94, oi_tag="LONG_BUILDUP"),
        flags=[],
    )
    setup = SetupInstance(
        setup_type=2, direction=Direction.LONG, ref=zone.ref, detected_idx=10, detected_ts=NOW,
        structural_price=24135.0, stop_source="rejection_wick", zone=zone, meta={},
    )
    plan = plan_mod.build(setup, ctx, 24180.0)
    leg = OptionLeg("2026-09-03", 2, 24200, "CE", 0.52, 13.4, 180.0, 179.5, 180.5,
                    1_450_000, 320_000, 117.0)
    counts = tally(BULL_READS, Direction.LONG, ConfluenceMode.REVERSAL)
    sized = sizing.size_option(plan, leg, cfg, market, 20.0, 20.0)
    leg.lots, leg.lot_size = sized.quantity, sized.lot_size
    leg.total_premium_outlay, leg.binding_cap = sized.total_premium_outlay, sized.binding_cap
    return pipeline._emit(ctx, setup, plan, counts, leg, sized, "24200 CE delta 0.52"), plan


def test_signal_matches_appendix_b(ready_cfg):
    signal, _ = build_signal(ready_cfg)
    payload = signal.to_dict()
    for field in (
        "signal_id", "timestamp_ist", "market", "underlying", "direction", "setup_type",
        "setup_ref", "regime", "counter_bias", "confluence_mode", "confluence_count",
        "indicator_reads", "timeframes", "entry_price", "stop_price", "stop_source",
        "target_price", "target_r", "trail", "risk_pct", "vol_factor", "atr_setup_tf",
        "leg", "chain_context", "flags", "mode", "reason_line",
    ):
        assert field in payload, f"Appendix B field missing: {field}"
    assert payload["underlying"] == "NIFTY50"
    assert payload["leg"]["type"] == "OPTION"
    assert payload["mode"] == "paper"
    assert payload["soul_sha256"] == soul_sha256(), "every signal is traceable to a brain revision"


def test_reason_line_states_the_underlying_plan_before_the_leg(ready_cfg):
    signal, _ = build_signal(ready_cfg)
    line = signal.reason_line
    assert line.index("underlying entry") < line.index("BUY 24200 CE")
    assert "SL 24,130" in line and "TP 24,280 (2.0R)" in line
    assert "premium stop 117" in line
    assert "reversal-mode" in line
    assert "max put OI 24,100 supports" in line


def test_sensex_signals_append_the_delay_warning(ready_cfg):
    signal, _ = build_signal(ready_cfg, market=Market.SENSEX)
    assert signal.reason_line.rstrip().endswith("price may be stale.")


def test_paper_trade_lifecycle_records_both_r_multiples(ready_cfg, tmp_path):
    cfg = ready_cfg.with_overrides(
        **{
            "paths.signals": str(tmp_path / "signals.jsonl"),
            "paths.trades": str(tmp_path / "trades.jsonl"),
            "paths.rejections": str(tmp_path / "rejections.jsonl"),
            "paths.overrides": str(tmp_path / "overrides.jsonl"),
        }
    )
    beast = Beast(cfg)
    signal, _ = build_signal(cfg)
    ctx = SimpleNamespace(cfg=cfg, market=Market.NIFTY, now=NOW, atr=20.0)

    beast._open_position(signal, ctx)
    position = beast.positions[Market.NIFTY]
    assert beast.risk.open_positions[Market.NIFTY] is Direction.LONG

    position.observe(high=24280.0, low=24180.0, premium=232.0)
    decision = ExitDecision(ExitReason.TP, 24280.0, "fixed R:R target hit", premium=232.0)
    record = beast._close_position(ctx, position, decision)

    assert record.exit_reason is ExitReason.TP
    assert record.underlying_r_multiple == pytest.approx(2.0)
    assert record.premium_r_multiple == pytest.approx((232.0 - 180.0) / (180.0 - 117.0))
    assert record.r_multiple == record.premium_r_multiple, "6.9 - premium R is the R of record"
    assert Market.NIFTY not in beast.positions
    assert beast.store.trades.count() == 1

    payload = next(iter(beast.store.trades.read()))
    for field in ("entry_premium", "exit_premium", "underlying_r_multiple", "premium_r_multiple",
                  "mae_r", "mfe_r", "delta_at_entry", "dte"):
        assert field in payload, f"Appendix C field missing: {field}"


def test_loss_feeds_the_daily_limit_and_the_level_blacklist(ready_cfg, tmp_path):
    cfg = ready_cfg.with_overrides(**{"paths.trades": str(tmp_path / "trades.jsonl")})
    beast = Beast(cfg)
    signal, _ = build_signal(cfg)
    ctx = SimpleNamespace(cfg=cfg, market=Market.NIFTY, now=NOW, atr=20.0)

    for _ in range(3):
        beast._open_position(signal, ctx)
        position = beast.positions[Market.NIFTY]
        position.observe(high=24180.0, low=24130.0, premium=140.0)
        beast._close_position(
            ctx, position, ExitDecision(ExitReason.SL, 24130.0, "stop-loss hit", premium=140.0)
        )

    paused, why = beast.risk.is_paused(Market.NIFTY, NOW)
    assert paused and "consecutive" in why
    assert beast.risk.level_blacklisted(Market.NIFTY, NOW, signal.setup_ref)


def test_engine_override_requires_the_typed_confirmation(ready_cfg, tmp_path):
    """Section 8 - Beast will not close a live position early without the friction step."""
    from beast.ops.override import OverrideRefused, OverrideRequest

    cfg = ready_cfg.with_overrides(
        **{
            "paths.trades": str(tmp_path / "trades.jsonl"),
            "paths.overrides": str(tmp_path / "overrides.jsonl"),
        }
    )
    beast = Beast(cfg)
    signal, _ = build_signal(cfg)
    ctx = SimpleNamespace(cfg=cfg, market=Market.NIFTY, now=NOW, atr=20.0)
    beast._open_position(signal, ctx)

    with pytest.raises(OverrideRefused):
        beast.request_override(
            Market.NIFTY,
            OverrideRequest(action="close_early", trade_id="t1", confirmation="just close it"),
            NOW,
        )
    assert Market.NIFTY in beast.positions, "a refused override leaves the position open"

    trade = beast.request_override(
        Market.NIFTY,
        OverrideRequest(action="close_early", trade_id="t1", confirmation=OVERRIDE_PHRASE),
        NOW,
        mark_price=24200.0,
        mark_premium=190.0,
        hypothetical_r=2.0,
    )
    assert trade.exit_reason is ExitReason.OVERRIDE
    assert trade.override is not None and trade.override.confirmed
    assert trade.hypothetical_r_if_held_to_target == 2.0
    assert Market.NIFTY not in beast.positions
    assert beast.overrides.weekly_report()["refused"] == 1
