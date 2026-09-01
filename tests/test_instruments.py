"""Section 5.7 - G8 strike selection, liquidity filters and the expiry-day block."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

from beast.constants import Direction, Market
from beast.entry import instruments
from beast.ops.immutable import ImmutableRuleViolation
from conftest import make_chain

NOW = datetime(2026, 9, 1, 10, 42)


def ctx_for(cfg, chain=None, direction=Direction.LONG, now=NOW, market=Market.NIFTY,
            futures_contract=None, next_chain=None):
    return SimpleNamespace(
        cfg=cfg, market=market, now=now, chain=chain, next_chain=next_chain,
        futures_contract=futures_contract, pending_direction=direction, atr=20.0,
    )


PLAN = SimpleNamespace(entry_price=24180.0, stop_price=24130.0)


def test_bullish_buys_a_call_inside_the_delta_band(ready_cfg):
    leg, detail = instruments.select_option_leg(PLAN, ctx_for(ready_cfg, make_chain()))
    assert leg.option_type == "CE"
    assert 0.45 <= leg.delta <= 0.65
    assert leg.premium_stop == pytest.approx(leg.mid_premium * 0.65, rel=1e-6)


def test_bearish_buys_a_put_never_sells(ready_cfg):
    leg, _ = instruments.select_option_leg(
        PLAN, ctx_for(ready_cfg, make_chain(), direction=Direction.SHORT)
    )
    assert leg.option_type == "PE"


def test_selling_options_is_impossible(ready_cfg):
    cfg = ready_cfg.with_overrides(**{"instruments.nifty.option_side": "short_permitted"})
    with pytest.raises(ImmutableRuleViolation):
        instruments.select_option_leg(PLAN, ctx_for(cfg, make_chain()))


def test_unset_liquidity_floor_fails_rather_than_passes(cfg):
    """5.7.3 - "an unset liquidity filter is treated as failing, not passing"."""
    populated = cfg.with_overrides(
        **{"capital": 500000.0, "instruments.nifty.lot_size": 75, "instruments.nifty.strike_interval": 50}
    )
    leg, detail = instruments.select_option_leg(PLAN, ctx_for(populated, make_chain()))
    assert leg is None
    assert "min_oi" in detail


def test_thin_strike_is_rejected_rather_than_downgraded(ready_cfg):
    chain = make_chain()
    for quote in chain.quotes:
        quote.oi = 10  # every strike illiquid
    leg, detail = instruments.select_option_leg(PLAN, ctx_for(ready_cfg, chain))
    assert leg is None
    assert "liquidity" in detail


def test_wide_spread_is_rejected(ready_cfg):
    chain = make_chain()
    for quote in chain.quotes:
        quote.bid, quote.ask = quote.mid - 20, quote.mid + 20
    leg, detail = instruments.select_option_leg(PLAN, ctx_for(ready_cfg, chain))
    assert leg is None
    assert "spread" in detail


def test_expiry_day_takes_atm_or_itm_only(ready_cfg):
    chain = make_chain(spot=24180.0, dte=0)
    leg, _ = instruments.select_option_leg(PLAN, ctx_for(ready_cfg, chain, now=NOW))
    assert leg is not None
    assert leg.strike <= chain.atm_strike(), "5.7.4 - no OTM strikes on expiry day"
    assert 0.50 <= leg.delta <= 0.75


def test_expiry_day_past_cutoff_rolls_to_the_next_weekly(ready_cfg):
    afternoon = NOW.replace(hour=14, minute=0)
    chain = make_chain(spot=24180.0, when=afternoon, dte=0)
    nxt = make_chain(spot=24180.0, when=afternoon, dte=7)

    leg, detail = instruments.select_option_leg(
        PLAN, ctx_for(ready_cfg, chain, now=afternoon)
    )
    assert leg is None and "roll to" in detail

    leg, _ = instruments.select_option_leg(
        PLAN, ctx_for(ready_cfg, chain, now=afternoon, next_chain=nxt)
    )
    assert leg is not None and leg.dte == 7


def test_gold_refuses_without_contract_specs(cfg):
    leg, detail = instruments.select_futures_contract(
        ctx_for(cfg, market=Market.GOLD, futures_contract={"symbol": "GCZ6", "days_to_expiry": 30})
    )
    assert leg is None
    assert "contract_multiplier" in detail


def test_gold_rollover_window_blocks_or_rolls(ready_cfg):
    inside = {"symbol": "GCZ6", "days_to_expiry": 2}
    leg, detail = instruments.select_futures_contract(
        ctx_for(ready_cfg, market=Market.GOLD, futures_contract=inside)
    )
    assert leg is None and "rollover window" in detail

    with_next = {"symbol": "GCZ6", "days_to_expiry": 2, "next_symbol": "GCG7"}
    leg, detail = instruments.select_futures_contract(
        ctx_for(ready_cfg, market=Market.GOLD, futures_contract=with_next)
    )
    assert leg.contract == "GCG7"
