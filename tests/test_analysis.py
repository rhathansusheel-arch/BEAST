"""Section 4 - indicators, the closed-candle rule, the level engine and the regime table."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from beast.analysis import indicators, levels, regime
from beast.constants import Direction, Market, Regime, Tier, ZoneKind
from conftest import make_bars


def test_closed_candle_rule_drops_the_forming_bar(cfg):
    bars = make_bars(n=30)
    now = bars.index[-1].to_pydatetime() + timedelta(seconds=30)
    closed = indicators.closed_only(bars, now, "1M")
    assert closed.index[-1] == bars.index[-2]


def test_timeframe_labels(cfg):
    assert indicators.tf_minutes("1M") == 1
    assert indicators.tf_minutes("15M") == 15
    assert indicators.tf_minutes("30M") == 30
    with pytest.raises(ValueError):
        indicators.tf_minutes("weekly")


def test_indicator_engine_refuses_premium_data(cfg):
    """Immutable Rule 9 / Section 3.1 - analysis is on the underlying, always."""
    from beast.ops.immutable import ImmutableRuleViolation

    bars = make_bars(n=60)
    with pytest.raises(ImmutableRuleViolation):
        indicators.compute(bars, cfg, "09:15", source="option_premium")


def test_indicators_are_causal(cfg):
    """A value at bar t must not change when later bars arrive (4.2)."""
    bars = make_bars(n=200)
    full = indicators.compute(bars, cfg, "09:15")
    partial = indicators.compute(bars.iloc[:150], cfg, "09:15")
    for column in ("adx", "rsi", "macd", "bb_mid", "atr", "vwap"):
        assert full[column].iloc[149] == pytest.approx(partial[column].iloc[149], rel=1e-9)


def test_atr_is_not_a_confluence_indicator():
    from beast.constants import CONFLUENCE_INDICATORS

    assert "atr" not in CONFLUENCE_INDICATORS
    assert len(CONFLUENCE_INDICATORS) == 6


def test_session_vwap_resets_at_the_session_open(cfg):
    day1 = make_bars(n=100, start=datetime(2026, 9, 1, 9, 15))
    day2 = make_bars(n=100, start=datetime(2026, 9, 2, 9, 15), base=float(day1["close"].iloc[-1]))
    bars = pd.concat([day1, day2])
    vwap = indicators.session_vwap(bars, "09:15")
    first_of_day2 = vwap.loc[day2.index[0]]
    typical = (day2["high"].iloc[0] + day2["low"].iloc[0] + day2["close"].iloc[0]) / 3
    assert first_of_day2 == pytest.approx(typical, rel=1e-6)


def test_swings_are_confirmed_only_after_n_following_closes(cfg):
    bars = make_bars(n=100)
    swings = levels.find_swings(bars, n=2, lookback=100)
    assert swings, "expected some fractals in a random walk"
    assert all(s.confirmed_idx == s.idx + 2 for s in swings)
    assert max(s.idx for s in swings) <= len(bars) - 3


def test_zone_flip_keeps_tier_and_resets_strength(cfg):
    bars = make_bars(n=40, base=100.0, noise=0.2)
    zone = levels.Zone(
        kind=ZoneKind.RESISTANCE, low=90.0, high=91.0, tier=Tier.A, source="price_structure",
        touches=4, strength=6,
    )
    bars.loc[bars.index[-1], "close"] = 95.0  # closes far beyond the zone
    flipped = levels.update_zones([zone], bars, atr_value=2.0, cfg=cfg)[0]
    assert flipped.kind is ZoneKind.SUPPORT
    assert flipped.tier is Tier.A
    assert flipped.strength == 1


def test_rejection_candle_accepts_a_close_beyond_the_zone(cfg):
    """4.5 - a strong rejection closes above a support zone, not inside its 0.25 ATR band."""
    index = pd.date_range(datetime(2026, 9, 1, 9, 15), periods=2, freq="5min")
    bars = pd.DataFrame(
        {"open": [100.0, 100.0], "high": [101.0, 101.0], "low": [95.0, 95.0], "close": [100.5, 100.5]},
        index=index,
    )
    zone = levels.Zone(kind=ZoneKind.SUPPORT, low=94.5, high=96.0, tier=Tier.A, source="session:low")
    assert levels.is_rejection_candle(bars, 1, zone, Direction.LONG)


def test_rejection_candle_rejects_a_close_that_stayed_below(cfg):
    index = pd.date_range(datetime(2026, 9, 1, 9, 15), periods=2, freq="5min")
    bars = pd.DataFrame(
        {"open": [100.0, 96.0], "high": [101.0, 96.5], "low": [95.0, 90.0], "close": [96.0, 91.0]},
        index=index,
    )
    zone = levels.Zone(kind=ZoneKind.SUPPORT, low=94.5, high=96.0, tier=Tier.A, source="session:low")
    assert not levels.is_rejection_candle(bars, 1, zone, Direction.LONG)


def test_level_refs_are_stable_across_cycles(cfg):
    """5.5 needs stable identity: the same zone must not re-arm every cycle."""
    a = levels.Zone(kind=ZoneKind.SUPPORT, low=100.0, high=101.0, tier=Tier.A, source="session:low")
    b = levels.Zone(kind=ZoneKind.SUPPORT, low=100.0, high=101.0, tier=Tier.A, source="session:low")
    assert a.ref == b.ref


def test_regime_table(cfg):
    def row(adx, plus_di, minus_di, close, bb_mid):
        return pd.DataFrame(
            [{"adx": adx, "plus_di": plus_di, "minus_di": minus_di, "close": close, "bb_mid": bb_mid}]
        )

    assert regime.classify(row(25, 30, 10, 105, 100), cfg).regime is Regime.TREND_UP
    assert regime.classify(row(25, 10, 30, 95, 100), cfg).regime is Regime.TREND_DOWN
    assert regime.classify(row(15, 30, 10, 105, 100), cfg).regime is Regime.RANGE
    # DI and BB disagreeing is RANGE even with a high ADX.
    assert regime.classify(row(40, 30, 10, 95, 100), cfg).regime is Regime.RANGE


def test_permitted_setups_follow_the_4_4_table(cfg):
    assert regime.permitted_setups(Regime.TREND_UP, Direction.LONG, cfg) == (1, 3, 4)
    assert regime.permitted_setups(Regime.TREND_UP, Direction.SHORT, cfg) == (2,)
    assert regime.permitted_setups(Regime.TREND_DOWN, Direction.SHORT, cfg) == (1, 3, 4)
    assert regime.permitted_setups(Regime.TREND_DOWN, Direction.LONG, cfg) == (2,)
    assert set(regime.permitted_setups(Regime.RANGE, Direction.LONG, cfg)) == {2, 3}


def test_counter_bias_requires_five_of_six(cfg):
    assert regime.is_counter_bias(Regime.TREND_UP, Direction.SHORT)
    assert not regime.is_counter_bias(Regime.RANGE, Direction.SHORT)
    assert regime.required_confluence(False, cfg) == 4
    assert regime.required_confluence(True, cfg) == 5
    assert regime.required_confluence(False, cfg, learning_tightened=True) == 5
