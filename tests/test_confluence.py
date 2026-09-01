"""Section 5.3 / 5.4 - two alignment columns, the conflict rule, and the machine-checkable
definitions behind the ambiguous phrases.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from beast.constants import ConfluenceMode, Direction, Read, SETUP_MODES
from beast.entry import confluence


def frame(**columns) -> pd.DataFrame:
    n = len(next(iter(columns.values())))
    index = pd.date_range(datetime(2026, 9, 1, 9, 15), periods=n, freq="5min")
    return pd.DataFrame(columns, index=index)


def test_setup_types_map_to_the_right_column():
    assert SETUP_MODES[1] is ConfluenceMode.TREND_CONTINUATION
    assert SETUP_MODES[2] is ConfluenceMode.REVERSAL
    assert SETUP_MODES[3] is ConfluenceMode.TREND_CONTINUATION
    assert SETUP_MODES[4] is ConfluenceMode.TREND_CONTINUATION


def test_slope_needs_more_than_half_a_point_over_three_candles(cfg):
    rising = frame(x=[20.0, 20.1, 20.2, 21.0])
    flat = frame(x=[20.0, 20.1, 20.2, 20.3])
    assert confluence.slope_read(rising["x"], 3, cfg) == "rising"
    assert confluence.slope_read(flat["x"], 3, cfg) == "flat"


def test_fresh_crossover_window_is_three_candles(cfg):
    fast = frame(f=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])["f"]
    slow = frame(s=[2.0, 1.5, 1.0, 1.0, 1.0, 1.0])["s"]
    assert confluence.fresh_cross(fast, slow, 2, cfg, bullish=True) == 1
    assert confluence.fresh_cross(fast, slow, 5, cfg, bullish=True) is None


def test_histogram_expanding_requires_two_growing_bars_with_matching_sign(cfg):
    growing = frame(h=[0.1, 0.2, 0.4])["h"]
    shrinking = frame(h=[0.4, 0.3, 0.2])["h"]
    wrong_sign = frame(h=[-0.1, -0.2, -0.4])["h"]
    assert confluence.histogram_expanding(growing, 2, cfg, bullish=True)
    assert not confluence.histogram_expanding(shrinking, 2, cfg, bullish=True)
    assert not confluence.histogram_expanding(wrong_sign, 2, cfg, bullish=True)
    assert confluence.histogram_expanding(wrong_sign, 2, cfg, bullish=False)


def test_walking_the_band_needs_two_of_the_last_three_closes(cfg):
    bars = frame(close=[100.0, 101.0, 99.0, 101.5], open=[100.0] * 4, high=[102.0] * 4, low=[98.0] * 4)
    band = frame(b=[100.5, 100.5, 100.5, 100.5])["b"]
    assert confluence.walking_band(bars, band, 3, atr_value=1.0, cfg=cfg, upper=True)


def test_adx_below_threshold_is_neutral_in_trend_mode_but_not_in_reversal(cfg):
    data = frame(
        adx=[10.0] * 6,
        plus_di=[30.0] * 6,
        minus_di=[10.0] * 6,
        stoch_k=[50.0] * 6,
        stoch_d=[50.0] * 6,
        macd=[0.0] * 6,
        macd_signal=[0.0] * 6,
        macd_hist=[0.0] * 6,
        rsi=[55.0] * 6,
        bb_mid=[100.0] * 6,
        bb_upper=[102.0] * 6,
        bb_lower=[98.0] * 6,
        close=[100.0] * 6,
        open=[100.0] * 6,
        high=[100.5] * 6,
        low=[99.5] * 6,
        vwap=[100.0] * 6,
    )
    trend = confluence.evaluate(data, 5, ConfluenceMode.TREND_CONTINUATION, cfg, 1.0)
    reversal = confluence.evaluate(data, 5, ConfluenceMode.REVERSAL, cfg, 1.0)
    assert trend["adx"] is Read.NEUTRAL, "ADX < 20 is the chop filter in trend-continuation"
    assert reversal["adx"] is Read.BULL, "reversals happen at low-ADX range extremes"


def test_conflict_rule_rejects_a_four_three_split(cfg):
    reads = {
        "adx": Read.BULL,
        "stoch": Read.BULL,
        "macd": Read.BULL,
        "rsi": Read.BULL,
        "bb": Read.BEAR,
        "vwap": Read.BEAR,
    }
    result = confluence.tally(reads, Direction.LONG, ConfluenceMode.TREND_CONTINUATION)
    assert (result.aligned, result.opposing, result.neutral) == (4, 2, 0)
    ok, _ = confluence.passes(result, 4, cfg)
    assert ok, "4 aligned against 2 opposing passes"

    reads["stoch"] = Read.BEAR
    reads["adx"] = Read.BULL
    reads["macd"] = Read.BULL
    reads["rsi"] = Read.BULL
    conflicted = confluence.tally(reads, Direction.LONG, ConfluenceMode.TREND_CONTINUATION)
    assert conflicted.opposing == 3
    ok, detail = confluence.passes(conflicted, 4, cfg)
    assert not ok and "conflict rule" in detail


def test_neutral_indicators_count_for_neither_side(cfg):
    reads = {k: Read.NEUTRAL for k in ("adx", "stoch", "macd", "rsi", "bb", "vwap")}
    reads["rsi"] = Read.BULL
    result = confluence.tally(reads, Direction.LONG, ConfluenceMode.REVERSAL)
    assert (result.aligned, result.opposing, result.neutral) == (1, 0, 5)
    ok, detail = confluence.passes(result, 4, cfg)
    assert not ok and "need 4" in detail


def test_every_read_is_recorded_including_the_misses(cfg):
    """Section 9 needs per-indicator hit rates, which requires storing the misses (5.3)."""
    reads = {
        "adx": Read.BULL, "stoch": Read.NEUTRAL, "macd": Read.BULL,
        "rsi": Read.BULL, "bb": Read.BEAR, "vwap": Read.BULL,
    }
    result = confluence.tally(reads, Direction.LONG, ConfluenceMode.TREND_CONTINUATION)
    assert set(result.read_strings()) == {"adx", "stoch", "macd", "rsi", "bb", "vwap"}
    assert result.neutral_names() == ["stoch"]
    assert result.opposing_names() == ["bb"]
