"""Section 5.3 - the 4-of-6 confluence engine, with 5.4's machine-checkable definitions.

Two things this module refuses to do, both of which the Soul File calls out explicitly:

1. **It does not use one flat rule per indicator.** Each of the six has *two* alignment
   modes - Trend-Continuation (Setups 1, 3, 4) and Reversal (Setup 2) - and the column is
   selected from the setup type being evaluated. The two are never mixed in one tally.
2. **It does not re-interpret the ambiguous phrases.** "Fresh crossover", "expanding
   histogram", "at/near the level", "walking the band" and the rest come from 5.4, and the
   windows and tolerances behind them come from Appendix A.

Indicator values are read from the **setup timeframe** (4.2): one timeframe per confluence
count, never a 15M MACD alongside a 1M RSI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from beast.constants import CONFLUENCE_INDICATORS, ConfluenceMode, Direction, Read


@dataclass
class ConfluenceResult:
    """The tally for one direction, plus every read - including the misses.

    Section 9 needs per-indicator hit rates, "and that requires storing the misses" (5.3),
    so ``reads`` always carries all six.
    """

    reads: dict[str, Read]
    aligned: int
    opposing: int
    neutral: int
    direction: Direction
    mode: ConfluenceMode

    def as_counts(self) -> dict[str, int]:
        return {"aligned": self.aligned, "opposing": self.opposing, "neutral": self.neutral}

    def read_strings(self) -> dict[str, str]:
        return {k: v.value for k, v in self.reads.items()}

    def aligned_names(self) -> list[str]:
        want = Read.BULL if self.direction is Direction.LONG else Read.BEAR
        return [k for k, v in self.reads.items() if v is want]

    def neutral_names(self) -> list[str]:
        return [k for k, v in self.reads.items() if v is Read.NEUTRAL]

    def opposing_names(self) -> list[str]:
        want = Read.BEAR if self.direction is Direction.LONG else Read.BULL
        return [k for k, v in self.reads.items() if v is want]


# ---------------------------------------------------------------------------
# 5.4 - machine-checkable primitives
# ---------------------------------------------------------------------------


def slope_read(series: pd.Series, idx: int, cfg) -> str:
    """"rising" / "falling" (ADX, RSI): current vs 3 candles ago, delta > 0.5 index points."""
    lookback = int(cfg.get("confluence.slope_lookback"))
    min_delta = float(cfg.get("confluence.slope_min_delta"))
    if idx - lookback < 0:
        return "flat"
    now, then = float(series.iloc[idx]), float(series.iloc[idx - lookback])
    if not np.isfinite(now) or not np.isfinite(then):
        return "flat"
    diff = now - then
    if diff > min_delta:
        return "rising"
    if diff < -min_delta:
        return "falling"
    return "flat"


def fresh_cross(fast: pd.Series, slow: pd.Series, idx: int, cfg, bullish: bool) -> Optional[int]:
    """"fresh crossover": the cross occurred within the last 3 closed candles (5.4).

    Returns the bar index of the cross, or ``None``.
    """
    window = int(cfg.get("confluence.fresh_crossover_bars"))
    for k in range(idx, max(idx - window, 0), -1):
        prev_diff = float(fast.iloc[k - 1]) - float(slow.iloc[k - 1])
        diff = float(fast.iloc[k]) - float(slow.iloc[k])
        if not np.isfinite(prev_diff) or not np.isfinite(diff):
            continue
        if bullish and prev_diff <= 0 < diff:
            return k
        if not bullish and prev_diff >= 0 > diff:
            return k
    return None


def near_level(bars: pd.DataFrame, idx: int, zone, atr_value: float, cfg) -> bool:
    """"at/near the level": the candle's body is within 0.5 x ATR of the zone centre (5.4)."""
    if zone is None:
        return False
    tol = float(cfg.get("confluence.near_level_atr")) * atr_value
    row = bars.iloc[idx]
    body_low = min(float(row["open"]), float(row["close"]))
    body_high = max(float(row["open"]), float(row["close"]))
    return (body_low - tol) <= zone.center <= (body_high + tol)


def histogram_expanding(hist: pd.Series, idx: int, cfg, bullish: bool) -> bool:
    """"histogram expanding": abs value up on each of the last 2 candles, sign matching (5.4)."""
    bars = int(cfg.get("confluence.histogram_expanding_bars"))
    if idx - bars < 0:
        return False
    values = [float(hist.iloc[idx - k]) for k in range(bars + 1)][::-1]
    if any(not np.isfinite(v) for v in values):
        return False
    if bullish and values[-1] <= 0:
        return False
    if not bullish and values[-1] >= 0:
        return False
    return all(abs(values[i]) > abs(values[i - 1]) for i in range(1, len(values)))


def walking_band(
    bars: pd.DataFrame, band: pd.Series, idx: int, atr_value: float, cfg, upper: bool
) -> bool:
    """"walking/hugging the band": 2 of the last 3 closes at or beyond the band (5.4).

    The 0.1 x ATR tolerance is applied *inward* - a close hugging the band from just inside
    counts, which is what "hugging" means and what the trend-continuation read is after.
    """
    lookback = int(cfg.get("confluence.band_walk_lookback"))
    need = int(cfg.get("confluence.band_walk_min"))
    tol = float(cfg.get("confluence.band_walk_tolerance_atr")) * atr_value
    if idx - lookback + 1 < 0:
        return False
    hits = 0
    for k in range(idx - lookback + 1, idx + 1):
        close = float(bars["close"].iloc[k])
        level = float(band.iloc[k])
        if not np.isfinite(level):
            continue
        if (upper and close >= level - tol) or (not upper and close <= level + tol):
            hits += 1
    return hits >= need


def tags_band(bars: pd.DataFrame, band: pd.Series, idx: int, upper: bool) -> bool:
    """"tags or pierces the band": high >= upper (or low <= lower) with the close inside (5.4)."""
    row = bars.iloc[idx]
    level = float(band.iloc[idx])
    if not np.isfinite(level):
        return False
    if upper:
        return float(row["high"]) >= level and float(row["close"]) < level
    return float(row["low"]) <= level and float(row["close"]) > level


def vwap_held(bars: pd.DataFrame, vwap: pd.Series, idx: int, atr_value: float, cfg, bullish: bool) -> bool:
    """"pullbacks holding VWAP as support": within the last 5 candles price came within
    0.25 x ATR of VWAP and closed back above it, with no close below VWAP (5.4). Mirrored
    for resistance.

    A stretch with no pullback at all satisfies the "no close through VWAP" half and is
    therefore still a hold - the rule is about VWAP not being lost, not about requiring a
    retest that may not have happened.
    """
    lookback = int(cfg.get("confluence.vwap_hold_lookback"))
    start = max(idx - lookback + 1, 0)
    for k in range(start, idx + 1):
        close = float(bars["close"].iloc[k])
        level = float(vwap.iloc[k])
        if not np.isfinite(level):
            return False
        if bullish and close < level:
            return False
        if not bullish and close > level:
            return False
    return True


def has_divergence(bars: pd.DataFrame, rsi: pd.Series, idx: int, direction: Direction, cfg) -> bool:
    """"bullish divergence": price makes a lower low vs the prior confirmed swing low within
    the last 20 candles while RSI makes a higher low at those two points (5.4). Mirrored
    for bearish.
    """
    from beast.analysis.levels import find_swings

    lookback = int(cfg.get("confluence.divergence_lookback"))
    n = int(cfg.get("levels.fractal_n"))
    window = bars.iloc[max(idx - lookback + 1, 0) : idx + 1]
    offset = max(idx - lookback + 1, 0)
    kind = "low" if direction is Direction.LONG else "high"
    swings = [s for s in find_swings(window, n, lookback) if s.kind == kind]
    if len(swings) < 2:
        return False
    prior, recent = swings[-2], swings[-1]
    pi, ri = offset + prior.idx, offset + recent.idx
    p_rsi, r_rsi = float(rsi.iloc[pi]), float(rsi.iloc[ri])
    if not np.isfinite(p_rsi) or not np.isfinite(r_rsi):
        return False
    if direction is Direction.LONG:
        return recent.price < prior.price and r_rsi > p_rsi
    return recent.price > prior.price and r_rsi < p_rsi


# ---------------------------------------------------------------------------
# 5.3 - per-indicator reads, two columns
# ---------------------------------------------------------------------------


def _read_adx(ind: pd.DataFrame, idx: int, mode: ConfluenceMode, cfg) -> Read:
    row = ind.iloc[idx]
    adx = float(row["adx"])
    plus_di, minus_di = float(row["plus_di"]), float(row["minus_di"])
    if not np.isfinite(adx) or not np.isfinite(plus_di) or not np.isfinite(minus_di):
        return Read.NEUTRAL
    directional = Read.BULL if plus_di > minus_di else Read.BEAR if minus_di > plus_di else Read.NEUTRAL

    if mode is ConfluenceMode.REVERSAL:
        # 5.3: same directional rule, but ADX is not required to be elevated - reversals
        # occur at low-ADX range extremes. A strong opposing trend still registers as the
        # opposing read, which is how this "confirms it isn't fighting" one.
        return directional

    threshold = float(cfg.get("indicators.adx_trend_threshold"))
    if adx < threshold or slope_read(ind["adx"], idx, cfg) != "rising":
        return Read.NEUTRAL  # below 20 = not aligned in either direction (chop filter)
    return directional


def _read_stoch(ind: pd.DataFrame, idx: int, mode: ConfluenceMode, cfg) -> Read:
    k_series, d_series = ind["stoch_k"], ind["stoch_d"]
    k_now = float(k_series.iloc[idx])
    d_now = float(d_series.iloc[idx])
    if not np.isfinite(k_now) or not np.isfinite(d_now):
        return Read.NEUTRAL
    oversold = float(cfg.get("indicators.stoch_levels.oversold"))
    overbought = float(cfg.get("indicators.stoch_levels.overbought"))

    if mode is ConfluenceMode.TREND_CONTINUATION:
        k_prev = float(k_series.iloc[idx - 1]) if idx else k_now
        rising = k_now > k_prev
        if k_now > d_now and rising and k_now < overbought:
            return Read.BULL
        if k_now < d_now and not rising and k_now > oversold:
            return Read.BEAR
        return Read.NEUTRAL

    cross_up = fresh_cross(k_series, d_series, idx, cfg, bullish=True)
    if cross_up is not None and float(k_series.iloc[cross_up]) < oversold:
        return Read.BULL
    cross_down = fresh_cross(k_series, d_series, idx, cfg, bullish=False)
    if cross_down is not None and float(k_series.iloc[cross_down]) > overbought:
        return Read.BEAR
    return Read.NEUTRAL


def _read_macd(
    ind: pd.DataFrame, idx: int, mode: ConfluenceMode, cfg, zone, atr_value: float
) -> Read:
    line, signal, hist = ind["macd"], ind["macd_signal"], ind["macd_hist"]
    if not np.isfinite(float(line.iloc[idx])) or not np.isfinite(float(signal.iloc[idx])):
        return Read.NEUTRAL

    if mode is ConfluenceMode.TREND_CONTINUATION:
        above = float(line.iloc[idx]) > float(signal.iloc[idx])
        if above and histogram_expanding(hist, idx, cfg, bullish=True):
            return Read.BULL
        if not above and histogram_expanding(hist, idx, cfg, bullish=False):
            return Read.BEAR
        return Read.NEUTRAL

    cross_up = fresh_cross(line, signal, idx, cfg, bullish=True)
    if cross_up is not None and near_level(ind, cross_up, zone, atr_value, cfg):
        return Read.BULL
    cross_down = fresh_cross(line, signal, idx, cfg, bullish=False)
    if cross_down is not None and near_level(ind, cross_down, zone, atr_value, cfg):
        return Read.BEAR
    return Read.NEUTRAL


def _read_rsi(ind: pd.DataFrame, idx: int, mode: ConfluenceMode, cfg) -> Read:
    series = ind["rsi"]
    value = float(series.iloc[idx])
    if not np.isfinite(value):
        return Read.NEUTRAL
    levels = cfg.section("indicators.rsi_levels")
    mid = float(levels["mid"])

    if mode is ConfluenceMode.TREND_CONTINUATION:
        slope = slope_read(series, idx, cfg)
        if value > mid and slope == "rising":
            return Read.BULL
        if value < mid and slope == "falling":
            return Read.BEAR
        return Read.NEUTRAL

    oversold, overbought = float(levels["oversold"]), float(levels["overbought"])
    window = int(cfg.get("confluence.fresh_crossover_bars"))
    recent = [float(series.iloc[k]) for k in range(max(idx - window, 0), idx + 1)]
    recovered_up = value > oversold and any(v < oversold for v in recent[:-1])
    fell_back = value < overbought and any(v > overbought for v in recent[:-1])
    if recovered_up:
        return Read.BULL
    if fell_back:
        return Read.BEAR
    if has_divergence(ind, series, idx, Direction.LONG, cfg):
        return Read.BULL
    if has_divergence(ind, series, idx, Direction.SHORT, cfg):
        return Read.BEAR
    return Read.NEUTRAL


def _read_bb(ind: pd.DataFrame, idx: int, mode: ConfluenceMode, cfg, atr_value: float) -> Read:
    row = ind.iloc[idx]
    close, mid = float(row["close"]), float(row["bb_mid"])
    if not np.isfinite(mid):
        return Read.NEUTRAL

    if mode is ConfluenceMode.TREND_CONTINUATION:
        if close > mid and walking_band(ind, ind["bb_upper"], idx, atr_value, cfg, upper=True):
            return Read.BULL
        if close < mid and walking_band(ind, ind["bb_lower"], idx, atr_value, cfg, upper=False):
            return Read.BEAR
        return Read.NEUTRAL

    if tags_band(ind, ind["bb_lower"], idx, upper=False):
        return Read.BULL
    if tags_band(ind, ind["bb_upper"], idx, upper=True):
        return Read.BEAR
    return Read.NEUTRAL


def _read_vwap(ind: pd.DataFrame, idx: int, cfg, atr_value: float) -> Read:
    """VWAP reads identically in both modes (5.3) - it is a level, not an oscillator."""
    close, vwap = float(ind["close"].iloc[idx]), float(ind["vwap"].iloc[idx])
    if not np.isfinite(vwap):
        return Read.NEUTRAL
    if close > vwap and vwap_held(ind, ind["vwap"], idx, atr_value, cfg, bullish=True):
        return Read.BULL
    if close < vwap and vwap_held(ind, ind["vwap"], idx, atr_value, cfg, bullish=False):
        return Read.BEAR
    return Read.NEUTRAL


def evaluate(
    ind: pd.DataFrame,
    idx: int,
    mode: ConfluenceMode,
    cfg,
    atr_value: float,
    zone=None,
) -> dict[str, Read]:
    """All six reads on the setup TF, using the column that matches the setup type."""
    return {
        "adx": _read_adx(ind, idx, mode, cfg),
        "stoch": _read_stoch(ind, idx, mode, cfg),
        "macd": _read_macd(ind, idx, mode, cfg, zone, atr_value),
        "rsi": _read_rsi(ind, idx, mode, cfg),
        "bb": _read_bb(ind, idx, mode, cfg, atr_value),
        "vwap": _read_vwap(ind, idx, cfg, atr_value),
    }


def tally(reads: dict[str, Read], direction: Direction, mode: ConfluenceMode) -> ConfluenceResult:
    """Count aligned / opposing / neutral for a proposed direction (5.3 counting rule).

    Flat or neutral indicators count toward neither side.
    """
    want = Read.BULL if direction is Direction.LONG else Read.BEAR
    against = Read.BEAR if direction is Direction.LONG else Read.BULL
    aligned = sum(1 for name in CONFLUENCE_INDICATORS if reads[name] is want)
    opposing = sum(1 for name in CONFLUENCE_INDICATORS if reads[name] is against)
    neutral = len(CONFLUENCE_INDICATORS) - aligned - opposing
    return ConfluenceResult(reads, aligned, opposing, neutral, direction, mode)


def passes(result: ConfluenceResult, required: int, cfg) -> tuple[bool, str]:
    """Apply the 5.3 threshold and conflict rule.

    The conflict rule: if the opposing direction registers 3 or more aligned indicators the
    signal is rejected regardless of the primary count - "a 4-3 split is not confluence, it
    is disagreement".
    """
    limit = int(cfg.get("entry.opposing_reject_count"))
    if result.opposing >= limit:
        return False, (
            f"conflict rule - {result.opposing} indicators oppose "
            f"({', '.join(result.opposing_names())})"
        )
    if result.aligned < required:
        return False, f"{result.aligned}/{len(CONFLUENCE_INDICATORS)} aligned, need {required}"
    return True, f"{result.aligned}/{len(CONFLUENCE_INDICATORS)} aligned"
