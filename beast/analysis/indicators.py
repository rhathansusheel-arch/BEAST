"""Section 4.1 indicators and the 4.2 closed-candle rule.

Every function here is pure and causal: the value at bar *t* uses only bars up to and
including *t*, so backtest and live behaviour are identical - which is the stated purpose
of the closed-candle rule in 4.2.

**Rule 9 (Section 13):** these run on the underlying only. :func:`compute` refuses a frame
that does not declare an underlying source, so option premium data cannot reach the
indicator engine.

ATR is computed here but is explicitly *not* a confluence indicator (4.1) - it is a utility
for buffers, sizing and stop distance. The six that count are in
``constants.CONFLUENCE_INDICATORS``.
"""

from __future__ import annotations

import re
from datetime import datetime, time, timedelta
from typing import Any

import numpy as np
import pandas as pd

from beast.ops.immutable import assert_analysis_source

OHLC = ("open", "high", "low", "close")


# ---------------------------------------------------------------------------
# timeframe helpers
# ---------------------------------------------------------------------------

_TF_RE = re.compile(r"^(\d+)\s*([MHD])$", re.IGNORECASE)


def tf_minutes(tf: str) -> int:
    """Minutes in a Section 4.3 timeframe label such as ``15M`` or ``1H``."""
    m = _TF_RE.match(str(tf).strip())
    if not m:
        raise ValueError(f"unrecognised timeframe: {tf!r}")
    n, unit = int(m.group(1)), m.group(2).upper()
    return n * {"M": 1, "H": 60, "D": 1440}[unit]


def closed_only(bars: pd.DataFrame, now: datetime, tf: str) -> pd.DataFrame:
    """Drop the in-progress candle (4.2).

    Bars are indexed by their *open* time, so a bar is closed once
    ``open_time + interval <= now``. No entry decision is ever made from a forming candle.
    """
    if bars.empty:
        return bars
    delta = timedelta(minutes=tf_minutes(tf))
    return bars[bars.index + delta <= now]


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------


def wilder_rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing - the average used by RSI, ATR and ADX."""
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def true_range(bars: pd.DataFrame) -> pd.Series:
    prev_close = bars["close"].shift(1)
    ranges = pd.concat(
        [
            bars["high"] - bars["low"],
            (bars["high"] - prev_close).abs(),
            (bars["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR(14) - utility only, never counted in the 4-of-6 (4.1)."""
    return wilder_rma(true_range(bars), period)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = wilder_rma(gain, period)
    avg_loss = wilder_rma(loss, period)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.where(avg_loss != 0.0, 100.0).where(avg_gain.notna(), np.nan)


def adx_di(bars: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """ADX(14), +DI(14), -DI(14) - trend strength and directional bias (4.1)."""
    up = bars["high"].diff()
    down = -bars["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=bars.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=bars.index)

    atr_ = wilder_rma(true_range(bars), period)
    plus_di = 100.0 * wilder_rma(plus_dm, period) / atr_.replace(0.0, np.nan)
    minus_di = 100.0 * wilder_rma(minus_dm, period) / atr_.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return pd.DataFrame(
        {"adx": wilder_rma(dx, period), "plus_di": plus_di, "minus_di": minus_di},
        index=bars.index,
    )


def stochastic(bars: pd.DataFrame, k: int = 14, smooth: int = 3, d: int = 3) -> pd.DataFrame:
    """Stochastic %K(14) smoothed by 3, %D(3) (4.1)."""
    low_k = bars["low"].rolling(k).min()
    high_k = bars["high"].rolling(k).max()
    raw = 100.0 * (bars["close"] - low_k) / (high_k - low_k).replace(0.0, np.nan)
    k_line = raw.rolling(smooth).mean()
    return pd.DataFrame({"stoch_k": k_line, "stoch_d": k_line.rolling(d).mean()}, index=bars.index)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD 12/26/9, EMA-based (4.1)."""
    line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sig = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame(
        {"macd": line, "macd_signal": sig, "macd_hist": line - sig}, index=close.index
    )


def bollinger(close: pd.Series, period: int = 20, stddev: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands - 20 SMA basis, 2.0 sigma (4.1)."""
    basis = close.rolling(period).mean()
    sd = close.rolling(period).std(ddof=0)
    return pd.DataFrame(
        {"bb_mid": basis, "bb_upper": basis + stddev * sd, "bb_lower": basis - stddev * sd},
        index=close.index,
    )


def session_ids(index: pd.DatetimeIndex, session_open: str) -> pd.Series:
    """Label each bar with the session it belongs to.

    A session runs from ``session_open`` on one calendar day to just before
    ``session_open`` on the next, so a VWAP anchored to it never bleeds across the open.
    """
    open_t = time.fromisoformat(session_open)
    minutes = index.hour * 60 + index.minute
    open_minutes = open_t.hour * 60 + open_t.minute
    day = pd.Series(index.normalize(), index=index)
    before_open = pd.Series(minutes < open_minutes, index=index)
    return day.where(~before_open, day - pd.Timedelta(days=1))


def session_vwap(bars: pd.DataFrame, session_open: str) -> pd.Series:
    """Session VWAP anchored to that market's session open (4.1).

    Volume is unreliable on index and spot data, so where volume is absent or zero the
    anchored average falls back to typical price - the level stays defined, which is what
    Setups 4 and the VWAP confluence rule need.
    """
    typical = (bars["high"] + bars["low"] + bars["close"]) / 3.0
    if "volume" in bars.columns:
        volume = bars["volume"].fillna(0.0).astype(float)
    else:
        volume = pd.Series(0.0, index=bars.index)
    sid = session_ids(bars.index, session_open)
    grouped_vol = volume.groupby(sid).cumsum()
    grouped_pv = (typical * volume).groupby(sid).cumsum()
    vwap = grouped_pv / grouped_vol.replace(0.0, np.nan)
    fallback = typical.groupby(sid).expanding().mean().reset_index(level=0, drop=True)
    return vwap.fillna(fallback)


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------


def compute(
    bars: pd.DataFrame,
    config,
    session_open: str,
    source: str = "underlying",
) -> pd.DataFrame:
    """Attach every Section 4.1 indicator to a frame of underlying bars.

    ``source`` must name an underlying feed. Passing option premium data raises under
    Immutable Rule 9 - "never run indicators on an option premium chart".
    """
    assert_analysis_source(source)
    missing = [c for c in OHLC if c not in bars.columns]
    if missing:
        raise ValueError(f"bars missing columns: {missing}")

    ind: dict[str, Any] = config.section("indicators")
    out = bars.copy()
    out = out.join(adx_di(bars, int(ind["adx_period"])))
    out = out.join(
        stochastic(
            bars,
            k=int(ind["stoch"]["k"]),
            smooth=int(ind["stoch"]["smooth"]),
            d=int(ind["stoch"]["d"]),
        )
    )
    out = out.join(
        macd(
            bars["close"],
            fast=int(ind["macd"]["fast"]),
            slow=int(ind["macd"]["slow"]),
            signal=int(ind["macd"]["signal"]),
        )
    )
    out["rsi"] = rsi(bars["close"], int(ind["rsi_period"]))
    out = out.join(
        bollinger(bars["close"], period=int(ind["bb"]["period"]), stddev=float(ind["bb"]["stddev"]))
    )
    out["atr"] = atr(bars, int(ind["atr_period"]))
    out["vwap"] = session_vwap(bars, session_open)
    return out
