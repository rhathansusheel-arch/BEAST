"""Indicator and feature computation (soul file 4.1).

Every indicator Beast uses is implemented here with Wilder's smoothing where the
original definition calls for it (ADX, +DI, -DI, RSI, ATR). The ``ta`` package is
a project dependency and is fine for exploratory work, but the confluence engine
reads *these* columns: a hand-rolled implementation is the only way to guarantee
that backtest and live produce byte-identical values, which is the whole premise
of the closed-candle rule in 4.2.

Two rules from the soul file are enforced structurally rather than by convention:

1. **Closed candles only.** :func:`compute_indicators` never looks forward, and
   :func:`closed_frame` drops a trailing in-progress bar. Entry decisions read
   ``df.iloc[-1]`` of a closed frame.
2. **ATR is a utility, never a confluence indicator.** It is computed and used
   for buffers, sizing and stop distance, and the confluence engine in
   ``core/confluence.py`` has no ATR branch. Do not add one (4.1).

All functions take and return pandas objects indexed by a timezone-aware
DatetimeIndex, with columns ``open``, ``high``, ``low``, ``close`` and
optionally ``volume``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.config import Config, get_config

OHLC_COLUMNS = ("open", "high", "low", "close")


# ---------------------------------------------------------------------------
# Frame hygiene
# ---------------------------------------------------------------------------


def validate_ohlc(df: pd.DataFrame) -> None:
    """Raise if ``df`` is not a usable OHLC frame.

    Raises:
        ValueError: Missing columns, unsorted index, or a naive index. A naive
            index is rejected because every session rule in the soul file is
            expressed in IST and silently mixing naive and aware timestamps is
            how session boundaries get missed.
    """
    missing = [col for col in OHLC_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"OHLC frame missing columns: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("OHLC frame must be indexed by a DatetimeIndex")
    if df.index.tz is None:
        raise ValueError("OHLC frame index must be timezone-aware (IST)")
    if not df.index.is_monotonic_increasing:
        raise ValueError("OHLC frame index must be sorted ascending")


def closed_frame(df: pd.DataFrame, now: pd.Timestamp | None = None,
                 interval: pd.Timedelta | None = None) -> pd.DataFrame:
    """Return only closed candles (soul file 4.2).

    Args:
        df: OHLC frame whose last row may still be forming.
        now: Current time. Defaults to the frame's own last index value, in
            which case the last bar is assumed closed (the normal case when a
            feed emits bars on close).
        interval: Bar interval. Inferred from the index when omitted.

    Returns:
        A frame with any in-progress trailing bar removed.
    """
    if df.empty:
        return df
    if now is None:
        return df
    if interval is None:
        interval = infer_interval(df)
    last_open = df.index[-1]
    if now < last_open + interval:
        return df.iloc[:-1]
    return df


def infer_interval(df: pd.DataFrame) -> pd.Timedelta:
    """Infer the bar interval as the modal gap between consecutive bars.

    The mode rather than the mean, because session breaks and weekend gaps would
    otherwise drag an average interval far above the true bar size.
    """
    if len(df) < 2:
        raise ValueError("Cannot infer interval from fewer than two bars")
    deltas = pd.Series(df.index[1:]) - pd.Series(df.index[:-1])
    return pd.Timedelta(deltas.mode().iloc[0])


def parse_timeframe(label: str) -> pd.Timedelta:
    """Convert a config timeframe label such as ``"15M"`` to a Timedelta."""
    text = str(label).strip().upper()
    unit = text[-1]
    value = int(text[:-1])
    if unit == "M":
        return pd.Timedelta(minutes=value)
    if unit == "H":
        return pd.Timedelta(hours=value)
    if unit == "D":
        return pd.Timedelta(days=value)
    raise ValueError(f"Unrecognised timeframe label: {label}")


def resample_ohlc(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Aggregate a lower timeframe into ``timeframe`` bars.

    Used by the backtester, which loads one 1-minute series and derives the
    trigger/setup/bias cascade from it, guaranteeing all three tiers come from
    exactly the same ticks.
    """
    validate_ohlc(df)
    rule = parse_timeframe(timeframe)
    agg: dict[str, Any] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in df.columns:
        agg["volume"] = "sum"
    out = df.resample(rule, label="left", closed="left").agg(agg)
    return out.dropna(subset=["open", "high", "low", "close"])


# ---------------------------------------------------------------------------
# Primitive smoothers
# ---------------------------------------------------------------------------


def wilder_rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's running moving average.

    Wilder's smoothing is an EMA with ``alpha = 1/period``. ADX, DI, RSI and ATR
    are all defined against it; using a plain EMA with ``alpha = 2/(n+1)``
    produces visibly different values and would make the ADX 20 threshold in the
    soul file mean something other than what the operator expects.
    """
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    """Wilder's true range."""
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range - the utility indicator (soul file 4.1).

    Used for stop buffers, zone widths, sizing and the many ``x ATR`` distances
    in the document. Never counted in the 4-of-6.
    """
    return wilder_rma(true_range(df), period)


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """ADX with +DI and -DI (soul file 4.1).

    Returns:
        Frame with columns ``adx``, ``plus_di``, ``minus_di``.
    """
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr_smoothed = wilder_rma(true_range(df), period)
    plus_dm_smoothed = wilder_rma(pd.Series(plus_dm, index=df.index), period)
    minus_dm_smoothed = wilder_rma(pd.Series(minus_dm, index=df.index), period)

    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100.0 * plus_dm_smoothed / tr_smoothed
        minus_di = 100.0 * minus_dm_smoothed / tr_smoothed
        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)

    return pd.DataFrame(
        {
            "adx": wilder_rma(dx.replace([np.inf, -np.inf], np.nan), period),
            "plus_di": plus_di,
            "minus_di": minus_di,
        },
        index=df.index,
    )


def rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's RSI on the close."""
    delta = df["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = wilder_rma(gain, period)
    avg_loss = wilder_rma(loss, period)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
    out = 100.0 - (100.0 / (1.0 + rs))
    # A window with no losses is RSI 100 and one with no gains is RSI 0; the
    # division above yields inf/NaN for both, so pin them explicitly.
    out[(avg_loss == 0) & (avg_gain > 0)] = 100.0
    out[(avg_gain == 0) & (avg_loss > 0)] = 0.0
    return out


def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26,
         signal: int = 9) -> pd.DataFrame:
    """EMA-based MACD.

    Returns:
        Frame with columns ``macd``, ``macd_signal``, ``macd_hist``.
    """
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    signal_line = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame(
        {"macd": line, "macd_signal": signal_line, "macd_hist": line - signal_line},
        index=df.index,
    )


def stochastic(df: pd.DataFrame, k: int = 14, smooth: int = 3,
               d: int = 3) -> pd.DataFrame:
    """Slow stochastic, %K(k) smoothed by ``smooth``, %D as its ``d``-SMA.

    Returns:
        Frame with columns ``stoch_k``, ``stoch_d``.
    """
    lowest = df["low"].rolling(k, min_periods=k).min()
    highest = df["high"].rolling(k, min_periods=k).max()
    span = (highest - lowest).replace(0.0, np.nan)
    raw_k = 100.0 * (df["close"] - lowest) / span
    slow_k = raw_k.rolling(smooth, min_periods=smooth).mean()
    return pd.DataFrame(
        {"stoch_k": slow_k, "stoch_d": slow_k.rolling(d, min_periods=d).mean()},
        index=df.index,
    )


def bollinger(df: pd.DataFrame, period: int = 20,
              stddev: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands on an SMA basis.

    Returns:
        Frame with columns ``bb_mid``, ``bb_upper``, ``bb_lower``, ``bb_width``.
    """
    mid = df["close"].rolling(period, min_periods=period).mean()
    # ddof=0: the population standard deviation, which is the Bollinger
    # definition. ddof=1 widens the bands slightly and shifts every band-touch
    # rule in 5.3.
    sd = df["close"].rolling(period, min_periods=period).std(ddof=0)
    upper = mid + stddev * sd
    lower = mid - stddev * sd
    return pd.DataFrame(
        {
            "bb_mid": mid,
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_width": upper - lower,
        },
        index=df.index,
    )


def session_vwap(df: pd.DataFrame, session_open: str, timezone: str) -> pd.Series:
    """Session-anchored VWAP (soul file 4.1).

    Anchored to that market's session open - 09:15 IST for the Indian session,
    05:00 IST for XAUUSD - and reset every session. A rolling or continuous VWAP
    would not be the same reference the setups in 5.2 use.

    Args:
        df: OHLC frame, optionally with a ``volume`` column.
        session_open: ``"HH:MM"`` in IST.
        timezone: Session timezone, normally ``Asia/Kolkata``.

    Returns:
        Series of VWAP values aligned to ``df``. When no volume column exists
        (spot gold, index spot), this degrades to a typical-price cumulative
        mean, which is the standard substitute and is stated on the signal.
    """
    local = df.index.tz_convert(timezone)
    open_hour, open_minute = (int(part) for part in session_open.split(":"))
    minutes_from_open = local.hour * 60 + local.minute - (open_hour * 60 + open_minute)
    # Bars before the session open belong to the previous session's anchor.
    session_date = np.where(minutes_from_open >= 0, local.date, (local - pd.Timedelta(days=1)).date)
    groups = pd.Series(session_date, index=df.index)

    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    if "volume" in df.columns and df["volume"].fillna(0).abs().sum() > 0:
        weights = df["volume"].fillna(0.0)
    else:
        weights = pd.Series(1.0, index=df.index)

    numerator = (typical * weights).groupby(groups).cumsum()
    denominator = weights.groupby(groups).cumsum().replace(0.0, np.nan)
    return numerator / denominator


# ---------------------------------------------------------------------------
# Combined computation
# ---------------------------------------------------------------------------


def compute_indicators(df: pd.DataFrame, market: str,
                       config: Config | None = None) -> pd.DataFrame:
    """Attach every soul-file indicator to ``df``.

    Args:
        df: OHLC frame for one timeframe.
        market: Used only to pick the correct session anchor for VWAP.
        config: Injected for tests; defaults to the singleton.

    Returns:
        A copy of ``df`` with the indicator columns appended. Column names are
        stable and are the contract the confluence engine reads.
    """
    cfg = config or get_config()
    validate_ohlc(df)
    out = df.copy()

    ind = cfg.section("indicators")
    out["atr"] = atr(df, int(ind["atr_period"]))

    adx_frame = adx(df, int(ind["adx_period"]))
    out[["adx", "plus_di", "minus_di"]] = adx_frame

    out["rsi"] = rsi(df, int(ind["rsi_period"]))

    macd_cfg = ind["macd"]
    out[["macd", "macd_signal", "macd_hist"]] = macd(
        df, int(macd_cfg["fast"]), int(macd_cfg["slow"]), int(macd_cfg["signal"])
    )

    stoch_cfg = ind["stoch"]
    out[["stoch_k", "stoch_d"]] = stochastic(
        df, int(stoch_cfg["k"]), int(stoch_cfg["smooth"]), int(stoch_cfg["d"])
    )

    bb_cfg = ind["bb"]
    out[["bb_mid", "bb_upper", "bb_lower", "bb_width"]] = bollinger(
        df, int(bb_cfg["period"]), float(bb_cfg["stddev"])
    )

    session = cfg.session(market)
    out["vwap"] = session_vwap(
        df, str(session["open"]), str(cfg.get("sessions.timezone"))
    )
    return out


def warmup_bars(config: Config | None = None) -> int:
    """Minimum bars needed before every indicator is defined.

    Beast refuses to evaluate a timeframe with fewer bars than this: a partially
    warmed indicator reads as neutral, which quietly changes the 4-of-6 count.
    """
    cfg = config or get_config()
    ind = cfg.section("indicators")
    macd_cfg = ind["macd"]
    stoch_cfg = ind["stoch"]
    candidates = [
        int(ind["atr_period"]) * 2,
        int(ind["adx_period"]) * 3,      # ADX is double-smoothed
        int(ind["rsi_period"]) * 2,
        int(macd_cfg["slow"]) + int(macd_cfg["signal"]),
        int(stoch_cfg["k"]) + int(stoch_cfg["smooth"]) + int(stoch_cfg["d"]),
        int(ind["bb"]["period"]),
    ]
    return max(candidates)


# ---------------------------------------------------------------------------
# HMM feature matrix
# ---------------------------------------------------------------------------


def hmm_features(df: pd.DataFrame, config: Config | None = None) -> pd.DataFrame:
    """Build the observation matrix for the HMM regime overlay.

    The features are deliberately volatility- and momentum-shaped rather than
    price-level shaped, so the fitted states are interpretable as market regimes
    (quiet trend, volatile trend, chop) instead of as price ranges that never
    recur.

    Returns:
        Frame with columns ``ret``, ``abs_ret``, ``atr_pct``, ``range_pct``,
        ``adx``, containing no NaNs. Rows before warmup are dropped, so the
        caller must align on the returned index rather than assume alignment
        with the input.
    """
    cfg = config or get_config()
    ind = cfg.section("indicators")

    close = df["close"]
    features = pd.DataFrame(index=df.index)
    features["ret"] = np.log(close / close.shift(1))
    features["abs_ret"] = features["ret"].abs()
    features["atr_pct"] = atr(df, int(ind["atr_period"])) / close
    features["range_pct"] = (df["high"] - df["low"]) / close
    features["adx"] = adx(df, int(ind["adx_period"]))["adx"] / 100.0
    return features.replace([np.inf, -np.inf], np.nan).dropna()
