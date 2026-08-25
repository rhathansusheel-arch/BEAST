"""Technical indicators and feature computation for the HMM engine.

All functions are pure: they take OHLCV bars (a DataFrame indexed by
timestamp with columns "open", "high", "low", "close", "volume") and return
new Series/DataFrames without mutating the input. Every rolling/window
computation at row t only uses bars up to and including t, which is what
keeps `compute_features` safe to feed into HMMEngine.predict_regime_filtered
in both backtesting and live inference without look-ahead bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from ta.momentum import ROCIndicator, RSIIndicator
from ta.trend import ADXIndicator, SMAIndicator
from ta.volatility import AverageTrueRange

RETURN_PERIODS: tuple[int, ...] = (1, 5, 20)
MOMENTUM_PERIODS: tuple[int, ...] = (10, 20)
ZSCORE_LOOKBACK = 252


def log_returns(close: pd.Series, period: int = 1) -> pd.Series:
    """Log return of `close` over `period` bars."""
    return np.log(close / close.shift(period))


def compute_returns(
    bars: pd.DataFrame, periods: tuple[int, ...] = RETURN_PERIODS
) -> pd.DataFrame:
    """Log returns over each period in `periods`, one column per period."""
    close = bars["close"]
    return pd.DataFrame(
        {f"return_{p}": log_returns(close, p) for p in periods},
        index=bars.index,
    )


def compute_volatility(
    bars: pd.DataFrame, short_window: int = 5, long_window: int = 20
) -> pd.DataFrame:
    """Realized volatility (rolling std of 1-period log returns, `long_window`)
    and the short/long volatility ratio used to flag vol regime shifts early."""
    r1 = log_returns(bars["close"], 1)
    vol_short = r1.rolling(short_window).std()
    vol_long = r1.rolling(long_window).std()
    return pd.DataFrame(
        {
            "realized_vol": vol_long,
            "vol_ratio": vol_short / vol_long,
        },
        index=bars.index,
    )


def compute_volume_features(
    bars: pd.DataFrame, zscore_window: int = 50, trend_window: int = 10
) -> pd.DataFrame:
    """Volume z-score vs its rolling `zscore_window` mean, and the slope of
    its rolling `trend_window` SMA."""
    volume = bars["volume"]
    vol_mean = volume.rolling(zscore_window).mean()
    vol_std = volume.rolling(zscore_window).std()
    volume_zscore = (volume - vol_mean) / vol_std

    volume_sma = volume.rolling(trend_window).mean()
    volume_trend = _rolling_slope(volume_sma, trend_window)

    return pd.DataFrame(
        {
            "volume_zscore": volume_zscore,
            "volume_trend": volume_trend,
        },
        index=bars.index,
    )


def compute_trend_indicators(
    bars: pd.DataFrame, adx_window: int = 14, sma_window: int = 50
) -> pd.DataFrame:
    """ADX (`adx_window`, trend strength) and the slope of the `sma_window` SMA."""
    adx = ADXIndicator(
        high=bars["high"], low=bars["low"], close=bars["close"], window=adx_window
    ).adx()
    sma = SMAIndicator(close=bars["close"], window=sma_window).sma_indicator()
    sma_slope = _rolling_slope(sma, sma_window)

    return pd.DataFrame(
        {
            "adx": adx,
            "sma_slope": sma_slope,
        },
        index=bars.index,
    )


def compute_mean_reversion_features(
    bars: pd.DataFrame, rsi_window: int = 14, sma_window: int = 200
) -> pd.DataFrame:
    """RSI(`rsi_window`) and % distance of price from its `sma_window` SMA.

    Both are raw here; `compute_features` standardizes them (along with every
    other column) via a 252-period rolling z-score, which is what actually
    produces the "RSI z-score" the HMM consumes.
    """
    close = bars["close"]
    rsi = RSIIndicator(close=close, window=rsi_window).rsi()
    sma = SMAIndicator(close=close, window=sma_window).sma_indicator()
    distance_from_sma = (close - sma) / sma

    return pd.DataFrame(
        {
            "rsi": rsi,
            "distance_from_sma200": distance_from_sma,
        },
        index=bars.index,
    )


def compute_momentum(
    bars: pd.DataFrame, periods: tuple[int, ...] = MOMENTUM_PERIODS
) -> pd.DataFrame:
    """Rate-of-change momentum over each period in `periods`."""
    close = bars["close"]
    return pd.DataFrame(
        {f"roc_{p}": ROCIndicator(close=close, window=p).roc() for p in periods},
        index=bars.index,
    )


def compute_range_features(bars: pd.DataFrame, atr_window: int = 14) -> pd.DataFrame:
    """ATR(`atr_window`) normalized by close price."""
    atr = AverageTrueRange(
        high=bars["high"], low=bars["low"], close=bars["close"], window=atr_window
    ).average_true_range()
    return pd.DataFrame({"normalized_atr": atr / bars["close"]}, index=bars.index)


def rolling_zscore(series: pd.Series, window: int = ZSCORE_LOOKBACK) -> pd.Series:
    """Standardize `series` using a trailing rolling mean/std — no look-ahead,
    since the mean/std at row t only ever see rows up to and including t."""
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Slope of a linear (least-squares) fit over the trailing `window` points."""
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def _slope(y: np.ndarray) -> float:
        return float(((y - y.mean()) * (x - x_mean)).sum() / x_var)

    return series.rolling(window).apply(_slope, raw=True)


def compute_features(bars: pd.DataFrame, zscore_window: int = ZSCORE_LOOKBACK) -> pd.DataFrame:
    """Assemble the full feature matrix consumed by HMMEngine.fit/predict.

    Computes returns, volatility, volume, trend, mean-reversion, momentum,
    and range features from `bars`, then standardizes every column with a
    rolling `zscore_window` z-score. Rows without a full `zscore_window` of
    trailing history (i.e. the warm-up period) are dropped.
    """
    blocks = [
        compute_returns(bars),
        compute_volatility(bars),
        compute_volume_features(bars),
        compute_trend_indicators(bars),
        compute_mean_reversion_features(bars),
        compute_momentum(bars),
        compute_range_features(bars),
    ]
    raw = pd.concat(blocks, axis=1)
    standardized = raw.apply(lambda col: rolling_zscore(col, zscore_window))
    return standardized.dropna()
