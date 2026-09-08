"""Feature computation for the volatility layer - intraday and session-aware.

Every function here is pure: same frame in, same frame out, no state, no clock,
no config singleton reached for behind the caller's back. That is what makes the
look-ahead tests meaningful.

Three rules this module exists to enforce
-----------------------------------------

**1. Reuse the indicator implementations, never a cached series.**
ADX, RSI and ATR already exist in :mod:`data.feature_engineering`, implemented
with Wilder's smoothing per soul file 4.1. They are imported, not reimplemented -
two ADX implementations in one repo means the volatility model and the entry
gates can disagree about the same bar.

But the soul file deliberately runs the same indicators on *different*
timeframes for different purposes: 4.4's regime classifier reads ADX on the
**bias** TF, 4.2's confluence reads indicators on the **setup** TF, and 4.1's
ATR is a **setup**-TF quantity used for stop buffers and sizing. This layer
needs ADX and ATR on the **bias** TF. So the functions are reused, parameterised
by whichever frame is passed in, and no series is ever shared across timeframes.
Appendix B already names the setup-TF one ``atr_setup_tf``; the bias-TF one here
is ``atr_bias_tf``, so the two can never be confused in a log.

**2. The overnight gap is not an intraday return.**
Every bar is tagged with a ``session_id``. The first bar of a session has no
valid one-bar return - the move from the previous close is a gap, not trading -
so that observation is dropped. Multi-bar returns are then built as rolling sums
of the surviving one-bar returns, which is what lets a 20-bar window span a
session boundary without the gap contaminating it. Note that 4.6 already treats
a gap wider than ``1.0 x ATR`` as a level-recompute event; this pipeline must
not silently absorb what that rule is trying to surface.

**3. Standardisation is causal.**
Rolling z-scores over ``zscore_lookback`` past-and-present bars. No centred
windows. No ``expanding()`` over the full frame at fit time. A perfectly causal
HMM fed by a scaler fitted on full history is still look-ahead - the bias just
moves one level down where it is harder to see.

Volume is per-market
--------------------
Nifty and Sensex index data has usable volume. XAUUSD spot is OTC: feeds report
tick counts, which measure the provider's update rate rather than traded size.
Volume features are therefore gated on ``regime.features.use_volume`` per
market, and :func:`assert_volume_policy` fails a training run whose matrix
contains volume columns for a market that disabled them.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from core.config import Config, get_config
from data.feature_engineering import adx as _adx
from data.feature_engineering import atr as _atr
from data.feature_engineering import rsi as _rsi
from data.feature_engineering import validate_ohlc

#: Bumped whenever the meaning of any feature changes. It is mixed into the
#: feature hash, so an old model refuses to load against a changed pipeline
#: rather than scoring new features against stale means.
FEATURE_PIPELINE_VERSION = "1.0.0"

# --- windows, in bias-TF bars ----------------------------------------------
# These are regime-layer windows and have no home in Appendix A; putting them
# there would duplicate nothing and clutter the operator's single config block.
# The indicator *periods* (ADX, RSI, ATR) are a different matter and are read
# from Appendix A's `indicators:` section, never hardcoded.

RETURN_HORIZONS = (1, 5, 20)
REALIZED_VOL_WINDOW = 20
VOL_RATIO_SHORT = 5
VOL_RATIO_LONG = 20
SMA_SLOPE_WINDOW = 50
SMA_SLOPE_LOOKBACK = 5
SMA_DISTANCE_WINDOW = 200
ROC_HORIZONS = (10, 20)
VOLUME_ZSCORE_WINDOW = 50
VOLUME_TREND_WINDOW = 10
VOLUME_TREND_LOOKBACK = 5

#: Feature columns produced when volume is disabled, in order.
BASE_FEATURES: tuple[str, ...] = (
    "ret_1", "ret_5", "ret_20",
    "realized_vol_20", "vol_ratio_5_20",
    "adx_bias_tf", "sma50_slope",
    "rsi_z", "sma200_distance_pct",
    "roc_10", "roc_20",
    "atr_bias_tf_norm",
)

#: Appended, in order, when volume is enabled for the market.
VOLUME_FEATURES: tuple[str, ...] = ("volume_z", "volume_trend")


def feature_columns(use_volume: bool) -> tuple[str, ...]:
    """The exact ordered feature list for a market.

    Ordering is part of the contract: a model's means and covariances are
    positional, so a reordered matrix would score silently and wrongly.
    """
    return BASE_FEATURES + (VOLUME_FEATURES if use_volume else ())


def feature_hash(columns: tuple[str, ...]) -> str:
    """Digest of the feature list plus the pipeline version.

    Checked when a persisted model is loaded. If the feature set changed and the
    model did not, the model is refused. Silent feature drift is how these
    systems rot: the numbers keep coming out, and they mean something else.
    """
    payload = f"{FEATURE_PIPELINE_VERSION}|" + ",".join(columns)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Session tagging
# ---------------------------------------------------------------------------


def session_ids(df: pd.DataFrame, session_open: str, timezone: str) -> pd.Series:
    """Tag every bar with the trading session it belongs to.

    Args:
        df: Bias-TF OHLC frame, tz-aware.
        session_open: ``"HH:MM"`` IST, from ``sessions.<family>.open``.
        timezone: Session timezone, normally ``Asia/Kolkata``.

    Returns:
        Series of ``pd.Timestamp`` session dates aligned to ``df``.

    Bars timestamped before the session open belong to the previous session's
    id, which matters for Gold: its window opens at 05:00 IST, so a bar at
    04:xx is the tail of the day before, not the head of today.
    """
    local = df.index.tz_convert(timezone)
    open_hour, open_minute = (int(part) for part in str(session_open).split(":"))
    minutes_from_open = (
        local.hour * 60 + local.minute - (open_hour * 60 + open_minute)
    )
    dates = np.where(
        minutes_from_open >= 0,
        local.normalize(),
        (local - pd.Timedelta(days=1)).normalize(),
    )
    return pd.Series(pd.to_datetime(dates), index=df.index, name="session_id")


def gap_clean_log_returns(close: pd.Series, sessions: pd.Series) -> pd.Series:
    """One-bar log returns with every session's first bar dropped.

    The move from the previous session's close to this session's open is a gap,
    not an intraday return. Feeding it to a volatility model teaches the model
    that overnight news is an intraday event.

    Returns:
        Series aligned to ``close``, ``NaN`` at each session's first bar.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.log(close / close.shift(1))
    returns = returns.replace([np.inf, -np.inf], np.nan)
    first_of_session = sessions.ne(sessions.shift(1))
    returns[first_of_session] = np.nan
    return returns


def _rolling_return(one_bar: pd.Series, horizon: int) -> pd.Series:
    """Sum ``horizon`` one-bar returns, treating an excluded gap as no move.

    A rolling *window* may span a session boundary - a 20-bar volatility read
    that reset every morning would be useless on a 25-bar session. What may not
    cross the boundary is an individual return observation, and that one is
    already ``NaN`` by the time it reaches here.
    """
    if horizon == 1:
        return one_bar
    filled = one_bar.fillna(0.0)
    summed = filled.rolling(horizon, min_periods=horizon).sum()
    # Keep the warm-up NaNs of the underlying series rather than reporting a
    # confident zero for a window that was mostly missing.
    valid = one_bar.notna().rolling(horizon, min_periods=horizon).sum()
    return summed.where(valid >= horizon - 1)


def _slope(series: pd.Series, lookback: int, scale: pd.Series) -> pd.Series:
    """Per-bar change in ``series`` over ``lookback`` bars, scaled by ``scale``.

    Scaling by price makes the slope comparable across Nifty at 24,000, Sensex
    at 81,000 and Gold at 2,500. An unscaled slope would hand the model a
    feature whose magnitude is mostly an artefact of the instrument's price.
    """
    return (series - series.shift(lookback)) / (lookback * scale)


# ---------------------------------------------------------------------------
# The feature matrix
# ---------------------------------------------------------------------------


def compute_features(df: pd.DataFrame, market: str,
                     config: Config | None = None,
                     use_volume: bool | None = None) -> pd.DataFrame:
    """Build the raw (unstandardised) feature matrix for one market.

    Args:
        df: **Bias-timeframe** OHLC frame, tz-aware and sorted. Passing a setup-
            or trigger-TF frame is a caller error this function cannot detect;
            the timeframe comes from ``timeframes.<family>.bias``.
        market: ``NIFTY50`` | ``SENSEX`` | ``XAUUSD`` and friends.
        config: Injected for tests.
        use_volume: Overrides ``regime.features.use_volume`` for this market.
            Present so a test can exercise both paths without editing config.

    Returns:
        Frame indexed like ``df`` with exactly the columns
        :func:`feature_columns` names, in that order, warm-up rows dropped.

    Raises:
        ValueError: ``df`` is not a valid OHLC frame, or volume features were
            requested for a frame with no ``volume`` column.

    The returned values are *raw*. Standardisation is a separate, causal step -
    :class:`CausalZScoreScaler` - because the scaler has to be fitted on the
    training window alone and then applied unchanged to live bars.
    """
    cfg = config or get_config()
    validate_ohlc(df)

    if use_volume is None:
        use_volume = cfg.regime_use_volume(market)

    indicators = cfg.section("indicators")
    session = cfg.session(market)
    sessions = session_ids(df, str(session["open"]), str(cfg.get("sessions.timezone")))

    close = df["close"]
    out = pd.DataFrame(index=df.index)

    # -- returns, gap-excluded ----------------------------------------------
    one_bar = gap_clean_log_returns(close, sessions)
    for horizon in RETURN_HORIZONS:
        out[f"ret_{horizon}"] = _rolling_return(one_bar, horizon)

    # -- volatility ----------------------------------------------------------
    out["realized_vol_20"] = one_bar.rolling(
        REALIZED_VOL_WINDOW, min_periods=REALIZED_VOL_WINDOW - 5
    ).std()
    short_vol = one_bar.rolling(VOL_RATIO_SHORT, min_periods=VOL_RATIO_SHORT - 1).std()
    long_vol = one_bar.rolling(VOL_RATIO_LONG, min_periods=VOL_RATIO_LONG - 5).std()
    out["vol_ratio_5_20"] = short_vol / long_vol.replace(0.0, np.nan)

    # -- trend ---------------------------------------------------------------
    # ADX on the BIAS timeframe. 4.4's regime classifier reads ADX on the same
    # timeframe but from its own call; neither caches a series for the other.
    out["adx_bias_tf"] = _adx(df, int(indicators["adx_period"]))["adx"] / 100.0
    sma50 = close.rolling(SMA_SLOPE_WINDOW, min_periods=SMA_SLOPE_WINDOW).mean()
    out["sma50_slope"] = _slope(sma50, SMA_SLOPE_LOOKBACK, close)

    # -- mean reversion ------------------------------------------------------
    rsi_series = _rsi(df, int(indicators["rsi_period"]))
    # A z-score of RSI rather than RSI itself: 4.1's 30/50/70 levels are entry
    # thresholds, and handing the model the raw level invites it to learn them.
    rsi_mean = rsi_series.rolling(SMA_SLOPE_WINDOW, min_periods=SMA_SLOPE_WINDOW).mean()
    rsi_std = rsi_series.rolling(SMA_SLOPE_WINDOW, min_periods=SMA_SLOPE_WINDOW).std()
    out["rsi_z"] = (rsi_series - rsi_mean) / rsi_std.replace(0.0, np.nan)
    sma200 = close.rolling(SMA_DISTANCE_WINDOW, min_periods=SMA_DISTANCE_WINDOW).mean()
    out["sma200_distance_pct"] = (close - sma200) / close

    # -- momentum ------------------------------------------------------------
    for horizon in ROC_HORIZONS:
        out[f"roc_{horizon}"] = np.expm1(_rolling_return(one_bar, horizon))

    # -- range ---------------------------------------------------------------
    # atr_bias_tf, named apart from Appendix B's atr_setup_tf on purpose.
    out["atr_bias_tf_norm"] = _atr(df, int(indicators["atr_period"])) / close

    # -- volume, per-market --------------------------------------------------
    if use_volume:
        if "volume" not in df.columns:
            raise ValueError(
                f"regime.features.use_volume is true for {market} but the bias-TF "
                f"frame has no volume column"
            )
        volume = df["volume"].astype(float)
        mean = volume.rolling(VOLUME_ZSCORE_WINDOW, min_periods=VOLUME_ZSCORE_WINDOW).mean()
        std = volume.rolling(VOLUME_ZSCORE_WINDOW, min_periods=VOLUME_ZSCORE_WINDOW).std()
        out["volume_z"] = (volume - mean) / std.replace(0.0, np.nan)
        volume_sma = volume.rolling(VOLUME_TREND_WINDOW, min_periods=VOLUME_TREND_WINDOW).mean()
        out["volume_trend"] = _slope(
            volume_sma, VOLUME_TREND_LOOKBACK, mean.replace(0.0, np.nan)
        )

    columns = feature_columns(bool(use_volume))
    out = out[list(columns)]
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def assert_volume_policy(features: pd.DataFrame, market: str,
                         config: Config | None = None) -> None:
    """Fail a training run whose matrix disagrees with the volume policy.

    Raises:
        ValueError: The market has ``use_volume: false`` but the matrix carries
            volume columns, or has it true and is missing them.

    Checked at train time rather than trusted, because the failure it catches is
    silent: a Gold model fitted on tick-count "volume" would train, converge,
    and produce plausible states derived partly from how often the data provider
    happened to publish.
    """
    cfg = config or get_config()
    expected = set(feature_columns(cfg.regime_use_volume(market)))
    present = set(features.columns)
    if present != expected:
        missing = sorted(expected - present)
        unexpected = sorted(present - expected)
        raise ValueError(
            f"feature matrix for {market} does not match the volume policy "
            f"(regime.features.use_volume). missing={missing} unexpected={unexpected}"
        )


# ---------------------------------------------------------------------------
# Causal standardisation
# ---------------------------------------------------------------------------


class CausalZScoreScaler:
    """Z-score standardisation that cannot see the future.

    Two modes, and the distinction is the whole point:

    * :meth:`fit_transform` - used at **training** time. Standardises each row
      against a trailing window ending at that row, so row *t* is scaled by
      statistics drawn from rows ``t - lookback + 1 .. t`` and nothing later.
    * :meth:`transform_last` - used at **inference** time on the live bar, using
      the same trailing-window rule.

    The alternative - one mean and standard deviation computed over the whole
    training frame - is what most implementations do, and it leaks: the mean of
    the full history is not knowable at any bar inside it. The model would be
    perfectly causal and the numbers feeding it would not.

    Args:
        lookback: Trailing window in bias-TF bars, from
            ``regime.features.zscore_lookback``.
        min_periods: Rows required before a z-score is emitted. Defaults to
            half the lookback, so the warm-up is bounded rather than a full
            window of discarded history.
    """

    def __init__(self, lookback: int, min_periods: int | None = None) -> None:
        if lookback < 2:
            raise ValueError("zscore_lookback must be at least 2")
        self.lookback = int(lookback)
        self.min_periods = int(min_periods if min_periods is not None else max(2, lookback // 2))

    def fit_transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Standardise every row against its own trailing window.

        Returns:
            Frame of the same shape and columns, rows whose window was too
            short dropped.
        """
        rolling = features.rolling(self.lookback, min_periods=self.min_periods)
        mean = rolling.mean()
        std = rolling.std()
        # A constant feature over the whole window has zero spread and no
        # information; emitting 0.0 rather than inf keeps it inert instead of
        # letting it dominate a full-covariance fit.
        scaled = (features - mean) / std.replace(0.0, np.nan)
        return scaled.replace([np.inf, -np.inf], np.nan).fillna(0.0).where(
            std.notna(), other=np.nan
        ).dropna()

    def transform_last(self, features: pd.DataFrame) -> np.ndarray:
        """Standardise only the final row, against the rows preceding it.

        Args:
            features: Raw feature frame ending at the bar being scored. Only
                the trailing ``lookback`` rows are read.

        Returns:
            1-D array of standardised values for the last row.

        Raises:
            ValueError: Fewer than ``min_periods`` rows are available.
        """
        window = features.tail(self.lookback)
        if len(window) < self.min_periods:
            raise ValueError(
                f"causal scaler needs at least {self.min_periods} bars, got {len(window)}"
            )
        mean = window.mean()
        std = window.std().replace(0.0, np.nan)
        scaled = (window.iloc[-1] - mean) / std
        return scaled.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
