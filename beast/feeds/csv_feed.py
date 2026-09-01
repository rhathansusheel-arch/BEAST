"""A CSV data adapter for the analysis layer.

Beast's analysis layer is instrument-agnostic by design (3.1), so the data layer's only job
is to hand it **underlying** OHLCV on the three Section 4.3 timeframes. This adapter builds
those from a single base-resolution CSV, which is enough to run paper mode and replays
without committing to a broker API.

Expected columns: ``timestamp,open,high,low,close[,volume]`` with an IST-naive or
IST-aware timestamp. Option chain snapshots and Gold contract specs are *not* synthesised
here - a missing chain means Nifty/Sensex signals reject at G8, which is the correct
behaviour, not a gap to paper over.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from beast.analysis.context import MarketFeed
from beast.analysis.indicators import tf_minutes
from beast.constants import Market

AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def load_bars(path: str | Path) -> pd.DataFrame:
    """Read a base-resolution OHLCV CSV indexed by bar open time."""
    df = pd.read_csv(path)
    stamp = next(c for c in df.columns if c.lower() in {"timestamp", "datetime", "date", "time"})
    df[stamp] = pd.to_datetime(df[stamp])
    df = df.set_index(stamp).sort_index()
    df.columns = [c.lower() for c in df.columns]
    if "volume" not in df.columns:
        df["volume"] = 0.0
    return df[["open", "high", "low", "close", "volume"]]


def resample(bars: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Aggregate base bars to a Section 4.3 timeframe, dropping empty periods."""
    rule = f"{tf_minutes(tf)}min"
    out = bars.resample(rule, label="left", closed="left").agg(AGG)
    return out.dropna(subset=["open", "high", "low", "close"])


def build_feed(
    market: Market,
    bars: pd.DataFrame,
    cfg,
    now: Optional[datetime] = None,
    chain=None,
    next_chain=None,
    spread: Optional[float] = None,
    futures_contract: Optional[dict] = None,
) -> MarketFeed:
    """Assemble a :class:`MarketFeed` on the market's own bias/setup/trigger cascade.

    ``atr_median`` is the median setup-TF ATR over the trailing ``risk.atr_median_days``
    sessions, which is the denominator of the Section 7 ``vol_factor``.
    """
    tfs = cfg.timeframes(market)
    now = now or bars.index[-1].to_pydatetime()
    window = bars[bars.index <= now]

    setup_bars = resample(window, tfs["setup"])
    atr_median = _atr_median(setup_bars, cfg, market)

    return MarketFeed(
        market=market,
        bias=resample(window, tfs["bias"]),
        setup=setup_bars,
        trigger=resample(window, tfs["trigger"]),
        chain=chain,
        next_chain=next_chain,
        spread=spread,
        futures_contract=futures_contract,
        atr_median=atr_median,
    )


def _atr_median(setup_bars: pd.DataFrame, cfg, market: Market) -> Optional[float]:
    from beast.analysis.indicators import atr, session_ids

    if setup_bars.empty:
        return None
    series = atr(setup_bars, int(cfg.get("indicators.atr_period")))
    sid = session_ids(setup_bars.index, cfg.session(market)["open"])
    days = int(cfg.get("risk.atr_median_days"))
    recent = sid.unique()[-days:]
    window = series[sid.isin(recent)].dropna()
    return float(window.median()) if not window.empty else None
