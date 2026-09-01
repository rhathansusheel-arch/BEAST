"""Section 4.5 - the level engine.

"The agent must not eyeball levels." Every object below has a deterministic definition, a
validity lifetime and an invalidation condition, exactly as specified:

* :class:`SwingPoint`  - fractal, confirmed only after N following candles close
* :class:`Zone`        - S/R zone from clustered swings, or a session-structural price
* :class:`Trendline`   - >= 3 anchors, least-squares fit, closes break it and wicks do not
* :class:`OrderBlock`  - last opposing candle before a structure-breaking impulse

This is shared infrastructure. Sections 5 and 6 consume what this module produces; they
never compute levels inline. :class:`LevelEngine` carries the state that has to survive
between cycles - order-block freshness, zone strength, flips and expiry.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from beast.constants import Direction, Tier, ZoneKind

# ---------------------------------------------------------------------------
# swing points
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SwingPoint:
    """A confirmed fractal (4.5).

    ``idx`` is the positional index of the pivot candle in the frame it was found in;
    ``confirmed_idx`` is the candle whose close confirmed it (pivot + N).
    """

    idx: int
    ts: datetime
    price: float
    kind: str  # "high" | "low"
    confirmed_idx: int


def find_swings(bars: pd.DataFrame, n: int, lookback: int) -> list[SwingPoint]:
    """Fractal swing points: the pivot is the extreme of the ``n`` candles either side.

    A swing is *confirmed only after the n following candles close* (4.5), so the last
    ``n`` candles of the frame can never contain a confirmed swing - which is what keeps
    this free of look-ahead.
    """
    if len(bars) < 2 * n + 1:
        return []
    start = max(n, len(bars) - lookback)
    highs = bars["high"].to_numpy()
    lows = bars["low"].to_numpy()
    out: list[SwingPoint] = []
    for i in range(start, len(bars) - n):
        window = slice(i - n, i + n + 1)
        ts = bars.index[i]
        if highs[i] == highs[window].max() and (highs[window] == highs[i]).sum() == 1:
            out.append(SwingPoint(i, ts, float(highs[i]), "high", i + n))
        if lows[i] == lows[window].min() and (lows[window] == lows[i]).sum() == 1:
            out.append(SwingPoint(i, ts, float(lows[i]), "low", i + n))
    return out


def last_swing(swings: Sequence[SwingPoint], kind: str, before_idx: int | None = None) -> Optional[SwingPoint]:
    """The most recent confirmed swing of ``kind``, optionally before a bar index."""
    candidates = [s for s in swings if s.kind == kind and (before_idx is None or s.idx < before_idx)]
    return max(candidates, key=lambda s: s.idx) if candidates else None


# ---------------------------------------------------------------------------
# support / resistance zones
# ---------------------------------------------------------------------------


@dataclass
class Zone:
    """A support or resistance zone (4.5).

    ``strength`` is "number of touches + 1 per prior rejection". It is reported on the
    signal for transparency and *does not gate the trade* (4.5).
    """

    kind: ZoneKind
    low: float
    high: float
    tier: Tier
    source: str  # price_structure | session | oi
    strength: int = 1
    touches: int = 1
    rejections: int = 0
    alive: bool = True
    flipped: bool = False
    created_ts: Optional[datetime] = None
    ref: str = ""

    def __post_init__(self) -> None:
        # Identity must be stable across cycles: 5.5's "one instance, one signal" and the
        # validity window both key off this ref, and a fresh uuid each cycle would re-arm
        # the same zone forever.
        if not self.ref:
            self.ref = f"zone:{self.source}:{self.kind.value}:{self.center:.1f}"

    @property
    def center(self) -> float:
        return (self.low + self.high) / 2.0

    @property
    def width(self) -> float:
        return self.high - self.low

    def contains(self, price: float) -> bool:
        return self.low <= price <= self.high

    def edge(self, direction: Direction) -> float:
        """The edge price must close beyond to trigger a reversal in ``direction`` (5.2)."""
        return self.high if direction is Direction.LONG else self.low

    def distance(self, price: float) -> float:
        if self.contains(price):
            return 0.0
        return self.low - price if price < self.low else price - self.high

    def score(self) -> int:
        return self.touches + self.rejections


def cluster_zones(
    swings: Iterable[SwingPoint],
    kind: str,
    atr_value: float,
    cfg,
    tier: Tier,
    source: str = "price_structure",
) -> list[Zone]:
    """Cluster >= 2 same-type swings within ``0.15 x ATR`` into zones of ``0.25 x ATR``."""
    cluster_tol = float(cfg.get("levels.sr_cluster_atr")) * atr_value
    width = float(cfg.get("levels.sr_zone_width_atr")) * atr_value
    pts = sorted((s for s in swings if s.kind == kind), key=lambda s: s.price)
    zones: list[Zone] = []
    bucket: list[SwingPoint] = []

    def flush() -> None:
        if len(bucket) < 2:
            return
        center = float(np.mean([p.price for p in bucket]))
        zones.append(
            Zone(
                kind=ZoneKind.RESISTANCE if kind == "high" else ZoneKind.SUPPORT,
                low=center - width / 2.0,
                high=center + width / 2.0,
                tier=tier,
                source=source,
                touches=len(bucket),
                strength=len(bucket),
                created_ts=max(p.ts for p in bucket),
            )
        )

    for point in pts:
        if bucket and point.price - bucket[-1].price > cluster_tol:
            flush()
            bucket = []
        bucket.append(point)
    flush()
    return zones


def session_zones(
    bars: pd.DataFrame,
    atr_value: float,
    cfg,
    session_open: str,
    include_overnight: bool = False,
) -> list[Zone]:
    """Session-structural Tier A prices (4.5).

    Prior-day high/low/close, current-session high/low, and for Gold the overnight
    (pre-session-open) high/low.
    """
    from beast.analysis.indicators import session_ids

    if bars.empty:
        return []
    width = float(cfg.get("levels.sr_zone_width_atr")) * atr_value
    sid = session_ids(bars.index, session_open)
    sessions = sid.unique()
    zones: list[Zone] = []

    def add(price: float, kind: ZoneKind, label: str) -> None:
        if price is None or not np.isfinite(price):
            return
        zones.append(
            Zone(
                kind=kind,
                low=price - width / 2.0,
                high=price + width / 2.0,
                tier=Tier.A,
                source=f"session:{label}",
                touches=1,
                strength=1,
                created_ts=bars.index[-1],
            )
        )

    current = bars[sid == sessions[-1]]
    add(float(current["high"].max()), ZoneKind.RESISTANCE, "session_high")
    add(float(current["low"].min()), ZoneKind.SUPPORT, "session_low")

    if len(sessions) > 1:
        prior = bars[sid == sessions[-2]]
        add(float(prior["high"].max()), ZoneKind.RESISTANCE, "prior_day_high")
        add(float(prior["low"].min()), ZoneKind.SUPPORT, "prior_day_low")
        close = float(prior["close"].iloc[-1])
        last = float(bars["close"].iloc[-1])
        add(close, ZoneKind.RESISTANCE if close > last else ZoneKind.SUPPORT, "prior_day_close")

    if include_overnight and len(sessions) > 1:
        # Gold: the pre-05:00 IST block of the current session date.
        overnight = bars[(sid == sessions[-1]) & (bars.index.hour < int(session_open.split(":")[0]))]
        if not overnight.empty:
            add(float(overnight["high"].max()), ZoneKind.RESISTANCE, "overnight_high")
            add(float(overnight["low"].min()), ZoneKind.SUPPORT, "overnight_low")
    return zones


def update_zones(zones: list[Zone], bars: pd.DataFrame, atr_value: float, cfg) -> list[Zone]:
    """Apply 4.5 zone invalidation and the flip rule.

    A zone dies once price closes beyond it by more than ``0.5 x ATR`` on the setup TF. A
    dead resistance becomes a candidate support (and vice versa), keeping its Tier and
    resetting strength to 1.
    """
    if bars.empty:
        return zones
    beyond = float(cfg.get("levels.zone_dead_close_atr")) * atr_value
    close = float(bars["close"].iloc[-1])
    out: list[Zone] = []
    for zone in zones:
        killed = (
            zone.kind is ZoneKind.RESISTANCE and close > zone.high + beyond
        ) or (zone.kind is ZoneKind.SUPPORT and close < zone.low - beyond)
        if not killed:
            out.append(zone)
            continue
        out.append(
            Zone(
                kind=ZoneKind.SUPPORT if zone.kind is ZoneKind.RESISTANCE else ZoneKind.RESISTANCE,
                low=zone.low,
                high=zone.high,
                tier=zone.tier,
                source=zone.source,
                strength=1,
                touches=1,
                rejections=0,
                flipped=True,
                created_ts=bars.index[-1],
            )
        )
    return out


def merge_convergent(price_zones: list[Zone], oi_zones: list[Zone], atr_value: float, cfg) -> tuple[list[Zone], bool]:
    """4.7.1 level-convergence bonus.

    Where an OI level lands within ``0.25 x ATR`` of a price-structure Tier A level, the
    merged zone's strength gets +2 - "the chart and the positioning agree". Returns the
    combined pool and whether any convergence occurred (recorded on the signal).
    """
    tol = float(cfg.get("options.oi_convergence_atr")) * atr_value
    bonus = int(cfg.get("options.oi_convergence_bonus"))
    pool = list(price_zones)
    converged = False
    for oi_zone in oi_zones:
        match = next(
            (
                z
                for z in pool
                if z.tier is Tier.A and abs(z.center - oi_zone.center) <= tol and z.kind is oi_zone.kind
            ),
            None,
        )
        if match is not None:
            match.strength += bonus
            match.source = f"{match.source}+{oi_zone.source}"
            converged = True
        else:
            pool.append(oi_zone)
    return pool, converged


# ---------------------------------------------------------------------------
# trendlines
# ---------------------------------------------------------------------------


@dataclass
class Trendline:
    """A least-squares trendline over >= 3 confirmed swings (4.5)."""

    kind: str  # "up" (from lows) | "down" (from highs)
    slope: float
    intercept: float
    anchors: tuple[int, ...]
    max_dev: float
    anchor_ts: tuple[str, ...] = ()
    ref: str = ""
    retired: bool = False

    def __post_init__(self) -> None:
        # Keyed on anchor timestamps, not bar positions: positions shift by one every time
        # a new candle closes, and an unstable ref would re-arm the same trendline forever.
        if not self.ref:
            anchors = self.anchor_ts or tuple(str(a) for a in self.anchors)
            self.ref = f"trendline:{self.kind}:{anchors[0]}:{anchors[-1]}"

    def value_at(self, idx: int) -> float:
        return self.slope * idx + self.intercept


def _fit(points: Sequence[tuple[int, float]]) -> tuple[float, float]:
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    slope, intercept = np.polyfit(xs, ys, 1)
    return float(slope), float(intercept)


def find_trendlines(
    bars: pd.DataFrame,
    swings: Sequence[SwingPoint],
    atr_value: float,
    cfg,
    max_candidates: int = 12,
) -> list[Trendline]:
    """Fit valid trendlines (4.5).

    Valid means: >= 3 confirmed anchors of the same type, maximum anchor deviation from
    the fitted line <= ``0.20 x ATR``, and **no candle has closed beyond the line** between
    the first and last anchor. Wicks through the line do not break it; closes do.

    Deviation is measured vertically, in price. A true perpendicular distance would mix
    price with bar count, which has no meaningful unit - the ATR tolerance is a price
    tolerance, so the measurement is taken in price.
    """
    min_touches = int(cfg.get("levels.trendline_min_touches"))
    tol = float(cfg.get("levels.trendline_max_dev_atr")) * atr_value
    closes = bars["close"].to_numpy()
    lines: list[Trendline] = []

    for kind, swing_kind in (("up", "low"), ("down", "high")):
        pts = [s for s in swings if s.kind == swing_kind][-max_candidates:]
        if len(pts) < min_touches:
            continue
        best: Optional[Trendline] = None
        for a, b in itertools.combinations(range(len(pts)), 2):
            seed = [(pts[a].idx, pts[a].price), (pts[b].idx, pts[b].price)]
            slope, intercept = _fit(seed)
            anchors = [
                p for p in pts if abs(p.price - (slope * p.idx + intercept)) <= tol
            ]
            if len(anchors) < min_touches:
                continue
            slope, intercept = _fit([(p.idx, p.price) for p in anchors])
            dev = max(abs(p.price - (slope * p.idx + intercept)) for p in anchors)
            if dev > tol:
                continue
            lo, hi = anchors[0].idx, anchors[-1].idx
            span = np.arange(lo, hi + 1)
            line_vals = slope * span + intercept
            if kind == "up" and np.any(closes[lo : hi + 1] < line_vals - 1e-9):
                continue
            if kind == "down" and np.any(closes[lo : hi + 1] > line_vals + 1e-9):
                continue
            candidate = Trendline(
                kind,
                slope,
                intercept,
                tuple(p.idx for p in anchors),
                dev,
                anchor_ts=tuple(p.ts.isoformat() for p in anchors),
            )
            if best is None or len(candidate.anchors) > len(best.anchors) or (
                len(candidate.anchors) == len(best.anchors) and candidate.anchors[-1] > best.anchors[-1]
            ):
                best = candidate
        if best is not None:
            lines.append(best)
    return lines


def trendline_broken(line: Trendline, bars: pd.DataFrame, idx: int, atr_value: float, cfg) -> Optional[Direction]:
    """Has the candle at ``idx`` closed beyond the line by >= ``0.10 x ATR`` (4.5, 5.2)?

    Returns the break direction, or ``None``. One such close retires the trendline.
    """
    thresh = float(cfg.get("levels.trendline_break_atr")) * atr_value
    close = float(bars["close"].iloc[idx])
    level = line.value_at(idx)
    if line.kind == "up" and close <= level - thresh:
        return Direction.SHORT
    if line.kind == "down" and close >= level + thresh:
        return Direction.LONG
    return None


# ---------------------------------------------------------------------------
# order blocks
# ---------------------------------------------------------------------------


@dataclass
class OrderBlock:
    """The last opposing candle before a structure-breaking impulse (4.5)."""

    direction: Direction  # direction of the impulse that created it
    low: float
    high: float
    created_idx: int
    created_ts: datetime
    expires_at: Optional[datetime] = None
    fresh: bool = True
    used: bool = False
    dead: bool = False
    touch_count: int = 0
    ref: str = ""

    def __post_init__(self) -> None:
        if not self.ref:
            self.ref = f"ob:{self.direction.value}:{self.created_ts.isoformat()}"

    def contains(self, price: float) -> bool:
        return self.low <= price <= self.high

    @property
    def far_edge(self) -> float:
        """The edge a close beyond which kills the block permanently (4.5)."""
        return self.low if self.direction is Direction.LONG else self.high

    def tradable(self) -> bool:
        return self.fresh and not self.used and not self.dead


def find_order_blocks(
    bars: pd.DataFrame,
    swings: Sequence[SwingPoint],
    atr_value: float,
    cfg,
    session_open: str,
    expiry_mode: str,
) -> list[OrderBlock]:
    """Detect order blocks (4.5).

    Impulse = >= 3 consecutive same-direction closes, **or** a move of >= ``1.5 x ATR``
    within 5 candles - and in either case the move must break a prior confirmed swing
    point. The order block is the last opposing-direction candle immediately before it.
    """
    if bars.empty:
        return []
    run_len = int(cfg.get("levels.ob_impulse_candles"))
    imp_atr = float(cfg.get("levels.ob_impulse_atr"))
    window = int(cfg.get("levels.ob_impulse_window"))
    body_mode = str(cfg.get("levels.ob_zone_mode")) == "body"

    o = bars["open"].to_numpy()
    h = bars["high"].to_numpy()
    low_ = bars["low"].to_numpy()
    c = bars["close"].to_numpy()
    n = len(bars)
    blocks: list[OrderBlock] = []

    for start in range(1, n):
        for direction, sign in ((Direction.LONG, 1), (Direction.SHORT, -1)):
            end = None
            run = 0
            for j in range(start, min(start + window, n)):
                if sign * (c[j] - o[j]) > 0:
                    run += 1
                else:
                    run = 0
                if run >= run_len:
                    end = j
                    break
                if sign * (c[j] - c[start - 1]) >= imp_atr * atr_value:
                    end = j
                    break
            if end is None:
                continue

            # structure break: the impulse must take out a prior confirmed swing.
            pivot = last_swing(swings, "high" if sign > 0 else "low", before_idx=start)
            if pivot is None:
                continue
            broke = h[start : end + 1].max() > pivot.price if sign > 0 else low_[start : end + 1].min() < pivot.price
            if not broke:
                continue

            ob_idx = None
            for k in range(start - 1, max(start - window, -1), -1):
                if sign * (c[k] - o[k]) < 0:
                    ob_idx = k
                    break
            if ob_idx is None:
                continue

            top = max(o[ob_idx], c[ob_idx]) if body_mode else h[ob_idx]
            bottom = min(o[ob_idx], c[ob_idx]) if body_mode else low_[ob_idx]
            ts = bars.index[ob_idx]
            expires = _ob_expiry(ts, expiry_mode, session_open)
            if any(b.created_idx == ob_idx and b.direction is direction for b in blocks):
                continue
            blocks.append(
                OrderBlock(
                    direction=direction,
                    low=float(bottom),
                    high=float(top),
                    created_idx=ob_idx,
                    created_ts=ts,
                    expires_at=expires,
                )
            )
    return blocks


def _ob_expiry(created: datetime, mode: str, session_open: str) -> Optional[datetime]:
    """4.5 - an untouched OB expires at session end (Indian) or after 24 hours (XAUUSD)."""
    if mode == "24h":
        return created + timedelta(hours=24)
    hour, minute = (int(x) for x in session_open.split(":"))
    anchor = created.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if created < anchor:
        anchor -= timedelta(days=1)
    return anchor + timedelta(days=1)


def update_order_blocks(
    blocks: list[OrderBlock], bars: pd.DataFrame, now: datetime, cfg
) -> list[OrderBlock]:
    """Apply freshness, invalidation and expiry (4.5).

    An OB is fresh until price trades into its zone once; after the first retest it is
    *used* and generates no further signals. A close beyond the far edge kills it
    permanently. An untouched OB expires on schedule.
    """
    retests_allowed = int(cfg.get("levels.ob_fresh_retests"))
    if bars.empty:
        return blocks
    last = bars.iloc[-1]
    for block in blocks:
        if block.dead:
            continue
        if block.expires_at is not None and now >= block.expires_at and block.fresh:
            block.dead = True
            continue
        close = float(last["close"])
        if (block.direction is Direction.LONG and close < block.far_edge) or (
            block.direction is Direction.SHORT and close > block.far_edge
        ):
            block.dead = True
            continue
        touched = float(last["low"]) <= block.high and float(last["high"]) >= block.low
        if touched:
            block.touch_count += 1
            if block.touch_count > retests_allowed:
                block.used = True
                block.fresh = False
    return blocks


# ---------------------------------------------------------------------------
# rejection candle (4.5) - used by Setups 2 and 3
# ---------------------------------------------------------------------------


def _closed_back_inside(close: float, zone: Zone, direction: Direction) -> bool:
    """"close back inside the zone" (4.5), read as *on the favourable side of the zone*.

    A zone is only ``0.25 x ATR`` wide, so a strong rejection off support routinely closes
    above the whole zone rather than within it. Taking "inside" literally would discard the
    cleanest rejections and keep only the weak ones, which inverts the rule's intent - the
    condition being tested is that price probed the level and came back, not that it
    stopped inside a three-point band.
    """
    if direction is Direction.LONG:
        return close >= zone.low
    return close <= zone.high


def is_rejection_candle(
    bars: pd.DataFrame,
    idx: int,
    zone: Zone,
    direction: Direction,
    rsi_series: Optional[pd.Series] = None,
    cfg=None,
) -> bool:
    """A candle satisfying **at least one** of the three 4.5 conditions.

    1. wick on the tested side >= 50% of total range, close back inside the zone; or
    2. an engulfing candle in the reversal direction (body engulfs the prior body); or
    3. a close back inside the zone after a wick pierce, with an RSI divergence (5.4).
    """
    row = bars.iloc[idx]
    o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
    rng = h - l
    if rng <= 0:
        return False

    if direction is Direction.LONG:
        wick = min(o, c) - l
        pierced = l <= zone.high
    else:
        wick = h - max(o, c)
        pierced = h >= zone.low
    inside = _closed_back_inside(c, zone, direction)

    if wick / rng >= 0.5 and inside:
        return True

    if idx > 0:
        prev = bars.iloc[idx - 1]
        po, pc = float(prev["open"]), float(prev["close"])
        body_low, body_high = min(o, c), max(o, c)
        pbody_low, pbody_high = min(po, pc), max(po, pc)
        engulfs = body_low <= pbody_low and body_high >= pbody_high
        bullish_body = c > o
        if engulfs and (bullish_body == (direction is Direction.LONG)):
            return True

    if pierced and inside and rsi_series is not None and cfg is not None:
        from beast.entry.confluence import has_divergence

        return has_divergence(bars, rsi_series, idx, direction, cfg)
    return False
