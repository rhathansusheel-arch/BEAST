"""Level engine - algorithmic definitions from soul file 4.5 and 4.7.1.

    The agent must not "eyeball" levels. Each object below has a deterministic
    definition, a validity lifetime, and an invalidation condition.

Sections 5 and 6 do not compute levels themselves; they consume what this module
produces. That is why the level engine is shared infrastructure with its own
state (:class:`LevelEngine`) rather than being recomputed inline per setup.

Everything here is expressed in underlying points. Option-chain OI levels are
built in ``core/option_chain.py`` and injected into the same Tier A pool via
:meth:`LevelEngine.merge_external_levels`, so they compete on the same strength
score as price structure - exactly as 4.7.1 specifies.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from core.config import Config, get_config
from core.schemas import (
    Direction,
    LevelKind,
    LevelTier,
    OrderBlock,
    Trendline,
    Zone,
)


@dataclass(frozen=True)
class Swing:
    """A confirmed fractal swing point.

    Attributes:
        index: Positional index into the frame it was found in.
        timestamp: Bar open time.
        price: The high (swing high) or low (swing low).
        is_high: True for a swing high.
    """

    index: int
    timestamp: datetime
    price: float
    is_high: bool


# ---------------------------------------------------------------------------
# Swing points
# ---------------------------------------------------------------------------


def find_swings(df: pd.DataFrame, fractal_n: int = 2,
                lookback: int = 100) -> list[Swing]:
    """Find confirmed fractal swing points (soul file 4.5).

    A swing high is a candle whose high is the highest of the ``fractal_n``
    candles either side of it; a swing low is the mirror. A swing is *confirmed*
    only after the ``fractal_n`` following candles have closed - which is why
    the last ``fractal_n`` bars can never contain a swing. This lag is the price
    of not repainting, and it is deliberate.

    Args:
        df: OHLC frame on the relevant timeframe.
        fractal_n: Candles either side that must be exceeded.
        lookback: Only the most recent ``lookback`` candles are scanned.

    Returns:
        Swings in ascending index order, highs and lows interleaved.
    """
    if len(df) < 2 * fractal_n + 1:
        return []

    start = max(0, len(df) - lookback)
    window = df.iloc[start:]
    highs = window["high"].to_numpy(dtype=float)
    lows = window["low"].to_numpy(dtype=float)
    times = window.index

    swings: list[Swing] = []
    # Stop `fractal_n` bars from the end: those swings are not yet confirmed.
    for pos in range(fractal_n, len(window) - fractal_n):
        left = slice(pos - fractal_n, pos)
        right = slice(pos + 1, pos + fractal_n + 1)

        if highs[pos] > highs[left].max() and highs[pos] > highs[right].max():
            swings.append(
                Swing(start + pos, times[pos].to_pydatetime(), float(highs[pos]), True)
            )
        if lows[pos] < lows[left].min() and lows[pos] < lows[right].min():
            swings.append(
                Swing(start + pos, times[pos].to_pydatetime(), float(lows[pos]), False)
            )

    swings.sort(key=lambda swing: swing.index)
    return swings


def last_swing(swings: Sequence[Swing], is_high: bool,
               before_index: int | None = None) -> Swing | None:
    """Return the most recent swing of the requested type."""
    for swing in reversed(swings):
        if swing.is_high != is_high:
            continue
        if before_index is not None and swing.index >= before_index:
            continue
        return swing
    return None


# ---------------------------------------------------------------------------
# Support / resistance zones
# ---------------------------------------------------------------------------


def cluster_swings(swings: Sequence[Swing], atr_value: float,
                   cluster_atr: float) -> list[list[Swing]]:
    """Group swings of the same type that sit within ``cluster_atr x ATR``.

    Single-member clusters are returned too; the caller drops them, because 4.5
    requires **>= 2 swing points of the same type** to form a zone.
    """
    if not swings or atr_value <= 0:
        return []
    tolerance = cluster_atr * atr_value
    ordered = sorted(swings, key=lambda swing: swing.price)

    clusters: list[list[Swing]] = [[ordered[0]]]
    for swing in ordered[1:]:
        # Compare against the running cluster mean so a long chain of swings
        # each just inside the tolerance cannot drift into one giant zone.
        current_mean = float(np.mean([member.price for member in clusters[-1]]))
        if abs(swing.price - current_mean) <= tolerance:
            clusters[-1].append(swing)
        else:
            clusters.append([swing])
    return clusters


def build_sr_zones(swings: Sequence[Swing], atr_value: float, tier: LevelTier,
                   config: Config | None = None) -> list[Zone]:
    """Build support/resistance zones from clustered swings (soul file 4.5).

    Zone width is ``0.25 x ATR`` centred on the mean of the clustered swings.
    Strength is the touch count; rejections add to it later as the engine
    observes them.

    Args:
        swings: Confirmed swings from one timeframe.
        atr_value: Current ATR on that timeframe.
        tier: A for bias-TF/structural levels, B for setup-TF only.
        config: Injected for tests.
    """
    cfg = config or get_config()
    if atr_value <= 0:
        return []

    cluster_atr = float(cfg.get("levels.sr_cluster_atr"))
    width_atr = float(cfg.get("levels.sr_zone_width_atr"))
    half_width = 0.5 * width_atr * atr_value

    zones: list[Zone] = []
    for is_high in (True, False):
        same_type = [swing for swing in swings if swing.is_high is is_high]
        for cluster in cluster_swings(same_type, atr_value, cluster_atr):
            if len(cluster) < 2:
                continue
            centre = float(np.mean([member.price for member in cluster]))
            zones.append(
                Zone(
                    zone_id=f"sr-{uuid.uuid4().hex[:10]}",
                    kind=LevelKind.SWING_CLUSTER,
                    tier=tier,
                    low=centre - half_width,
                    high=centre + half_width,
                    centre=centre,
                    strength=float(len(cluster)),
                    is_support=not is_high,
                    touches=len(cluster),
                    created_at=max(member.timestamp for member in cluster),
                )
            )
    return zones


def structural_zones(setup_df: pd.DataFrame, atr_value: float, market: str,
                     config: Config | None = None) -> list[Zone]:
    """Build the session-structural Tier A zones (soul file 4.5).

    Prior-day high/low/close and current-session high/low for every market, plus
    the overnight (pre-05:00 IST) high/low for Gold.

    Args:
        setup_df: Setup-timeframe OHLC frame covering at least two sessions.
        atr_value: Current setup-TF ATR, used for the zone width.
        market: Determines the session anchor and whether overnight levels apply.
    """
    cfg = config or get_config()
    if setup_df.empty or atr_value <= 0:
        return []

    timezone = str(cfg.get("sessions.timezone"))
    session = cfg.session(market)
    half_width = 0.5 * float(cfg.get("levels.sr_zone_width_atr")) * atr_value

    local = setup_df.index.tz_convert(timezone)
    open_hour, open_minute = (int(part) for part in str(session["open"]).split(":"))
    open_minutes = open_hour * 60 + open_minute
    minutes = local.hour * 60 + local.minute
    # Bars before the session open are attributed to the previous session.
    session_day = pd.Series(
        np.where(minutes >= open_minutes, local.date, (local - pd.Timedelta(days=1)).date),
        index=setup_df.index,
    )

    days = list(dict.fromkeys(session_day.tolist()))
    if not days:
        return []
    today = days[-1]
    today_mask = session_day == today

    def make(kind: LevelKind, price: float, is_support: bool) -> Zone:
        return Zone(
            zone_id=f"{kind.value}-{today}",
            kind=kind,
            tier=LevelTier.A,
            low=price - half_width,
            high=price + half_width,
            centre=float(price),
            strength=1.0,
            is_support=is_support,
            touches=1,
            created_at=setup_df.index[-1].to_pydatetime(),
        )

    zones: list[Zone] = []
    last_price = float(setup_df["close"].iloc[-1])

    if len(days) >= 2:
        prior = setup_df[session_day == days[-2]]
        if not prior.empty:
            prior_high = float(prior["high"].max())
            prior_low = float(prior["low"].min())
            prior_close = float(prior["close"].iloc[-1])
            zones.append(make(LevelKind.PRIOR_DAY_HIGH, prior_high, False))
            zones.append(make(LevelKind.PRIOR_DAY_LOW, prior_low, True))
            zones.append(
                make(LevelKind.PRIOR_DAY_CLOSE, prior_close, prior_close < last_price)
            )

    session_frame = setup_df[today_mask]
    if not session_frame.empty:
        zones.append(make(LevelKind.SESSION_HIGH, float(session_frame["high"].max()), False))
        zones.append(make(LevelKind.SESSION_LOW, float(session_frame["low"].min()), True))

    # Gold only: the overnight range before the 05:00 IST window start.
    if cfg.market_family(market) == "gold":
        overnight_mask = today_mask & (minutes < open_minutes)
        overnight = setup_df[overnight_mask]
        if not overnight.empty:
            zones.append(make(LevelKind.OVERNIGHT_HIGH, float(overnight["high"].max()), False))
            zones.append(make(LevelKind.OVERNIGHT_LOW, float(overnight["low"].min()), True))

    return zones


# ---------------------------------------------------------------------------
# Trendlines
# ---------------------------------------------------------------------------


def fit_trendlines(df: pd.DataFrame, swings: Sequence[Swing], atr_value: float,
                   config: Config | None = None) -> list[Trendline]:
    """Fit valid trendlines through confirmed swings (soul file 4.5).

    A trendline needs >= 3 confirmed swing points of the same type, fitted by
    least squares. It is valid only when the maximum perpendicular deviation of
    any anchor is ``<= 0.20 x ATR`` and no candle has *closed* beyond the line
    between the first and last anchor. Wicks through the line do not break it.

    The search is anchored on the most recent swings and walks backwards, so the
    lines returned are the ones price is currently interacting with rather than
    every line that could be drawn on the chart.

    Returns:
        Valid, unbroken trendlines, most-recently-anchored first.
    """
    cfg = config or get_config()
    if atr_value <= 0 or len(df) < 3:
        return []

    min_touches = int(cfg.get("levels.trendline_min_touches"))
    max_dev = float(cfg.get("levels.trendline_max_dev_atr")) * atr_value

    closes = df["close"].to_numpy(dtype=float)
    lines: list[Trendline] = []

    for is_high in (True, False):
        points = [swing for swing in swings if swing.is_high is is_high]
        if len(points) < min_touches:
            continue

        # Try the longest available anchor chains first, then shorter ones.
        for size in range(len(points), min_touches - 1, -1):
            for start in range(0, len(points) - size + 1):
                chain = points[start : start + size]
                xs = np.array([point.index for point in chain], dtype=float)
                ys = np.array([point.price for point in chain], dtype=float)
                if len(np.unique(xs)) < 2:
                    continue

                slope, intercept = np.polyfit(xs - xs[0], ys, 1)
                fitted = intercept + slope * (xs - xs[0])
                perpendicular = np.abs(ys - fitted) / np.sqrt(1.0 + slope**2)
                if perpendicular.max() > max_dev:
                    continue

                first, last = int(chain[0].index), int(chain[-1].index)
                span = np.arange(first, last + 1, dtype=float)
                line_values = intercept + slope * (span - xs[0])
                segment_closes = closes[first : last + 1]
                if is_high:
                    violated = np.any(segment_closes > line_values)
                else:
                    violated = np.any(segment_closes < line_values)
                if violated:
                    continue

                lines.append(
                    Trendline(
                        line_id=f"tl-{uuid.uuid4().hex[:10]}",
                        is_support=not is_high,
                        slope=float(slope),
                        intercept=float(intercept),
                        anchor_indices=[int(point.index) for point in chain],
                        anchor_prices=[float(point.price) for point in chain],
                        start_index=first,
                        max_deviation=float(perpendicular.max()),
                    )
                )
                break  # one line per swing type per size; take the longest chain
            if lines and lines[-1].is_support == (not is_high):
                break

    return lines


def trendline_broken(line: Trendline, df: pd.DataFrame, index: int,
                     atr_value: float, config: Config | None = None) -> bool:
    """True when the candle at ``index`` closed beyond ``line`` by the break threshold.

    A close beyond the line by ``>= 0.10 x ATR`` retires the trendline and may
    generate a Setup 1 signal (soul file 4.5, 5.2).
    """
    cfg = config or get_config()
    threshold = float(cfg.get("levels.trendline_break_atr")) * atr_value
    close = float(df["close"].iloc[index])
    line_price = line.value_at(index)
    if line.is_support:
        return close < line_price - threshold
    return close > line_price + threshold


# ---------------------------------------------------------------------------
# Order blocks
# ---------------------------------------------------------------------------


def detect_order_blocks(df: pd.DataFrame, swings: Sequence[Swing], atr_value: float,
                        session_end: datetime | None = None,
                        config: Config | None = None) -> list[OrderBlock]:
    """Detect order blocks (soul file 4.5).

    An **impulse** is >= 3 consecutive same-direction closes, or a move of
    ``>= 1.5 x ATR`` within 5 candles - and in either case the move must break a
    prior confirmed swing point. The **order block** is the last
    opposing-direction candle immediately before that impulse.

    Args:
        df: Setup-timeframe frame.
        swings: Confirmed swings on the same frame, for the structure-break test.
        atr_value: Current setup-TF ATR.
        session_end: When the current session ends, for ``ob_expiry: session``.

    Returns:
        Order blocks in formation order, oldest first.
    """
    cfg = config or get_config()
    if atr_value <= 0 or len(df) < 6:
        return []

    consecutive = int(cfg.get("levels.ob_impulse_candles"))
    impulse_atr = float(cfg.get("levels.ob_impulse_atr"))
    window = int(cfg.get("levels.ob_impulse_window"))
    zone_mode = str(cfg.get("levels.ob_zone_mode"))
    expiry_mode = str(cfg.get("levels.ob_expiry"))

    opens = df["open"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    bullish = closes > opens

    swing_highs = [swing for swing in swings if swing.is_high]
    swing_lows = [swing for swing in swings if not swing.is_high]

    def breaks_structure(direction: Direction, start: int, end: int) -> bool:
        """True when the leg [start, end] closed beyond a prior confirmed swing."""
        if direction is Direction.LONG:
            prior = [swing for swing in swing_highs if swing.index < start]
            if not prior:
                return False
            return float(closes[start : end + 1].max()) > prior[-1].price
        prior = [swing for swing in swing_lows if swing.index < start]
        if not prior:
            return False
        return float(closes[start : end + 1].min()) < prior[-1].price

    blocks: list[OrderBlock] = []
    seen_starts: set[tuple[int, str]] = set()

    for start in range(1, len(df)):
        for direction, is_up in ((Direction.LONG, True), (Direction.SHORT, False)):
            end: int | None = None

            # Impulse form A: N consecutive same-direction closes.
            run_end = start + consecutive - 1
            if run_end < len(df) and bool(np.all(bullish[start : run_end + 1] == is_up)):
                end = run_end

            # Impulse form B: a >= 1.5 ATR move inside `window` candles.
            if end is None:
                far = min(start + window - 1, len(df) - 1)
                if far > start:
                    if is_up:
                        move = float(highs[start : far + 1].max() - lows[start])
                    else:
                        move = float(highs[start] - lows[start : far + 1].min())
                    if move >= impulse_atr * atr_value:
                        end = far

            if end is None:
                continue
            if not breaks_structure(direction, start, end):
                continue

            key = (start, direction.value)
            if key in seen_starts:
                continue

            # The order block is the last opposing-direction candle before the leg.
            ob_index: int | None = None
            for back in range(start - 1, max(-1, start - 1 - window), -1):
                if bool(bullish[back]) != is_up:
                    ob_index = back
                    break
            if ob_index is None:
                continue

            seen_starts.add(key)
            if zone_mode == "wick":
                low, high = float(lows[ob_index]), float(highs[ob_index])
            else:
                low = float(min(opens[ob_index], closes[ob_index]))
                high = float(max(opens[ob_index], closes[ob_index]))
            if high <= low:  # a doji body has no zone to retest
                continue

            formed_at = df.index[ob_index].to_pydatetime()
            if expiry_mode == "session":
                expires_at = session_end
            else:
                expires_at = formed_at + timedelta(hours=24)

            blocks.append(
                OrderBlock(
                    ob_id=f"ob-{uuid.uuid4().hex[:10]}",
                    direction=direction,
                    low=low,
                    high=high,
                    formed_index=ob_index,
                    formed_at=formed_at,
                    expires_at=expires_at,
                )
            )

    return blocks


# ---------------------------------------------------------------------------
# Rejection candle
# ---------------------------------------------------------------------------


def is_engulfing(df: pd.DataFrame, index: int, direction: Direction) -> bool:
    """True when the candle's body engulfs the prior candle's body."""
    if index < 1:
        return False
    open_now, close_now = float(df["open"].iloc[index]), float(df["close"].iloc[index])
    open_prev, close_prev = float(df["open"].iloc[index - 1]), float(df["close"].iloc[index - 1])
    body_low, body_high = min(open_now, close_now), max(open_now, close_now)
    prev_low, prev_high = min(open_prev, close_prev), max(open_prev, close_prev)
    if body_low > prev_low or body_high < prev_high:
        return False
    return close_now > open_now if direction is Direction.LONG else close_now < open_now


def is_rejection_candle(df: pd.DataFrame, index: int, zone: Zone,
                        direction: Direction, has_divergence: bool = False,
                        config: Config | None = None) -> tuple[bool, str]:
    """Test the 4.5 rejection-candle definition.

    A candle qualifies on **at least one** of three grounds:

    1. The wick on the tested side is >= 50% of the candle's total range and the
       close is back inside the zone.
    2. It is an engulfing candle in the reversal direction.
    3. It closes back inside the zone after any wick pierce, combined with an
       RSI divergence at the level (per 5.4).

    Args:
        direction: The reversal direction being tested - ``LONG`` at support.
        has_divergence: Result of the 5.4 divergence check, supplied by the
            caller so this module stays free of indicator dependencies.

    Returns:
        ``(qualifies, reason)`` - the reason string is recorded on the signal.
    """
    cfg = config or get_config()
    wick_ratio = float(cfg.get("levels.rejection_wick_ratio"))

    high = float(df["high"].iloc[index])
    low = float(df["low"].iloc[index])
    open_price = float(df["open"].iloc[index])
    close = float(df["close"].iloc[index])
    total_range = high - low
    if total_range <= 0:
        return False, ""

    body_low, body_high = min(open_price, close), max(open_price, close)
    if direction is Direction.LONG:
        tested_wick = body_low - low
        pierced = low < zone.low
    else:
        tested_wick = high - body_high
        pierced = high > zone.high

    closed_inside = zone.contains(close)

    if closed_inside and tested_wick / total_range >= wick_ratio:
        return True, f"wick {tested_wick / total_range:.0%} of range, close inside zone"
    if is_engulfing(df, index, direction):
        return True, "engulfing candle in the reversal direction"
    if closed_inside and pierced and has_divergence:
        return True, "close back inside zone after pierce, with RSI divergence"
    return False, ""


# ---------------------------------------------------------------------------
# Stateful engine
# ---------------------------------------------------------------------------


class LevelEngine:
    """Maintains the level pool for one instrument across bars.

    Zones, trendlines and order blocks each have a lifetime, so they cannot be
    recomputed from scratch every bar without losing the state the soul file
    depends on: order-block freshness (4.5), the per-session re-entry cap (5.5),
    and zone flips. This class owns that state.

    Attributes:
        market: Market key, e.g. ``"NIFTY50"``.
        zones: Live zones, Tier A and B pooled together.
        trendlines: Live trendlines on the setup timeframe.
        order_blocks: Live order blocks.
        blacklisted: Zone ids retired for the session by the re-entry cap.
    """

    def __init__(self, market: str, config: Config | None = None) -> None:
        self.cfg = config or get_config()
        self.market = market
        self.zones: list[Zone] = []
        self.trendlines: list[Trendline] = []
        self.order_blocks: list[OrderBlock] = []
        self.blacklisted: set[str] = set()
        self._failed_attempts: dict[str, int] = {}
        self._session_day: object | None = None

    # -- session handling ----------------------------------------------------

    def roll_session(self, session_day: object) -> None:
        """Reset per-session state when a new session begins.

        Order blocks expire at the end of the session in which they formed
        (Indian) and the level blacklist is per-session (5.5), so both are
        cleared here rather than aging out bar by bar.
        """
        if self._session_day == session_day:
            return
        self._session_day = session_day
        self.blacklisted.clear()
        self._failed_attempts.clear()
        if str(self.cfg.get("levels.ob_expiry")) == "session":
            self.order_blocks = []

    # -- rebuild -------------------------------------------------------------

    def rebuild(self, setup_df: pd.DataFrame, bias_df: pd.DataFrame,
                atr_setup: float, session_end: datetime | None = None) -> None:
        """Recompute the level pool on a setup-timeframe close (soul file 4.2).

        Tier A comes from the bias timeframe and from session structure; Tier B
        from the setup timeframe only. Existing zones are carried forward so
        that touch counts and strength accumulate rather than resetting.
        """
        fractal_n = int(self.cfg.get("levels.fractal_n"))
        lookback = int(self.cfg.get("levels.swing_lookback"))

        setup_swings = find_swings(setup_df, fractal_n, lookback)
        bias_swings = find_swings(bias_df, fractal_n, lookback) if len(bias_df) else []

        fresh: list[Zone] = []
        fresh.extend(build_sr_zones(bias_swings, atr_setup, LevelTier.A, self.cfg))
        fresh.extend(structural_zones(setup_df, atr_setup, self.market, self.cfg))
        fresh.extend(build_sr_zones(setup_swings, atr_setup, LevelTier.B, self.cfg))

        self.zones = self._merge_zones(self.zones, fresh, atr_setup)
        self.trendlines = fit_trendlines(setup_df, setup_swings, atr_setup, self.cfg)

        known = {block.ob_id for block in self.order_blocks}
        for block in detect_order_blocks(
            setup_df, setup_swings, atr_setup, session_end, self.cfg
        ):
            if not any(
                abs(block.low - existing.low) < 1e-9
                and abs(block.high - existing.high) < 1e-9
                and block.direction is existing.direction
                for existing in self.order_blocks
            ):
                known.add(block.ob_id)
                self.order_blocks.append(block)

        self._age_out(setup_df, atr_setup)

    def _merge_zones(self, existing: list[Zone], fresh: list[Zone],
                     atr_value: float) -> list[Zone]:
        """Fold newly-detected zones into the live pool.

        A fresh zone that overlaps a live one of the same polarity is treated as
        another touch of that level rather than a second zone - otherwise the
        pool fills with near-duplicates and the "next opposing level" test at G7
        picks an arbitrary one of them.
        """
        merged = [zone for zone in existing if zone.alive]
        tolerance = 0.5 * float(self.cfg.get("levels.sr_zone_width_atr")) * atr_value

        for candidate in fresh:
            match = next(
                (
                    zone
                    for zone in merged
                    if zone.is_support == candidate.is_support
                    and abs(zone.centre - candidate.centre) <= tolerance
                ),
                None,
            )
            if match is None:
                merged.append(candidate)
                continue
            match.touches = max(match.touches, candidate.touches)
            match.strength = max(match.strength, candidate.strength)
            if candidate.tier is LevelTier.A:
                match.tier = LevelTier.A
        return merged

    def _age_out(self, setup_df: pd.DataFrame, atr_value: float) -> None:
        """Apply every invalidation rule in 4.5."""
        close = float(setup_df["close"].iloc[-1])
        index = len(setup_df) - 1

        # Zones: a close beyond by > 0.5 x ATR kills the zone. A dead resistance
        # becomes a candidate support (flip), keeping Tier and resetting strength.
        threshold = float(self.cfg.get("levels.sr_invalidate_atr")) * atr_value
        flipped: list[Zone] = []
        for zone in self.zones:
            if not zone.alive:
                continue
            broke_up = close > zone.high + threshold
            broke_down = close < zone.low - threshold
            if (zone.is_support and broke_down) or (not zone.is_support and broke_up):
                zone.alive = False
                flipped.append(
                    Zone(
                        zone_id=f"flip-{uuid.uuid4().hex[:10]}",
                        kind=zone.kind,
                        tier=zone.tier,
                        low=zone.low,
                        high=zone.high,
                        centre=zone.centre,
                        strength=1.0,
                        is_support=not zone.is_support,
                        touches=1,
                        created_at=setup_df.index[-1].to_pydatetime(),
                        flipped_from=zone.zone_id,
                    )
                )
        self.zones = [zone for zone in self.zones if zone.alive] + flipped

        # Trendlines: one close beyond by >= 0.10 x ATR retires the line.
        for line in self.trendlines:
            if line.alive and trendline_broken(line, setup_df, index, atr_value, self.cfg):
                line.alive = False
                line.broken_at_index = index

        # Order blocks: freshness, the far-edge kill, and expiry.
        fresh_retests = int(self.cfg.get("levels.ob_fresh_retests"))
        now = setup_df.index[-1].to_pydatetime()
        low = float(setup_df["low"].iloc[-1])
        high = float(setup_df["high"].iloc[-1])
        for block in self.order_blocks:
            if not block.alive:
                continue
            if block.expires_at is not None and now >= block.expires_at:
                block.alive = False
                continue
            if block.direction is Direction.LONG and close < block.far_edge:
                block.alive = False
                continue
            if block.direction is Direction.SHORT and close > block.far_edge:
                block.alive = False
                continue
            if low <= block.high and high >= block.low:
                block.retests += 1
                if block.retests > fresh_retests:
                    block.fresh = False
        self.order_blocks = [block for block in self.order_blocks if block.alive]

    # -- external levels (option chain) --------------------------------------

    def merge_external_levels(self, external: Iterable[Zone], atr_value: float) -> None:
        """Merge OI-derived Tier A levels into the pool (soul file 4.7.1).

        Where an OI level lands within ``0.25 x ATR`` of a price-structure Tier A
        level, the merged zone's strength gets ``+2``. That is the highest-quality
        level Beast can identify - the chart and the positioning agree.
        """
        convergence_atr = float(self.cfg.get("options.level_convergence_atr")) * atr_value
        bonus = float(self.cfg.get("options.level_convergence_bonus"))

        # Drop the previous cycle's OI levels; the chain is re-snapshotted each
        # setup-TF close and stale OI walls must not linger in the pool.
        oi_kinds = {
            LevelKind.MAX_CALL_OI,
            LevelKind.MAX_PUT_OI,
            LevelKind.SECOND_MAX_CALL_OI,
            LevelKind.SECOND_MAX_PUT_OI,
            LevelKind.MAX_OI_CHANGE_CALL,
            LevelKind.MAX_OI_CHANGE_PUT,
        }
        self.zones = [zone for zone in self.zones if zone.kind not in oi_kinds]

        for level in external:
            for zone in self.zones:
                if zone.tier is not LevelTier.A or not zone.alive:
                    continue
                if abs(zone.centre - level.centre) <= convergence_atr:
                    zone.strength += bonus
                    zone.converged_with.append(level.zone_id)
                    level.strength += bonus
                    level.converged_with.append(zone.zone_id)
            self.zones.append(level)

    # -- queries -------------------------------------------------------------

    def live_zones(self, tier: LevelTier | None = None) -> list[Zone]:
        """Return live, non-blacklisted zones, optionally filtered by tier."""
        return [
            zone
            for zone in self.zones
            if zone.alive
            and zone.zone_id not in self.blacklisted
            and (tier is None or zone.tier is tier)
        ]

    def next_opposing_level(self, price: float, direction: Direction,
                            tier: LevelTier | None = LevelTier.A) -> Zone | None:
        """The nearest level standing between ``price`` and the trade's target.

        This is the G7 feasibility test's input (soul file 6.2). For a long, that
        is the nearest resistance above; for a short, the nearest support below.
        """
        candidates = []
        for zone in self.live_zones(tier):
            if direction is Direction.LONG and not zone.is_support and zone.low > price:
                candidates.append((zone.low - price, zone))
            elif direction is Direction.SHORT and zone.is_support and zone.high < price:
                candidates.append((price - zone.high, zone))
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])[1]

    def zones_containing(self, price: float,
                         tier: LevelTier | None = None) -> list[Zone]:
        """Live zones whose bounds contain ``price``, strongest first."""
        hits = [zone for zone in self.live_zones(tier) if zone.contains(price)]
        return sorted(hits, key=lambda zone: zone.strength, reverse=True)

    def fresh_order_blocks(self, direction: Direction | None = None) -> list[OrderBlock]:
        """Live, fresh, unexpired order blocks."""
        return [
            block
            for block in self.order_blocks
            if block.alive and block.fresh and (direction is None or block.direction is direction)
        ]

    def register_failed_attempt(self, zone_id: str) -> bool:
        """Record a failed attempt at a level; blacklist it at the cap (5.5).

        Returns:
            True when this attempt tipped the level onto the blacklist.
        """
        cap = int(self.cfg.get("entry.level_reentry_cap"))
        self._failed_attempts[zone_id] = self._failed_attempts.get(zone_id, 0) + 1
        if self._failed_attempts[zone_id] >= cap:
            self.blacklisted.add(zone_id)
            return True
        return False

    def attempts_at(self, zone_id: str) -> int:
        """How many failed attempts this level has taken this session."""
        return self._failed_attempts.get(zone_id, 0)
