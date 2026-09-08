"""The 4-of-6 confluence engine (soul file 5.3) and its 5.4 definitions.

Two things about this module are non-negotiable and are worth stating up front,
because getting either wrong silently changes what Beast trades:

1. **Two alignment modes, selected by setup type.** Setup 2 (reversal at S/R)
   reads the *Reversal* column; setups 1, 3 and 4 read the *Trend-Continuation*
   column. There is no single flat rule per indicator, and the two columns are
   never mixed inside one count.
2. **Six indicators, on the setup timeframe, on the underlying.** ADX+DI,
   Stochastic, MACD, RSI, Bollinger Bands and session VWAP. ATR is a utility and
   is never counted. The option chain is context and never adds to or subtracts
   from the count (4.7).

Section 5.4 turns every judgement call in the 5.3 table ("fresh crossover",
"expanding histogram", "walking the band") into a machine-checkable predicate.
Those predicates live here as module-level functions so they are testable in
isolation and so no caller can re-interpret them.
"""

from __future__ import annotations

import pandas as pd

from core.config import Config, get_config
from core.schemas import (
    ConfluenceMode,
    ConfluenceResult,
    Direction,
    IndicatorRead,
    Zone,
)

INDICATOR_NAMES = ("adx", "stoch", "macd", "rsi", "bb", "vwap")


# ---------------------------------------------------------------------------
# Section 5.4 - machine-checkable definitions
# ---------------------------------------------------------------------------


def is_rising(series: pd.Series, lookback: int, min_delta: float) -> bool:
    """"rising" per 5.4: current vs ``lookback`` candles ago, by more than ``min_delta``."""
    if len(series) <= lookback:
        return False
    now = float(series.iloc[-1])
    then = float(series.iloc[-1 - lookback])
    if pd.isna(now) or pd.isna(then):
        return False
    return (now - then) > min_delta


def is_falling(series: pd.Series, lookback: int, min_delta: float) -> bool:
    """"falling" per 5.4 - the mirror of :func:`is_rising`."""
    if len(series) <= lookback:
        return False
    now = float(series.iloc[-1])
    then = float(series.iloc[-1 - lookback])
    if pd.isna(now) or pd.isna(then):
        return False
    return (then - now) > min_delta


def crossed_above(fast: pd.Series, slow: pd.Series, within: int) -> int | None:
    """Index offset of a fresh upward cross, or ``None``.

    Returns:
        ``0`` when the cross happened on the latest closed candle, ``1`` for the
        one before it, and so on up to ``within - 1``. ``None`` when no cross
        occurred inside the window - which is what makes a crossover "fresh"
        (5.4: the cross occurred within the last 3 closed candles).
    """
    if len(fast) < within + 2:
        return None
    for offset in range(within):
        now = -1 - offset
        prev = now - 1
        if pd.isna(fast.iloc[now]) or pd.isna(slow.iloc[now]):
            continue
        if pd.isna(fast.iloc[prev]) or pd.isna(slow.iloc[prev]):
            continue
        if fast.iloc[prev] <= slow.iloc[prev] and fast.iloc[now] > slow.iloc[now]:
            return offset
    return None


def crossed_below(fast: pd.Series, slow: pd.Series, within: int) -> int | None:
    """Index offset of a fresh downward cross, or ``None``."""
    if len(fast) < within + 2:
        return None
    for offset in range(within):
        now = -1 - offset
        prev = now - 1
        if pd.isna(fast.iloc[now]) or pd.isna(slow.iloc[now]):
            continue
        if pd.isna(fast.iloc[prev]) or pd.isna(slow.iloc[prev]):
            continue
        if fast.iloc[prev] >= slow.iloc[prev] and fast.iloc[now] < slow.iloc[now]:
            return offset
    return None


def histogram_expanding(hist: pd.Series, direction: Direction, bars: int) -> bool:
    """"histogram expanding" per 5.4.

    The absolute histogram value must have increased on each of the last ``bars``
    candles, and its sign must match the trade direction.
    """
    if len(hist) < bars + 1:
        return False
    window = hist.iloc[-(bars + 1):]
    if window.isna().any():
        return False
    sign_ok = (window.iloc[-1] > 0) if direction is Direction.LONG else (window.iloc[-1] < 0)
    if not sign_ok:
        return False
    magnitudes = window.abs().to_list()
    return all(later > earlier for earlier, later in zip(magnitudes, magnitudes[1:]))


def walking_band(df: pd.DataFrame, direction: Direction, atr_value: float,
                 lookback: int, min_closes: int, tolerance_atr: float) -> bool:
    """"walking/hugging the band" per 5.4.

    At least ``min_closes`` of the last ``lookback`` closes are above the upper
    band minus a ``tolerance_atr x ATR`` cushion (bullish), or below the lower
    band plus that cushion (bearish).
    """
    if len(df) < lookback:
        return False
    window = df.iloc[-lookback:]
    tolerance = tolerance_atr * atr_value
    if direction is Direction.LONG:
        hits = (window["close"] >= window["bb_upper"] - tolerance).sum()
    else:
        hits = (window["close"] <= window["bb_lower"] + tolerance).sum()
    return int(hits) >= min_closes


def tags_band(df: pd.DataFrame, direction: Direction) -> bool:
    """"tags or pierces the band" per 5.4, with the close back inside.

    For a bullish reversal the candle's low must reach the lower band; for a
    bearish reversal the high must reach the upper band.
    """
    if df.empty:
        return False
    row = df.iloc[-1]
    if any(pd.isna(row.get(col)) for col in ("bb_upper", "bb_lower")):
        return False
    if direction is Direction.LONG:
        return bool(row["low"] <= row["bb_lower"] and row["close"] > row["bb_lower"])
    return bool(row["high"] >= row["bb_upper"] and row["close"] < row["bb_upper"])


def vwap_holding(df: pd.DataFrame, direction: Direction, atr_value: float,
                 lookback: int, proximity_atr: float) -> bool:
    """"pullbacks holding VWAP as support/resistance" per 5.4.

    Within the last ``lookback`` candles price traded to within
    ``proximity_atr x ATR`` of VWAP and closed back above it, with no close
    below it. Mirrored for resistance.
    """
    if len(df) < lookback:
        return False
    window = df.iloc[-lookback:]
    if window["vwap"].isna().any():
        return False
    proximity = proximity_atr * atr_value

    if direction is Direction.LONG:
        if (window["close"] < window["vwap"]).any():
            return False
        return bool(((window["low"] - window["vwap"]).abs() <= proximity).any())
    if (window["close"] > window["vwap"]).any():
        return False
    return bool(((window["high"] - window["vwap"]).abs() <= proximity).any())


def rsi_divergence(df: pd.DataFrame, direction: Direction, lookback: int,
                   fractal_n: int = 2) -> bool:
    """"bullish/bearish divergence" per 5.4.

    Bullish: price makes a lower low versus the prior confirmed swing low within
    the last ``lookback`` candles, while RSI makes a higher low at those two
    points. Mirrored for bearish.

    The swing points are the same confirmed fractals the level engine uses, so
    "the prior confirmed swing low" means the same thing everywhere in the system.
    """
    from core.levels import find_swings  # local import avoids a circular dependency

    if len(df) < lookback or "rsi" not in df.columns:
        return False
    window = df.iloc[-lookback:]
    swings = find_swings(window, fractal_n, lookback)
    points = [swing for swing in swings if swing.is_high is (direction is Direction.SHORT)]
    if len(points) < 2:
        return False

    previous, latest = points[-2], points[-1]
    # find_swings indexes against the frame it was handed, so index into `window`.
    rsi_prev = float(window["rsi"].iloc[previous.index])
    rsi_last = float(window["rsi"].iloc[latest.index])
    if pd.isna(rsi_prev) or pd.isna(rsi_last):
        return False

    if direction is Direction.LONG:
        return latest.price < previous.price and rsi_last > rsi_prev
    return latest.price > previous.price and rsi_last < rsi_prev


def near_level(df: pd.DataFrame, offset: int, zone: Zone | None, atr_value: float,
               near_atr: float) -> bool:
    """"at/near the level" per 5.4.

    The crossing candle's body must sit within ``near_atr x ATR`` of the S/R zone
    centre. When no zone is supplied the test cannot be satisfied - a reversal
    setup always has a level, so a missing one means the caller made a mistake
    and the safe answer is "not aligned".
    """
    if zone is None or atr_value <= 0:
        return False
    position = -1 - offset
    if len(df) < abs(position):
        return False
    row = df.iloc[position]
    body_low = min(float(row["open"]), float(row["close"]))
    body_high = max(float(row["open"]), float(row["close"]))
    tolerance = near_atr * atr_value
    # Distance from the body (as an interval) to the zone centre.
    if body_low <= zone.centre <= body_high:
        return True
    return min(abs(zone.centre - body_low), abs(zone.centre - body_high)) <= tolerance


# ---------------------------------------------------------------------------
# Per-indicator reads
# ---------------------------------------------------------------------------


class ConfluenceEngine:
    """Evaluates the 5.3 table for one setup-timeframe frame.

    Args:
        config: Injected for tests; defaults to the singleton.

    The engine is stateless between calls - every read is derived from the frame
    it is handed. That is what makes backtest and live identical.
    """

    def __init__(self, config: Config | None = None) -> None:
        self.cfg = config or get_config()

    # -- public API ----------------------------------------------------------

    def evaluate(self, df: pd.DataFrame, direction: Direction, mode: ConfluenceMode,
                 zone: Zone | None = None, required: int | None = None) -> ConfluenceResult:
        """Count aligned indicators for ``direction`` under ``mode``.

        Args:
            df: Setup-timeframe frame with indicator columns attached, already
                trimmed to closed candles.
            direction: The direction being proposed.
            mode: Which column of the 5.3 table to read.
            zone: The S/R zone under test - required by the reversal column's
                "at/near the level" conditions.
            required: Confluence threshold. Defaults to ``entry.min_confluence``;
                callers pass 5 for counter-bias reversals (4.4).

        Returns:
            A :class:`ConfluenceResult` carrying all six reads, including the
            neutrals - section 9 needs the misses to compute per-indicator hit
            rates, so nothing is discarded.
        """
        if required is None:
            required = int(self.cfg.get("entry.min_confluence"))

        atr_value = float(df["atr"].iloc[-1]) if "atr" in df.columns else 0.0
        notes: dict[str, str] = {}

        if mode is ConfluenceMode.REVERSAL:
            reads = {
                "adx": self._adx_reversal(df, notes),
                "stoch": self._stoch_reversal(df, notes),
                "macd": self._macd_reversal(df, zone, atr_value, notes),
                "rsi": self._rsi_reversal(df, zone, atr_value, notes),
                "bb": self._bb_reversal(df, notes),
                "vwap": self._vwap_read(df, atr_value, notes),
            }
        else:
            reads = {
                "adx": self._adx_trend(df, notes),
                "stoch": self._stoch_trend(df, notes),
                "macd": self._macd_trend(df, notes),
                "rsi": self._rsi_trend(df, notes),
                "bb": self._bb_trend(df, atr_value, notes),
                "vwap": self._vwap_read(df, atr_value, notes),
            }

        wanted = IndicatorRead.BULL if direction is Direction.LONG else IndicatorRead.BEAR
        against = IndicatorRead.BEAR if direction is Direction.LONG else IndicatorRead.BULL

        aligned = sum(1 for read in reads.values() if read is wanted)
        opposing = sum(1 for read in reads.values() if read is against)
        neutral = sum(1 for read in reads.values() if read is IndicatorRead.NEUTRAL)

        # Conflict rule (5.3): a 4-3 split is not confluence, it is disagreement.
        conflict_threshold = int(self.cfg.get("entry.opposing_reject_count"))
        rejected_by_conflict = opposing >= conflict_threshold
        passed = aligned >= required and not rejected_by_conflict

        return ConfluenceResult(
            mode=mode,
            direction=direction,
            reads=reads,
            aligned=aligned,
            opposing=opposing,
            neutral=neutral,
            required=required,
            passed=passed,
            rejected_by_conflict=rejected_by_conflict,
            notes=notes,
        )

    # -- trend-continuation column -------------------------------------------

    def _adx_trend(self, df: pd.DataFrame, notes: dict[str, str]) -> IndicatorRead:
        """ADX >= 20 and rising, with the DI pair agreeing. Below 20 is neutral."""
        threshold = float(self.cfg.get("indicators.adx_trend_threshold"))
        defs = self.cfg.section("entry")["definitions"]
        row = df.iloc[-1]
        if any(pd.isna(row.get(col)) for col in ("adx", "plus_di", "minus_di")):
            return IndicatorRead.NEUTRAL

        if float(row["adx"]) < threshold:
            notes["adx"] = f"ADX {row['adx']:.1f} below {threshold:.0f} - chop filter, neutral"
            return IndicatorRead.NEUTRAL
        if not is_rising(df["adx"], int(defs["rising_lookback"]), float(defs["rising_min_delta"])):
            notes["adx"] = f"ADX {row['adx']:.1f} not rising"
            return IndicatorRead.NEUTRAL

        if float(row["plus_di"]) > float(row["minus_di"]):
            notes["adx"] = f"ADX {row['adx']:.1f} rising, +DI > -DI"
            return IndicatorRead.BULL
        notes["adx"] = f"ADX {row['adx']:.1f} rising, -DI > +DI"
        return IndicatorRead.BEAR

    def _stoch_trend(self, df: pd.DataFrame, notes: dict[str, str]) -> IndicatorRead:
        """%K vs %D with momentum, and not yet at the far extreme."""
        levels = self.cfg.section("indicators")["stoch_levels"]
        row, prev = df.iloc[-1], df.iloc[-2] if len(df) > 1 else df.iloc[-1]
        if any(pd.isna(row.get(col)) for col in ("stoch_k", "stoch_d")):
            return IndicatorRead.NEUTRAL

        k_now, d_now = float(row["stoch_k"]), float(row["stoch_d"])
        k_prev = float(prev["stoch_k"]) if not pd.isna(prev["stoch_k"]) else k_now
        rising = k_now > k_prev

        if k_now > d_now and rising and k_now < float(levels["overbought"]):
            notes["stoch"] = f"%K {k_now:.0f} > %D {d_now:.0f}, rising, below overbought"
            return IndicatorRead.BULL
        if k_now < d_now and not rising and k_now > float(levels["oversold"]):
            notes["stoch"] = f"%K {k_now:.0f} < %D {d_now:.0f}, falling, above oversold"
            return IndicatorRead.BEAR
        return IndicatorRead.NEUTRAL

    def _macd_trend(self, df: pd.DataFrame, notes: dict[str, str]) -> IndicatorRead:
        """Line versus signal, with an expanding histogram of matching sign."""
        defs = self.cfg.section("entry")["definitions"]
        bars = int(defs["histogram_expand_bars"])
        row = df.iloc[-1]
        if any(pd.isna(row.get(col)) for col in ("macd", "macd_signal", "macd_hist")):
            return IndicatorRead.NEUTRAL

        above = float(row["macd"]) > float(row["macd_signal"])
        if above and histogram_expanding(df["macd_hist"], Direction.LONG, bars):
            notes["macd"] = "MACD above signal, histogram expanding positive"
            return IndicatorRead.BULL
        if not above and histogram_expanding(df["macd_hist"], Direction.SHORT, bars):
            notes["macd"] = "MACD below signal, histogram expanding negative"
            return IndicatorRead.BEAR
        return IndicatorRead.NEUTRAL

    def _rsi_trend(self, df: pd.DataFrame, notes: dict[str, str]) -> IndicatorRead:
        """RSI either side of 50, and moving that way."""
        mid = float(self.cfg.get("indicators.rsi_levels")["mid"])
        defs = self.cfg.section("entry")["definitions"]
        lookback = int(defs["rising_lookback"])
        min_delta = float(defs["rising_min_delta"])
        value = df["rsi"].iloc[-1]
        if pd.isna(value):
            return IndicatorRead.NEUTRAL

        if float(value) > mid and is_rising(df["rsi"], lookback, min_delta):
            notes["rsi"] = f"RSI {value:.0f} above {mid:.0f} and rising"
            return IndicatorRead.BULL
        if float(value) < mid and is_falling(df["rsi"], lookback, min_delta):
            notes["rsi"] = f"RSI {value:.0f} below {mid:.0f} and falling"
            return IndicatorRead.BEAR
        return IndicatorRead.NEUTRAL

    def _bb_trend(self, df: pd.DataFrame, atr_value: float,
                  notes: dict[str, str]) -> IndicatorRead:
        """Close the right side of the basis, and walking the matching band."""
        defs = self.cfg.section("entry")["definitions"]
        lookback = int(defs["band_walk_lookback"])
        min_closes = int(defs["band_walk_min_closes"])
        tolerance = float(defs["band_walk_tolerance_atr"])
        row = df.iloc[-1]
        if any(pd.isna(row.get(col)) for col in ("bb_mid", "bb_upper", "bb_lower")):
            return IndicatorRead.NEUTRAL

        close = float(row["close"])
        if close > float(row["bb_mid"]) and walking_band(
            df, Direction.LONG, atr_value, lookback, min_closes, tolerance
        ):
            notes["bb"] = "close above BB basis, hugging the upper band"
            return IndicatorRead.BULL
        if close < float(row["bb_mid"]) and walking_band(
            df, Direction.SHORT, atr_value, lookback, min_closes, tolerance
        ):
            notes["bb"] = "close below BB basis, hugging the lower band"
            return IndicatorRead.BEAR
        return IndicatorRead.NEUTRAL

    # -- reversal column ------------------------------------------------------

    def _adx_reversal(self, df: pd.DataFrame, notes: dict[str, str]) -> IndicatorRead:
        """Directional rule only - ADX need not be elevated at a range extreme."""
        row = df.iloc[-1]
        if any(pd.isna(row.get(col)) for col in ("plus_di", "minus_di")):
            return IndicatorRead.NEUTRAL
        plus, minus = float(row["plus_di"]), float(row["minus_di"])
        if plus > minus:
            notes["adx"] = f"+DI {plus:.0f} > -DI {minus:.0f} (ADX {row['adx']:.0f})"
            return IndicatorRead.BULL
        if minus > plus:
            notes["adx"] = f"-DI {minus:.0f} > +DI {plus:.0f} (ADX {row['adx']:.0f})"
            return IndicatorRead.BEAR
        return IndicatorRead.NEUTRAL

    def _stoch_reversal(self, df: pd.DataFrame, notes: dict[str, str]) -> IndicatorRead:
        """A fresh %K/%D cross out of the oversold or overbought extreme."""
        levels = self.cfg.section("indicators")["stoch_levels"]
        defs = self.cfg.section("entry")["definitions"]
        within = int(defs["fresh_crossover_bars"])

        up = crossed_above(df["stoch_k"], df["stoch_d"], within)
        if up is not None:
            k_at_cross = float(df["stoch_k"].iloc[-1 - up])
            if k_at_cross < float(levels["oversold"]):
                notes["stoch"] = f"fresh %K cross above %D from {k_at_cross:.0f} (oversold)"
                return IndicatorRead.BULL

        down = crossed_below(df["stoch_k"], df["stoch_d"], within)
        if down is not None:
            k_at_cross = float(df["stoch_k"].iloc[-1 - down])
            if k_at_cross > float(levels["overbought"]):
                notes["stoch"] = f"fresh %K cross below %D from {k_at_cross:.0f} (overbought)"
                return IndicatorRead.BEAR
        return IndicatorRead.NEUTRAL

    def _macd_reversal(self, df: pd.DataFrame, zone: Zone | None, atr_value: float,
                       notes: dict[str, str]) -> IndicatorRead:
        """A fresh MACD crossover occurring at or near the level under test."""
        defs = self.cfg.section("entry")["definitions"]
        within = int(defs["fresh_crossover_bars"])
        near_atr = float(defs["near_level_atr"])

        up = crossed_above(df["macd"], df["macd_signal"], within)
        if up is not None and near_level(df, up, zone, atr_value, near_atr):
            notes["macd"] = f"fresh bullish MACD cross {up} bar(s) ago, at the level"
            return IndicatorRead.BULL

        down = crossed_below(df["macd"], df["macd_signal"], within)
        if down is not None and near_level(df, down, zone, atr_value, near_atr):
            notes["macd"] = f"fresh bearish MACD cross {down} bar(s) ago, at the level"
            return IndicatorRead.BEAR
        return IndicatorRead.NEUTRAL

    def _rsi_reversal(self, df: pd.DataFrame, zone: Zone | None, atr_value: float,
                      notes: dict[str, str]) -> IndicatorRead:
        """Recovery across the 30/70 line, or a divergence at the level."""
        levels = self.cfg.section("indicators")["rsi_levels"]
        defs = self.cfg.section("entry")["definitions"]
        within = int(defs["fresh_crossover_bars"])
        lookback = int(defs["divergence_lookback"])
        fractal_n = int(self.cfg.get("levels.fractal_n"))

        oversold = float(levels["oversold"])
        overbought = float(levels["overbought"])
        series = df["rsi"]

        recovered_up = False
        recovered_down = False
        if len(series) > within + 1:
            recent = series.iloc[-(within + 1):]
            if not recent.isna().any():
                recovered_up = bool((recent.iloc[:-1] < oversold).any() and recent.iloc[-1] > oversold)
                recovered_down = bool((recent.iloc[:-1] > overbought).any() and recent.iloc[-1] < overbought)

        if recovered_up:
            notes["rsi"] = f"RSI recovered above {oversold:.0f}"
            return IndicatorRead.BULL
        if recovered_down:
            notes["rsi"] = f"RSI fell back below {overbought:.0f}"
            return IndicatorRead.BEAR

        if rsi_divergence(df, Direction.LONG, lookback, fractal_n):
            notes["rsi"] = "bullish RSI divergence at the level"
            return IndicatorRead.BULL
        if rsi_divergence(df, Direction.SHORT, lookback, fractal_n):
            notes["rsi"] = "bearish RSI divergence at the level"
            return IndicatorRead.BEAR
        return IndicatorRead.NEUTRAL

    def _bb_reversal(self, df: pd.DataFrame, notes: dict[str, str]) -> IndicatorRead:
        """A band tag or pierce with the close back inside."""
        if tags_band(df, Direction.LONG):
            notes["bb"] = "pierced the lower band, closed back inside"
            return IndicatorRead.BULL
        if tags_band(df, Direction.SHORT):
            notes["bb"] = "pierced the upper band, closed back inside"
            return IndicatorRead.BEAR
        return IndicatorRead.NEUTRAL

    # -- shared ---------------------------------------------------------------

    def _vwap_read(self, df: pd.DataFrame, atr_value: float,
                   notes: dict[str, str]) -> IndicatorRead:
        """VWAP reads the same in both modes.

        The soul file is explicit: VWAP is a level/reference, not a momentum
        oscillator, so its bullish/bearish read does not change by setup type.
        """
        defs = self.cfg.section("entry")["definitions"]
        lookback = int(defs["vwap_hold_lookback"])
        proximity = float(defs["vwap_hold_atr"])
        row = df.iloc[-1]
        if pd.isna(row.get("vwap")):
            return IndicatorRead.NEUTRAL

        close, vwap = float(row["close"]), float(row["vwap"])
        if close > vwap and vwap_holding(df, Direction.LONG, atr_value, lookback, proximity):
            notes["vwap"] = "above VWAP, pullbacks holding it as support"
            return IndicatorRead.BULL
        if close < vwap and vwap_holding(df, Direction.SHORT, atr_value, lookback, proximity):
            notes["vwap"] = "below VWAP, rallies rejected at VWAP"
            return IndicatorRead.BEAR
        return IndicatorRead.NEUTRAL
