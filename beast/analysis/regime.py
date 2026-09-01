"""Section 4.4 - the regime classifier, evaluated on the bias timeframe.

On each bias-TF close the market is exactly one of ``TREND_UP``, ``TREND_DOWN`` or
``RANGE``. The regime decides which setups are permitted in which direction; it never
generates an entry itself (4.3: "Bias / regime ... never generates entries").
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from beast.constants import Direction, REGIME_PERMITS, Regime


@dataclass(frozen=True)
class RegimeState:
    regime: Regime
    adx: float
    plus_di: float
    minus_di: float
    close: float
    bb_mid: float

    def describe(self) -> str:
        return (
            f"{self.regime.value} (ADX {self.adx:.1f}, +DI {self.plus_di:.1f}, "
            f"-DI {self.minus_di:.1f}, close {self.close:.2f} vs BB mid {self.bb_mid:.2f})"
        )


def classify(bias_indicators: pd.DataFrame, cfg) -> RegimeState:
    """Classify the regime from the last closed bias-TF candle (4.4).

    ``TREND_UP``   - ADX >= threshold **and** +DI > -DI **and** close > BB middle band
    ``TREND_DOWN`` - ADX >= threshold **and** -DI > +DI **and** close < BB middle band
    ``RANGE``      - ADX below threshold, or the DI/BB conditions disagree
    """
    if bias_indicators.empty:
        # No bias-TF history yet. RANGE is the honest answer - it is the bucket 4.4 uses
        # when the trend conditions are not met - and it suppresses Setups 1 and 4. The
        # data integrity gate (4.6) separately blocks entries while history is missing.
        nan = float("nan")
        return RegimeState(Regime.RANGE, nan, nan, nan, nan, nan)

    row = bias_indicators.iloc[-1]
    threshold = float(cfg.get("indicators.adx_trend_threshold"))
    adx = float(row["adx"])
    plus_di, minus_di = float(row["plus_di"]), float(row["minus_di"])
    close, bb_mid = float(row["close"]), float(row["bb_mid"])

    if adx >= threshold and plus_di > minus_di and close > bb_mid:
        regime = Regime.TREND_UP
    elif adx >= threshold and minus_di > plus_di and close < bb_mid:
        regime = Regime.TREND_DOWN
    else:
        regime = Regime.RANGE
    return RegimeState(regime, adx, plus_di, minus_di, close, bb_mid)


def permitted_setups(regime: Regime, direction: Direction, cfg) -> tuple[int, ...]:
    """Which setup types may be taken in ``direction`` under ``regime`` (4.4 table).

    In ``RANGE`` the permitted set is config-driven (``entry.range_regime_allowed_setups``,
    default Setups 2 and 3 - Setups 1 and 4 are suppressed).
    """
    if regime is Regime.RANGE:
        return tuple(int(s) for s in cfg.get("entry.range_regime_allowed_setups"))
    return REGIME_PERMITS[regime].get(direction, ())


def is_counter_bias(regime: Regime, direction: Direction) -> bool:
    """Is this direction against the prevailing bias-TF regime? (4.4)

    Counter-bias trades are Setup 2 only, need 5-of-6 confluence instead of 4, and must be
    reversing at a Tier A level. In ``RANGE`` there is no prevailing bias to fight.
    """
    if regime is Regime.TREND_UP:
        return direction is Direction.SHORT
    if regime is Regime.TREND_DOWN:
        return direction is Direction.LONG
    return False


def required_confluence(counter_bias: bool, cfg, learning_tightened: bool = False) -> int:
    """The 4-of-6 confluence bar, raised to 5 where the Soul File says so.

    Two - and only two - things raise it: a counter-bias reversal (4.4) and Section 9's
    learning loop reducing confidence in an underperforming setup. Section 9 is explicit
    that this is the *only* lever learning may pull.
    """
    base = int(cfg.get("entry.min_confluence"))
    if counter_bias:
        base = max(base, int(cfg.get("entry.min_confluence_counter_bias")))
    if learning_tightened:
        base = max(base, int(cfg.get("learning.tighten_to_confluence")))
    return base
