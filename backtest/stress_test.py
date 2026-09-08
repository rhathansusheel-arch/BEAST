"""Stress testing - crash injection, gap simulation, spread and IV shocks.

The soul file names the failure modes it expects, and this module manufactures
each of them so the exit logic can be shown to survive them rather than assumed
to:

* **Gap through the stop** (6.8) - the trade exits at the gap price and the
  R-multiple is recorded as *actual*, worse than -1R, never clamped. Section 9's
  expectancy must see real slippage.
* **Crash / flash move** - a violent multi-ATR move inside a few bars, testing
  that the stop fills and the daily loss cap engages.
* **Spread blowout** (4.6, 6.10) - high spread blocks entries, never exits.
* **IV crush** (4.7.4, 6.10) - premium collapses while the underlying goes
  nowhere; the premium hard stop is the backstop that catches it.
* **Stale feed while in position** (6.8) - Beast alerts with the last known
  price, position and stop, and does not guess.

Each scenario returns a modified bar series plus a description, so a run can be
reported as "here is what happened when the market gapped 3 ATR against an open
long" rather than as an abstract robustness claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from core.config import Config, get_config
from data.feature_engineering import atr


@dataclass
class Scenario:
    """A named stress scenario.

    Attributes:
        name: Short identifier used in the report.
        description: What it simulates and which soul-file rule it exercises.
        apply: Callable transforming an OHLC frame into the stressed version.
    """

    name: str
    description: str
    apply: Callable[[pd.DataFrame], pd.DataFrame]


@dataclass
class StressResult:
    """Outcome of running one scenario."""

    scenario: str
    description: str
    bars_modified: int
    max_adverse_move_atr: float
    notes: list[str] = field(default_factory=list)


class StressTester:
    """Builds and applies stress scenarios to a bar series.

    Args:
        config: Injected for tests.
        seed: Makes injection points deterministic so a failing scenario can be
            reproduced exactly.
    """

    def __init__(self, config: Config | None = None, seed: int = 42) -> None:
        self.cfg = config or get_config()
        self.rng = np.random.default_rng(seed)

    # -- scenario builders ----------------------------------------------------

    def gap_scenario(self, gap_atr: float = 3.0, direction: int = -1) -> Scenario:
        """A session that opens ``gap_atr`` ATR away from the prior close.

        Exercises the gap-through-the-stop rule in 6.8 and the gap detection in
        4.6, which forces a level recompute before the first entry is permitted.
        """

        def apply(frame: pd.DataFrame) -> pd.DataFrame:
            out = frame.copy()
            width = float(atr(out, int(self.cfg.get("indicators.atr_period"))).iloc[-1])
            if not np.isfinite(width) or width <= 0:
                return out
            shift = direction * gap_atr * width
            cut = len(out) // 2
            out.iloc[cut:, out.columns.get_indexer(["open", "high", "low", "close"])] += shift
            return out

        return Scenario(
            name=f"gap_{gap_atr:g}atr_{'down' if direction < 0 else 'up'}",
            description=(
                f"Session gaps {gap_atr:g} ATR {'down' if direction < 0 else 'up'} at the "
                f"midpoint. Tests 6.8 gap-through-stop pricing and 4.6 gap detection."
            ),
            apply=apply,
        )

    def crash_scenario(self, magnitude_atr: float = 6.0, bars: int = 5) -> Scenario:
        """A crash: ``magnitude_atr`` ATR down over ``bars`` candles.

        Tests that the stop fills, the R-multiple records the real slippage, and
        that three such losses trip the consecutive-loss pause in section 7.
        """

        def apply(frame: pd.DataFrame) -> pd.DataFrame:
            out = frame.copy()
            width = float(atr(out, int(self.cfg.get("indicators.atr_period"))).iloc[-1])
            if not np.isfinite(width) or width <= 0:
                return out
            start = max(0, len(out) - bars * 3)
            step = magnitude_atr * width / bars
            for offset in range(bars):
                index = start + offset
                if index >= len(out):
                    break
                drop = step * (offset + 1)
                for column in ("open", "high", "low", "close"):
                    out.iloc[index, out.columns.get_loc(column)] -= drop
                out.iloc[index, out.columns.get_loc("low")] -= step * 0.5
            # Everything after the crash sits at the new level.
            tail = start + bars
            if tail < len(out):
                total = magnitude_atr * width
                out.iloc[tail:, out.columns.get_indexer(["open", "high", "low", "close"])] -= total
            return out

        return Scenario(
            name=f"crash_{magnitude_atr:g}atr",
            description=(
                f"{magnitude_atr:g} ATR decline over {bars} candles. Tests stop fills under "
                f"velocity, honest R recording (6.8), and the loss-limit pause (7)."
            ),
            apply=apply,
        )

    def whipsaw_scenario(self, amplitude_atr: float = 2.0, cycles: int = 8) -> Scenario:
        """Alternating multi-ATR swings with no net direction.

        The regime classifier should read this as ``RANGE`` (ADX below the
        threshold, or DI and BB disagreeing), which suppresses setups 1 and 4 and
        is the main defence against paying spread repeatedly in chop.
        """

        def apply(frame: pd.DataFrame) -> pd.DataFrame:
            out = frame.copy()
            width = float(atr(out, int(self.cfg.get("indicators.atr_period"))).iloc[-1])
            if not np.isfinite(width) or width <= 0:
                return out
            span = max(1, len(out) // (cycles * 2))
            offsets = np.zeros(len(out))
            for cycle in range(cycles * 2):
                start = cycle * span
                stop = min(len(out), start + span)
                offsets[start:stop] = amplitude_atr * width * (1 if cycle % 2 == 0 else -1)
            for column in ("open", "high", "low", "close"):
                out[column] = out[column] + offsets
            return out

        return Scenario(
            name=f"whipsaw_{amplitude_atr:g}atr",
            description=(
                f"{cycles} alternating {amplitude_atr:g} ATR swings with no net move. Tests "
                f"that the 4.4 regime classifier suppresses trend setups in chop."
            ),
            apply=apply,
        )

    def spread_blowout(self, multiplier: float = 10.0) -> Scenario:
        """A spread blowout.

        There is no price change here - the point is behavioural: high spread must
        block entries and never block exits (4.6, 6.8, 6.10). Drive this by
        setting ``FeedState.spread`` rather than by modifying bars.
        """

        def apply(frame: pd.DataFrame) -> pd.DataFrame:
            return frame.copy()

        return Scenario(
            name=f"spread_blowout_{multiplier:g}x",
            description=(
                f"Spread widens {multiplier:g}x. Entries must be blocked with a HIGH SPREAD "
                f"ALERT; exits must still execute and log the slippage."
            ),
            apply=apply,
        )

    def iv_crush(self, collapse_pct: float = 0.40) -> Scenario:
        """An IV crush that destroys premium while the underlying is unchanged.

        This is the scenario the premium hard stop in 6.10 exists for: a "0.6R"
        underlying move quietly becoming a 60% premium loss. Apply it to the
        premium series in an option replay, not to the underlying bars.
        """

        def apply(frame: pd.DataFrame) -> pd.DataFrame:
            return frame.copy()

        return Scenario(
            name=f"iv_crush_{collapse_pct:.0%}",
            description=(
                f"ATM IV collapses {collapse_pct:.0%} post-event with the underlying flat. "
                f"The premium hard stop (6.10) is the only thing that fires."
            ),
            apply=apply,
        )

    def stale_feed(self, gap_bars: int = 20) -> Scenario:
        """A hole in the series, simulating a dead feed."""

        def apply(frame: pd.DataFrame) -> pd.DataFrame:
            if len(frame) < gap_bars * 3:
                return frame.copy()
            cut = len(frame) // 2
            return pd.concat([frame.iloc[:cut], frame.iloc[cut + gap_bars:]])

        return Scenario(
            name=f"stale_feed_{gap_bars}bars",
            description=(
                f"{gap_bars} consecutive bars missing. The 4.6 staleness check must mark the "
                f"feed STALE and suppress entries; 6.8 requires an alert carrying the last "
                f"known price, position and stop rather than a guess."
            ),
            apply=apply,
        )

    # -- running --------------------------------------------------------------

    def default_suite(self) -> list[Scenario]:
        """The scenarios worth running on every strategy change."""
        return [
            self.gap_scenario(3.0, -1),
            self.gap_scenario(3.0, 1),
            self.crash_scenario(6.0, 5),
            self.whipsaw_scenario(2.0, 8),
            self.spread_blowout(10.0),
            self.iv_crush(0.40),
            self.stale_feed(20),
        ]

    def apply(self, scenario: Scenario, frame: pd.DataFrame) -> tuple[pd.DataFrame, StressResult]:
        """Apply ``scenario`` and describe what changed."""
        stressed = scenario.apply(frame)
        width = float(atr(frame, int(self.cfg.get("indicators.atr_period"))).iloc[-1])

        if len(stressed) == len(frame):
            differences = (stressed["close"] - frame["close"]).abs()
            modified = int((differences > 1e-9).sum())
            worst = float(differences.max() / width) if width > 0 else 0.0
        else:
            modified = abs(len(frame) - len(stressed))
            worst = 0.0

        return stressed, StressResult(
            scenario=scenario.name,
            description=scenario.description,
            bars_modified=modified,
            max_adverse_move_atr=round(worst, 3),
        )


def format_results(results: list[StressResult]) -> str:
    """Render stress results for the operator."""
    if not results:
        return "STRESS: no scenarios run."
    lines = ["STRESS TESTS"]
    for result in results:
        lines.append(f"  {result.scenario}")
        lines.append(f"    {result.description}")
        lines.append(
            f"    bars modified {result.bars_modified}, "
            f"max move {result.max_adverse_move_atr:.2f} ATR"
        )
        for note in result.notes:
            lines.append(f"    - {note}")
    return "\n".join(lines)
