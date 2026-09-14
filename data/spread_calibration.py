"""Auto-calibration of ``data.gold_spread_max`` from the live book (D-82).

The gold spread ceiling is a risk tolerance, not a fact any API can supply, and
it is broker-specific: one venue quotes XAUUSD at 0.12, another at 0.45, and a
number typed in without looking either blocks every entry or blocks none. So
when the operator leaves ``data.gold_spread_max`` unset and
``data.gold_spread_auto_calibrate`` on, Beast samples the spread for a short
burn-in after the bridge comes up, takes the 90th percentile of what it saw,
multiplies by a safety factor and uses *that* as the ceiling - logging exactly
what it chose and why - until the operator replaces it with a number of their
own. A value set in the config always wins; calibration never overwrites one.

Sampling is incremental: a burst of a few quotes per loop cycle, spread over
``data.gold_spread_calibration_seconds`` of wall time, so the exit path is
never blocked for the length of the window. Until the window closes the
ceiling stays ``null`` and the existing rule applies - no gold entries.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger("beast.data.spread")


@dataclass
class SpreadCalibrator:
    """Collects live spread samples and decides a ceiling when the window closes.

    Args:
        window_seconds: Wall-clock length of the burn-in.
        multiplier: Safety factor applied to the p90 (2-3x is the brief's range).
        burst: Quotes taken per :meth:`sample` call.
        burst_gap_seconds: Pause between the quotes of one burst.
        min_samples: Fewer than this at the end of the window extends it.
        clock: Monotonic clock, injected by tests.
        sleep: Injected by tests so bursts do not really wait.
    """

    window_seconds: float = 60.0
    multiplier: float = 2.5
    burst: int = 5
    burst_gap_seconds: float = 0.2
    min_samples: int = 20
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    samples: list[float] = field(default_factory=list)
    started_at: float | None = None
    result: float | None = None
    p90: float | None = None

    @property
    def done(self) -> bool:
        return self.result is not None

    def sample(self, quote_fn: Callable[[], object | None]) -> float | None:
        """Take one burst. Returns the ceiling once decided, else ``None``.

        ``quote_fn`` returns an object with a ``spread`` attribute (a
        ``broker.Quote``) or ``None``. Non-finite and non-positive spreads -
        a one-sided book, a closed market - are skipped, not recorded as zero.
        """
        if self.done:
            return self.result
        now = self.clock()
        if self.started_at is None:
            self.started_at = now
            logger.warning(
                "gold spread ceiling is unset; auto-calibrating from the live book for "
                "%.0fs (p90 x %.1f). No gold entries until it is decided.",
                self.window_seconds, self.multiplier,
            )
        for index in range(max(1, self.burst)):
            quote = quote_fn()
            spread = getattr(quote, "spread", None) if quote is not None else None
            if spread is not None and math.isfinite(spread) and spread > 0:
                self.samples.append(float(spread))
            if index + 1 < self.burst and self.burst_gap_seconds > 0:
                self.sleep(self.burst_gap_seconds)
        elapsed = self.clock() - self.started_at
        if elapsed < self.window_seconds or len(self.samples) < self.min_samples:
            return None
        return self._decide(elapsed)

    def _decide(self, elapsed: float) -> float:
        ordered = sorted(self.samples)
        rank = min(len(ordered) - 1, int(math.ceil(0.9 * len(ordered))) - 1)
        self.p90 = ordered[max(0, rank)]
        self.result = round(self.p90 * self.multiplier, 6)
        logger.warning(
            "gold spread ceiling AUTO-CALIBRATED to %.4f: p90 %.4f of %d samples over %.0fs "
            "(min %.4f, max %.4f) x %.1f safety multiplier. Override with data.gold_spread_max "
            "once these numbers have been reviewed.",
            self.result, self.p90, len(ordered), elapsed, ordered[0], ordered[-1],
            self.multiplier,
        )
        return self.result

    def describe(self) -> str:
        if not self.done:
            return f"AUTO-CALIBRATING ({len(self.samples)} samples so far)"
        return (f"AUTO-CALIBRATED {self.result:.4f} = p90 {self.p90:.4f} x {self.multiplier:.1f} "
                f"over {len(self.samples)} samples")
