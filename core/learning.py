"""Self-learning and performance tracking - soul file section 9.

The learning here is deliberately bounded, and the bound is the point:

    Setup types or conditions that consistently underperform can have their
    **confidence weighting reduced** - implemented as raising that setup's
    required confluence from 4 to 5 out of 6, never as changing any other rule.
    [...] These weightings never change the hard risk caps in Section 7.

So this module can do exactly one thing to Beast's behaviour: raise a setup's
confluence requirement by one, for one market, when its expectancy over a rolling
30-trade sample turns negative - and lower it back when expectancy recovers over
the following 30. Everything else it produces is reporting.

It cannot invent setups, change stops, alter position sizing, or touch any cap in
section 7. That constraint is enforced structurally: the only value this module
hands back to the pipeline is an integer confluence requirement, through
:meth:`PerformanceTracker.confluence_requirement`.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from statistics import mean
from typing import Any

from core.config import Config, get_config
from monitoring.journal import Journal

ROLLING_SAMPLE = 30


@dataclass
class SetupStats:
    """Performance of one setup+market pair.

    Attributes:
        trades: Number of closed trades in the sample.
        win_rate: Fraction of trades with a positive R.
        avg_r: Mean R-multiple - the expectancy.
        total_r: Sum of R-multiples.
        best_r / worst_r: Extremes in the sample.
    """

    setup_type: int
    market: str
    trades: int = 0
    win_rate: float = 0.0
    avg_r: float = 0.0
    total_r: float = 0.0
    best_r: float = 0.0
    worst_r: float = 0.0

    @property
    def negative_expectancy(self) -> bool:
        """True when the sample is full and expectancy is negative."""
        return self.trades >= ROLLING_SAMPLE and self.avg_r < 0


@dataclass
class OptionDiagnostics:
    """Options-specific breakdowns (soul file section 9).

    The ``premium_vs_underlying_gap`` is the instrument-selection diagnostic
    described in 6.9: if the analysis is sound but premium R lags badly, the fix
    is in strike selection (5.7), not in section 5.
    """

    by_dte: dict[str, dict[str, float]] = field(default_factory=dict)
    by_delta_bucket: dict[str, dict[str, float]] = field(default_factory=dict)
    by_oi_tag: dict[str, dict[str, float]] = field(default_factory=dict)
    expiry_day_vs_normal: dict[str, dict[str, float]] = field(default_factory=dict)
    premium_vs_underlying_gap: dict[str, float] = field(default_factory=dict)


class PerformanceTracker:
    """Reads the journal and produces section 9's outputs.

    Args:
        journal: The trade journal to read from.
        config: Injected for tests.
    """

    def __init__(self, journal: Journal, config: Config | None = None) -> None:
        self.cfg = config or get_config()
        self.journal = journal
        # Pairs currently held at the raised requirement, and the sample size at
        # which they were demoted - used to implement "reverts when expectancy
        # recovers over the following 30".
        self._demoted: dict[tuple[int, str], int] = {}

    # -- the one behavioural lever -------------------------------------------

    def confluence_requirement(self, setup_type: int, market: str) -> int | None:
        """Return a raised confluence requirement, or ``None`` for the default.

        Trigger: negative expectancy over a rolling 30-trade sample for that
        setup+market pair. Reverts when expectancy recovers over the following
        30. The only permitted values are ``None`` (default 4-of-6) and 5.
        """
        key = (int(setup_type), market)
        stats = self.setup_stats(setup_type, market, limit=ROLLING_SAMPLE)

        if key in self._demoted:
            if stats.trades >= ROLLING_SAMPLE and stats.avg_r >= 0:
                self._demoted.pop(key)
                return None
            return int(self.cfg.get("entry.min_confluence_counter_bias"))

        if stats.negative_expectancy:
            self._demoted[key] = stats.trades
            return int(self.cfg.get("entry.min_confluence_counter_bias"))
        return None

    def demoted_pairs(self) -> list[tuple[int, str]]:
        """Setup+market pairs currently running at the raised requirement."""
        return sorted(self._demoted.keys())

    # -- aggregate statistics -------------------------------------------------

    def setup_stats(self, setup_type: int, market: str,
                    limit: int = ROLLING_SAMPLE) -> SetupStats:
        """Rolling statistics for one setup+market pair."""
        rows = self.journal.recent_trades(market=market, setup_type=setup_type, limit=limit)
        return self._summarise(rows, setup_type, market)

    def all_setup_stats(self, markets: list[str],
                        limit: int = ROLLING_SAMPLE) -> list[SetupStats]:
        """Statistics for every setup+market pair, for the weekly report."""
        return [
            self.setup_stats(setup_type, market, limit)
            for market in markets
            for setup_type in (1, 2, 3, 4)
        ]

    def _summarise(self, rows: list[dict[str, Any]], setup_type: int,
                   market: str) -> SetupStats:
        """Fold trade rows into a :class:`SetupStats`."""
        values = [self._r_of_record(row) for row in rows]
        values = [value for value in values if value is not None]
        if not values:
            return SetupStats(setup_type, market)
        wins = sum(1 for value in values if value > 0)
        return SetupStats(
            setup_type=setup_type,
            market=market,
            trades=len(values),
            win_rate=round(wins / len(values), 4),
            avg_r=round(mean(values), 4),
            total_r=round(sum(values), 4),
            best_r=round(max(values), 4),
            worst_r=round(min(values), 4),
        )

    @staticmethod
    def _r_of_record(row: dict[str, Any]) -> float | None:
        """The R-multiple of record (6.9): premium-based for options."""
        for key in ("premium_r_multiple", "r_multiple", "underlying_r_multiple"):
            value = row.get(key)
            if value is not None:
                return float(value)
        return None

    # -- options diagnostics --------------------------------------------------

    def option_diagnostics(self, market: str, limit: int = 200) -> OptionDiagnostics:
        """Break option performance down by DTE, delta, OI tag and expiry day."""
        rows = self.journal.recent_trades(market=market, limit=limit)
        rows = [row for row in rows if row.get("delta_at_entry") is not None]
        diagnostics = OptionDiagnostics()
        if not rows:
            return diagnostics

        def bucket_dte(row: dict[str, Any]) -> str:
            dte = row.get("dte")
            if dte is None:
                return "unknown"
            return "0" if dte == 0 else ("1" if dte == 1 else "2+")

        def bucket_delta(row: dict[str, Any]) -> str:
            delta = float(row.get("delta_at_entry") or 0.0)
            if delta < 0.45:
                return "<0.45"
            if delta < 0.55:
                return "0.45-0.55"
            if delta <= 0.65:
                return "0.55-0.65"
            return ">0.65"

        diagnostics.by_dte = self._group(rows, bucket_dte)
        diagnostics.by_delta_bucket = self._group(rows, bucket_delta)
        diagnostics.by_oi_tag = self._group(rows, lambda row: str(row.get("oi_tag") or "UNKNOWN"))
        diagnostics.expiry_day_vs_normal = self._group(
            rows, lambda row: "expiry_day" if row.get("expiry_day") else "normal"
        )

        # The premium-R vs underlying-R gap, per setup type.
        gaps: dict[int, list[float]] = defaultdict(list)
        for row in rows:
            premium_r = row.get("premium_r_multiple")
            underlying_r = row.get("underlying_r_multiple")
            if premium_r is None or underlying_r is None:
                continue
            gaps[int(row["setup_type"])].append(float(underlying_r) - float(premium_r))
        diagnostics.premium_vs_underlying_gap = {
            f"setup_{setup}": round(mean(values), 4) for setup, values in gaps.items() if values
        }
        return diagnostics

    def _group(self, rows: list[dict[str, Any]], key_fn) -> dict[str, dict[str, float]]:
        """Group rows by ``key_fn`` and summarise win rate and average R."""
        buckets: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            value = self._r_of_record(row)
            if value is not None:
                buckets[key_fn(row)].append(value)
        return {
            name: {
                "trades": len(values),
                "win_rate": round(sum(1 for v in values if v > 0) / len(values), 4),
                "avg_r": round(mean(values), 4),
            }
            for name, values in buckets.items()
            if values
        }

    # -- weekly report --------------------------------------------------------

    def weekly_summary(self, markets: list[str], now: datetime | None = None,
                       days: int = 7) -> dict[str, Any]:
        """Assemble the section 9 weekly performance summary.

        Includes win rate, average R, best and worst setup type, override count,
        the gate histogram (so the operator can see whether Beast is missing
        trades at G5 or G7), and any alerts raised.
        """
        now = now or datetime.now()
        start = now - timedelta(days=days)
        rows = self.journal.trades_between(start, now)

        values = [value for value in (self._r_of_record(row) for row in rows) if value is not None]
        stats = self.all_setup_stats(markets, limit=ROLLING_SAMPLE)
        ranked = [item for item in stats if item.trades > 0]

        return {
            "window": {"from": start.isoformat(), "to": now.isoformat(), "days": days},
            "trades": len(values),
            "win_rate": round(sum(1 for v in values if v > 0) / len(values), 4) if values else 0.0,
            "avg_r": round(mean(values), 4) if values else 0.0,
            "total_r": round(sum(values), 4) if values else 0.0,
            "best_setup": (
                max(ranked, key=lambda item: item.avg_r).__dict__ if ranked else None
            ),
            "worst_setup": (
                min(ranked, key=lambda item: item.avg_r).__dict__ if ranked else None
            ),
            "override_count": self.journal.override_count(since=start),
            "gate_histogram": self.journal.gate_histogram(since=start),
            "demoted_pairs": [
                {"setup_type": setup, "market": market}
                for setup, market in self.demoted_pairs()
            ],
            "option_diagnostics": {
                market: self.option_diagnostics(market).__dict__
                for market in markets
                if self.cfg.market_family(market) == "indian"
            },
        }

    def format_weekly_summary(self, summary: dict[str, Any]) -> str:
        """Render the weekly summary in the section 11 tone: factual, no padding."""
        lines = [
            f"WEEKLY REVIEW - {summary['window']['days']} days to "
            f"{summary['window']['to'][:10]}",
            f"  trades {summary['trades']} | win rate {summary['win_rate']:.0%} | "
            f"avg R {summary['avg_r']:+.2f} | total R {summary['total_r']:+.2f}",
        ]
        if summary.get("best_setup"):
            best = summary["best_setup"]
            lines.append(
                f"  best: setup {best['setup_type']} {best['market']} "
                f"{best['avg_r']:+.2f}R over {best['trades']}"
            )
        if summary.get("worst_setup"):
            worst = summary["worst_setup"]
            lines.append(
                f"  worst: setup {worst['setup_type']} {worst['market']} "
                f"{worst['avg_r']:+.2f}R over {worst['trades']}"
            )
        lines.append(f"  overrides: {summary['override_count']}")
        if summary["gate_histogram"]:
            gates = ", ".join(
                f"{gate} {count}" for gate, count in summary["gate_histogram"].items()
            )
            lines.append(f"  rejections by gate: {gates}")
        if summary["demoted_pairs"]:
            pairs = ", ".join(
                f"setup {item['setup_type']} on {item['market']}"
                for item in summary["demoted_pairs"]
            )
            lines.append(f"  raised to 5-of-6 on negative expectancy: {pairs}")
        return "\n".join(lines)
