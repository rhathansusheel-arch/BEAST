"""Performance measurement - Sharpe, drawdown, regime breakdown, benchmarks.

Everything is computed in **R-multiples** rather than currency. That is not a
stylistic choice: the soul file expresses every exit rule in R (6.1-6.3), risk
per trade differs by market (3% Indian, 2% Gold), and the R-multiple of record
for an option is premium-based (6.9). Mixing currencies across markets would make
the aggregate meaningless; R is the common unit the whole document is written in.

Two numbers deserve particular attention when reading a report:

* **Expectancy** (average R). This is what section 9's learning loop triggers on -
  negative expectancy over a rolling 30-trade sample for a setup+market pair.
* **The premium-vs-underlying R gap** for options. If underlying R is consistently
  better than premium R, strike selection (5.7) is leaking edge, not the analysis.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Any, Iterable, Sequence

import pandas as pd

from core.config import Config, get_config
from core.schemas import ExitReason, TradeRecord


@dataclass
class PerformanceReport:
    """Aggregate performance over a set of trades."""

    trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    expectancy_r: float = 0.0
    total_r: float = 0.0
    profit_factor: float = 0.0
    sharpe: float = 0.0
    max_drawdown_r: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_win_r: float = 0.0
    avg_loss_r: float = 0.0
    largest_win_r: float = 0.0
    largest_loss_r: float = 0.0
    avg_bars_held: float = 0.0
    trail_activation_rate: float = 0.0
    by_setup: dict[str, dict[str, float]] = field(default_factory=dict)
    by_regime: dict[str, dict[str, float]] = field(default_factory=dict)
    by_exit_reason: dict[str, int] = field(default_factory=dict)
    by_direction: dict[str, dict[str, float]] = field(default_factory=dict)
    premium_vs_underlying: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def r_of_record(trade: TradeRecord) -> float:
    """The R-multiple of record: premium-based for options (soul file 6.9)."""
    if trade.premium_r_multiple is not None:
        return float(trade.premium_r_multiple)
    if trade.r_multiple is not None:
        return float(trade.r_multiple)
    return float(trade.underlying_r_multiple or 0.0)


def analyse(trades: Sequence[TradeRecord],
            config: Config | None = None) -> PerformanceReport:
    """Compute the full report for ``trades``."""
    cfg = config or get_config()
    report = PerformanceReport()
    closed = [trade for trade in trades if trade.exit_time is not None]
    if not closed:
        return report

    values = [r_of_record(trade) for trade in closed]
    wins = [value for value in values if value > 0]
    losses = [value for value in values if value <= 0]

    report.trades = len(values)
    report.wins = len(wins)
    report.losses = len(losses)
    report.win_rate = round(len(wins) / len(values), 4)
    report.expectancy_r = round(mean(values), 4)
    report.total_r = round(sum(values), 4)
    report.avg_win_r = round(mean(wins), 4) if wins else 0.0
    report.avg_loss_r = round(mean(losses), 4) if losses else 0.0
    report.largest_win_r = round(max(values), 4)
    report.largest_loss_r = round(min(values), 4)
    report.avg_bars_held = round(mean([trade.bars_held for trade in closed]), 2)
    report.trail_activation_rate = round(
        sum(1 for trade in closed if trade.trail_activated) / len(closed), 4
    )

    gross_loss = abs(sum(losses))
    report.profit_factor = (
        round(sum(wins) / gross_loss, 4) if gross_loss > 0 else float("inf") if wins else 0.0
    )
    report.sharpe = sharpe_ratio(values, cfg)

    curve = pd.Series(values).cumsum()
    drawdown = curve - curve.cummax()
    report.max_drawdown_r = round(float(drawdown.min()), 4) if len(drawdown) else 0.0
    peak = float(curve.cummax().max()) if len(curve) else 0.0
    report.max_drawdown_pct = (
        round(abs(report.max_drawdown_r) / peak, 4) if peak > 0 else 0.0
    )

    report.by_setup = _group(closed, lambda t: f"setup_{int(t.signal.setup_type)}")
    report.by_regime = _group(closed, lambda t: t.signal.regime.value)
    report.by_direction = _group(closed, lambda t: t.signal.direction.value)
    report.by_exit_reason = _count(closed, lambda t: (t.exit_reason or ExitReason.SESSION).value)
    report.premium_vs_underlying = _premium_gap(closed)
    return report


def sharpe_ratio(values: Sequence[float], config: Config | None = None) -> float:
    """Annualised Sharpe of the per-trade R series.

    Per-trade R is already a risk-normalised return, so the risk-free rate is
    subtracted per trade after scaling it by the observed trade frequency rather
    than being applied as an annual figure to a per-trade series.
    """
    cfg = config or get_config()
    if len(values) < 2:
        return 0.0
    periods = float(cfg.get("backtest.bars_per_year"))
    risk_free = float(cfg.get("backtest.risk_free_rate")) / periods

    excess = [value - risk_free for value in values]
    deviation = pstdev(excess)
    if deviation == 0:
        return 0.0
    return round(mean(excess) / deviation * math.sqrt(periods), 4)


def _group(trades: Iterable[TradeRecord], key_fn) -> dict[str, dict[str, float]]:
    """Group trades and summarise each bucket."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for trade in trades:
        buckets[key_fn(trade)].append(r_of_record(trade))
    return {
        name: {
            "trades": len(values),
            "win_rate": round(sum(1 for v in values if v > 0) / len(values), 4),
            "expectancy_r": round(mean(values), 4),
            "total_r": round(sum(values), 4),
        }
        for name, values in sorted(buckets.items())
        if values
    }


def _count(trades: Iterable[TradeRecord], key_fn) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for trade in trades:
        counts[key_fn(trade)] += 1
    return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))


def _premium_gap(trades: Iterable[TradeRecord]) -> dict[str, float]:
    """Mean ``underlying_r - premium_r`` per setup type (soul file 6.9).

    A consistently positive gap means the analysis was right and the instrument
    gave the edge back - that is a 5.7 strike-selection problem, not a section 5
    problem, and the report should say so rather than leaving it to be inferred.
    """
    gaps: dict[str, list[float]] = defaultdict(list)
    for trade in trades:
        if trade.premium_r_multiple is None or trade.underlying_r_multiple is None:
            continue
        gaps[f"setup_{int(trade.signal.setup_type)}"].append(
            float(trade.underlying_r_multiple) - float(trade.premium_r_multiple)
        )
    return {name: round(mean(values), 4) for name, values in sorted(gaps.items()) if values}


def buy_and_hold_benchmark(bars: pd.DataFrame, risk_points: float) -> float:
    """Buy-and-hold return over the same window, expressed in R.

    Args:
        bars: The underlying series the strategy traded.
        risk_points: A representative 1R distance, normally the median stop
            distance from the run being compared.

    Comparing an intraday strategy to buy-and-hold is only fair with that caveat
    attached, which is why this returns a single number the report labels rather
    than something dressed up as an alpha.
    """
    if bars.empty or risk_points <= 0:
        return 0.0
    move = float(bars["close"].iloc[-1]) - float(bars["close"].iloc[0])
    return round(move / risk_points, 4)


def format_report(report: PerformanceReport, title: str = "PERFORMANCE") -> str:
    """Render a report in the section 11 tone: factual, dense, no padding."""
    if report.trades == 0:
        return f"{title}: no closed trades."

    lines = [
        f"{title}",
        f"  trades {report.trades} | win rate {report.win_rate:.0%} | "
        f"expectancy {report.expectancy_r:+.3f}R | total {report.total_r:+.2f}R",
        f"  profit factor {report.profit_factor:.2f} | Sharpe {report.sharpe:.2f} | "
        f"max drawdown {report.max_drawdown_r:.2f}R",
        f"  avg win {report.avg_win_r:+.2f}R | avg loss {report.avg_loss_r:+.2f}R | "
        f"trail armed on {report.trail_activation_rate:.0%} | "
        f"avg {report.avg_bars_held:.0f} bars held",
    ]

    if report.by_setup:
        lines.append("  by setup:")
        for name, stats in report.by_setup.items():
            lines.append(
                f"    {name:<9} {stats['trades']:>4} trades  "
                f"win {stats['win_rate']:.0%}  exp {stats['expectancy_r']:+.3f}R"
            )
    if report.by_regime:
        lines.append("  by regime:")
        for name, stats in report.by_regime.items():
            lines.append(
                f"    {name:<12} {stats['trades']:>4} trades  "
                f"win {stats['win_rate']:.0%}  exp {stats['expectancy_r']:+.3f}R"
            )
    if report.by_exit_reason:
        reasons = ", ".join(f"{name} {count}" for name, count in report.by_exit_reason.items())
        lines.append(f"  exits: {reasons}")
    if report.premium_vs_underlying:
        gaps = ", ".join(
            f"{name} {value:+.3f}R" for name, value in report.premium_vs_underlying.items()
        )
        lines.append(f"  underlying-R minus premium-R: {gaps}")
        worst = max(report.premium_vs_underlying.values())
        if worst > 0.25:
            lines.append(
                "  note: underlying R is materially ahead of premium R. The edge is "
                "leaking in strike selection (5.7), not in the analysis."
            )
    return "\n".join(lines)
