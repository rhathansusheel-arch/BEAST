"""Section 9 - self-learning and performance tracking, bounded exactly as specified.

What learning may do: raise an underperforming setup's required confluence from 4 to 5 out
of 6. That is the whole lever.

What learning may **not** do: change any other rule, touch the hard risk caps in Section 7,
or invent new setup types. "Learning is confined to *how strictly* it applies the setups
already defined in Section 5, not *what* setups exist."

Trigger: negative expectancy over a rolling 30-trade sample for that setup+market pair.
It reverts when expectancy recovers over the following 30.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from beast.constants import Gate, Market


@dataclass
class SetupStats:
    trades: int = 0
    wins: int = 0
    total_r: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades else 0.0

    @property
    def expectancy(self) -> float:
        return self.total_r / self.trades if self.trades else 0.0


class Learning:
    """Rolling per-setup, per-market performance, and the one weighting it controls."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.sample = int(cfg.get("learning.sample_size"))
        self._r_history: dict[tuple[str, int], list[float]] = defaultdict(list)
        self._tightened: set[tuple[str, int]] = set()
        self._gate_counts: dict[str, int] = defaultdict(int)
        self._option_buckets: dict[str, list[float]] = defaultdict(list)

    # -- ingestion -------------------------------------------------------------

    def record_trade(self, record) -> None:
        """Ingest a closed trade (Appendix C) and re-evaluate the confluence weighting."""
        key = (record.market.value, record.setup_type)
        r = record.r_multiple if record.r_multiple is not None else 0.0
        history = self._r_history[key]
        history.append(r)

        if record.market.is_option_market:
            leg = record.signal.leg.get("_option_only", {})
            self._option_buckets[f"dte:{record.dte}"].append(r)
            self._option_buckets[f"delta:{_delta_bucket(leg.get('delta'))}"].append(r)
            tag = record.signal.chain_context.oi_tag if record.signal.chain_context else None
            if tag:
                self._option_buckets[f"oi_tag:{tag}"].append(r)
            if record.underlying_r_multiple is not None and record.premium_r_multiple is not None:
                self._option_buckets["premium_vs_underlying_gap"].append(
                    record.underlying_r_multiple - record.premium_r_multiple
                )

        if not self.cfg.get("learning.enabled"):
            return
        if len(history) < self.sample:
            return
        window = history[-self.sample :]
        expectancy = sum(window) / len(window)
        if expectancy < 0:
            self._tightened.add(key)
        elif key in self._tightened:
            # Reverts when expectancy recovers over the following 30 (Section 9).
            self._tightened.discard(key)

    def record_rejection(self, rejection) -> None:
        """Gate analysis (Section 9) - where signals die, not only what got taken."""
        self._gate_counts[rejection.failed_gate.value] += 1

    # -- queries ---------------------------------------------------------------

    def is_tightened(self, market: Market, setup_type: int) -> bool:
        """Does this setup+market currently require 5 of 6 instead of 4?"""
        return (market.value, setup_type) in self._tightened

    def expectancy(self, market: Market, setup_type: int) -> Optional[float]:
        history = self._r_history.get((market.value, setup_type))
        if not history:
            return None
        window = history[-self.sample :]
        return sum(window) / len(window)

    def stats(self, market: Market, setup_type: int) -> SetupStats:
        history = self._r_history.get((market.value, setup_type), [])
        return SetupStats(
            trades=len(history),
            wins=sum(1 for r in history if r > 0),
            total_r=sum(history),
        )

    def weekly_summary(self, override_count: int = 0, alerts: Optional[list[str]] = None) -> dict:
        """Section 9's weekly summary: win rate, average R, best/worst setup, overrides."""
        per_setup = {
            f"{market}:{setup}": {
                "trades": len(history),
                "win_rate": sum(1 for r in history if r > 0) / len(history) if history else 0.0,
                "avg_r": sum(history) / len(history) if history else 0.0,
                "requires_5_of_6": (market, setup) in self._tightened,
            }
            for (market, setup), history in self._r_history.items()
        }
        ranked = sorted(per_setup.items(), key=lambda kv: kv[1]["avg_r"], reverse=True)
        all_r = [r for history in self._r_history.values() for r in history]
        return {
            "trades": len(all_r),
            "win_rate": sum(1 for r in all_r if r > 0) / len(all_r) if all_r else 0.0,
            "avg_r": sum(all_r) / len(all_r) if all_r else 0.0,
            "best_setup": ranked[0][0] if ranked else None,
            "worst_setup": ranked[-1][0] if ranked else None,
            "per_setup": per_setup,
            "gate_rejections": dict(self._gate_counts),
            "override_count": override_count,
            "alerts": alerts or [],
            "option_buckets": {
                k: {"n": len(v), "avg_r": sum(v) / len(v)} for k, v in self._option_buckets.items() if v
            },
        }

    def gate_report(self) -> dict[str, int]:
        """Where candidates died, by gate ID (5.1, Section 9)."""
        return {g.value: self._gate_counts.get(g.value, 0) for g in Gate}


def _delta_bucket(delta: Optional[float]) -> str:
    if delta is None:
        return "unknown"
    return f"{int(delta * 10) / 10:.1f}"
