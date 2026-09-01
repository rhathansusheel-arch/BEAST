"""Instruction precedence - the Soul File wins.

Beast has one source of truth: ``beast/soul/BEAST_SOUL_v3.md``. Where any other
instruction conflicts with it - a config value, this code, a CLI flag, or the operator
speaking - the Soul File governs. This module makes that machine-readable:

* :func:`authority` ranks an instruction source.
* :func:`resolve` picks the winner between two conflicting sources and says why.
* :func:`soul_sha256` stamps every signal and trade with the exact brain revision that
  produced it, so behaviour is always traceable to a specific Soul File state.

The prose version, and the reasoning, is in PRECEDENCE.md.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SOUL_PATH = Path(__file__).resolve().parent.parent / "soul" / "BEAST_SOUL_v3.md"

#: Lower rank number = higher authority. Section 13 outranks the rest of the Soul File;
#: the Soul File outranks everything else, and the operator ranks last.
AUTHORITY: dict[str, int] = {
    "soul.immutable": 1,  # Section 13
    "soul": 2,  # every other Soul File section
    "config": 3,  # config/beast.yaml - may only set flagged defaults
    "code": 4,  # beast/*.py - implements 1-3; disagreement is a bug
    "operator": 5,  # chat, CLI flags, prompts - routes through Section 8
    "legacy": 6,  # the unrelated regime-trader scaffold
}


class PrecedenceError(Exception):
    """The Soul File is missing or unreadable - Beast has no brain to follow."""


@dataclass(frozen=True)
class Resolution:
    """The outcome of a conflict between two instruction sources."""

    winner: str
    loser: str
    reason: str

    def __str__(self) -> str:
        return f"{self.winner} governs over {self.loser}: {self.reason}"


def authority(source: str) -> int:
    """Rank an instruction source. Unknown sources are treated as operator-level."""
    return AUTHORITY.get(source, AUTHORITY["operator"])


def resolve(source_a: str, source_b: str) -> Resolution:
    """Return which of two conflicting instruction sources governs.

    Ties are impossible by construction - a source never conflicts with itself; if two
    instructions from the same source disagree, that is a bug in that source, and the
    caller is told so.
    """
    ra, rb = authority(source_a), authority(source_b)
    if ra == rb:
        return Resolution(
            winner=source_a,
            loser=source_b,
            reason=f"both are {source_a!r}; a source contradicting itself is a defect in it",
        )
    winner, loser = (source_a, source_b) if ra < rb else (source_b, source_a)
    if authority(winner) <= AUTHORITY["soul"]:
        reason = "the Soul File is the single source of truth (PRECEDENCE.md rank 1-2)"
    else:
        reason = f"{winner!r} outranks {loser!r} (PRECEDENCE.md)"
    return Resolution(winner=winner, loser=loser, reason=reason)


def soul_text() -> str:
    if not SOUL_PATH.exists():
        raise PrecedenceError(f"Soul File missing at {SOUL_PATH}. Beast will not trade blind.")
    return SOUL_PATH.read_text(encoding="utf-8")


def soul_sha256() -> str:
    """SHA-256 of the Soul File, stamped onto every signal and trade record."""
    return hashlib.sha256(soul_text().encode("utf-8")).hexdigest()


def soul_version() -> str:
    """The ``**Version:**`` line from the Soul File header, or ``unknown``."""
    for line in soul_text().splitlines():
        if line.startswith("**Version:**"):
            return line.split("**Version:**", 1)[1].strip()
    return "unknown"


def integrity_report(config) -> dict[str, Any]:
    """A single readable statement of what Beast is currently governed by.

    Surfaces the Soul File revision, the operating mode, and every Appendix A blocker that
    is still ``null`` - which is exactly the set of things preventing Beast from trading.
    """
    return {
        "soul_path": str(SOUL_PATH),
        "soul_version": soul_version(),
        "soul_sha256": soul_sha256(),
        "config_path": str(config.source) if config.source else None,
        "mode": config.mode,
        "unresolved_blockers": config.unresolved_blockers(),
        "precedence": [
            src for src, _ in sorted(AUTHORITY.items(), key=lambda kv: kv[1])
        ],
    }
