"""Appendix B / C persistence.

Three append-only JSONL journals, because Section 9's learning loop and Section 11's
reporting both read from them and neither may quietly lose a row:

* ``signals.jsonl``     - every emitted Signal (Appendix B)
* ``trades.jsonl``      - every closed trade (Appendix C)
* ``rejections.jsonl``  - every candidate killed at a gate, with its gate ID (5.1)
* ``overrides.jsonl``   - every Section 8 override attempt, confirmed or refused

Paper mode writes exactly what live mode writes (Section 10) so the data is comparable
later.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


class Journal:
    """An append-only JSONL file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def read(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return iter(())
        with open(self.path, "r", encoding="utf-8") as fh:
            return iter([json.loads(line) for line in fh if line.strip()])

    def count(self) -> int:
        return sum(1 for _ in self.read())


class Store:
    """The four journals, wired to the Appendix A ``paths`` block."""

    def __init__(self, cfg) -> None:
        base = Path(cfg.source).parent.parent if cfg.source else Path(".")
        self.signals = Journal(base / str(cfg.get("paths.signals")))
        self.trades = Journal(base / str(cfg.get("paths.trades")))
        self.rejections = Journal(base / str(cfg.get("paths.rejections")))
        self.overrides = Journal(base / str(cfg.get("paths.overrides")))

    def record_signal(self, signal) -> None:
        self.signals.append(signal.to_dict())

    def record_trade(self, record) -> None:
        self.trades.append(record.to_dict())

    def record_rejection(self, rejection) -> None:
        self.rejections.append(rejection.to_dict())

    def record_override(self, record) -> None:
        self.overrides.append(record.to_dict())
