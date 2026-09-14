"""``heartbeat.json`` - one line of truth per loop cycle.

Written atomically (``.tmp`` then ``os.replace``) so the watchdog never reads a
half-written file. Written unconditionally at the end of every cycle,
including one that raised, so a heartbeat carrying ``last_error`` is
distinguishable from no heartbeat at all - the first means Beast is alive and
struggling, the second means it is gone.

``clean_shutdown: true`` on the final heartbeat is the watchdog's signal that
the stop was intentional and must not be undone.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class Heartbeat:
    written_at: str
    pid: int
    mode: str
    markets: list[str]
    loop_cycle_count: int
    open_positions: list[dict[str, Any]] = field(default_factory=list)
    feed_ok: dict[str, bool] = field(default_factory=dict)
    entry_pause_reason: dict[str, str] = field(default_factory=dict)
    last_error: str | None = None
    session_day: str | None = None
    capital: float = 0.0
    realised_pnl_today: float = 0.0
    clean_shutdown: bool = False
    mt5_state: str | None = None

    @property
    def age_seconds(self) -> float:
        try:
            return (datetime.now() - datetime.fromisoformat(self.written_at)).total_seconds()
        except (TypeError, ValueError):
            return float("inf")


def write_heartbeat(path: str | Path, beat: Heartbeat) -> Path:
    """Atomic write. Never raises into the trading loop."""
    target = Path(path)
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(asdict(beat), handle, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
    return target


def read_heartbeat(path: str | Path) -> Heartbeat | None:
    """The last heartbeat, or None when absent or unreadable."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    known = {k: v for k, v in data.items() if k in Heartbeat.__dataclass_fields__}
    try:
        return Heartbeat(**known)
    except TypeError:
        return None
