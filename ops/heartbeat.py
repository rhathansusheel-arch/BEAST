"""``heartbeat.json`` - Beast's live state, one file, written every cycle.

Two readers, one writer, one cadence (D-72):

* the watchdog reads ``written_at``, ``clean_shutdown``, ``positions`` and
  ``system.last_error`` to decide whether Beast is alive;
* the dashboard reads everything else to show the session.

It is **display and supervision only**. The trading loop never reads it. If
it is corrupt, missing or a week old, Beast trades exactly the same - which
is what keeps a cosmetic file from becoming safety-critical. Recovery state
lives in ``state_snapshot.json`` and keeps its own meaning.

Written atomically: a ``.tmp`` in the **same directory** (a temp file on
another mount makes ``os.replace`` a copy, not a rename), fsync, replace.
``allow_nan=False`` so a NaN in the payload fails at write time instead of
shipping a file no non-Python reader can parse.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import sys
import time
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2


class BeastJSONEncoder(json.JSONEncoder):
    """numpy scalars, pandas/datetime with tzinfo, Enum, Decimal, dataclasses.

    NaN and infinity raise: a NaN reaching the dashboard is a bug upstream,
    and ``json.dump(allow_nan=False)`` only checks floats it sees directly,
    not numpy ones, so the check is repeated here for those.
    """

    def default(self, o: Any) -> Any:  # noqa: D401 - json API name
        try:
            import numpy as np
        except ImportError:  # pragma: no cover
            np = None
        if np is not None:
            if isinstance(o, np.bool_):
                return bool(o)
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                value = float(o)
                if math.isnan(value) or math.isinf(value):
                    raise ValueError(f"non-finite float in live state: {o!r}")
                return value
            if isinstance(o, np.ndarray):
                return o.tolist()
        try:
            import pandas as pd
            if isinstance(o, pd.Timestamp):
                if o.tzinfo is None:
                    o = o.tz_localize(timezone.utc)
                return o.isoformat()
            if o is pd.NaT:
                return None
        except ImportError:  # pragma: no cover
            pass
        if isinstance(o, datetime):
            if o.tzinfo is None:
                o = o.replace(tzinfo=timezone.utc)
            return o.isoformat()
        if isinstance(o, date):
            return o.isoformat()
        if isinstance(o, Enum):
            return o.value
        if isinstance(o, Decimal):
            return float(o)
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            return dataclasses.asdict(o)
        if isinstance(o, (set, frozenset)):
            return sorted(o, key=str)
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> Path:
    """Serialise and replace atomically. Raises on any failure - the caller decides."""
    target = Path(path)
    tmp = target.with_name(target.name + ".tmp")     # same directory, same mount
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, cls=BeastJSONEncoder, allow_nan=False, separators=(",", ":"))
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        _replace(tmp, target)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return target


def _replace(src: Path, dst: Path) -> None:
    """``os.replace`` - atomic on POSIX; on Windows, retried briefly.

    Windows refuses to rename over a file another process holds open without
    FILE_SHARE_DELETE, which a plain reader does not set. The VPS is Linux and
    never hits this; the retry keeps the developer box honest without
    changing the on-disk guarantee anywhere.
    """
    if sys.platform != "win32":
        os.replace(src, dst)
        return
    deadline = time.monotonic() + 2.0
    delay = 0.001
    while True:
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 0.05)


def read_live_state(path: str | Path) -> dict[str, Any] | None:
    """The file as a dict, or None when missing or unparseable.

    Malformed JSON is None on purpose: the reader must say DOWN rather than
    render the previous value as if it were current.
    """
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def written_at(state: dict[str, Any] | None) -> datetime | None:
    """``written_at`` as a tz-aware datetime, or None."""
    if not state:
        return None
    try:
        stamp = datetime.fromisoformat(str(state.get("written_at")))
    except (TypeError, ValueError):
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def age_seconds(state: dict[str, Any] | None, now: datetime | None = None) -> float | None:
    stamp = written_at(state)
    if stamp is None:
        return None
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return (now - stamp).total_seconds()


# -- staleness (B3) ----------------------------------------------------------

LIVE, LAGGING, STALE, DOWN, HALTED, STOPPED = (
    "LIVE", "LAGGING", "STALE", "DOWN", "HALTED", "STOPPED")


def classify(state: dict[str, Any] | None, loop_interval: float,
             now: datetime | None = None) -> tuple[str, float | None]:
    """The one verdict a phone glance must get right.

    Returns ``(state_label, age_seconds)``. Precedence: no file or unparseable
    is DOWN; a clean shutdown is STOPPED whatever its age; a KILL flag is
    HALTED; then age against the loop interval - under 2x is LIVE, 2-6x is
    LAGGING, beyond is STALE.
    """
    if state is None:
        return DOWN, None
    age = age_seconds(state, now)
    if age is None:
        return DOWN, None
    if state.get("clean_shutdown"):
        return STOPPED, age
    if (state.get("system") or {}).get("kill_flag"):
        return HALTED, age
    interval = max(1.0, float(loop_interval or 1.0))
    if age < 2 * interval:
        return LIVE, age
    if age <= 6 * interval:
        return LAGGING, age
    return STALE, age


# -- the watchdog's view -----------------------------------------------------


@dataclasses.dataclass
class Heartbeat:
    """What the watchdog needs, projected out of the live-state file."""

    written_at: str
    clean_shutdown: bool = False
    last_error: str | None = None
    open_positions: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    loop_cycle_count: int = 0
    pid: int = 0


def write_heartbeat(path: str | Path, beat: Heartbeat) -> Path:
    """Write a minimal v2 file from the watchdog's view (used by tests and tools)."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "written_at": beat.written_at,
        "pid": beat.pid,
        "cycle_count": beat.loop_cycle_count,
        "clean_shutdown": beat.clean_shutdown,
        "positions": [{"market": p.get("market"), "direction": p.get("direction"),
                       "volume": p.get("volume"),
                       "stop_present_at_broker": bool(p.get("sl_present"))}
                      for p in beat.open_positions],
        "system": {"last_error": ({"message": beat.last_error} if beat.last_error else None)},
    }
    return write_json_atomic(path, payload)


def read_heartbeat(path: str | Path) -> Heartbeat | None:
    state = read_live_state(path)
    if not state or "written_at" not in state:
        return None
    system = state.get("system") or {}
    error = system.get("last_error")
    positions = []
    for row in state.get("positions") or []:
        positions.append({
            "market": row.get("market"),
            "direction": row.get("direction"),
            "volume": row.get("volume"),
            "sl_present": bool(row.get("stop_present_at_broker", row.get("sl_present"))),
        })
    return Heartbeat(
        written_at=str(state["written_at"]),
        clean_shutdown=bool(state.get("clean_shutdown", False)),
        last_error=(error.get("message") if isinstance(error, dict) else error),
        open_positions=positions,
        loop_cycle_count=int(state.get("cycle_count", state.get("loop_cycle_count", 0)) or 0),
        pid=int(state.get("pid", 0) or 0),
    )
