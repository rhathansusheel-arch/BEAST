"""``state_snapshot.json`` - what Beast believed when it last stopped.

Written on every clean shutdown and read at the next startup. Its purpose is
narrow and worth stating precisely, because a snapshot used for the wrong thing
is more dangerous than no snapshot at all.

What it is for
--------------
Restoring the **day's risk counters** across a restart: realised P&L, the
consecutive-loss count, the session pause and its reason, and per-market
cooldowns. A restart must not reset the daily loss cap. That is how a 15% cap
becomes a 30% day, and it is the single most expensive thing a crash-restart can
silently do.

What it is emphatically not for
-------------------------------
**Positions.** The broker is the authority on what Beast owns, always. This file
records what Beast *believed*, which is a different and weaker claim: it is
written at shutdown, so it is wrong by construction if anything filled during or
after the shutdown, and it is absent entirely after a hard kill. Positions are
therefore reconciled from the broker at startup, and this file is used only to
detect and alert on a disagreement.

Two guards
----------
* **Same-session-day only.** Counters are restored only when the snapshot's
  session day matches today's. A snapshot from Friday must not carry Friday's
  losses into Monday - the cap is session-scoped (soul file 7).
* **Clean-exit marker.** ``clean_exit`` is written last, on the clean path only.
  Its absence means the previous run died without shutting down, which is
  something the operator has to be told rather than something to paper over.

Atomic write
------------
Temp file plus ``os.replace``, which is atomic on both POSIX and Windows. A
reader can therefore never see a half-written snapshot, only the old one or the
new one. A crash mid-write costs the newest snapshot, never the file.

The fuller reconcile that this file is one input to - broker positions first,
resting stops verified before anything else, in-flight setup state explicitly
cleared, counters restored from the database rather than from here - belongs to
the ops layer and is not built yet. Until it is, this module states plainly what
it does and does not guarantee.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from core.config import Config, get_config

LOGGER = logging.getLogger("beast.session_state")

#: Bumped when the file's shape changes. A snapshot written by an older Beast is
#: ignored rather than half-read - a partially understood snapshot restores a
#: subset of the counters, which is worse than restoring none of them.
SNAPSHOT_VERSION = 2


@dataclass
class MarketCounters:
    """One family's session-scoped risk counters (soul file 7)."""

    family: str
    session_day: str | None = None
    realised_pnl: float = 0.0
    consecutive_losses: int = 0
    trades_today: int = 0
    paused: bool = False
    pause_reason: str = ""
    cooldown_until: dict[str, str] = field(default_factory=dict)


@dataclass
class SessionSnapshot:
    """Everything carried across a restart.

    Attributes:
        version: :data:`SNAPSHOT_VERSION` at write time.
        written_at: ISO timestamp.
        clean_exit: True only when written by an orderly shutdown. Absent or
            false means the previous run died.
        mode: ``paper`` or ``live``, so a snapshot is never restored across a
            mode change.
        capital: Capital in force at shutdown.
        counters: Per-family risk counters.
        believed_positions: What Beast thought it held. Diagnostic only - the
            broker is the authority.
        vol_states: Last ``VolState`` per market, for the dashboard and for the
            session-boundary carry.
        model_versions: The frozen model version per market, so a restart on a
            different model is visible.
    """

    version: int = SNAPSHOT_VERSION
    written_at: str = ""
    clean_exit: bool = False
    mode: str = "paper"
    capital: float = 0.0
    counters: dict[str, dict[str, Any]] = field(default_factory=dict)
    believed_positions: list[dict[str, Any]] = field(default_factory=list)
    vol_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    model_versions: dict[str, str] = field(default_factory=dict)
    session_summary: dict[str, Any] = field(default_factory=dict)


def snapshot_path(config: Config | None = None) -> Path:
    """Where the snapshot lives.

    Overridable by ``BEAST_STATE_DIR`` so a test never touches the real one.
    """
    cfg = config or get_config()
    directory = os.environ.get("BEAST_STATE_DIR") or cfg.get("monitoring.log_dir", "./logs")
    return Path(directory) / "state_snapshot.json"


def build_snapshot(risk, positions, vol_states: dict[str, Any] | None = None,
                   model_versions: dict[str, str] | None = None,
                   session_summary: dict[str, Any] | None = None,
                   clean_exit: bool = True,
                   config: Config | None = None) -> SessionSnapshot:
    """Capture the current state.

    Args:
        risk: The live :class:`~core.risk_manager.RiskManager`.
        positions: The live :class:`~broker.position_tracker.PositionTracker`.
        vol_states: ``{market: VolState}`` - stored via ``to_dict``.
        model_versions: ``{market: model_version}``.
        session_summary: Whatever the runner wants printed on the next start.
        clean_exit: False when writing from the crash path, so the next start
            knows the difference.
    """
    cfg = config or get_config()
    counters: dict[str, dict[str, Any]] = {}
    for family, state in risk.state.items():
        counters[family] = asdict(MarketCounters(
            family=family,
            session_day=str(state.session_day) if state.session_day else None,
            realised_pnl=float(state.realised_pnl),
            consecutive_losses=int(state.consecutive_losses),
            trades_today=int(state.trades_today),
            paused=bool(state.paused),
            pause_reason=str(state.pause_reason),
            cooldown_until={
                market: moment.isoformat()
                for market, moment in state.cooldown_until.items()
            },
        ))

    return SessionSnapshot(
        written_at=datetime.now().isoformat(),
        clean_exit=clean_exit,
        mode=cfg.mode,
        capital=float(risk.capital),
        counters=counters,
        believed_positions=positions.snapshot(),
        vol_states={
            market: state.to_dict() for market, state in (vol_states or {}).items()
            if state is not None
        },
        model_versions=dict(model_versions or {}),
        session_summary=dict(session_summary or {}),
    )


def save_snapshot(snapshot: SessionSnapshot, config: Config | None = None,
                  path: Path | None = None) -> Path:
    """Write atomically: temp file, flush, fsync, ``os.replace``.

    Returns:
        The path written.

    Never raises. A snapshot that cannot be written is logged and the process
    continues - failing to save state must not turn an orderly shutdown into a
    crash, which would lose the very thing the save was protecting.
    """
    target = path or snapshot_path(config)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=target.parent,
            prefix=".state_snapshot.", suffix=".tmp", delete=False,
        )
        try:
            json.dump(asdict(snapshot), handle, indent=2, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            handle.close()
        os.replace(handle.name, target)
    except OSError as error:
        LOGGER.error("could not write %s: %s", target, error)
    return target


def load_snapshot(config: Config | None = None,
                  path: Path | None = None) -> SessionSnapshot | None:
    """Read the snapshot, or return ``None``.

    Returns ``None`` for a missing file, unreadable JSON, or a version this
    build does not understand. Every one of those is logged. None of them is an
    error: a first run has no snapshot, and a corrupt one is better discarded
    than half-trusted.
    """
    target = path or snapshot_path(config)
    if not target.exists():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        LOGGER.warning("ignoring unreadable %s: %s", target, error)
        return None

    version = int(payload.get("version", 0))
    if version != SNAPSHOT_VERSION:
        LOGGER.warning(
            "ignoring %s: written by snapshot version %d, this build reads %d",
            target, version, SNAPSHOT_VERSION,
        )
        return None

    known = set(SessionSnapshot.__dataclass_fields__)
    return SessionSnapshot(**{k: v for k, v in payload.items() if k in known})


def restore_counters(snapshot: SessionSnapshot, risk, session_days: dict[str, Any],
                     config: Config | None = None) -> list[str]:
    """Restore session-scoped risk counters, guarded by session day and mode.

    Args:
        snapshot: What was loaded.
        risk: The live risk manager to restore into.
        session_days: ``{family: today's session day}``, from the session clocks.

    Returns:
        Human-readable notes about what was and was not restored, for the
        startup log. Never silent: a counter that was *not* restored matters as
        much as one that was.

    A restart must not reset the daily loss cap - that is how a 15% cap becomes
    a 30% day. But a Friday snapshot must not carry Friday's losses into Monday
    either, because the cap is session-scoped. Both are handled by comparing
    session days per family.
    """
    cfg = config or get_config()
    notes: list[str] = []

    if snapshot.mode != cfg.mode:
        return [
            f"snapshot was written in {snapshot.mode} mode, running in "
            f"{cfg.mode} - counters NOT restored"
        ]

    for family, stored in snapshot.counters.items():
        state = risk.state.get(family)
        if state is None:
            continue
        today = session_days.get(family)
        stored_day = stored.get("session_day")
        if today is None or stored_day is None or str(today) != str(stored_day):
            notes.append(
                f"{family}: snapshot is from session {stored_day}, today is "
                f"{today} - counters start fresh"
            )
            continue

        state.session_day = today
        state.realised_pnl = float(stored.get("realised_pnl", 0.0))
        state.consecutive_losses = int(stored.get("consecutive_losses", 0))
        state.trades_today = int(stored.get("trades_today", 0))
        state.paused = bool(stored.get("paused", False))
        state.pause_reason = str(stored.get("pause_reason", ""))
        state.cooldown_until = {
            market: datetime.fromisoformat(value)
            for market, value in (stored.get("cooldown_until") or {}).items()
        }
        notes.append(
            f"{family}: resumed mid-session - realised {state.realised_pnl:+,.0f}, "
            f"{state.consecutive_losses} consecutive losses"
            + (f", PAUSED ({state.pause_reason})" if state.paused else "")
        )

    return notes


def compare_positions(snapshot: SessionSnapshot,
                      broker_markets: list[str]) -> list[str]:
    """Report disagreements between what Beast believed and what the broker holds.

    Args:
        broker_markets: Markets the broker says currently hold a position.

    Returns:
        One line per disagreement, empty when they match.

    The broker is always right. This function never changes anything - it exists
    so that a disagreement is alerted rather than discovered later, because a
    position Beast does not know about is a position with no managed stop.
    """
    believed = {
        str(item.get("market")) for item in snapshot.believed_positions
        if item.get("market")
    }
    actual = set(broker_markets)
    lines: list[str] = []
    for market in sorted(believed - actual):
        lines.append(
            f"{market}: snapshot believed a position, broker reports none - "
            f"it closed while Beast was down"
        )
    for market in sorted(actual - believed):
        lines.append(
            f"{market}: broker reports a position the snapshot did not know "
            f"about - verify it has a resting stop before trading"
        )
    return lines
