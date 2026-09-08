"""Structured logging - four rotating streams, one shared runtime context.

Beast produces two kinds of output and they must not be mixed:

* **Operator-facing lines** - the section 11 signal lines and alerts. Short,
  factual, no padding, in the strict-risk-manager tone the soul file specifies.
* **Machine-facing records** - JSON lines carrying the full Appendix B/C
  payloads, which the learning loop and the weekly report read back.

Both go through the same logger so the ordering on disk reflects what actually
happened, but the JSON records carry an ``event`` key so they can be filtered
out of a human read.

Four streams
------------
========== =========================================================
``main``   Everything. The complete ordered record of the session.
``trades`` Signals, fills and closed trades - the Appendix B/C stream.
``alerts`` Anything the operator was paged about.
``regime`` ``vol_state`` and section 4.4 ``regime`` transitions.
========== =========================================================

Every record also lands in ``main``, so a stream is a filtered view rather than
a partition. Reconstructing a session from four files that each hold a disjoint
slice means merging by timestamp and hoping the clocks agree; reading one file
does not.

Rotation is 10 MB per file with 30 backups. Thirty backups rather than thirty
*days* because a logger cannot know how many days 300 MB will cover - a quiet
week and a volatile one differ by an order of magnitude - and a size-and-count
policy has a bound a disk-space alert can be written against. Section 9's
history lives in the SQLite journal, not in these files, so rotating a log away
loses telemetry rather than trade records.

The runtime context
-------------------
Every record carries the same six fields: timestamp, ``vol_state`` and its
probability, equity, open positions, and the day's P&L. They are not passed at
each call site - that would be six extra arguments on every log line and they
would drift - but injected by :class:`RuntimeContextFilter` from a process-wide
:class:`RuntimeContext` that the runner refreshes once per cycle.

That matters when reading a log after the fact. The question is almost never
"what happened" but "what was true when it happened": a stale-feed warning at
14:31 reads very differently when the same line says equity was down 12% on the
day and a position was open.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from core.config import Config, get_config

LOGGER_NAME = "beast"

#: Rotation policy. 10 MB x (1 + 30) files caps each stream at ~310 MB.
MAX_BYTES = 10 * 1024 * 1024
BACKUP_COUNT = 30

#: ``event`` values routed to each stream, beyond ``main`` which takes them all.
TRADE_EVENTS = frozenset({"signal", "trade", "order", "fill", "exit"})
ALERT_EVENTS = frozenset({"alert", "pause", "override", "blocker"})
REGIME_EVENTS = frozenset({"vol_state", "regime", "model"})


@dataclass
class RuntimeContext:
    """What was true at the moment a record was written.

    Refreshed once per cycle by the runner and read by every log record. It is
    a snapshot of state, never a source of it - nothing reads its way back into
    a trading decision, so a stale context degrades a log line and nothing else.

    Attributes:
        vol_state: ``CALM | NORMAL | TURBULENT | UNKNOWN`` per market.
        vol_probability: Filtered posterior for the reported state, per market.
        regime: Section 4.4's ``TREND_UP | TREND_DOWN | RANGE``, per market.
        equity: Capital in force.
        open_positions: Markets currently holding a position.
        daily_pnl: Realised P&L today, per session family.
        daily_pnl_pct: The same as a fraction of equity.
        mode: ``paper`` or ``live``.
    """

    vol_state: dict[str, str] = field(default_factory=dict)
    vol_probability: dict[str, float] = field(default_factory=dict)
    regime: dict[str, str] = field(default_factory=dict)
    equity: float = 0.0
    open_positions: list[str] = field(default_factory=list)
    daily_pnl: dict[str, float] = field(default_factory=dict)
    daily_pnl_pct: dict[str, float] = field(default_factory=dict)
    mode: str = "paper"

    def as_record(self) -> dict[str, Any]:
        """The context block embedded in every JSON line."""
        return {
            "vol_state": dict(self.vol_state),
            "vol_probability": {
                market: round(value, 4)
                for market, value in self.vol_probability.items()
            },
            "regime": dict(self.regime),
            "equity": round(self.equity, 2),
            "open_positions": list(self.open_positions),
            "daily_pnl": {k: round(v, 2) for k, v in self.daily_pnl.items()},
            "daily_pnl_pct": {
                k: round(v, 5) for k, v in self.daily_pnl_pct.items()
            },
            "mode": self.mode,
        }


_context = RuntimeContext()
_context_lock = threading.Lock()


def set_runtime_context(**fields: Any) -> RuntimeContext:
    """Update the process-wide context. Unknown keys are ignored.

    Called once per cycle by the runner. Ignoring unknown keys rather than
    raising is deliberate: a caller passing a field this build does not carry
    should lose the field, not the log line.
    """
    with _context_lock:
        for key, value in fields.items():
            if hasattr(_context, key):
                setattr(_context, key, value)
        return _context


def get_runtime_context() -> RuntimeContext:
    """Return the process-wide context."""
    return _context


def reset_runtime_context() -> None:
    """Clear the context - used by tests so one does not leak into the next."""
    global _context
    with _context_lock:
        _context = RuntimeContext()


class RuntimeContextFilter(logging.Filter):
    """Attaches the runtime context to every record passing through."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.runtime = get_runtime_context().as_record()
        return True


class StreamFilter(logging.Filter):
    """Admits only the ``event`` types belonging to one stream."""

    def __init__(self, events: frozenset[str]) -> None:
        super().__init__()
        self.events = events

    def filter(self, record: logging.LogRecord) -> bool:
        return getattr(record, "event", "message") in self.events


class JsonLineFormatter(logging.Formatter):
    """One JSON object per line, context included."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "event": getattr(record, "event", "message"),
            "logger": record.name,
        }
        payload = getattr(record, "payload", None)
        if payload is None:
            entry["message"] = record.getMessage()
        else:
            entry["message"] = record.getMessage()
            entry["payload"] = payload
        entry["runtime"] = getattr(record, "runtime", None) or {}
        return json.dumps(entry, default=str)


class ConsoleFormatter(logging.Formatter):
    """Terse human formatting: ``HH:MM:SS LEVEL message``."""

    def format(self, record: logging.LogRecord) -> str:
        stamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        return f"{stamp} {record.levelname:<7} {record.getMessage()}"


def _rotating(path: Path, events: frozenset[str] | None = None) -> RotatingFileHandler:
    """Build one rotating JSON stream, optionally filtered to ``events``."""
    handler = RotatingFileHandler(
        path, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    handler.setFormatter(JsonLineFormatter())
    if events is not None:
        handler.addFilter(StreamFilter(events))
    return handler


def log_directory(config: Config | None = None) -> Path:
    """Resolve ``monitoring.log_dir``, relative to the project root."""
    cfg = config or get_config()
    directory = Path(str(cfg.get("monitoring.log_dir")))
    if not directory.is_absolute():
        directory = Path(__file__).resolve().parent.parent / directory
    return directory


def setup_logging(config: Config | None = None,
                  quiet_console: bool = False) -> logging.Logger:
    """Configure and return the Beast logger.

    Args:
        config: Injected for tests.
        quiet_console: Suppress console output - used when the rich dashboard
            owns the terminal and log lines would corrupt the display.

    Returns:
        The configured ``beast`` logger. Calling this twice is safe: existing
        handlers are closed and replaced rather than duplicated, which matters
        because a leaked ``RotatingFileHandler`` holds an open file that Windows
        will not let the rotation rename.
    """
    cfg = config or get_config()
    directory = log_directory(cfg)
    directory.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(
        getattr(logging, str(cfg.get("monitoring.log_level")).upper(), logging.INFO)
    )
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    for existing in list(logger.filters):
        logger.removeFilter(existing)
    logger.propagate = False

    logger.addFilter(RuntimeContextFilter())
    logger.addHandler(_rotating(directory / "main.log"))
    logger.addHandler(_rotating(directory / "trades.log", TRADE_EVENTS))
    logger.addHandler(_rotating(directory / "alerts.log", ALERT_EVENTS))
    logger.addHandler(_rotating(directory / "regime.log", REGIME_EVENTS))

    if not quiet_console:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(ConsoleFormatter())
        logger.addHandler(console)

    return logger


def get_logger() -> logging.Logger:
    """Return the Beast logger, configuring it on first use."""
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        return setup_logging()
    return logger


def log_event(event: str, payload: dict[str, Any], message: str = "",
              level: int = logging.INFO) -> None:
    """Emit a structured record.

    Args:
        event: Record type. Routes the record to a stream - see
            :data:`TRADE_EVENTS`, :data:`ALERT_EVENTS`, :data:`REGIME_EVENTS`.
            An unrecognised event still reaches ``main.log``.
        payload: The Appendix B/C dictionary, or any structured detail.
        message: Optional human-readable line logged alongside it.
        level: Logging level.
    """
    get_logger().log(level, message or event,
                     extra={"event": event, "payload": payload})


def log_signal(signal_dict: dict[str, Any], reason_line: str) -> None:
    """Log an emitted signal - the operator line plus the Appendix B record."""
    log_event("signal", signal_dict, reason_line)


def log_trade(trade_dict: dict[str, Any], message: str = "") -> None:
    """Log a closed trade - the full Appendix C record."""
    log_event("trade", trade_dict, message or "trade closed")


def log_rejection(rejection_dict: dict[str, Any]) -> None:
    """Log a gate rejection at DEBUG - these are high-volume by design.

    Section 9 reads them from the journal, not from the console; surfacing every
    G0 rejection in the terminal would bury the signals.
    """
    log_event(
        "rejection",
        rejection_dict,
        f"rejected at {rejection_dict.get('failed_gate')}: "
        f"{rejection_dict.get('gate_detail')}",
        level=logging.DEBUG,
    )


def log_alert(message: str, payload: dict[str, Any] | None = None) -> None:
    """Log an operator alert - spread, data delay, blackout, loss-limit pause."""
    log_event("alert", payload or {}, message, level=logging.WARNING)


def log_vol_state(market: str, state: Any) -> None:
    """Log a volatility-state reading to the regime stream.

    Args:
        state: A :class:`~core.regime.contracts.VolState`, or anything with a
            ``to_dict``.
    """
    payload = state.to_dict() if hasattr(state, "to_dict") else dict(state)
    log_event(
        "vol_state", payload,
        f"{market} vol_state={payload.get('vol_state')} "
        f"confirmed={payload.get('vol_state_confirmed')} "
        f"p={payload.get('vol_state_probability')}",
    )


def log_regime_change(market: str, previous: str, current: str,
                      detail: str = "") -> None:
    """Log a section 4.4 regime transition to the regime stream."""
    log_event(
        "regime",
        {"market": market, "from": previous, "to": current, "detail": detail},
        f"{market} regime {previous} -> {current}"
        + (f" ({detail})" if detail else ""),
    )


def log_model_event(market: str, action: str, payload: dict[str, Any]) -> None:
    """Log a model load, refusal or retrain to the regime stream."""
    log_event("model", {"market": market, "action": action, **payload},
              f"{market} model {action}")
