"""Persistence for Appendix B and C records.

Three tables, matching the soul file's three record types:

* ``signals``    - Appendix B, every emitted signal.
* ``trades``     - Appendix C, the signal plus everything the exit produced.
* ``rejections`` - the separate rejection table, keyed by failing gate ID.

SQLite rather than flat files because section 9 asks aggregate questions ("win
rate per setup type per market over the trailing 30 trades", "where do signals
die") that are one query here and a parsing exercise anywhere else. The full JSON
payload is stored alongside the indexed columns, so nothing in the Appendix
schemas is lost even where it is not a column.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from core.config import Config, get_config
from core.schemas import Rejection, Signal, TradeRecord

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    signal_id      TEXT PRIMARY KEY,
    timestamp_ist  TEXT NOT NULL,
    market         TEXT NOT NULL,
    direction      TEXT NOT NULL,
    setup_type     INTEGER NOT NULL,
    regime         TEXT,
    counter_bias   INTEGER,
    confluence_aligned INTEGER,
    entry_price    REAL,
    stop_price     REAL,
    target_price   REAL,
    leg_type       TEXT,
    mode           TEXT,
    reason_line    TEXT,
    payload        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trades (
    signal_id      TEXT PRIMARY KEY,
    market         TEXT NOT NULL,
    setup_type     INTEGER NOT NULL,
    direction      TEXT NOT NULL,
    entry_time     TEXT NOT NULL,
    exit_time      TEXT,
    exit_reason    TEXT,
    r_multiple     REAL,
    underlying_r_multiple REAL,
    premium_r_multiple    REAL,
    mae_r          REAL,
    mfe_r          REAL,
    bars_held      INTEGER,
    trail_activated INTEGER,
    dte            INTEGER,
    delta_at_entry REAL,
    oi_tag         TEXT,
    expiry_day     INTEGER,
    overridden     INTEGER DEFAULT 0,
    payload        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rejections (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp      TEXT NOT NULL,
    instrument     TEXT NOT NULL,
    setup_type     INTEGER,
    direction      TEXT,
    failed_gate    TEXT NOT NULL,
    gate_detail    TEXT,
    confluence_aligned INTEGER,
    payload        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trades_market_setup ON trades (market, setup_type);
CREATE INDEX IF NOT EXISTS idx_trades_entry ON trades (entry_time);
CREATE INDEX IF NOT EXISTS idx_rejections_gate ON rejections (failed_gate);
CREATE INDEX IF NOT EXISTS idx_rejections_ts ON rejections (timestamp);
"""


class Journal:
    """SQLite-backed trade journal.

    Args:
        config: Injected for tests.
        path: Overrides ``monitoring.journal_db`` - pass ``":memory:"`` in tests.
    """

    def __init__(self, config: Config | None = None, path: str | Path | None = None) -> None:
        self.cfg = config or get_config()
        raw = Path(str(path if path is not None else self.cfg.get("monitoring.journal_db")))
        if str(raw) != ":memory:" and not raw.is_absolute():
            raw = Path(__file__).resolve().parent.parent / raw
        if str(raw) != ":memory:":
            raw.parent.mkdir(parents=True, exist_ok=True)
        self.path = str(raw)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with closing(self._conn.cursor()) as cursor:
            cursor.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # -- writes --------------------------------------------------------------

    def record_signal(self, signal: Signal) -> None:
        """Persist an Appendix B record."""
        payload = signal.to_dict()
        self._conn.execute(
            """INSERT OR REPLACE INTO signals VALUES
               (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                signal.signal_id,
                payload["timestamp_ist"],
                signal.market,
                signal.direction.value,
                int(signal.setup_type),
                signal.regime.value,
                int(signal.counter_bias),
                int(signal.confluence_count.get("aligned", 0)),
                signal.entry_price,
                signal.stop_price,
                signal.target_price,
                signal.leg_type,
                signal.mode,
                signal.reason_line,
                json.dumps(payload, default=str),
            ),
        )
        self._conn.commit()

    def record_trade(self, trade: TradeRecord) -> None:
        """Persist an Appendix C record.

        The R-multiple *of record* is the premium-based one for options - that is
        the actual money - with the underlying R stored alongside it (6.9). The
        gap between the two is the strike-selection diagnostic.
        """
        payload = trade.to_dict()
        chain = payload.get("chain_context") or {}
        self._conn.execute(
            """INSERT OR REPLACE INTO trades VALUES
               (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                trade.signal.signal_id,
                trade.signal.market,
                int(trade.signal.setup_type),
                trade.signal.direction.value,
                payload["entry_time"],
                payload["exit_time"],
                payload["exit_reason"],
                trade.r_multiple,
                trade.underlying_r_multiple,
                trade.premium_r_multiple,
                trade.mae_r,
                trade.mfe_r,
                trade.bars_held,
                int(trade.trail_activated),
                trade.dte,
                trade.delta_at_entry,
                chain.get("oi_tag"),
                int("EXPIRY_DAY" in payload.get("flags", [])),
                int(trade.override is not None),
                json.dumps(payload, default=str),
            ),
        )
        self._conn.commit()

    def record_rejection(self, rejection: Rejection) -> None:
        """Persist a rejection-log row."""
        payload = rejection.to_dict()
        counts = payload.get("confluence_count") or {}
        self._conn.execute(
            """INSERT INTO rejections
               (timestamp, instrument, setup_type, direction, failed_gate,
                gate_detail, confluence_aligned, payload)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                payload["timestamp"],
                payload["instrument"],
                payload["setup_type"],
                payload["direction"],
                payload["failed_gate"],
                payload["gate_detail"],
                counts.get("aligned"),
                json.dumps(payload, default=str),
            ),
        )
        self._conn.commit()

    def record_rejections(self, rejections: Iterable[Rejection]) -> None:
        """Persist several rejections in one transaction."""
        for rejection in rejections:
            self.record_rejection(rejection)

    # -- reads ---------------------------------------------------------------

    def recent_trades(self, market: str | None = None, setup_type: int | None = None,
                      limit: int = 30) -> list[dict[str, Any]]:
        """The most recent closed trades, newest first.

        Section 9's rolling window is 30 trades per setup+market pair, which is
        why ``limit`` defaults to 30.
        """
        clauses, params = ["exit_time IS NOT NULL"], []
        if market:
            clauses.append("market = ?")
            params.append(market)
        if setup_type is not None:
            clauses.append("setup_type = ?")
            params.append(setup_type)
        query = (
            f"SELECT * FROM trades WHERE {' AND '.join(clauses)} "
            f"ORDER BY entry_time DESC LIMIT ?"
        )
        params.append(limit)
        with closing(self._conn.cursor()) as cursor:
            return [dict(row) for row in cursor.execute(query, params).fetchall()]

    def trades_between(self, start: datetime, end: datetime) -> list[dict[str, Any]]:
        """Closed trades with an entry time in ``[start, end]``."""
        with closing(self._conn.cursor()) as cursor:
            rows = cursor.execute(
                "SELECT * FROM trades WHERE entry_time BETWEEN ? AND ? "
                "AND exit_time IS NOT NULL ORDER BY entry_time",
                (start.isoformat(), end.isoformat()),
            ).fetchall()
        return [dict(row) for row in rows]

    def gate_histogram(self, since: datetime | None = None) -> dict[str, int]:
        """Rejections per gate - section 9's "where do signals die" query."""
        query = "SELECT failed_gate, COUNT(*) AS n FROM rejections"
        params: list[Any] = []
        if since:
            query += " WHERE timestamp >= ?"
            params.append(since.isoformat())
        query += " GROUP BY failed_gate ORDER BY n DESC"
        with closing(self._conn.cursor()) as cursor:
            return {row["failed_gate"]: row["n"] for row in cursor.execute(query, params)}

    def override_count(self, since: datetime | None = None) -> int:
        """How many trades carry an override record (section 8, point 4)."""
        query = "SELECT COUNT(*) AS n FROM trades WHERE overridden = 1"
        params: list[Any] = []
        if since:
            query += " AND entry_time >= ?"
            params.append(since.isoformat())
        with closing(self._conn.cursor()) as cursor:
            return int(cursor.execute(query, params).fetchone()["n"])

    def open_signals_without_trades(self) -> list[dict[str, Any]]:
        """Signals with no matching trade row - used to reconcile after a restart."""
        with closing(self._conn.cursor()) as cursor:
            rows = cursor.execute(
                "SELECT s.* FROM signals s LEFT JOIN trades t USING (signal_id) "
                "WHERE t.signal_id IS NULL ORDER BY s.timestamp_ist DESC"
            ).fetchall()
        return [dict(row) for row in rows]
