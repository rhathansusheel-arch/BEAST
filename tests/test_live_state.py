"""The live-state writer and reader - D-72 to D-75."""

from __future__ import annotations

import json
import math
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from core.session_state import (
    SessionSnapshot, compare_positions, load_snapshot, restore_counters, save_snapshot,
)
from ops.heartbeat import (
    DOWN, HALTED, LAGGING, LIVE, STALE, STOPPED, BeastJSONEncoder, classify,
    read_live_state, write_json_atomic,
)

T0 = datetime(2026, 9, 14, 12, 0, 0, tzinfo=timezone.utc)


# -- 1. valid JSON, no NaN ---------------------------------------------------------

def test_encoder_handles_every_awkward_type_and_no_nan_escapes(tmp_path):
    class Colour(Enum):
        RED = "red"

    payload = {
        "np_int": np.int64(7), "np_float": np.float32(1.5), "np_bool": np.bool_(True),
        "np_arr": np.array([1, 2]), "ts": pd.Timestamp("2026-09-14 12:00", tz="Asia/Kolkata"),
        "naive_dt": datetime(2026, 9, 14, 12, 0), "enum": Colour.RED, "dec": Decimal("2.5"),
        "nat": pd.NaT, "set": {"b", "a"},
    }
    write_json_atomic(tmp_path / "s.json", payload)
    text = (tmp_path / "s.json").read_text()
    got = json.loads(text)                              # plain json.loads, no cls
    assert got["np_int"] == 7 and got["np_bool"] is True and got["np_arr"] == [1, 2]
    assert got["ts"].endswith("+05:30"), "timestamps carry tzinfo"
    assert got["naive_dt"].endswith("+00:00"), "a naive datetime is stamped UTC, never bare"
    assert got["enum"] == "red" and got["dec"] == 2.5 and got["nat"] is None
    assert "NaN" not in text and "Infinity" not in text


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), np.float64("nan"), np.float32("inf")])
def test_non_finite_floats_fail_at_write_time(tmp_path, bad):
    with pytest.raises(ValueError):
        write_json_atomic(tmp_path / "s.json", {"x": bad})
    assert not (tmp_path / "s.json").exists(), "nothing half-written"
    assert not (tmp_path / "s.json.tmp").exists(), "temp file cleaned up"


# -- 2. atomic: same directory, never a partial parse ------------------------------

def test_temp_file_is_in_the_same_directory(tmp_path, monkeypatch):
    seen = []
    real_replace = __import__("os").replace

    def spy(src, dst):
        seen.append((Path(src).parent, Path(dst).parent, Path(src).name))
        return real_replace(src, dst)

    monkeypatch.setattr("ops.heartbeat.os.replace", spy)
    write_json_atomic(tmp_path / "sub" / "s.json", {"a": 1})
    src_dir, dst_dir, name = seen[0]
    assert src_dir == dst_dir, "a temp file on another mount makes os.replace a copy"
    assert name.endswith(".tmp")


def test_reader_never_sees_a_partial_file_across_1000_writes(tmp_path):
    path = tmp_path / "s.json"
    write_json_atomic(path, {"n": 0, "pad": "x" * 2000})
    failures, stop = [], threading.Event()

    def reader():
        while not stop.is_set():
            try:
                got = json.loads(path.read_text(encoding="utf-8"))
                assert "n" in got and len(got["pad"]) == 2000
            except PermissionError:
                pass                # Windows: the writer's rename is mid-flight; not a partial parse
            except Exception as error:
                failures.append(repr(error))
            time.sleep(0.001)       # a real reader polls every few seconds, not in a hot loop

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    try:
        for i in range(1, 1001):
            write_json_atomic(path, {"n": i, "pad": "x" * 2000})
    finally:
        stop.set()
        thread.join(timeout=10)
    assert failures == [], failures[:3]
    assert json.loads(path.read_text())["n"] == 1000


# -- 3. write failure never propagates ------------------------------------------

@pytest.mark.parametrize("error", [PermissionError("read-only"), OSError(28, "No space left on device")])
def test_write_failure_is_logged_once_and_swallowed(cfg, tmp_path, monkeypatch, caplog, error):
    from tests.test_ops import bare_runner
    runner = bare_runner(cfg, tmp_path)

    def boom(path, payload):
        raise error

    monkeypatch.setattr("main.write_json_atomic", boom)
    import logging
    with caplog.at_level(logging.WARNING, logger="t"):
        for i in range(5):
            runner._write_heartbeat(T0 + timedelta(seconds=i))          # must not raise
    assert runner._heartbeat_failures == 5
    warnings = [r for r in caplog.records if "live state not written" in r.getMessage()]
    assert len(warnings) == 1, "one warning per five minutes, not one per cycle"
    assert "trading continues" in warnings[0].getMessage()


# -- 4 and 5. runs on an errored cycle, and when no market is open --------------

def test_writer_runs_on_an_errored_cycle_with_last_error(cfg, tmp_path):
    from tests.test_ops import bare_runner
    runner = bare_runner(cfg, tmp_path)
    runner._tick_body = lambda now: (_ for _ in ()).throw(RuntimeError("cycle exploded"))
    with pytest.raises(RuntimeError):
        runner.tick(T0)
    state = read_live_state(tmp_path / "heartbeat.json")
    assert state["system"]["last_error"]["message"].endswith("cycle exploded")
    assert state["cycle_count"] == 1


def test_writer_runs_when_no_market_is_open(cfg, tmp_path):
    from tests.test_ops import bare_runner
    runner = bare_runner(cfg, tmp_path)             # clock.is_open -> False
    runner.tick(T0)
    state = read_live_state(tmp_path / "heartbeat.json")
    assert state is not None
    assert state["markets"][0]["is_open"] is False
    assert state["schema_version"] == 2
    assert state["written_at"].endswith("+00:00")


# -- 6. staleness classification ------------------------------------------------

def _state(age_s: float, **extra) -> dict:
    s = {"written_at": (T0 - timedelta(seconds=age_s)).isoformat(), "system": {}}
    s.update(extra)
    return s


@pytest.mark.parametrize("age,expected", [
    (0, LIVE), (9.9, LIVE), (10.0, LAGGING), (29.9, LAGGING), (30.0, LAGGING), (30.1, STALE),
    (3600, STALE),
])
def test_age_boundaries_against_a_5s_loop(age, expected):
    verdict, got_age = classify(_state(age), loop_interval=5.0, now=T0)
    assert verdict == expected
    assert got_age == pytest.approx(age)


def test_missing_file_and_malformed_json_are_down(tmp_path):
    assert classify(None, 5.0, now=T0) == (DOWN, None)
    (tmp_path / "bad.json").write_text("{not json")
    assert read_live_state(tmp_path / "bad.json") is None
    assert classify(read_live_state(tmp_path / "bad.json"), 5.0, now=T0)[0] == DOWN
    assert classify({"no_written_at": 1}, 5.0, now=T0)[0] == DOWN


def test_halted_and_stopped_take_precedence_over_age():
    halted = _state(0, system={"kill_flag": {"mode": "halt"}})
    assert classify(halted, 5.0, now=T0)[0] == HALTED
    stopped = _state(99999, clean_shutdown=True)
    assert classify(stopped, 5.0, now=T0)[0] == STOPPED, "a clean stop is never STALE"


# -- 7 and 8. the reader's file handling ------------------------------------------

def test_reader_handles_missing_file_malformed_json_and_locked_db(tmp_path):
    from monitoring.streamlit_app import read_journal, tail_log_stream
    assert read_live_state(tmp_path / "nope.json") is None
    assert read_journal(tmp_path / "nope.sqlite", "SELECT 1") == ([], None)

    import sqlite3
    db = tmp_path / "j.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE signals (signal_id TEXT)")
    conn.execute("INSERT INTO signals VALUES ('a')")
    conn.commit()
    conn.execute("BEGIN EXCLUSIVE")                  # the trading loop mid-write
    rows, note = read_journal(db, "SELECT * FROM signals")
    assert rows == [] and note and "busy" in note, "a lock is reported, not raised"
    conn.rollback(); conn.close()
    rows, note = read_journal(db, "SELECT * FROM signals")
    assert rows == [{"signal_id": "a"}] and note is None

    log = tmp_path / "alerts.log"
    log.write_text('{"ts": "1", "message": "one"}\n{"ts": "2", "message": "two"}\n{"ts": "3", "mess')
    got = tail_log_stream(log, max_kb=256)
    assert [e["message"] for e in got] == ["two", "one"], "the truncated last line is skipped"


def test_log_tail_reads_only_the_configured_bytes_of_a_big_file(tmp_path, monkeypatch):
    from monitoring import streamlit_app
    log = tmp_path / "main.log"
    line = json.dumps({"ts": "2026-09-14T00:00:00", "message": "x" * 80}) + "\n"
    with log.open("w") as handle:
        for _ in range(10 * 1024 * 1024 // len(line)):
            handle.write(line)
    assert log.stat().st_size > 9 * 1024 * 1024

    read_sizes = []
    real_open = Path.open

    def spy_open(self, *args, **kwargs):
        handle = real_open(self, *args, **kwargs)
        if self == log:
            real_read = handle.read

            def counted_read(*a, **k):
                data = real_read(*a, **k)
                read_sizes.append(len(data))
                return data
            handle.read = counted_read
        return handle

    monkeypatch.setattr(Path, "open", spy_open)
    got = streamlit_app.tail_log_stream(log, max_kb=64, limit=500)
    assert sum(read_sizes) <= 64 * 1024, f"read {sum(read_sizes)} bytes of a 10 MB file"
    assert got and all(e["message"] == "x" * 80 for e in got)


# -- 9. recovery semantics unchanged ----------------------------------------------

def test_snapshot_recovery_semantics_survive_periodic_writes(cfg, tmp_path):
    from core.risk_manager import RiskManager
    live = SessionSnapshot(written_at=T0.isoformat(), clean_exit=False, in_progress=True,
                           mode="paper", capital=500_000.0,
                           counters={"gold": {"session_day": "2026-09-14", "realised_pnl": -1200.0,
                                              "consecutive_losses": 2, "trades_today": 3,
                                              "paused": False, "pause_reason": "",
                                              "cooldown_until": {}}},
                           believed_positions=[{"market": "XAUUSD"}])
    path = tmp_path / "state_snapshot.json"
    save_snapshot(live, cfg, path)
    loaded = load_snapshot(cfg, path)
    assert loaded is not None
    assert loaded.clean_exit is False, "a mid-session write is never a clean exit"
    assert loaded.in_progress is True

    risk = RiskManager(cfg, capital=500_000)
    notes = restore_counters(loaded, risk, {"gold": "2026-09-14", "indian": "2026-09-14"}, cfg)
    assert risk.state["gold"].realised_pnl == -1200.0
    assert risk.state["gold"].consecutive_losses == 2
    assert compare_positions(loaded, []) == [
        "XAUUSD: snapshot believed a position, broker reports none - it closed while Beast was down"
    ]
    assert compare_positions(loaded, ["XAUUSD"]) == []


def test_clean_exit_is_only_true_from_shutdown(cfg, tmp_path):
    from tests.test_ops import bare_runner
    runner = bare_runner(cfg, tmp_path)
    written = []
    runner._save_snapshot = lambda clean_exit, in_progress=False, quiet=False: written.append(
        (clean_exit, in_progress))
    runner._maybe_periodic_snapshot = type(runner)._maybe_periodic_snapshot.__get__(runner)
    runner._last_periodic_snapshot = None
    runner._last_snapshot_fingerprint = None
    runner._maybe_periodic_snapshot(T0)
    runner._maybe_periodic_snapshot(T0 + timedelta(seconds=1))        # not due, unchanged
    runner._entry_pause_reason["XAUUSD"] = "feed down"                # a state change
    runner._maybe_periodic_snapshot(T0 + timedelta(seconds=2))
    assert written == [(False, True), (False, True)], "periodic and event writes: never clean_exit"


# -- 10. no widget reaches an order path --------------------------------------

def test_no_widget_in_the_app_can_reach_an_order_path():
    source = Path("monitoring/streamlit_app.py").read_text(encoding="utf-8")
    code = "\n".join(l for l in source.splitlines() if not l.strip().startswith(("#", '"""')))
    forbidden = ["place_order", "order_send", "cancel_order", "close_position", "OrderRequest",
                 "OrderExecutor", "PositionTracker", "set_flag(", "clear_flag(", "MT5Adapter",
                 "st.button", "st.form"]
    hits = [f for f in forbidden if f in code]
    assert hits == [], f"the dashboard must have no path to an order: {hits}"
    assert not re.search(r"from broker\.|import broker", code)


def test_config_forbids_a_public_bind():
    text = Path(".streamlit/config.toml").read_text(encoding="utf-8")
    assert 'address = "127.0.0.1"' in text
    assert "0.0.0.0" not in text.replace('"0.0.0.0" is forbidden', "")
