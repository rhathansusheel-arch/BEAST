"""The ops layer - Part B, step 6. No systemd, no network, no Wine."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from ops import EXIT_CLEAN, EXIT_DELIBERATE_HALT, EXIT_KILL, EXIT_STARTUP_FAILURE
from ops.heartbeat import Heartbeat, read_heartbeat, write_heartbeat
from ops.killswitch import FLATTEN_PHRASE, clear_flag, read_flag, set_flag
from ops.preflight import Check, Preflight
from ops.watchdog import Watchdog

T0 = datetime(2026, 9, 14, 12, 0, 0)


def beat(age_s=0, **kw) -> Heartbeat:
    base = dict(written_at=(T0 - timedelta(seconds=age_s)).isoformat(timespec="seconds"),
                pid=1, loop_cycle_count=1)
    base.update(kw)
    return Heartbeat(**base)


class Alerts:
    def __init__(self): self.sent = []
    def send(self, kind, market, message): self.sent.append((str(kind), market, message))


def ops_cfg(cfg, tmp_path, **over):
    cfg.data["ops"] = {
        "heartbeat_path": str(tmp_path / "heartbeat.json"),
        "kill_flag_path": str(tmp_path / "KILL"),
        "heartbeat_stale_seconds": 90, "max_restarts": 3, "restart_window_minutes": 30,
        "beast_unit": "beast.service", "dashboard_url": "http://127.0.0.1:1/",
        "max_clock_drift_seconds": 5, "min_free_disk_mb": 100,
    }
    cfg.data["ops"].update(over)
    return cfg


def watchdog(cfg, tmp_path, exit_code=None, restart_ok=True, now=None):
    calls = []
    wd = Watchdog(
        ops_cfg(cfg, tmp_path), Alerts(),
        systemctl=lambda action, unit: calls.append((action, unit)) or restart_ok,
        exit_status=lambda unit: exit_code,
        now=now or (lambda: T0),
        tcp_check=lambda h, p: True, http_check=lambda u: True, terminal_check=lambda: True,
    )
    return wd, calls


# -- heartbeat -------------------------------------------------------------------

def test_heartbeat_roundtrip_is_atomic_and_tolerant(tmp_path):
    path = tmp_path / "hb.json"
    write_heartbeat(path, beat(last_error="boom"))
    assert not path.with_suffix(".json.tmp").exists()
    got = read_heartbeat(path)
    assert got is not None and got.last_error == "boom" and got.clean_shutdown is False
    path.write_text('{"written_at": "x", "unknown_key": 1, "pid": 2, "cycle_count": 0}')
    assert read_heartbeat(path) is not None, "unknown keys are ignored, not fatal"


def bare_runner(cfg, tmp_path):
    """A BeastRunner with just enough state for tick()'s ops hooks."""
    from main import BeastRunner
    import logging
    c = ops_cfg(cfg, tmp_path)
    c.section("broker")["symbols"] = ["XAUUSD"]
    runner = BeastRunner.__new__(BeastRunner)
    runner.cfg = c; runner.markets = ["XAUUSD"]; runner.brokers = {}
    runner.stats = SimpleNamespace(cycles=0, errors=0, started_at=T0)
    runner.risk = SimpleNamespace(state={}, capital=1.0)
    runner.positions = SimpleNamespace(open_markets=lambda: [], get=lambda m: None)
    runner.clocks = {"XAUUSD": SimpleNamespace(is_open=lambda now: False,
                                                session_day=lambda now: T0.date())}
    runner.data = SimpleNamespace(feed=lambda m: None)
    runner.journal = SimpleNamespace(recent_signals=lambda n: [], recent_rejections=lambda n: [],
                                     trades_between=lambda a, b: [])
    runner._feed_failures = {}; runner._entry_pause_reason = {}; runner._kill_mode = None
    runner._last_cycle_error = None; runner.running = True; runner._exit_code = None
    runner._reconcile_report = None; runner._last_dashboard_state = {}
    runner._last_cycle_ms = 0.0; runner._heartbeat_write_ms = 0.0
    runner._heartbeat_failures = 0; runner._heartbeat_warned_at = None
    runner._error_streak = 0
    runner._loop_interval = lambda: 5.0
    runner.logger = logging.getLogger("t")
    runner._maybe_periodic_snapshot = lambda now: None
    # the per-cycle helpers _tick_body calls after the market loop
    for name in ("_check_circuit_breakers", "_check_broker_sessions", "_maybe_retrain",
                 "_maybe_weekly_review"):
        setattr(runner, name, lambda now: None)
    runner._refresh_dashboard = lambda now, d, r: None
    return runner


def test_runner_writes_a_heartbeat_even_when_the_cycle_raises(cfg, tmp_path):
    runner = bare_runner(cfg, tmp_path)
    runner._tick_body = lambda now: (_ for _ in ()).throw(RuntimeError("cycle exploded"))

    with pytest.raises(RuntimeError):
        runner.tick(T0)

    got = read_heartbeat(tmp_path / "heartbeat.json")
    assert got is not None
    assert "cycle exploded" in (got.last_error or "")
    assert got.loop_cycle_count == 1


# -- watchdog --------------------------------------------------------------------

def test_stale_heartbeat_is_rechecked_once_then_restarted(cfg, tmp_path):
    wd, calls = watchdog(cfg, tmp_path)
    write_heartbeat(wd.heartbeat_path, beat(age_s=300))
    assert wd.check()["action"] == "recheck"
    assert calls == [], "a slow cycle is not a dead process"
    assert wd.check()["action"] == "restarted"
    assert calls == [("restart", "beast.service")]


def test_no_restart_on_clean_shutdown(cfg, tmp_path):
    wd, calls = watchdog(cfg, tmp_path)
    write_heartbeat(wd.heartbeat_path, beat(age_s=3600, clean_shutdown=True))
    for _ in range(3):
        assert wd.check()["action"].startswith("none")
    assert calls == []


def test_no_restart_through_a_kill_flag(cfg, tmp_path):
    wd, calls = watchdog(cfg, tmp_path)
    write_heartbeat(wd.heartbeat_path, beat(age_s=3600))
    set_flag(wd.kill_path, "halt", "maintenance")
    for _ in range(3):
        assert "KILL" in wd.check()["action"]
    assert calls == []


@pytest.mark.parametrize("code", [EXIT_CLEAN, EXIT_STARTUP_FAILURE, EXIT_DELIBERATE_HALT, EXIT_KILL])
def test_no_restart_on_deliberate_exit_codes(cfg, tmp_path, code):
    wd, calls = watchdog(cfg, tmp_path, exit_code=code)
    write_heartbeat(wd.heartbeat_path, beat(age_s=3600))
    for _ in range(3):
        wd.check()
    assert calls == []
    assert any("deliberately" in m for _, _, m in wd.alerts.sent)


def test_crash_loop_guard_trips_and_pages(cfg, tmp_path):
    clock = {"t": T0}
    wd, calls = watchdog(cfg, tmp_path, now=lambda: clock["t"])
    for _ in range(3):
        write_heartbeat(wd.heartbeat_path, beat(age_s=3600))
        wd.check(); wd.check()                      # recheck then restart
        clock["t"] += timedelta(minutes=1)
    assert len(calls) == 3
    write_heartbeat(wd.heartbeat_path, beat(age_s=3600))
    wd.check(); action = wd.check()["action"]
    assert "crash-loop" in action
    assert len(calls) == 3, "no fourth restart"
    assert wd.crash_loop_tripped
    assert any("CRASH LOOP" in m for _, _, m in wd.alerts.sent)


def test_restart_alert_names_an_open_position(cfg, tmp_path):
    wd, calls = watchdog(cfg, tmp_path)
    write_heartbeat(wd.heartbeat_path, beat(
        age_s=3600, open_positions=[{"market": "XAUUSD", "direction": "LONG",
                                     "volume": 0.1, "sl_present": False}]))
    wd.check(); wd.check()
    kinds = [k for k, _, _ in wd.alerts.sent]
    msgs = [m for _, _, m in wd.alerts.sent]
    assert any("CIRCUIT_BREAKER" in k for k in kinds), "open position => critical"
    assert any("XAUUSD" in m and "stop=MISSING" in m for m in msgs)


# -- kill switch -----------------------------------------------------------------

def test_halt_pauses_entries_and_leaves_exits_running(cfg, tmp_path):
    runner = bare_runner(cfg, tmp_path)
    runner._alert = lambda *a, **k: None
    runner.positions = SimpleNamespace(open_markets=lambda: ["XAUUSD"])

    set_flag(tmp_path / "KILL", "halt", "test")
    runner._poll_kill_flag(T0)
    assert runner._entry_pause_reason["XAUUSD"].startswith("KILL halt")
    assert runner.running is True, "halt must not stop the loop - exits keep running"
    assert runner._exit_code is None

    clear_flag(tmp_path / "KILL")
    runner._poll_kill_flag(T0)
    assert "XAUUSD" not in runner._entry_pause_reason


def test_flatten_refuses_without_the_section_8_phrase(tmp_path):
    with pytest.raises(PermissionError):
        set_flag(tmp_path / "KILL", "flatten", "panic", confirmation="yes")
    assert read_flag(tmp_path / "KILL") is None, "nothing may be written without the phrase"
    set_flag(tmp_path / "KILL", "flatten", "panic", confirmation=FLATTEN_PHRASE)
    assert read_flag(tmp_path / "KILL")["mode"] == "flatten"


def test_flatten_goes_through_the_override_guard_and_exits_3(cfg, tmp_path):
    closed = []

    class Guard:
        def request(self, request, position, exits):
            ok = request.confirmation_text == FLATTEN_PHRASE
            return SimpleNamespace(accepted=ok, message="ok" if ok else "refused", record="rec")

    runner = bare_runner(cfg, tmp_path)
    runner._alert = lambda *a, **k: None
    runner.overrides = Guard()
    pos = SimpleNamespace(last_underlying=2500.0, entry_underlying=2500.0)
    runner.positions = SimpleNamespace(
        open_markets=lambda: ["XAUUSD"], get=lambda m: pos, exits=None,
        close=lambda m, d, now, override=None: closed.append((m, d.reason.value, override)) or (None, []))

    # written by hand to bypass set_flag's own guard: the LOOP must refuse too
    (tmp_path / "KILL").write_text(json.dumps({"mode": "flatten", "reason": "t",
                                               "confirmation": "wrong", "set_by": "x", "set_at": "y"}))
    runner._poll_kill_flag(T0)
    assert closed == [], "the loop must not close without the phrase either"

    (tmp_path / "KILL").write_text(json.dumps({"mode": "flatten", "reason": "t",
                                               "confirmation": FLATTEN_PHRASE, "set_by": "x", "set_at": "y"}))
    runner._poll_kill_flag(T0)
    assert closed == [("XAUUSD", "OVERRIDE", "rec")]
    assert runner.running is False and runner._exit_code == EXIT_KILL


# -- preflight -------------------------------------------------------------------

class FakeLink:
    def __init__(self, demo=True, connects=True):
        self.state = SimpleNamespace(value="READY" if connects else "HALTED")
        self.facts = SimpleNamespace(is_demo=demo, login=1, server="Demo", terminal_build=6191,
                                     latency_p50_ms=5.0, server_utc_offset_hours=0.0)
        self._connects = connects
    def connect(self, wait_seconds=0): return self._connects
    def spec(self, market): return SimpleNamespace(name="XAUUSD")
    def call(self, method, *a, **k): return SimpleNamespace(time=datetime.now().timestamp())
    def disconnect(self): pass


def preflight(cfg, tmp_path, link=None, ntp=(True, "yes"), disk=10_000, env=None):
    c = ops_cfg(cfg, tmp_path)
    c.section("broker")["symbols"] = ["XAUUSD"]
    c.section("instruments")["gold"].update({
        "trade": "cfd", "venue": "x", "symbol": "XAUUSD", "contract_multiplier": 100.0,
        "tick_size": 0.01, "tick_value": 1.0, "point": 0.01, "volume_min": 0.01,
        "volume_step": 0.01, "volume_max": 100.0, "stops_level_points": 100,
        "account_currency": "USD"})
    c.section("data")["gold_spread_max"] = 0.35
    c.section("monitoring")["journal_db"] = str(tmp_path / "j.sqlite")
    c.section("monitoring")["log_dir"] = str(tmp_path / "logs")
    environ = {"MT5_GOLD_LOGIN": "1", "MT5_GOLD_PASSWORD": "p", "MT5_GOLD_SERVER": "s"}
    environ.update(env or {})
    return Preflight(c, mt5=link or FakeLink(), ntp_status=lambda: ntp,
                     disk_free_mb=lambda p: disk, environ=environ)


def test_preflight_passes_on_a_healthy_box(cfg, tmp_path):
    checks = preflight(cfg, tmp_path).run()
    assert all(c.ok for c in checks), [c for c in checks if not c.ok]


def test_preflight_fails_on_a_blocker(cfg, tmp_path):
    p = preflight(cfg, tmp_path)
    p.cfg.section("instruments")["gold"]["tick_value"] = None
    assert not p.check_blockers().ok


def test_preflight_fails_on_clock_drift(cfg, tmp_path):
    assert not preflight(cfg, tmp_path, ntp=(False, "no")).run()[4].ok
    assert not preflight(cfg, tmp_path, ntp=(None, "no timedatectl")).run()[4].ok


def test_preflight_fails_on_low_disk(cfg, tmp_path):
    assert not preflight(cfg, tmp_path, disk=10).check_disk().ok


def test_preflight_fails_on_a_stale_kill_flag(cfg, tmp_path):
    p = preflight(cfg, tmp_path)
    set_flag(tmp_path / "KILL", "halt", "left over")
    assert not p.check_kill_flag().ok


def test_preflight_fails_on_a_real_account_in_paper_mode(cfg, tmp_path):
    p = preflight(cfg, tmp_path, link=FakeLink(demo=False))
    assert not p.check_bridge().ok
    assert "REAL" in p.check_bridge().detail


def test_preflight_fails_when_allow_live_is_set_in_demo_mode(cfg, tmp_path):
    p = preflight(cfg, tmp_path, env={"BEAST_ALLOW_LIVE": "1"})
    assert not p.check_env().ok
