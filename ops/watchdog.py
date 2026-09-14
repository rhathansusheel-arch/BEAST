"""The watchdog. It restarts processes and raises alerts; that is all.

    python -m ops.watchdog            # run forever
    python -m ops.watchdog --once     # one pass, exit code 0 if all healthy

Supervises four things and treats them differently:

* **Beast** - ``heartbeat.json`` age. Stale past ``ops.heartbeat_stale_seconds``
  -> re-read once (a slow cycle is not a dead process) -> restart the unit.
* **MT5 terminal** - process alive and ``terminal_info()`` answers over the
  bridge. Reported; restarted via its unit when dead.
* **RPyC bridge** - TCP connect on loopback.
* **Dashboard** - HTTP 200 on its local health URL. Reported only; a dead
  dashboard is not a trading problem.

It will **not** restart Beast when the last heartbeat says
``clean_shutdown: true``, when the KILL flag is set, or when Beast's unit
exited with a deliberate code (0-3, see ``ops/__init__.py``). More than
``ops.max_restarts`` inside ``ops.restart_window_minutes`` trips the
crash-loop guard: it stops restarting, raises CRITICAL, and stays up
reporting. Flapping a trading process against a live venue is worse than
being down.

If the last heartbeat shows an open position, every restart alert is CRITICAL
and names the position and whether its stop was present, because until Beast
is back and the reconcile has run, that position is running on its venue-side
stop alone.

Trading decisions never happen here. The watchdog holds no broker handle and
cannot place, modify or cancel anything.
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
import urllib.request
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ops import NO_RESTART_CODES  # noqa: E402
from ops.heartbeat import Heartbeat, read_heartbeat  # noqa: E402
from ops.killswitch import read_flag  # noqa: E402


class Watchdog:
    """One supervisor, injectable for tests.

    Args:
        cfg: Beast config.
        alerts: Anything with ``send(kind, market, message)`` - normally
            ``monitoring.alerts.AlertManager``; the test passes a recorder.
        systemctl: ``(action, unit) -> bool``.
        exit_status: ``(unit) -> int | None`` - the unit's last exit code.
        now: Clock, injectable.
    """

    def __init__(self, cfg, alerts, systemctl: Callable[[str, str], bool] | None = None,
                 exit_status: Callable[[str], int | None] | None = None,
                 now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
                 tcp_check: Callable[[str, int], bool] | None = None,
                 http_check: Callable[[str], bool] | None = None,
                 terminal_check: Callable[[], bool | None] | None = None) -> None:
        self.cfg = cfg
        self.alerts = alerts
        self.systemctl = systemctl or _systemctl
        self.exit_status = exit_status or _unit_exit_status
        self.now = now
        self.tcp_check = tcp_check or _tcp_ok
        self.http_check = http_check or _http_ok
        self.terminal_check = terminal_check or self._terminal_ok
        self.restarts: deque[datetime] = deque()
        self.crash_loop_tripped = False
        self._stale_seen_once = False
        self._deliberate_seen: tuple[int, datetime] | None = None
        self.last_report: dict[str, Any] = {}

    # -- settings ------------------------------------------------------------

    def _s(self, key: str, default: Any) -> Any:
        return self.cfg.get(f"ops.{key}", default)

    @property
    def heartbeat_path(self) -> Path:
        raw = Path(str(self._s("heartbeat_path", "./heartbeat.json")))
        return raw if raw.is_absolute() else PROJECT_ROOT / raw

    @property
    def kill_path(self) -> Path:
        raw = Path(str(self._s("kill_flag_path", "./KILL")))
        return raw if raw.is_absolute() else PROJECT_ROOT / raw

    # -- one pass ------------------------------------------------------------

    def check(self) -> dict[str, Any]:
        """Inspect everything once; restart Beast if - and only if - it should be."""
        report: dict[str, Any] = {"at": self.now().isoformat(timespec="seconds")}
        beat = read_heartbeat(self.heartbeat_path)
        report["heartbeat_age"] = None if beat is None else round(self._age(beat), 1)
        report["beast"] = self._beast_verdict(beat)
        report["bridge"] = self.tcp_check(
            str(self.cfg.get("broker.mt5.profiles.gold.host", "127.0.0.1")),
            int(self.cfg.get("broker.mt5.profiles.gold.port", 18812)),
        )
        report["terminal"] = self.terminal_check()
        report["dashboard"] = self.http_check(
            str(self._s("dashboard_url", "http://127.0.0.1:8501/_stcore/health")))
        report["action"] = self._act(beat, report)
        self.last_report = report
        return report

    def _beast_verdict(self, beat: Heartbeat | None) -> str:
        if beat is None:
            return "no heartbeat"
        if beat.clean_shutdown:
            return "clean shutdown"
        age = self._age(beat)
        limit = float(self._s("heartbeat_stale_seconds", 90))
        if age > limit:
            return f"stale ({age:.0f}s > {limit:.0f}s)"
        return "alive" + (f" (last_error: {beat.last_error})" if beat.last_error else "")

    def _act(self, beat: Heartbeat | None, report: dict[str, Any]) -> str:
        unit = str(self._s("beast_unit", "beast.service"))
        verdict = report["beast"]

        if not (verdict.startswith("stale") or verdict == "no heartbeat"):
            self._stale_seen_once = False
            return "none"

        # Reasons never to restart, cheapest first.
        if beat is not None and beat.clean_shutdown:
            return "none: clean shutdown"
        flag = read_flag(self.kill_path)
        if flag is not None:
            return f"none: KILL flag ({flag.get('mode')})"
        code = self.exit_status(unit)
        if code is not None and code in NO_RESTART_CODES:
            # Once per exit, then a reminder - not once per 15-second pass. The
            # 2026-09-14 log holds 125 identical pages for one preflight refusal;
            # the operator learned nothing from the 124 after the first.
            now = self.now()
            reminder = timedelta(minutes=float(self._s("watchdog_reminder_minutes", 15)))
            seen = self._deliberate_seen
            if seen is None or seen[0] != code or now - seen[1] >= reminder:
                self._deliberate_seen = (code, now)
                self._page("SYSTEM", f"Beast exited deliberately (code {code}); not restarting. "
                                     f"Operator action needed.", critical=True)
            return f"none: deliberate exit {code}"
        self._deliberate_seen = None
        if self.crash_loop_tripped:
            return "none: crash-loop guard"

        # A slow cycle is not a dead process: confirm on the next pass.
        if not self._stale_seen_once:
            self._stale_seen_once = True
            return "recheck"
        self._stale_seen_once = False

        if self._crash_looping():
            self.crash_loop_tripped = True
            self._page("SYSTEM",
                       f"CRASH LOOP: {len(self.restarts)} restarts in "
                       f"{self._s('restart_window_minutes', 30)} min. Watchdog has STOPPED "
                       f"restarting Beast. {self._position_note(beat)}", critical=True)
            return "none: crash-loop guard tripped"

        ok = self.systemctl("restart", unit)
        self.restarts.append(self.now())
        self._page("SYSTEM",
                   f"Beast heartbeat {verdict}; restart {'issued' if ok else 'FAILED'} "
                   f"({len(self.restarts)}/{self._s('max_restarts', 3)} in window). "
                   f"{self._position_note(beat)}",
                   critical=bool(beat and beat.open_positions) or not ok)
        return "restarted" if ok else "restart failed"

    # -- helpers -------------------------------------------------------------

    def _crash_looping(self) -> bool:
        window = timedelta(minutes=float(self._s("restart_window_minutes", 30)))
        cutoff = self.now() - window
        while self.restarts and self.restarts[0] < cutoff:
            self.restarts.popleft()
        return len(self.restarts) >= int(self._s("max_restarts", 3))

    def _age(self, beat: Heartbeat) -> float:
        try:
            stamp = datetime.fromisoformat(beat.written_at)
        except (TypeError, ValueError):
            return float("inf")
        now = self.now()
        # v2 stamps are UTC with tzinfo; a v1 or test stamp is naive. Compare
        # like with like rather than raising on the mix.
        if stamp.tzinfo is not None and now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        elif stamp.tzinfo is None and now.tzinfo is not None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return (now - stamp).total_seconds()

    @staticmethod
    def _position_note(beat: Heartbeat | None) -> str:
        if beat is None or not beat.open_positions:
            return "No open position in the last heartbeat."
        parts = []
        for p in beat.open_positions:
            parts.append(f"{p.get('market')} {p.get('direction')} {p.get('volume')} "
                         f"stop={'PRESENT' if p.get('sl_present') else 'MISSING'}")
        return ("OPEN POSITION running on its venue-side stop alone until the reconcile "
                "runs: " + "; ".join(parts))

    def _page(self, market: str, message: str, critical: bool) -> None:
        try:
            from monitoring.alerts import AlertKind
            kind = AlertKind.CIRCUIT_BREAKER if critical else AlertKind.ERROR
            self.alerts.send(kind, market, f"[watchdog] {message}")
        except Exception:
            self.alerts.send("WATCHDOG", market, message)

    def _terminal_ok(self) -> bool | None:
        """Is the terminal process alive and the bridge able to see its module?

        ``terminal_info()`` only answers inside an ``initialize()`` session,
        and opening one would give the watchdog a broker session - the one
        thing it must never hold. So this checks what can be checked without
        one: the terminal process exists, and the Wine-side Python can import
        ``MetaTrader5`` over the bridge. Whether the session works is Beast's
        handshake's job, and its heartbeat carries ``mt5_state`` for that.
        Returns None when the bridge itself is down.
        """
        alive = _process_alive("terminal64")
        try:
            import rpyc
            conn = rpyc.classic.connect(
                str(self.cfg.get("broker.mt5.profiles.gold.host", "127.0.0.1")),
                int(self.cfg.get("broker.mt5.profiles.gold.port", 18812)))
            conn._config["sync_request_timeout"] = 10
            try:
                conn.modules.MetaTrader5.__version__
            finally:
                conn.close()
        except Exception:
            return None
        return alive

    # -- loop ----------------------------------------------------------------

    def run(self, once: bool = False) -> int:
        interval = float(self._s("watchdog_interval_seconds", 15))
        while True:
            report = self.check()
            print(f"{report['at']}  beast={report['beast']}  bridge={report['bridge']}  "
                  f"terminal={report['terminal']}  dashboard={report['dashboard']}  "
                  f"action={report['action']}", flush=True)
            if once:
                healthy = report["beast"].startswith("alive") and report["bridge"] and \
                    report["terminal"] and report["dashboard"]
                return 0 if healthy else 1
            time.sleep(interval)


# -- process-level helpers -----------------------------------------------------


def _systemctl(action: str, unit: str) -> bool:
    try:
        return subprocess.run(["systemctl", action, unit], timeout=60,
                              capture_output=True).returncode == 0
    except Exception:
        return False


def _unit_exit_status(unit: str) -> int | None:
    """The unit's last exit code, or None while it is running.

    A failed ``ExecStartPre`` (preflight) never runs the main process, so
    ``ExecMainStatus`` stays 0 while ``Result`` says ``exit-code``. That is a
    startup failure, code 1 - not a clean exit - and is reported as such.
    """
    try:
        out = subprocess.run(["systemctl", "show", unit, "-p", "ExecMainStatus",
                              "-p", "ActiveState", "-p", "Result"],
                             timeout=10, capture_output=True, text=True).stdout
        props = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
        if props.get("ActiveState") == "active":
            return None                      # still running: no exit code yet
        code = int(props.get("ExecMainStatus", "") or 0)
        if code == 0 and props.get("Result") in ("exit-code", "start-limit-hit"):
            return 1                         # control process (preflight) blocked the start
        return code
    except Exception:
        return None


def _process_alive(name: str) -> bool:
    try:
        return subprocess.run(["pgrep", "-f", name], capture_output=True,
                              timeout=5).returncode == 0
    except Exception:
        return False


def _tcp_ok(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=3):
            return True
    except OSError:
        return False


def _http_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return 200 <= response.status < 300
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Beast watchdog")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    from core.config import get_config
    from monitoring.alerts import AlertManager
    cfg = get_config()
    return Watchdog(cfg, AlertManager(cfg)).run(once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())
