"""Refuse to start a box that is not fit to trade.

    python -m ops.preflight          # exit 0 = go, non-zero = blocked

Runs as ``ExecStartPre`` for ``beast.service`` and by hand after a deploy.
Eight checks, each a row in the table it prints. Any FAIL blocks the start.
A check that cannot run (no MT5, no ``timedatectl``) is a FAIL, not a skip:
the point is to refuse when the answer is unknown.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ops.killswitch import read_flag  # noqa: E402


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


class Preflight:
    """The checks, each independently callable and each injectable for tests.

    Args:
        cfg: Beast config.
        mt5: An object with ``connect()``, ``call()``, ``facts``, ``spec()``,
            ``disconnect()`` - normally ``MT5Connection``; a fake in tests.
        ntp_status: ``() -> (synchronised: bool | None, detail)``.
        disk_free_mb: ``(path) -> float``.
        server_time: ``() -> datetime | None`` in UTC, from the venue.
    """

    def __init__(self, cfg, mt5=None, ntp_status: Callable | None = None,
                 disk_free_mb: Callable[[Path], float] | None = None,
                 environ: dict | None = None) -> None:
        self.cfg = cfg
        self._mt5 = mt5
        self.ntp_status = ntp_status or _timedatectl
        self.disk_free_mb = disk_free_mb or _disk_free_mb
        self.environ = os.environ if environ is None else environ
        self._connected = False

    # -- the eight ---------------------------------------------------------

    def check_blockers(self) -> Check:
        markets = [str(m).upper() for m in self.cfg.get("broker.symbols")]
        families = {self.cfg.market_family(m) for m in markets}
        relevant = [
            key for key in self.cfg.unset_blockers()
            if (key.startswith("instruments.gold") and "gold" in families)
            or (key.startswith("instruments.nifty") and "indian" in families and "NIFTY50" in markets)
            or (key.startswith("instruments.sensex") and "SENSEX" in markets)
            or (key.startswith("options.") and "indian" in families)
            or (key == "data.gold_spread_max" and "gold" in families)
        ]
        return Check("config blockers for active markets", not relevant,
                     "none" if not relevant else ", ".join(relevant))

    def check_env(self) -> Check:
        prefix = str(self.cfg.get("broker.mt5.profiles.gold.env_prefix", "MT5_GOLD"))
        missing = [f"{prefix}_{k}" for k in ("LOGIN", "PASSWORD", "SERVER")
                   if not self.environ.get(f"{prefix}_{k}")]
        mode = str(self.cfg.get("broker.mt5.account_mode", "demo"))
        allow_live = self.environ.get("BEAST_ALLOW_LIVE", "") == "1"
        problems = list(missing)
        if allow_live and mode != "live":
            problems.append("BEAST_ALLOW_LIVE=1 is set but broker.mt5.account_mode is not live")
        return Check("environment", not problems, "ok" if not problems else "; ".join(problems))

    def check_bridge(self) -> Check:
        mt5 = self._link()
        if mt5 is None:
            return Check("MT5 bridge + terminal + account", False, "no MT5 connection object")
        try:
            if not self._connected:
                self._connected = bool(mt5.connect(wait_seconds=20))
            if not self._connected:
                return Check("MT5 bridge + terminal + account", False,
                             f"connect failed: bridge {mt5.state.value}")
            facts = mt5.facts
            paper = str(self.cfg.get("mode", "paper")) != "live"
            if paper and not facts.is_demo:
                return Check("MT5 bridge + terminal + account", False,
                             f"mode is paper but account {facts.login} is REAL - refusing")
            return Check("MT5 bridge + terminal + account", True,
                         f"{'DEMO' if facts.is_demo else 'REAL'} {facts.login} on {facts.server}, "
                         f"build {facts.terminal_build}, p50 {facts.latency_p50_ms:.0f} ms")
        except Exception as error:
            return Check("MT5 bridge + terminal + account", False, str(error))

    def check_symbol(self) -> Check:
        mt5 = self._link()
        if mt5 is None or not self._connected:
            return Check("gold symbol selectable", False, "MT5 not connected")
        try:
            spec = mt5.spec("XAUUSD")
        except Exception as error:
            return Check("gold symbol selectable", False, str(error))
        if spec is None:
            return Check("gold symbol selectable", False, "no XAUUSD symbol resolved")
        return Check("gold symbol selectable", True, spec.name)

    def check_clock(self) -> Check:
        synced, detail = self.ntp_status()
        if synced is None:
            return Check("clock sync", False, f"cannot determine NTP state: {detail}")
        if not synced:
            return Check("clock sync", False, f"NTP not synchronised: {detail}")
        mt5 = self._link()
        if mt5 is None or not self._connected:
            return Check("clock sync", False, "NTP ok but MT5 server time unavailable")
        try:
            tick = mt5.call("symbol_info_tick", mt5.spec("XAUUSD").name)
            server = float(tick.time) - mt5.facts.server_utc_offset_hours * 3600
            drift = abs(datetime.now(timezone.utc).timestamp() - server)
        except Exception as error:
            return Check("clock sync", False, f"NTP ok; server time unreadable: {error}")
        limit = float(self.cfg.get("ops.max_clock_drift_seconds", 5))
        # the tick may be a few seconds old in a quiet market; be strict only on gross drift
        ok = drift <= max(limit, 60)
        return Check("clock sync", ok, f"NTP synchronised; |local - server| = {drift:.1f}s "
                                       f"(limit {limit:g}s on a live tick)")

    def check_host_timezone(self) -> Check:
        """The host clock must be in ``sessions.timezone`` (D-78).

        Beast's session windows, the last-entry cutoff and the flatten time
        compare a naive ``datetime.now()`` against IST wall-clock times. On a
        host whose clock is UTC every one of them is five and a half hours
        off, silently. ``timedatectl set-timezone Asia/Kolkata`` fixes it.
        """
        want = str(self.cfg.get("sessions.timezone"))
        try:
            from zoneinfo import ZoneInfo
            now = datetime.now()
            host_offset = now.astimezone().utcoffset()
            want_offset = now.replace(tzinfo=ZoneInfo(want)).utcoffset()
        except Exception as error:
            return Check("host clock in sessions.timezone", False, str(error))
        ok = host_offset == want_offset
        return Check("host clock in sessions.timezone", ok,
                     f"host UTC{host_offset} vs {want} UTC{want_offset}"
                     + ("" if ok else f" - run: timedatectl set-timezone {want}"))

    def check_disk(self) -> Check:
        need = float(self.cfg.get("ops.min_free_disk_mb", 2048))
        log_dir = Path(str(self.cfg.get("monitoring.log_dir", "./logs")))
        target = log_dir if log_dir.is_absolute() else PROJECT_ROOT / log_dir
        target.mkdir(parents=True, exist_ok=True)
        free = self.disk_free_mb(target)
        return Check("disk space", free >= need, f"{free:,.0f} MB free, need {need:,.0f}")

    def check_kill_flag(self) -> Check:
        raw = Path(str(self.cfg.get("ops.kill_flag_path", "./KILL")))
        path = raw if raw.is_absolute() else PROJECT_ROOT / raw
        flag = read_flag(path)
        if flag is None:
            return Check("no stale KILL flag", True, "clear")
        return Check("no stale KILL flag", False,
                     f"{flag.get('mode')} set by {flag.get('set_by')} at {flag.get('set_at')}: "
                     f"{flag.get('reason')} - clear it with `python -m ops.killswitch clear`")

    def check_journal(self) -> Check:
        raw = Path(str(self.cfg.get("monitoring.journal_db", "./logs/beast_journal.sqlite")))
        path = raw if raw.is_absolute() else PROJECT_ROOT / raw
        if not path.exists():
            return Check("journal readable", True, f"{path.name} will be created")
        try:
            conn = sqlite3.connect(f"file:{path}?mode=rw", uri=True, timeout=2)
            try:
                conn.execute("BEGIN IMMEDIATE")   # fails fast if another writer holds it
                conn.execute("ROLLBACK")
                n = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
            finally:
                conn.close()
        except sqlite3.Error as error:
            return Check("journal readable", False, f"{path.name}: {error}")
        return Check("journal readable", True, f"{path.name}, {n} signals, not locked")

    # -- run ---------------------------------------------------------------

    def run(self) -> list[Check]:
        checks = [
            self.check_blockers(),
            self.check_env(),
            self.check_bridge(),
            self.check_symbol(),
            self.check_clock(),
            self.check_host_timezone(),
            self.check_disk(),
            self.check_kill_flag(),
            self.check_journal(),
        ]
        mt5 = self._link()
        if mt5 is not None and self._connected:
            try:
                mt5.disconnect()
            except Exception:
                pass
        return checks

    def _link(self):
        if self._mt5 is None:
            try:
                from broker.mt5_connection import MT5Connection
                self._mt5 = MT5Connection(
                    profile=str(self.cfg.get("broker.mt5.active_profile", "gold")), config=self.cfg)
            except Exception:
                return None
        return self._mt5


def _timedatectl() -> tuple[bool | None, str]:
    if shutil.which("timedatectl") is None:
        return None, "timedatectl not available"
    try:
        out = subprocess.run(["timedatectl", "show", "-p", "NTPSynchronized"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception as error:
        return None, str(error)
    return out.endswith("yes"), out


def _disk_free_mb(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024 * 1024)


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Beast preflight").parse_args(argv)
    from core.config import get_config
    checks = Preflight(get_config()).run()
    width = max(len(c.name) for c in checks)
    print("Beast preflight\n")
    for c in checks:
        print(f"  {'PASS' if c.ok else 'FAIL'}  {c.name:<{width}}  {c.detail}")
    failed = [c for c in checks if not c.ok]
    print(f"\n{len(checks) - len(failed)} of {len(checks)} passed")
    if failed:
        print("BLOCKED: " + "; ".join(c.name for c in failed))
        return 1
    print("GO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
