"""Which layer of the MT5 chain is down? Read-only, safe to run any time.

    python scripts/diagnose_mt5.py            # one PASS/FAIL line per layer
    python scripts/diagnose_mt5.py --json     # the same as a JSON document

Walks the chain from the outside in - supervisor units, processes, the RPyC
port, the bridge module, ``initialize``, ``terminal_info``, ``account_info``,
the symbol candidates, the tick stream - and stops describing at the first
layer that fails, because everything past it will fail for the same reason.
The verdict is one reason code from the table below plus the command that
fixes it. ``last_error()`` is printed verbatim after every failure.

Reason codes (``ops.mt5_reason``):

    BRIDGE_DOWN          mt5linux/RPyC server unreachable      retry with backoff
    TERMINAL_DOWN        Wine MT5 process not running/IPC dead alert; cannot self-heal
    NOT_LOGGED_IN        login refused or account_info None    alert; credentials
    NO_TRADE_PERMISSION  terminal_info().trade_allowed false   alert; Algo Trading toggle
    SYMBOL_NOT_FOUND     no candidate resolves                 alert with the list tried
    SYMBOL_NOT_SELECTED  exists, not in Market Watch           self-heals: symbol_select
    NO_TICKS             connected, tick stream stale          wait; escalate if open

The one thing it writes is ``symbol_select(name, True)`` when a candidate exists
but is not in Market Watch - that is the self-heal for SYMBOL_NOT_SELECTED and
it is the fix, not a side effect. Nothing here places, modifies or cancels an
order, and the password is never printed.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import get_config  # noqa: E402

DEFAULT_CANDIDATES = ["XAUUSD", "XAUUSD.m", "XAUUSDm", "XAUUSD.raw", "GOLD", "GOLD.m",
                      "XAUUSD_i", "XAUUSD-ECN"]

REMEDIATION = {
    "BRIDGE_DOWN": "systemctl restart beast-rpyc; then re-run scripts/diagnose_mt5.py",
    "TERMINAL_DOWN": "systemctl restart beast-mt5 (Wine terminal); see docs/vps-ops.md",
    "NOT_LOGGED_IN": "the account was refused: open a new demo account in the terminal, "
                     "update MT5_GOLD_LOGIN/_SERVER (and the password) in /home/beast-agent/.env, "
                     "restart beast-mt5",
    "NO_TRADE_PERMISSION": "enable Algo Trading in the terminal (Ctrl+E) or on the account",
    "SYMBOL_NOT_FOUND": "set broker.mt5.profiles.gold.symbols.XAUUSD to a name from the list above",
    "SYMBOL_NOT_SELECTED": "already healed by this run (symbol_select); re-run to confirm",
    "NO_TICKS": "wait if the market is closed; if open, check the terminal's connection",
    "TERMINAL_NOT_CONNECTED": "the terminal is running but has no link to the trade server: "
                              "check the VPS network and the broker server name",
}

#: initialize() last_error codes -> layer.
INIT_ERRORS = {
    -6: "NOT_LOGGED_IN",           # Terminal: Authorization failed
    -10005: "TERMINAL_DOWN",       # IPC timeout: terminal never became ready
    -10004: "TERMINAL_DOWN",       # no IPC connection
    -10003: "TERMINAL_DOWN",       # IPC initialisation failed / wrong path
}


class Report:
    def __init__(self, as_json: bool) -> None:
        self.as_json = as_json
        self.rows: list[dict] = []
        self.verdict: str | None = None

    def add(self, ok: bool | None, name: str, detail: str = "", code: str | None = None) -> None:
        mark = {True: "PASS", False: "FAIL", None: "SKIP"}[ok]
        self.rows.append({"check": name, "result": mark, "detail": detail, "code": code})
        if ok is False and self.verdict is None:
            self.verdict = code or "UNKNOWN"
        if not self.as_json:
            tail = f"  [{code}]" if (ok is False and code) else ""
            print(f"  {mark}  {name:<34s} {detail}{tail}")

    def finish(self) -> int:
        remedy = REMEDIATION.get(self.verdict or "", "")
        if self.as_json:
            print(json.dumps({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                              "verdict": self.verdict or "ALL_PASS", "remediation": remedy,
                              "checks": self.rows}, indent=2))
        else:
            passed = sum(1 for r in self.rows if r["result"] == "PASS")
            total = sum(1 for r in self.rows if r["result"] != "SKIP")
            print(f"\n{passed} of {total} checks passed")
            if self.verdict:
                print(f"verdict: {self.verdict}")
                print(f"fix:     {remedy}")
            else:
                print("verdict: ALL_PASS - the chain is up end to end")
        return 0 if self.verdict is None else 1


def mask(value) -> str:
    text = str(value)
    return text if len(text) <= 4 else "*" * (len(text) - 4) + text[-4:]


def unit_state(unit: str) -> str | None:
    if not unit or platform.system() == "Windows":
        return None
    try:
        out = subprocess.run(["systemctl", "is-active", unit], capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() or out.stderr.strip()
    except Exception as error:
        return f"unknown ({error})"


def process_alive(pattern: str) -> int | None:
    """Count of matching processes, or None when it cannot be determined."""
    try:
        if platform.system() == "Windows":
            out = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {pattern}", "/NH"],
                                 capture_output=True, text=True, timeout=10).stdout
            return sum(1 for line in out.splitlines() if pattern.lower() in line.lower())
        out = subprocess.run(["pgrep", "-fc", pattern], capture_output=True, text=True,
                             timeout=10).stdout.strip()
        return int(out or 0)
    except Exception:
        return None


def port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def last_error(mt5) -> str:
    try:
        code, text = mt5.last_error()
        return f"last_error() = ({int(code)}, {str(text)!r})"
    except Exception as error:
        return f"last_error() unavailable: {error}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose Beast's MT5 chain, layer by layer.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--candidates", nargs="*", default=None,
                        help="symbol names to probe (default: the common gold spellings)")
    parser.add_argument("--tick-stale-seconds", type=float, default=120.0)
    args = parser.parse_args(argv)
    report = Report(args.json)

    cfg = get_config()
    profile = str(cfg.get("broker.mt5.active_profile", "gold"))
    host = str(cfg.get(f"broker.mt5.profiles.{profile}.host", "127.0.0.1"))
    port = int(cfg.get(f"broker.mt5.profiles.{profile}.port", 18812))
    prefix = str(cfg.get(f"broker.mt5.profiles.{profile}.env_prefix", f"MT5_{profile.upper()}"))
    explicit = cfg.get("broker.mt5.bridge.enabled", None)
    bridged = bool(explicit) if explicit is not None else platform.system() != "Windows"
    configured = str(cfg.get(f"broker.mt5.profiles.{profile}.symbols.XAUUSD", "") or "")
    candidates = list(args.candidates) if args.candidates else list(DEFAULT_CANDIDATES)
    if configured and configured not in candidates:
        candidates.insert(0, configured)
    server_unit = str(cfg.get("broker.mt5.units.server", "") or "beast-rpyc")
    terminal_unit = str(cfg.get("broker.mt5.units.terminal", "") or "beast-mt5")

    if not args.json:
        print(f"Beast - MT5 chain diagnostic  ({datetime.now().isoformat(timespec='seconds')} host time)")
        print(f"  transport: {'RPyC bridge to Wine' if bridged else 'native MetaTrader5 package'} "
              f"at {host}:{port} | profile {profile} | env prefix {prefix}")
        present = [k for k in ("LOGIN", "PASSWORD", "SERVER", "TERMINAL_PATH", "PORTABLE")
                   if os.environ.get(f"{prefix}_{k}")]
        login_env = os.environ.get(f"{prefix}_LOGIN", "")
        print(f"  env: {', '.join(f'{prefix}_{k}' for k in present) or 'none'}"
              f" | login {mask(login_env) if login_env else '-'}"
              f" | server {os.environ.get(f'{prefix}_SERVER', '-')}\n")

    # -- 1. server process and port -----------------------------------------
    state = unit_state(server_unit)
    if bridged:
        listening = port_open(host, port)
        detail = f"unit {server_unit}={state or 'n/a'}; {host}:{port} {'listening' if listening else 'NOT listening'}"
        report.add(listening, "mt5linux/RPyC server", detail, "BRIDGE_DOWN")
    else:
        report.add(None, "mt5linux/RPyC server", "native transport on Windows - no bridge")

    # -- 2. terminal process -------------------------------------------------
    count = process_alive("terminal64.exe" if platform.system() == "Windows" else "terminal64")
    tstate = unit_state(terminal_unit)
    detail = f"unit {terminal_unit}={tstate or 'n/a'}; {count if count is not None else '?'} terminal64 process(es)"
    report.add(None if count is None else count > 0, "Wine MT5 terminal process", detail,
               "TERMINAL_DOWN")
    if report.verdict and bridged:
        return report.finish()

    # -- 3. bridge client ------------------------------------------------------
    mt5 = None
    conn = None
    try:
        if bridged:
            import rpyc
            conn = rpyc.classic.connect(host, port)
            conn._config["sync_request_timeout"] = 60
            mt5 = conn.modules.MetaTrader5
        else:
            import MetaTrader5 as mt5  # type: ignore
        version = getattr(mt5, "__version__", "?")
        report.add(True, "Python client reaches MetaTrader5 module", f"package {version}")
    except Exception as error:
        report.add(False, "Python client reaches MetaTrader5 module", str(error)[:200],
                   "BRIDGE_DOWN")
        return report.finish()

    try:
        # -- 4. initialize + terminal_info -----------------------------------
        raw_login = os.environ.get(f"{prefix}_LOGIN", "")
        kwargs = {"timeout": int(cfg.get("broker.mt5.timeouts.connect_seconds", 30)) * 1000}
        missing = [k for k in ("LOGIN", "PASSWORD", "SERVER") if not os.environ.get(f"{prefix}_{k}")]
        if missing:
            report.add(False, "initialize() with account", f"missing env: {', '.join(missing)}",
                       "NOT_LOGGED_IN")
            return report.finish()
        try:
            kwargs["login"] = int(raw_login.strip())
        except ValueError:
            report.add(False, "initialize() with account", f"{prefix}_LOGIN is not numeric",
                       "NOT_LOGGED_IN")
            return report.finish()
        kwargs["password"] = os.environ[f"{prefix}_PASSWORD"]
        kwargs["server"] = os.environ[f"{prefix}_SERVER"]
        if os.environ.get(f"{prefix}_TERMINAL_PATH"):
            kwargs["path"] = os.environ[f"{prefix}_TERMINAL_PATH"]
        if os.environ.get(f"{prefix}_PORTABLE", "").strip().lower() in ("1", "true", "yes"):
            kwargs["portable"] = True

        started = time.monotonic()
        ok = bool(mt5.initialize(**kwargs))
        elapsed = time.monotonic() - started
        if not ok:
            err = last_error(mt5)
            code = None
            try:
                code = int(mt5.last_error()[0])
            except Exception:
                pass
            report.add(False, "initialize() with account",
                       f"{err} after {elapsed:.1f}s (login {mask(kwargs['login'])} on "
                       f"{kwargs['server']})", INIT_ERRORS.get(code, "TERMINAL_DOWN"))
            return report.finish()

        terminal = mt5.terminal_info()
        if terminal is None:
            report.add(False, "terminal_info()", last_error(mt5), "TERMINAL_DOWN")
            return report.finish()
        connected = bool(getattr(terminal, "connected", False))
        trade_allowed = bool(getattr(terminal, "trade_allowed", False))
        report.add(True, "initialize() with account",
                   f"{elapsed:.1f}s; build {getattr(terminal, 'build', '?')}, "
                   f"{getattr(terminal, 'company', '')}")
        report.add(connected, "terminal_info().connected",
                   "terminal has a link to the trade server" if connected
                   else "terminal is NOT connected to the trade server",
                   "TERMINAL_NOT_CONNECTED")
        report.add(trade_allowed, "terminal_info().trade_allowed",
                   "Algo Trading enabled" if trade_allowed else "Algo Trading is OFF",
                   "NO_TRADE_PERMISSION")

        # -- 5. account_info -------------------------------------------------
        account = mt5.account_info()
        if account is None:
            report.add(False, "account_info()", f"None - not logged in; {last_error(mt5)}",
                       "NOT_LOGGED_IN")
            return report.finish()
        demo = int(getattr(account, "trade_mode", -1)) == int(getattr(mt5, "ACCOUNT_TRADE_MODE_DEMO", 0))
        report.add(True, "account_info()",
                   f"login {mask(getattr(account, 'login', ''))} on {getattr(account, 'server', '')} "
                   f"({getattr(account, 'company', '')}), {getattr(account, 'currency', '')}, "
                   f"{'DEMO' if demo else 'REAL'}, trade_allowed={bool(getattr(account, 'trade_allowed', False))}, "
                   f"trade_expert={bool(getattr(account, 'trade_expert', False))}")

        # -- 6. symbols ------------------------------------------------------
        found, healed, resolved = [], [], None
        for name in candidates:
            info = mt5.symbol_info(name)
            if info is None:
                continue
            visible = bool(getattr(info, "visible", False))
            if not visible:
                if mt5.symbol_select(name, True):
                    healed.append(name)
                    visible = True
            found.append(f"{name}{'' if visible else ' (not selectable)'}")
            if resolved is None and visible:
                resolved = name
        try:
            listed = sorted({str(getattr(s, "name", "")) for grp in ("*XAU*", "*GOLD*")
                             for s in (mt5.symbols_get(group=grp) or ())})
        except Exception:
            listed = []
        if resolved is None:
            report.add(False, "gold symbol candidates",
                       f"none of {candidates} resolve; symbols_get(*XAU*/*GOLD*) = {listed[:12]}",
                       "SYMBOL_NOT_FOUND")
            return report.finish()
        report.add(True, "gold symbol candidates",
                   f"resolved {resolved}; found {found}; account lists {listed[:12]}"
                   + (f"; selected into Market Watch: {healed}" if healed else ""))
        if healed:
            report.add(True, "SYMBOL_NOT_SELECTED self-heal",
                       f"symbol_select({healed[0]}, True) succeeded - the earlier symptom is fixed")

        info = mt5.symbol_info(resolved)
        report.add(True, f"symbol_info({resolved})",
                   f"contract_size {getattr(info, 'trade_contract_size', '?')}, "
                   f"tick {getattr(info, 'trade_tick_size', '?')}/{getattr(info, 'trade_tick_value', '?')}, "
                   f"point {getattr(info, 'point', '?')}, digits {getattr(info, 'digits', '?')}, "
                   f"lots {getattr(info, 'volume_min', '?')}-{getattr(info, 'volume_max', '?')} "
                   f"step {getattr(info, 'volume_step', '?')}, stops_level {getattr(info, 'trade_stops_level', '?')}, "
                   f"trade_mode {getattr(info, 'trade_mode', '?')}")

        # -- 7. tick ---------------------------------------------------------
        tick = mt5.symbol_info_tick(resolved)
        if tick is None:
            report.add(False, f"symbol_info_tick({resolved})", last_error(mt5), "NO_TICKS")
            return report.finish()
        tick_time = float(getattr(tick, "time", 0) or 0)
        age = time.time() - tick_time
        try:
            from core.session import SessionClock
            session_open = SessionClock("XAUUSD", cfg).is_open(datetime.now())
        except Exception:
            session_open = None
        stale = age > args.tick_stale_seconds
        detail = (f"bid {getattr(tick, 'bid', '?')} ask {getattr(tick, 'ask', '?')}, "
                  f"stamped {datetime.fromtimestamp(tick_time, timezone.utc).isoformat(timespec='seconds')}, "
                  f"~{age:.0f}s old by host clock (server offset not removed); "
                  f"config gold session {'OPEN' if session_open else 'closed' if session_open is False else '?'}")
        if stale and session_open:
            report.add(False, f"symbol_info_tick({resolved})", detail, "NO_TICKS")
        elif stale:
            report.add(None, f"symbol_info_tick({resolved})", detail + " - stale but session closed")
        else:
            report.add(True, f"symbol_info_tick({resolved})", detail)
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
    return report.finish()


if __name__ == "__main__":
    raise SystemExit(main())
