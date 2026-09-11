"""The only module that owns an MT5 client. Everything else goes through it.

Topology on the VPS::

    MT5 terminal (Wine) <-> mt5linux server (Wine side) <-> RPyC on 127.0.0.1
                                                        <-> Beast (Linux Python)

Four links, each of which fails differently, and a naive adapter reports all four
as "no data". This module exists so that a break is named: the RPyC server is
unreachable, the terminal lost its broker connection, Algo Trading is off, or a
call hung inside Wine. An operator woken at 03:00 needs the layer, not a
traceback.

Why one worker thread rather than a lock
----------------------------------------
The MT5 API is not thread-safe and RPyC classic connections are not either, so
calls must be serialised. A plain lock serialises but cannot bound a call: a Wine
hang holds the lock forever and every caller blocks behind it, which is the
silent stall this design is built to prevent. A single worker thread plus
``Future.result(timeout=...)`` gives both properties at once - strict FIFO order,
and a caller that always gets control back.

A timed-out call is still running inside that worker. The thread is therefore
treated as poisoned: it is abandoned (it is a daemon, so it cannot keep the
process alive) and :meth:`connect` builds a fresh one. Reusing a worker whose
queue sits behind a hung Wine call would time out every subsequent request and
look like a dead broker.

Netrefs
-------
Across the bridge, results are RPyC netrefs: every attribute read is another
round trip over the socket. ``account_info().equity`` inside a loop is a network
call nobody wrote. Results are therefore materialised at this boundary, once,
before anything downstream touches them.

Credentials
-----------
Environment only - ``MT5_GOLD_LOGIN``, ``MT5_GOLD_PASSWORD``, ``MT5_GOLD_SERVER``
for the ``gold`` profile. ``core.config`` loads ``.env`` into the environment at
import, so a ``.env`` file still works. The password is never logged, never put
in an exception message, and never echoed back in a spec report.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from core.config import Config, get_config

logger = logging.getLogger("beast.broker.mt5.connection")


class ConnectionState(str, Enum):
    """Where the bridge is. Every transition is logged and emitted."""

    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    READY = "READY"
    DEGRADED = "DEGRADED"
    HALTED = "HALTED"


class BridgeError(Exception):
    """Base for every failure this layer reports. Never raises a bare MT5 None."""


class BridgeUnavailable(BridgeError):
    """The mt5linux server or the terminal could not be reached at all."""


class BridgeMisconfigured(BridgeError):
    """A settings or credentials fault. Retrying cannot fix it, so nothing does.

    Kept apart from :class:`BridgeUnavailable` because the two deserve opposite
    treatment on startup: an unreachable terminal is worth waiting out, since on
    a reboot it may simply be slower than Beast. A missing password will still be
    missing in three minutes, and retrying only delays the alert.
    """


class BridgeTimeout(BridgeError):
    """A call exceeded its budget. Usually a Wine hang, occasionally a slow feed."""


class BridgeCallFailed(BridgeError):
    """MT5 returned None. Carries ``last_error()`` so the layer is identifiable."""


class AccountModeViolation(BridgeError):
    """The logged-in account does not match ``broker.mt5.account_mode``."""


@dataclass
class SymbolSpec:
    """The static facts about a symbol, fetched once per session.

    Cached because none of it changes intraday, and re-reading it per cycle is
    the kind of hidden round trip this module exists to remove.
    """

    name: str
    digits: int
    point: float
    tick_size: float
    tick_value: float
    contract_size: float
    volume_min: float
    volume_max: float
    volume_step: float
    stops_level: int
    freeze_level: int
    filling_mask: int
    trade_mode: int

    def report(self) -> str:
        modes = [label for bit, label in ((1, "FOK"), (2, "IOC"), (4, "BOC"))
                 if self.filling_mask & bit]
        return (
            f"{self.name}: digits={self.digits} point={self.point} "
            f"tick={self.tick_size}/{self.tick_value} contract={self.contract_size} "
            f"lots={self.volume_min}-{self.volume_max} step={self.volume_step} "
            f"stops_level={self.stops_level} freeze_level={self.freeze_level} "
            f"filling={'|'.join(modes) or 'none'}"
        )


@dataclass
class SessionFacts:
    """What the handshake established. Rebuilt on every reconnect."""

    login: int = 0
    server: str = ""
    currency: str = ""
    is_demo: bool = False
    margin_mode: int = -1
    hedging: bool = False
    server_utc_offset_hours: float = 0.0
    terminal_build: int = 0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    specs: dict[str, SymbolSpec] = field(default_factory=dict)


class MT5Connection:
    """Owns the mt5linux/MetaTrader5 client for one terminal profile.

    Args:
        profile: Key under ``broker.mt5.profiles``. Only ``gold`` is wired today;
            a second terminal is a second profile and no code change.
        config: Injected for tests.
        client: A pre-built client (a fake in tests, or an already-connected
            handle). When supplied, no import and no ``initialize`` happen.
        on_event: Called as ``(state, reason)`` on every transition, so
            ``main.py`` can raise alerts without this module importing the alert
            layer.
    """

    def __init__(self, profile: str = "gold", config: Config | None = None,
                 client: object | None = None,
                 on_event: Callable[[str, str], None] | None = None) -> None:
        self.cfg = config or get_config()
        self.profile = profile
        self.on_event = on_event
        self.state = ConnectionState.DISCONNECTED
        self.facts = SessionFacts()

        self._client = client
        self._injected = client is not None
        self._worker: ThreadPoolExecutor | None = None
        self._worker_poisoned = False
        self._state_lock = threading.Lock()
        self._consecutive_failures = 0

    # -- config helpers ------------------------------------------------------

    def _setting(self, key: str, default: Any = None) -> Any:
        return self.cfg.get(f"broker.mt5.{key}", default)

    def _profile_setting(self, key: str, default: Any = None) -> Any:
        return self.cfg.get(f"broker.mt5.profiles.{self.profile}.{key}", default)

    def _timeout(self, call_class: str) -> float:
        return float(self._setting(f"timeouts.{call_class}_seconds", 10))

    # -- state ---------------------------------------------------------------

    def _set_state(self, state: ConnectionState, reason: str) -> None:
        """Move to ``state`` and announce it. Repeat states are not re-announced."""
        with self._state_lock:
            if self.state is state:
                return
            previous, self.state = self.state, state

        level = logger.error if state in (ConnectionState.HALTED,
                                          ConnectionState.DEGRADED) else logger.info
        level("MT5 bridge %s -> %s: %s", previous.value, state.value, reason)
        if self.on_event is not None:
            try:
                self.on_event(state.value, reason)
            except Exception as error:
                logger.error("MT5 state event handler raised: %s", error)

    def _record_success(self) -> None:
        self._consecutive_failures = 0
        if self.state is ConnectionState.DEGRADED:
            self._set_state(ConnectionState.READY, "call succeeded after a failure")

    def _record_failure(self, reason: str) -> None:
        """Escalate DEGRADED then HALTED. Halting blocks entries, never exits."""
        self._consecutive_failures += 1
        limit = int(self._setting("health.failures_before_halt", 3))
        if self._consecutive_failures >= limit:
            self._set_state(
                ConnectionState.HALTED,
                f"{self._consecutive_failures} consecutive failures: {reason}",
            )
        else:
            self._set_state(ConnectionState.DEGRADED, reason)

    @property
    def is_ready(self) -> bool:
        return self.state is ConnectionState.READY

    # -- the call boundary ---------------------------------------------------

    def call(self, method: str, *args, call_class: str = "fast",
             allow_none: bool = False, **kwargs) -> Any:
        """Invoke one MT5 function, serialised, bounded and materialised.

        Args:
            method: Name of the function on the MT5 client.
            call_class: Timeout budget - ``fast`` for ticks and info calls,
                ``history`` for bar fetches, ``connect`` for the handshake.
            allow_none: True when None is a legitimate answer (an empty
                ``positions_get`` is not a failure). Otherwise None is turned
                into :class:`BridgeCallFailed` carrying ``last_error()``, because
                a None that reaches Beast becomes an empty frame and then a
                silent no-trade.

        Raises:
            BridgeUnavailable: No client, or the worker is gone.
            BridgeTimeout: The call outlived its budget - the worker is poisoned
                and a reconnect is required.
            BridgeCallFailed: MT5 answered None, or the call raised.
        """
        if self._client is None:
            raise BridgeUnavailable(f"MT5 client for profile {self.profile} is not connected")

        # A previous call timed out and its thread is still stuck inside Wine.
        # Rebuild before submitting, or every later call - including the
        # reconciliation that decides whether a timed-out order actually landed -
        # would queue behind the hung one and time out too.
        if self._worker is None or self._worker_poisoned:
            self._start_worker()

        future: Future = self._worker.submit(self._invoke, method, args, kwargs, allow_none)
        budget = self._timeout(call_class)
        started = time.monotonic()
        try:
            result = future.result(timeout=budget)
        except FutureTimeout:
            self._worker_poisoned = True
            reason = f"{method} exceeded {budget:g}s - the terminal or Wine is hung"
            self._record_failure(reason)
            raise BridgeTimeout(reason) from None
        except BridgeError as error:
            self._record_failure(f"{method}: {error}")
            raise
        except Exception as error:
            self._record_failure(f"{method} raised: {error}")
            raise BridgeCallFailed(f"{method} raised: {error}") from error

        self._record_success()
        elapsed = (time.monotonic() - started) * 1000.0
        if elapsed > float(self._setting("health.latency_warn_ms", 750)):
            logger.warning("MT5 %s took %.0fms", method, elapsed)
        return result

    def _invoke(self, method: str, args: tuple, kwargs: dict, allow_none: bool) -> Any:
        """Run inside the single worker thread. Never call this directly."""
        function = getattr(self._client, method, None)
        if function is None:
            raise BridgeCallFailed(f"the MT5 client has no {method}")

        value = function(*args, **kwargs)
        if value is None and not allow_none:
            raise BridgeCallFailed(f"{method} returned None: {self._last_error()}")
        return self._materialise(value)

    def _last_error(self) -> str:
        """Read MT5's error slot. Already inside the worker, so no recursion."""
        try:
            return str(self._client.last_error())
        except Exception:
            return "last_error() unavailable"

    def _materialise(self, value: Any) -> Any:
        """Pull a netref across the wire once, instead of per attribute read."""
        if value is None or not self._is_bridge():
            return value
        try:
            import rpyc
            return rpyc.classic.obtain(value)
        except Exception:
            # Not a netref, or rpyc cannot copy it. The caller still gets a
            # working object; it just costs round trips, which the latency
            # baseline in the handshake will show.
            return value

    def _is_bridge(self) -> bool:
        """Whether to reach MT5 through mt5linux rather than the native package.

        ``broker.mt5.bridge.enabled`` is honoured when set. When it is ``null``
        the platform decides: the native ``MetaTrader5`` package ships only
        ``win_amd64`` wheels, so on anything but Windows the bridge is not a
        preference but the only transport that can exist. Deciding it here
        means one committed config serves both the Windows dev box and the
        Linux VPS.
        """
        explicit = self._setting("bridge.enabled", None)
        if explicit is not None:
            return bool(explicit)
        if self._profile_setting("bridge", None) is not None:
            return bool(self._profile_setting("bridge"))
        return sys.platform != "win32"

    # -- session -------------------------------------------------------------

    def connect(self, wait_seconds: float | None = None) -> bool:
        """Bring the bridge up and run the startup handshake.

        Retries for a bounded window because on a VPS reboot the terminal and
        the mt5linux server may still be starting when Beast is already running.
        Bounded, because a connect that retries forever is indistinguishable
        from a hang.
        """
        self._set_state(ConnectionState.CONNECTING, f"profile {self.profile}")
        deadline = time.monotonic() + float(
            wait_seconds if wait_seconds is not None
            else self._setting("ready_wait_seconds", 180)
        )
        delay = 2.0
        last_reason = "not attempted"

        while True:
            try:
                self._start_worker()
                if not self._injected:
                    self._open_client()
                    self._initialise()
                self.handshake()
                self._set_state(ConnectionState.READY, "handshake complete")
                return True
            except (BridgeMisconfigured, AccountModeViolation) as error:
                # Neither a missing credential nor the wrong kind of account
                # becomes correct by waiting. Fail now, loudly.
                self._set_state(ConnectionState.HALTED, str(error))
                return False
            except BridgeError as error:
                last_reason = str(error)
                logger.warning("MT5 connect attempt failed: %s", last_reason)

            if time.monotonic() >= deadline:
                self._set_state(ConnectionState.HALTED,
                                f"could not connect within the window: {last_reason}")
                return False
            time.sleep(delay)
            delay = min(delay * 2.0, 30.0)

    def disconnect(self) -> None:
        """Release the terminal handle and stop the worker."""
        if self._client is not None and not self._injected:
            try:
                self._client.shutdown()
            except Exception as error:
                logger.warning("MT5 shutdown raised: %s", error)
        self._stop_worker()
        self._set_state(ConnectionState.DISCONNECTED, "disconnected")

    def _start_worker(self) -> None:
        """Create the serialising worker, replacing one poisoned by a timeout."""
        if self._worker is not None and not self._worker_poisoned:
            return
        if self._worker is not None:
            # Abandon it: the hung call still holds the thread. It is a daemon
            # thread, so it cannot keep the interpreter alive.
            self._worker.shutdown(wait=False)
        self._worker = ThreadPoolExecutor(max_workers=1,
                                          thread_name_prefix=f"mt5-{self.profile}")
        self._worker_poisoned = False

    def _stop_worker(self) -> None:
        if self._worker is not None:
            self._worker.shutdown(wait=False)
        self._worker = None
        self._worker_poisoned = False

    def _open_client(self) -> None:
        """Import the bridge or the native package and build the client."""
        if self._is_bridge():
            try:
                from mt5linux import MetaTrader5
            except ImportError as error:
                raise BridgeUnavailable(
                    "mt5linux is not installed on the Linux side"
                ) from error
            host = str(self._profile_setting("host", "127.0.0.1"))
            port = int(self._profile_setting("port", 18812))
            try:
                self._client = MetaTrader5(host=host, port=port)
            except Exception as error:
                raise BridgeUnavailable(
                    f"mt5linux server at {host}:{port} is unreachable - is the "
                    f"unit running? ({error})"
                ) from error
            return

        try:
            import MetaTrader5
        except ImportError as error:
            raise BridgeUnavailable(
                "MetaTrader5 is not installed. It is Windows-only; on Linux "
                "enable broker.mt5.bridge and use mt5linux."
            ) from error
        self._client = MetaTrader5

    def _initialise(self) -> None:
        """Log the terminal in. Credentials are read from the environment only."""
        prefix = str(self._profile_setting("env_prefix", f"MT5_{self.profile.upper()}"))
        raw_login = os.environ.get(f"{prefix}_LOGIN")
        password = os.environ.get(f"{prefix}_PASSWORD")
        server = os.environ.get(f"{prefix}_SERVER")

        missing = [f"{prefix}_{name}" for name, value in
                   (("LOGIN", raw_login), ("PASSWORD", password), ("SERVER", server))
                   if not value]
        if missing:
            raise BridgeMisconfigured(f"missing environment variables: {', '.join(missing)}")

        try:
            login = int(str(raw_login).strip())
        except ValueError:
            raise BridgeMisconfigured(
                f"{prefix}_LOGIN must be the numeric account number"
            ) from None

        kwargs: dict[str, Any] = {
            "login": login,
            "password": str(password),
            "server": str(server),
            "timeout": int(self._setting("timeouts.connect_seconds", 30)) * 1000,
        }
        path = os.environ.get(f"{prefix}_TERMINAL_PATH")
        if path:
            kwargs["path"] = path

        if not self.call("initialize", call_class="connect", **kwargs):
            raise BridgeUnavailable(
                f"initialize failed for account {login} on {server}. The terminal "
                f"must be running and logged in, and the server string must match "
                f"the broker exactly."
            )

    # -- handshake (spec step 2) ---------------------------------------------

    def handshake(self) -> SessionFacts:
        """Verify the whole chain before Beast is allowed to trade.

        Ordered so the cheapest disqualifier fires first, and so that every
        failure names its own layer rather than reporting "no data".
        """
        facts = SessionFacts()

        terminal = self.call("terminal_info", call_class="connect")
        if not bool(getattr(terminal, "connected", False)):
            raise BridgeUnavailable("the terminal is running but has no broker connection")
        if not bool(getattr(terminal, "trade_allowed", False)):
            raise BridgeUnavailable(
                "Algo Trading is disabled in the terminal (Ctrl+E) - order_send "
                "would return retcode 10027"
            )
        facts.terminal_build = int(getattr(terminal, "build", 0) or 0)

        account = self.call("account_info", call_class="connect")
        facts.login = int(getattr(account, "login", 0) or 0)
        facts.server = str(getattr(account, "server", ""))
        facts.currency = str(getattr(account, "currency", ""))
        if not bool(getattr(account, "trade_allowed", False)):
            raise BridgeUnavailable(f"trading is disabled on account {facts.login}")
        if not bool(getattr(account, "trade_expert", False)):
            raise BridgeUnavailable(
                f"expert trading is disabled on account {facts.login} - the broker "
                f"blocks automated orders on this account"
            )

        demo_mode = int(getattr(self._client, "ACCOUNT_TRADE_MODE_DEMO", 0))
        facts.is_demo = int(getattr(account, "trade_mode", -1)) == demo_mode
        self._guard_account_mode(facts)

        facts.margin_mode = int(getattr(account, "margin_mode", -1))
        hedging_mode = int(getattr(self._client, "ACCOUNT_MARGIN_MODE_RETAIL_HEDGING", 2))
        facts.hedging = facts.margin_mode == hedging_mode

        facts.server_utc_offset_hours = self._measure_offset()
        facts.latency_p50_ms, facts.latency_p95_ms = self._measure_latency()

        self.facts = facts
        logger.info(
            "MT5 %s ready: account %s on %s (%s, %s), build %s, server UTC%+g, "
            "latency p50 %.0fms p95 %.0fms",
            self.profile, facts.login, facts.server,
            "DEMO" if facts.is_demo else "REAL",
            "hedging" if facts.hedging else "netting",
            facts.terminal_build, facts.server_utc_offset_hours,
            facts.latency_p50_ms, facts.latency_p95_ms,
        )
        return facts

    def _guard_account_mode(self, facts: SessionFacts) -> None:
        """Refuse to trade an account that is not the configured kind.

        ``live`` needs both the config value and ``BEAST_ALLOW_LIVE=1``, because
        one switch is one typo away from a funded account.
        """
        mode = str(self._setting("account_mode", "demo")).lower()
        if mode == "demo" and not facts.is_demo:
            raise AccountModeViolation(
                f"account_mode is demo but account {facts.login} on {facts.server} "
                f"is not a demo account"
            )
        if mode == "live":
            if facts.is_demo:
                raise AccountModeViolation(
                    f"account_mode is live but account {facts.login} is a demo account"
                )
            if os.environ.get("BEAST_ALLOW_LIVE") != "1":
                raise AccountModeViolation(
                    "account_mode is live but BEAST_ALLOW_LIVE is not set to 1"
                )
        if mode not in ("demo", "live"):
            raise AccountModeViolation(f"broker.mt5.account_mode must be demo or live, got {mode!r}")

    def _measure_offset(self) -> float:
        """Hours the broker's clock runs ahead of real UTC.

        MT5 stamps every bar and tick in server time. Measured rather than
        configured, because a broker that shifts for daylight saving would
        otherwise put every candle in the wrong session for a week.
        """
        symbol = self.resolve_symbol(str(self._setting("offset_probe_symbol", "XAUUSD")))
        if symbol is None:
            return float(self._setting("server_utc_offset_hours", 0))
        tick = self.call("symbol_info_tick", symbol)
        server_epoch = float(getattr(tick, "time", 0) or 0)
        if server_epoch <= 0:
            return float(self._setting("server_utc_offset_hours", 0))
        drift = (server_epoch - datetime.now(timezone.utc).timestamp()) / 3600.0
        return float(round(drift))

    def _measure_latency(self) -> tuple[float, float]:
        """Baseline round-trip on the cheapest call there is."""
        samples: list[float] = []
        for _ in range(int(self._setting("health.latency_samples", 50))):
            started = time.monotonic()
            try:
                self.call("terminal_info")
            except BridgeError:
                break
            samples.append((time.monotonic() - started) * 1000.0)
        if not samples:
            return 0.0, 0.0
        samples.sort()
        p50 = samples[len(samples) // 2]
        p95 = samples[min(len(samples) - 1, int(len(samples) * 0.95))]
        return p50, p95

    # -- symbols -------------------------------------------------------------

    def resolve_symbol(self, market: str) -> str | None:
        """Map a Beast market name onto this broker's symbol, once per session."""
        key = market.upper()
        cached = self.facts.specs.get(key)
        if cached is not None:
            return cached.name

        configured = self._profile_setting(f"symbols.{key}", "")
        candidates = [str(configured)] if configured else [key]
        if not configured:
            candidates.extend(self._search_gold(key))

        for name in candidates:
            spec = self._load_spec(name)
            if spec is not None:
                self.facts.specs[key] = spec
                if name != key:
                    logger.info("MT5 resolved %s to the broker symbol %s", key, name)
                logger.info("MT5 spec %s", spec.report())
                return name
        return None

    def spec(self, market: str) -> SymbolSpec | None:
        """Cached static facts for ``market``, resolving the symbol if needed."""
        key = market.upper()
        if key not in self.facts.specs:
            self.resolve_symbol(key)
        return self.facts.specs.get(key)

    def _search_gold(self, market: str) -> list[str]:
        if "XAU" not in market and "GOLD" not in market:
            return []
        found: list[str] = []
        for group in ("*XAU*", "*GOLD*"):
            try:
                matches = self.call("symbols_get", group=group, allow_none=True) or ()
            except BridgeError:
                continue
            for item in matches:
                name = str(getattr(item, "name", ""))
                if name and name not in found:
                    found.append(name)
        usd = [name for name in found if "USD" in name.upper()]
        return sorted(usd or found, key=lambda name: (len(name), name))

    def _load_spec(self, name: str) -> SymbolSpec | None:
        """Fetch and cache one symbol's static facts, selecting it in Market Watch.

        Probed with ``allow_none``: discovery tries candidate names that are
        *expected* to miss, and a miss is not a bridge fault. Counting one would
        let a broker whose gold is named ``XAUUSD.m`` halt a healthy connection
        during startup.
        """
        try:
            info = self.call("symbol_info", name, allow_none=True)
        except BridgeError:
            return None
        if info is None:
            return None
        try:
            if not self.call("symbol_select", name, True):
                logger.error("MT5 could not select %s into Market Watch", name)
                return None
        except BridgeError:
            return None

        return SymbolSpec(
            name=name,
            digits=int(getattr(info, "digits", 0) or 0),
            point=float(getattr(info, "point", 0.0) or 0.0),
            tick_size=float(getattr(info, "trade_tick_size", 0.0) or 0.0),
            tick_value=float(getattr(info, "trade_tick_value", 0.0) or 0.0),
            contract_size=float(getattr(info, "trade_contract_size", 0.0) or 0.0),
            volume_min=float(getattr(info, "volume_min", 0.0) or 0.0),
            volume_max=float(getattr(info, "volume_max", 0.0) or 0.0),
            volume_step=float(getattr(info, "volume_step", 0.0) or 0.0),
            stops_level=int(getattr(info, "trade_stops_level", 0) or 0),
            freeze_level=int(getattr(info, "trade_freeze_level", 0) or 0),
            filling_mask=int(getattr(info, "filling_mode", 0) or 0),
            trade_mode=int(getattr(info, "trade_mode", 0) or 0),
        )

    # -- health (spec step 5) ------------------------------------------------

    def health_check(self) -> bool:
        """One cheap probe. Drives DEGRADED and HALTED through :meth:`call`.

        Returns:
            True when the terminal answered, is connected to the broker and
            still permits automated trading. False escalates - three in a row
            halts new entries while exits keep running, which is Beast's
            existing rule for a broken feed (``main.py`` FEED_FAILURES_BEFORE_PAUSE).
        """
        try:
            terminal = self.call("terminal_info")
        except BridgeError as error:
            logger.error("MT5 health check failed: %s", error)
            return False

        if not bool(getattr(terminal, "connected", False)):
            self._record_failure("the terminal lost its broker connection")
            return False
        if not bool(getattr(terminal, "trade_allowed", False)):
            self._record_failure("Algo Trading was switched off in the terminal")
            return False
        return True

    def reconnect(self) -> bool:
        """Tear down and rebuild, then re-run the handshake.

        The caller must reconcile broker positions against Beast's state before
        returning to normal operation - this method restores the pipe, not the
        truth about what is open.
        """
        logger.warning("MT5 %s reconnecting", self.profile)
        try:
            self.disconnect()
        except Exception as error:
            logger.warning("MT5 disconnect during reconnect raised: %s", error)
        self._client = None if not self._injected else self._client
        self.facts = SessionFacts()
        return self.connect()
