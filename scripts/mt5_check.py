"""Verify the MT5 bridge end to end, from the machine Beast runs on.

    python scripts/mt5_check.py                    # read-only, safe anywhere
    python scripts/mt5_check.py --place-test-order # DEMO only: open, modify, close
    python scripts/mt5_check.py --failure-drill    # DEMO only: kill the bridge, time recovery

Read-only mode reports everything Step 0 and Step 2 of the brief ask for and
exits 0 only when every check passes, so it works as a deployment gate.

The two active modes refuse to run against anything but a demo account, and they
check that with MT5's own answer rather than with a config value the operator
also controls.

This is an operator tool, not a pytest case: every check needs a live terminal, a
real login and an open market. The pure-logic equivalents live in
tests/test_mt5_connection.py and run against a fake client.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from broker import OrderRequest, OrderSide, OrderType
from broker.mt5_adapter import MT5Adapter
from broker.mt5_connection import ConnectionState, MT5Connection
from core.config import get_config

PASS, FAIL, WARN, INFO = "[ OK ]", "[FAIL]", "[WARN]", "[    ]"

results: list[tuple[str, bool]] = []
transitions: list[tuple[float, str, str]] = []


def report(mark: str, headline: str, *details: str) -> None:
    print(f"{mark} {headline}")
    for line in details:
        print(f"       {line}")
    if mark in (PASS, FAIL):
        results.append((headline, mark is PASS))


def mask(value) -> str:
    """Show only the tail of an account number - logs get shared."""
    text = str(value)
    return text if len(text) <= 4 else "*" * (len(text) - 4) + text[-4:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Beast's MetaTrader 5 bridge.")
    parser.add_argument("--market", default="XAUUSD")
    parser.add_argument("--place-test-order", action="store_true",
                        help="DEMO only: open a minimum-volume position, modify its "
                             "stop, close it, and confirm nothing is left open")
    parser.add_argument("--failure-drill", action="store_true",
                        help="DEMO only: stop the mt5linux unit, time the halt and "
                             "the recovery, then restart it")
    parser.add_argument("--lots", type=float, default=None)
    return parser.parse_args()


def on_state(state: str, reason: str) -> None:
    transitions.append((time.monotonic(), state, reason))


# -- read-only checks --------------------------------------------------------

def check_environment(link: MT5Connection) -> None:
    cfg = get_config()
    profile = link.profile
    host = cfg.get(f"broker.mt5.profiles.{profile}.host", "?")
    port = cfg.get(f"broker.mt5.profiles.{profile}.port", "?")
    bridged = link._is_bridge()

    version = "native MetaTrader5 package (Windows)"
    if bridged:
        try:
            import rpyc
            version = f"RPyC {rpyc.__version__} classic bridge to the Wine-side MetaTrader5"
        except ImportError:
            version = "rpyc NOT INSTALLED"
    report(INFO, f"profile {profile}: {version}", f"endpoint {host}:{port}")

    if bridged and str(host) not in ("127.0.0.1", "localhost", "::1"):
        report(FAIL, f"RPyC endpoint is {host}, not loopback",
               "Anyone who can reach that port controls the trading account, and",
               "RPyC classic mode allows arbitrary code execution. Bind to",
               "127.0.0.1 and block the port in the VPS firewall before trading.")
    elif bridged:
        report(PASS, "RPyC endpoint is loopback-only")


def check_handshake(link: MT5Connection) -> bool:
    if not link.connect(wait_seconds=30):
        report(FAIL, "Handshake failed",
               "The log line above names the layer that broke.")
        return False

    facts = link.facts
    report(PASS, f"Connected: account {mask(facts.login)} on {facts.server}",
           f"terminal build {facts.terminal_build}, currency {facts.currency}",
           f"{'DEMO' if facts.is_demo else 'REAL'} account, "
           f"{'hedging' if facts.hedging else 'netting'} margin mode")
    report(PASS, f"Server clock is UTC{facts.server_utc_offset_hours:+g}",
           "All bar and tick timestamps are corrected by this before use.")
    report(PASS, f"Latency p50 {facts.latency_p50_ms:.0f}ms / "
                 f"p95 {facts.latency_p95_ms:.0f}ms")
    return True


def check_symbol(link: MT5Connection, market: str) -> bool:
    spec = link.spec(market)
    if spec is None:
        report(FAIL, f"No MT5 symbol for {market}",
               f"Set broker.mt5.profiles.{link.profile}.symbols.{market.upper()}")
        return False
    report(PASS, f"{market} -> {spec.name}", spec.report())
    if not spec.filling_mask:
        report(WARN, "The symbol advertises no filling mode",
               "order_send will fall back to RETURN and may return retcode 10030.")
    return True


def check_data(adapter: MT5Adapter, market: str) -> None:
    quote = adapter.quote(market)
    if quote is None:
        report(FAIL, "No live quote", "The market may be closed.")
    else:
        report(PASS, "Live quote",
               f"bid {quote.bid}, ask {quote.ask}, spread {quote.spread:.5f}",
               f"stamped {quote.timestamp}")

    started = time.monotonic()
    frame = adapter.history(market, "1M", 1000)
    elapsed = time.monotonic() - started
    if frame.empty:
        report(FAIL, "No M1 history", "A freshly started terminal may still be syncing.")
        return
    report(PASS, f"{len(frame)} M1 bars in {elapsed:.2f}s",
           f"{frame.index[0]} -> {frame.index[-1]}")
    if len(frame) < 1000:
        report(WARN, f"Asked for 1000 M1 bars, got {len(frame)}",
               "The terminal's 'Max bars in chart' setting caps history.")

    # Session VWAP reads volume; on a spot CFD real_volume is usually absent and
    # tick_volume is a publishing rate, not traded size. Reported, never swapped.
    if float(frame["volume"].abs().sum()) == 0.0:
        report(WARN, "Volume is zero on this feed",
               "Session VWAP has no meaningful input. Decision needed.")
    else:
        report(PASS, "Volume is populated (tick_volume unless real_volume exists)",
               "Confirm which one before trusting Session VWAP.")


# -- active checks -----------------------------------------------------------

def check_order(adapter: MT5Adapter, market: str, lots: float | None) -> None:
    link = adapter.link
    if not link.facts.is_demo:
        report(FAIL, "Refusing the test order: this is not a demo account")
        return

    spec = link.spec(market)
    quote = adapter.quote(market)
    if spec is None or quote is None:
        report(FAIL, "Cannot price a test order")
        return

    volume = lots if lots is not None else spec.volume_min
    # Park the stop well outside the broker's minimum distance so the test
    # exercises the order path, not the stops-level rejection.
    distance = max(spec.stops_level * spec.point * 3, quote.ask * 0.002)
    stop = round(quote.ask - distance, spec.digits)

    print(f"\n--- DEMO test order: BUY {volume} {spec.name}, SL {stop} ---")
    result = adapter.place_order(OrderRequest(
        symbol=market, side=OrderSide.BUY, quantity=1, order_type=OrderType.MARKET,
        tag="beast-check", metadata={"volume_lots": volume, "sl": stop,
                                     "trade_id": "beast-check"},
    ))
    if not result.accepted:
        report(FAIL, "Test order rejected", result.message)
        return
    report(PASS, "Test order filled", result.message)

    positions = [p for p in (link.call("positions_get", allow_none=True) or ())
                 if int(getattr(p, "magic", -1)) == int(get_config().get("broker.mt5.magic"))]
    if not positions:
        report(FAIL, "Filled, but no matching position is open",
               "Check the terminal by hand before trading.")
        return
    position = positions[0]
    report(PASS, f"Position {position.ticket} open",
           f"{position.volume} lots at {position.price_open}, SL {position.sl}")

    tightened = round(float(position.sl) + distance * 0.25, spec.digits)
    modified = link.call("order_send", {
        "action": link._client.TRADE_ACTION_SLTP,
        "symbol": spec.name,
        "position": int(position.ticket),
        "sl": tightened,
    })
    if int(getattr(modified, "retcode", -1)) == 10009:
        report(PASS, f"Stop modified to {tightened}")
    else:
        report(FAIL, "Could not modify the stop",
               f"retcode {getattr(modified, 'retcode', '?')}")

    closed = adapter.close_position(int(position.ticket))
    report(PASS if closed.accepted else FAIL,
           "Position closed" if closed.accepted else "Could not close the position",
           closed.message)

    still_open = [p for p in (link.call("positions_get", allow_none=True) or ())
                  if int(getattr(p, "ticket", -1)) == int(position.ticket)]
    report(PASS if not still_open else FAIL,
           "Nothing left open" if not still_open
           else f"Position {position.ticket} is STILL OPEN - close it by hand")


def check_failure_drill(adapter: MT5Adapter) -> None:
    """Stop the bridge, prove Beast halts loudly, restart it, time the recovery."""
    cfg = get_config()
    unit = str(cfg.get("broker.mt5.units.server", "") or "")
    if not unit:
        report(FAIL, "No systemd unit configured for the mt5linux server",
               "Set broker.mt5.units.server (e.g. mt5linux.service). The drill",
               "only runs on the VPS that hosts the bridge.")
        return
    if not adapter.link.facts.is_demo:
        report(FAIL, "Refusing the failure drill: this is not a demo account")
        return

    def systemctl(action: str) -> bool:
        try:
            return subprocess.run(["systemctl", action, unit], timeout=30).returncode == 0
        except Exception as error:
            report(FAIL, f"systemctl {action} {unit} failed", str(error))
            return False

    print(f"\n--- failure drill: stopping {unit} ---")
    if not systemctl("stop"):
        return

    limit = int(cfg.get("broker.mt5.health.failures_before_halt", 3))
    interval = float(cfg.get("broker.mt5.health.interval_seconds", 5))
    started = time.monotonic()
    for _ in range(limit + 2):
        adapter.link.health_check()
        if adapter.link.state is ConnectionState.HALTED:
            break
        time.sleep(interval)
    halt_seconds = time.monotonic() - started

    if adapter.link.state is ConnectionState.HALTED:
        report(PASS, f"HALTED after {halt_seconds:.1f}s",
               f"{limit} failed health checks at {interval:g}s apart")
    else:
        report(FAIL, f"Did not halt after {halt_seconds:.1f}s",
               f"state is {adapter.link.state.value} - entries would still be allowed")

    print(f"--- restarting {unit} ---")
    systemctl("start")
    started = time.monotonic()
    recovered = adapter.link.reconnect()
    recovery_seconds = time.monotonic() - started
    report(PASS if recovered else FAIL,
           f"Reconnect {'succeeded' if recovered else 'failed'} in {recovery_seconds:.1f}s")

    if recovered:
        positions = adapter.link.call("positions_get", allow_none=True) or ()
        report(PASS, f"Reconciled: broker reports {len(positions)} open position(s)",
               "Beast must compare this against its own state before resuming.")

    print("\nstate transitions during the drill:")
    base = transitions[0][0] if transitions else 0.0
    for at, state, reason in transitions:
        print(f"       +{at - base:6.1f}s  {state:<12} {reason}")


def main() -> int:
    args = parse_args()
    cfg = get_config()
    print(f"Beast - MetaTrader 5 bridge check ({args.market})\n")

    link = MT5Connection(profile=str(cfg.get("broker.mt5.active_profile", "gold")),
                         config=cfg, on_event=on_state)
    adapter = MT5Adapter(config=cfg, connection=link)

    try:
        check_environment(link)
        if check_handshake(link):
            if check_symbol(link, args.market):
                check_data(adapter, args.market)
                if args.place_test_order:
                    check_order(adapter, args.market, args.lots)
                if args.failure_drill:
                    check_failure_drill(adapter)
    finally:
        link.disconnect()

    failed = [name for name, ok in results if not ok]
    print(f"\n{len(results) - len(failed)} of {len(results)} checks passed")
    if failed:
        print("failed: " + "; ".join(failed))
        return 1
    print("Bridge is healthy. Route XAUUSD to 'mt5' in config/beast_config.yaml "
          "when you are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
