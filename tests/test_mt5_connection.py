"""MT5 bridge and adapter tests, against a fake client.

Everything here runs without Wine, without a terminal and without a network.
The fake implements only the calls the bridge actually makes, and each test
drives it into one failure mode.

The rule these tests exist to protect, above all others: **no code path may
place an order on a REAL account.** ``test_account_mode_guard_*`` and
``test_real_account_never_reaches_order_send`` are the ones to read first if the
handshake is ever refactored.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from broker import OrderRequest, OrderSide, OrderType
from broker.mt5_adapter import MT5Adapter
from broker.mt5_connection import (
    AccountModeViolation,
    BridgeCallFailed,
    BridgeMisconfigured,
    BridgeTimeout,
    BridgeUnavailable,
    ConnectionState,
    MT5Connection,
)

RATE_DTYPE = [("time", "<i8"), ("open", "<f8"), ("high", "<f8"), ("low", "<f8"),
              ("close", "<f8"), ("tick_volume", "<u8"), ("spread", "<i4"),
              ("real_volume", "<u8")]

DEMO, CONTEST, REAL = 0, 1, 2


class FakeMT5:
    """A minimal stand-in for the MetaTrader5 module."""

    ACCOUNT_TRADE_MODE_DEMO = DEMO
    ACCOUNT_MARGIN_MODE_RETAIL_HEDGING = 2
    TIMEFRAME_M1, TIMEFRAME_M15, TIMEFRAME_M30 = 1, 15, 30
    TRADE_ACTION_DEAL, TRADE_ACTION_SLTP, TRADE_ACTION_REMOVE = 1, 6, 8
    ORDER_TYPE_BUY, ORDER_TYPE_SELL = 0, 1
    ORDER_TIME_GTC = 0
    ORDER_FILLING_FOK, ORDER_FILLING_IOC, ORDER_FILLING_RETURN = 0, 1, 2

    def __init__(self, trade_mode: int = DEMO, filling_mask: int = 2,
                 margin_mode: int = 0, stops_level: int = 0,
                 retcodes: list[int] | None = None, hang_on: str | None = None):
        self.trade_mode = trade_mode
        self.filling_mask = filling_mask
        self.margin_mode = margin_mode
        self.stops_level = stops_level
        self.retcodes = retcodes or []
        self.hang_on = hang_on
        self.sent: list[dict] = []
        self.calls: list[str] = []
        self.call_order: list[tuple[str, float]] = []
        self.positions: list[SimpleNamespace] = []
        self.deals: list[SimpleNamespace] = []

    # -- session
    def initialize(self, **kwargs):
        self.init_kwargs = kwargs
        return True

    def shutdown(self):
        return True

    def last_error(self):
        return (1, "fake error")

    def terminal_info(self):
        self._record("terminal_info")
        return SimpleNamespace(connected=True, trade_allowed=True, build=4620)

    def account_info(self):
        self._record("account_info")
        return SimpleNamespace(login=778899, server="Fake-Demo", currency="USD",
                               equity=10000.0, balance=10000.0, leverage=500,
                               trade_mode=self.trade_mode, trade_allowed=True,
                               trade_expert=True, margin_mode=self.margin_mode)

    # -- symbols
    def symbol_info(self, name):
        self._record("symbol_info")
        if name not in ("XAUUSD.m", "XAUEUR"):
            return None
        return SimpleNamespace(
            name=name, digits=2, point=0.01, volume_min=0.01, volume_max=50.0,
            volume_step=0.01, trade_contract_size=100.0, trade_tick_size=0.01,
            trade_tick_value=1.0, trade_stops_level=self.stops_level,
            trade_freeze_level=0, filling_mode=self.filling_mask, trade_mode=4)

    def symbol_select(self, name, enable):
        return name in ("XAUUSD.m", "XAUEUR")

    def symbols_get(self, group=""):
        return [SimpleNamespace(name="XAUUSD.m"), SimpleNamespace(name="XAUEUR")]

    def symbol_info_tick(self, name):
        self._record("symbol_info_tick")
        return SimpleNamespace(time=1757500800, bid=2500.10, ask=2500.45, last=0.0)

    # -- data
    def copy_rates_from_pos(self, name, timeframe, start, count):
        self._record("copy_rates_from_pos")
        assert start == 1, "the forming bar must never be requested"
        return np.array(
            [(1757500000 + i * 60, 2500 + i, 2506 + i, 2494 + i, 2503 + i, 90 + i, 3, 0)
             for i in range(count)], dtype=RATE_DTYPE)

    # -- orders
    def order_check(self, payload):
        return SimpleNamespace(retcode=0, comment="ok")

    def order_send(self, payload):
        self._record("order_send")
        self.sent.append(dict(payload))
        retcode = self.retcodes.pop(0) if self.retcodes else 10009
        return SimpleNamespace(retcode=retcode, order=5150, deal=1,
                               volume=payload.get("volume"), price=2500.45,
                               comment="done")

    def positions_get(self, symbol=None):
        return list(self.positions)

    def orders_get(self, symbol=None):
        return []

    def history_deals_get(self, *args, **kwargs):
        return list(self.deals)

    def _record(self, name: str) -> None:
        if self.hang_on == name:
            time.sleep(30)
        self.calls.append(name)
        self.call_order.append((name, time.monotonic()))


def _mt5_config(cfg, **overrides):
    """Point the config at the fake and shorten every timeout.

    Runs with both paper switches off so the order-path tests exercise
    transmission to the FAKE; the paper-mode tests flip ``mode`` back.
    """
    cfg.data["mode"] = "live"
    cfg.section("broker")["paper_trading"] = False
    section = cfg.section("broker")["mt5"]
    section.update({"account_mode": "demo", "active_profile": "gold",
                    "magic": 20260910, "comment_limit": 31, "refresh_bars": 5})
    section["timeouts"] = {"fast_seconds": 2, "history_seconds": 2, "connect_seconds": 2}
    section["health"] = {"interval_seconds": 1, "latency_warn_ms": 5000,
                         "latency_samples": 3, "failures_before_halt": 3}
    section.update(overrides)
    return cfg


def _link(cfg, client, **overrides) -> MT5Connection:
    _mt5_config(cfg, **overrides)
    return MT5Connection(profile="gold", config=cfg, client=client)


def _connected(cfg, client, **overrides) -> MT5Connection:
    link = _link(cfg, client, **overrides)
    assert link.connect(wait_seconds=1)
    return link


# -- call discipline ---------------------------------------------------------

def test_calls_are_serialised(cfg):
    """Concurrent callers must not interleave: MT5 is not thread-safe."""
    client = FakeMT5()
    link = _connected(cfg, client)

    overlaps = []
    active = {"count": 0}
    lock = threading.Lock()
    original = client.terminal_info

    def watched():
        with lock:
            active["count"] += 1
            if active["count"] > 1:
                overlaps.append(True)
        time.sleep(0.01)
        with lock:
            active["count"] -= 1
        return original()

    client.terminal_info = watched
    threads = [threading.Thread(target=lambda: link.call("terminal_info"))
               for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not overlaps, "two MT5 calls ran at the same time"


def test_timeout_raises_typed_error_and_poisons_the_worker(cfg):
    """A hung Wine call must return control to the caller, not block forever."""
    client = FakeMT5(hang_on="terminal_info")
    link = _link(cfg, client)
    link._start_worker()

    with pytest.raises(BridgeTimeout):
        link.call("terminal_info")

    assert link._worker_poisoned, "a timed-out worker must not be reused"
    assert link.state in (ConnectionState.DEGRADED, ConnectionState.HALTED)


def test_none_becomes_a_typed_error_not_a_silent_none(cfg):
    client = FakeMT5()
    link = _connected(cfg, client)
    with pytest.raises(BridgeCallFailed) as error:
        link.call("symbol_info", "NOPE")
    assert "fake error" in str(error.value), "last_error must be attached"


def test_allow_none_permits_a_legitimately_empty_answer(cfg):
    client = FakeMT5()
    link = _connected(cfg, client)
    assert link.call("positions_get", allow_none=True) == []


# -- state machine -----------------------------------------------------------

def test_state_transitions_are_emitted(cfg):
    seen: list[tuple[str, str]] = []
    client = FakeMT5()
    _mt5_config(cfg)
    link = MT5Connection(profile="gold", config=cfg, client=client,
                         on_event=lambda state, reason: seen.append((state, reason)))
    assert link.connect(wait_seconds=1)

    states = [state for state, _ in seen]
    assert states[0] == ConnectionState.CONNECTING.value
    assert ConnectionState.READY.value in states


def test_three_failures_halt(cfg):
    client = FakeMT5()
    link = _connected(cfg, client)
    for _ in range(3):
        link._record_failure("bridge down")
    assert link.state is ConnectionState.HALTED


def test_success_clears_degraded(cfg):
    client = FakeMT5()
    link = _connected(cfg, client)
    link._record_failure("one blip")
    assert link.state is ConnectionState.DEGRADED
    link.call("terminal_info")
    assert link.state is ConnectionState.READY


def test_health_check_fails_when_algo_trading_is_off(cfg):
    client = FakeMT5()
    link = _connected(cfg, client)
    client.terminal_info = lambda: SimpleNamespace(connected=True, trade_allowed=False,
                                                   build=4620)
    assert link.health_check() is False


# -- the account-mode guard --------------------------------------------------

def test_account_mode_guard_refuses_a_real_account_in_demo_mode(cfg):
    link = _link(cfg, FakeMT5(trade_mode=REAL), account_mode="demo")
    link._start_worker()
    with pytest.raises(AccountModeViolation):
        link.handshake()


def test_account_mode_guard_refuses_live_without_the_env_switch(cfg, monkeypatch):
    monkeypatch.delenv("BEAST_ALLOW_LIVE", raising=False)
    link = _link(cfg, FakeMT5(trade_mode=REAL), account_mode="live")
    link._start_worker()
    with pytest.raises(AccountModeViolation):
        link.handshake()


def test_account_mode_guard_refuses_a_demo_account_in_live_mode(cfg, monkeypatch):
    monkeypatch.setenv("BEAST_ALLOW_LIVE", "1")
    link = _link(cfg, FakeMT5(trade_mode=DEMO), account_mode="live")
    link._start_worker()
    with pytest.raises(AccountModeViolation):
        link.handshake()


def test_missing_credentials_fail_immediately_without_retrying(cfg, monkeypatch):
    """Waiting cannot conjure a password, so the retry window must be skipped."""
    for name in ("MT5_GOLD_LOGIN", "MT5_GOLD_PASSWORD", "MT5_GOLD_SERVER"):
        monkeypatch.delenv(name, raising=False)
    _mt5_config(cfg)
    link = MT5Connection(profile="gold", config=cfg)
    link._client = FakeMT5()          # a client exists; the credentials do not

    started = time.monotonic()
    assert link.connect(wait_seconds=30) is False
    assert time.monotonic() - started < 5, "a misconfiguration must not be retried"
    assert link.state is ConnectionState.HALTED


def test_misconfiguration_is_typed_separately_from_unavailability(cfg, monkeypatch):
    monkeypatch.delenv("MT5_GOLD_PASSWORD", raising=False)
    monkeypatch.setenv("MT5_GOLD_LOGIN", "123")
    monkeypatch.setenv("MT5_GOLD_SERVER", "Fake-Demo")
    _mt5_config(cfg)
    link = MT5Connection(profile="gold", config=cfg)
    link._client = FakeMT5()
    link._start_worker()

    with pytest.raises(BridgeMisconfigured):
        link._initialise()


def test_portable_flag_reaches_initialize(cfg, monkeypatch):
    monkeypatch.setenv("MT5_GOLD_LOGIN", "778899")
    monkeypatch.setenv("MT5_GOLD_PASSWORD", "pw")
    monkeypatch.setenv("MT5_GOLD_SERVER", "Fake-Demo")
    monkeypatch.setenv("MT5_GOLD_PORTABLE", "1")
    monkeypatch.setenv("MT5_GOLD_TERMINAL_PATH", r"C:\MT5\terminal64.exe")
    client = FakeMT5()
    link = _link(cfg, client)
    link._injected = False
    link._start_worker()
    link._initialise()
    assert client.init_kwargs["portable"] is True
    assert client.init_kwargs["path"] == r"C:\MT5\terminal64.exe"
    assert client.init_kwargs["login"] == 778899


def test_initialize_failure_reports_the_terminals_own_error(cfg, monkeypatch):
    """-6 from the terminal must reach the log verbatim, and must not be retried."""
    monkeypatch.setenv("MT5_GOLD_LOGIN", "778899")
    monkeypatch.setenv("MT5_GOLD_PASSWORD", "pw")
    monkeypatch.setenv("MT5_GOLD_SERVER", "Fake-Demo")
    client = FakeMT5()
    client.initialize = lambda **kw: False
    client.last_error = lambda: (-6, "Terminal: Authorization failed")
    link = _link(cfg, client)
    link._injected = False
    link._start_worker()
    with pytest.raises(BridgeMisconfigured) as error:
        link._initialise()
    assert "Authorization failed" in str(error.value)
    assert "rejected the login" in str(error.value)


def test_a_missing_terminal_is_retried_not_fatal(cfg, monkeypatch):
    """-10003 means the terminal is not up yet - that one is worth waiting for."""
    monkeypatch.setenv("MT5_GOLD_LOGIN", "778899")
    monkeypatch.setenv("MT5_GOLD_PASSWORD", "pw")
    monkeypatch.setenv("MT5_GOLD_SERVER", "Fake-Demo")
    client = FakeMT5()
    client.initialize = lambda **kw: False
    client.last_error = lambda: (-10003, "IPC initialize failed")
    link = _link(cfg, client)
    link._injected = False
    link._start_worker()
    with pytest.raises(BridgeUnavailable):
        link._initialise()


def test_real_account_never_reaches_order_send(cfg):
    """The guard runs at connect, so a real account never becomes tradable."""
    client = FakeMT5(trade_mode=REAL)
    link = _link(cfg, client, account_mode="demo", ready_wait_seconds=0)
    assert link.connect(wait_seconds=0) is False

    adapter = MT5Adapter(config=cfg, connection=link)
    result = adapter.place_order(OrderRequest(symbol="XAUUSD", side=OrderSide.BUY,
                                              quantity=1))
    assert result.accepted is False
    assert client.sent == [], "an order reached a REAL account"


# -- data --------------------------------------------------------------------

def test_history_excludes_the_forming_bar_and_is_tz_aware(cfg):
    adapter = MT5Adapter(config=cfg, connection=_connected(cfg, FakeMT5()))
    frame = adapter.history("XAUUSD", "1M", 10)

    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert frame.index.tz is not None
    assert frame.index.is_monotonic_increasing
    assert len(frame) == 10


def test_history_merges_instead_of_refetching(cfg):
    client = FakeMT5()
    adapter = MT5Adapter(config=cfg, connection=_connected(cfg, client))

    adapter.history("XAUUSD", "1M", 100)
    first = client.calls.count("copy_rates_from_pos")
    frame = adapter.history("XAUUSD", "1M", 100)

    assert client.calls.count("copy_rates_from_pos") == first + 1
    assert len(frame) == 100, "the merge must still serve the full window"


def test_server_time_offset_is_applied(cfg):
    link = _connected(cfg, FakeMT5())
    adapter = MT5Adapter(config=cfg, connection=link)

    link.facts.server_utc_offset_hours = 0
    base = adapter.server_time(1757500800)
    link.facts.server_utc_offset_hours = 3
    shifted = adapter.server_time(1757500800)

    assert (base - shifted).total_seconds() == 3 * 3600


# -- volume ------------------------------------------------------------------

def test_volume_rounds_down_never_up(cfg):
    client = FakeMT5()
    adapter = MT5Adapter(config=cfg, connection=_connected(cfg, client))

    adapter.place_order(OrderRequest(symbol="XAUUSD", side=OrderSide.BUY, quantity=1,
                                     metadata={"volume_lots": 0.129}))
    assert client.sent[-1]["volume"] == 0.12, "0.129 lots must round DOWN to 0.12"


def test_sub_minimum_volume_skips_the_trade(cfg):
    client = FakeMT5()
    adapter = MT5Adapter(config=cfg, connection=_connected(cfg, client))

    result = adapter.place_order(OrderRequest(symbol="XAUUSD", side=OrderSide.BUY,
                                              quantity=1,
                                              metadata={"volume_lots": 0.004}))
    assert result.accepted is False
    assert client.sent == [], "a sub-minimum size must never be rounded up and sent"


# -- filling mode ------------------------------------------------------------

@pytest.mark.parametrize("mask,expected", [(1, 0), (2, 1), (4, 2)])
def test_filling_mask_maps_to_the_request_enum(cfg, mask, expected):
    """The symbol mask and the request enum are different numbering schemes."""
    client = FakeMT5(filling_mask=mask)
    adapter = MT5Adapter(config=cfg, connection=_connected(cfg, client))
    adapter.place_order(OrderRequest(symbol="XAUUSD", side=OrderSide.BUY, quantity=1))
    assert client.sent[-1]["type_filling"] == expected


# -- stops -------------------------------------------------------------------

def test_stop_inside_the_brokers_minimum_distance_is_refused(cfg):
    client = FakeMT5(stops_level=100)      # 100 points x 0.01 = 1.00 minimum
    adapter = MT5Adapter(config=cfg, connection=_connected(cfg, client))

    result = adapter.place_order(OrderRequest(
        symbol="XAUUSD", side=OrderSide.BUY, quantity=1,
        metadata={"sl": 2500.20}))          # only 0.25 below the ask

    assert result.accepted is False
    assert "stops_level" in result.message
    assert client.sent == []


def test_stop_on_the_wrong_side_is_refused(cfg):
    client = FakeMT5()
    adapter = MT5Adapter(config=cfg, connection=_connected(cfg, client))
    result = adapter.place_order(OrderRequest(
        symbol="XAUUSD", side=OrderSide.BUY, quantity=1, metadata={"sl": 2600.0}))
    assert result.accepted is False
    assert "wrong side" in result.message


def test_valid_stop_is_attached_server_side(cfg):
    client = FakeMT5()
    adapter = MT5Adapter(config=cfg, connection=_connected(cfg, client))
    adapter.place_order(OrderRequest(symbol="XAUUSD", side=OrderSide.BUY, quantity=1,
                                     metadata={"sl": 2480.0}))
    assert client.sent[-1]["sl"] == 2480.0, "the SL must rest at the broker"


# -- idempotency -------------------------------------------------------------

def test_timeout_reconciles_before_it_would_resend(cfg):
    """A timed-out submit that actually executed must not be sent twice."""
    client = FakeMT5(hang_on="order_send")
    client.positions = [SimpleNamespace(ticket=4242, magic=20260910,
                                        comment="beast-trade-7", price_open=2500.45,
                                        symbol="XAUUSD.m", volume=0.01, type=0, sl=0.0)]
    adapter = MT5Adapter(config=cfg, connection=_connected(cfg, client))

    result = adapter.place_order(OrderRequest(
        symbol="XAUUSD", side=OrderSide.BUY, quantity=1,
        metadata={"trade_id": "beast-trade-7"}))

    assert result.accepted is True
    assert result.order_id == "4242"
    assert "recovered" in result.message


def test_timeout_with_no_matching_order_reports_failure(cfg):
    client = FakeMT5(hang_on="order_send")
    adapter = MT5Adapter(config=cfg, connection=_connected(cfg, client))

    result = adapter.place_order(OrderRequest(
        symbol="XAUUSD", side=OrderSide.BUY, quantity=1,
        metadata={"trade_id": "beast-trade-9"}))

    assert result.accepted is False
    assert "no matching order" in result.message


def test_requote_is_retried_with_a_fresh_tick(cfg):
    client = FakeMT5(retcodes=[10004, 10009])
    adapter = MT5Adapter(config=cfg, connection=_connected(cfg, client))

    result = adapter.place_order(OrderRequest(symbol="XAUUSD", side=OrderSide.BUY,
                                              quantity=1))
    assert result.accepted is True
    assert len(client.sent) == 2, "a requote must be re-priced and re-sent once"


def test_a_hard_rejection_is_not_retried(cfg):
    client = FakeMT5(retcodes=[10019])          # insufficient funds
    adapter = MT5Adapter(config=cfg, connection=_connected(cfg, client))

    result = adapter.place_order(OrderRequest(symbol="XAUUSD", side=OrderSide.BUY,
                                              quantity=1))
    assert result.accepted is False
    assert "insufficient funds" in result.message
    assert len(client.sent) == 1, "a funding rejection must never be retried"


# -- closing -----------------------------------------------------------------

@pytest.mark.parametrize("margin_mode,hedging", [(0, False), (2, True)])
def test_close_pins_the_position_ticket_on_both_margin_modes(cfg, margin_mode, hedging):
    """Passing `position` closes the intended leg instead of opening an offset."""
    client = FakeMT5(margin_mode=margin_mode)
    client.positions = [SimpleNamespace(ticket=99, magic=20260910, comment="x",
                                        price_open=2500.0, symbol="XAUUSD.m",
                                        volume=0.05, type=0, sl=0.0)]
    link = _connected(cfg, client)
    assert link.facts.hedging is hedging

    adapter = MT5Adapter(config=cfg, connection=link)
    result = adapter.close_position(99)

    assert result.accepted is True
    sent = client.sent[-1]
    assert sent["position"] == 99
    assert sent["type"] == FakeMT5.ORDER_TYPE_SELL, "a long is closed by a sell"
    assert sent["volume"] == 0.05


# -- paper mode --------------------------------------------------------------

def test_paper_mode_prices_from_the_live_tick_and_transmits_nothing(cfg):
    """The BrokerClient contract: refuse to transmit while mode is paper."""
    client = FakeMT5()
    adapter = MT5Adapter(config=cfg, connection=_connected(cfg, client))
    cfg.data["mode"] = "paper"

    result = adapter.place_order(OrderRequest(symbol="XAUUSD", side=OrderSide.BUY,
                                              quantity=1, tag="sig-1"))

    assert result.accepted is True
    assert result.paper is True
    assert client.sent == [], "paper mode must never call order_send"
    assert result.average_price > 2500.45, "a paper buy fills at the ask plus slippage"
    assert result.order_id == "paper-sig-1"


def test_paper_mode_still_refuses_what_live_would_refuse(cfg):
    """Validation runs before the paper branch, so a bad size is bad on paper too."""
    client = FakeMT5()
    adapter = MT5Adapter(config=cfg, connection=_connected(cfg, client))
    cfg.data["mode"] = "paper"

    result = adapter.place_order(OrderRequest(symbol="XAUUSD", side=OrderSide.BUY,
                                              quantity=1,
                                              metadata={"volume_lots": 0.004}))
    assert result.accepted is False
    assert client.sent == []


def test_bridge_is_chosen_by_platform_when_unset(cfg, monkeypatch):
    """A null bridge.enabled means Windows -> native, anything else -> mt5linux."""
    _mt5_config(cfg)
    cfg.section("broker")["mt5"]["bridge"] = {"enabled": None}
    link = MT5Connection(profile="gold", config=cfg, client=FakeMT5())

    monkeypatch.setattr("broker.mt5_connection.sys.platform", "linux")
    assert link._is_bridge() is True
    monkeypatch.setattr("broker.mt5_connection.sys.platform", "win32")
    assert link._is_bridge() is False

    cfg.section("broker")["mt5"]["bridge"] = {"enabled": True}
    assert link._is_bridge() is True, "an explicit setting always wins"


# -- interface conformance ---------------------------------------------------

def test_futures_contracts_raises_for_the_gold_path_to_catch(cfg):
    adapter = MT5Adapter(config=cfg, connection=_connected(cfg, FakeMT5()))
    with pytest.raises(NotImplementedError):
        adapter.futures_contracts("XAUUSD")


def test_limit_orders_are_refused_rather_than_half_implemented(cfg):
    client = FakeMT5()
    adapter = MT5Adapter(config=cfg, connection=_connected(cfg, client))
    result = adapter.place_order(OrderRequest(symbol="XAUUSD", side=OrderSide.BUY,
                                              quantity=1, order_type=OrderType.LIMIT,
                                              limit_price=2400.0))
    assert result.accepted is False
    assert client.sent == []


def test_comment_is_trimmed_to_the_broker_limit(cfg):
    client = FakeMT5()
    adapter = MT5Adapter(config=cfg, connection=_connected(cfg, client), )
    adapter.place_order(OrderRequest(symbol="XAUUSD", side=OrderSide.BUY, quantity=1,
                                     metadata={"trade_id": "x" * 80}))
    assert len(client.sent[-1]["comment"]) == 31
