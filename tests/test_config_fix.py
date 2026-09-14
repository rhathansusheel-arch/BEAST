"""MT5 session-drop and config-blocker fixes - DECISIONS D-79 to D-82.

Everything here runs against fakes: no bridge, no Wine, no file outside
``tmp_path``. The guarantees, in order:

* D-79 an XAUUSD-only agent is never blocked by, or told about, Nifty/Sensex
  keys, and never instantiates the Zerodha adapter;
* D-80 a dropped bridge is retried with capped exponential backoff, pages once
  on the drop, again only after the reminder interval, and announces its
  return as ``API_RESTORED``; a restore re-validates specs and reconciles;
* D-81 gold specs are cached per account, a changed value is ``SPEC_DRIFT``
  and never applied, a connected venue with no symbol is ``SYMBOL_UNRESOLVED``;
* D-82 ``data.gold_spread_max`` is auto-calibrated from the live book, listed
  as unreviewed, and ``mode: live`` is refused while it is.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from core.config import ConfigError
from data.spread_calibration import SpreadCalibrator
from tests.test_gold_cfd import RecordingAlerts, cfd_config, spec_runner

T0 = datetime(2026, 9, 14, 12, 0, 0)


def spec(**over):
    base = dict(name="XAUUSD", contract_size=100.0, tick_size=0.01, tick_value=1.0, point=0.01,
                volume_min=0.01, volume_step=0.01, volume_max=100.0, stops_level=100,
                filling_mask=1)
    base.update(over)
    return SimpleNamespace(**base)


def mt5_with(spec_obj, login=5055782131, server="MetaQuotes-Demo"):
    link = SimpleNamespace(
        spec=lambda market: spec_obj,
        facts=SimpleNamespace(company="MetaQuotes Ltd.", server=server, currency="USD",
                              login=login, specs={}),
    )
    return SimpleNamespace(is_connected=lambda: True, link=link, name="mt5")


def gold_null(cfg):
    """The shipped shape: every gold spec null, account currency declared."""
    cfd_config(cfg)
    section = cfg.section("instruments")["gold"]
    for key in ("venue", "symbol", "contract_multiplier", "tick_size", "tick_value", "point",
                "volume_min", "volume_step", "volume_max", "stops_level_points", "filling_mode"):
        section[key] = None
    return section


# -- D-79: XAUUSD-only scoping ---------------------------------------------------

def test_blockers_are_scoped_to_the_active_markets(raw_config):
    everything = raw_config.unset_blockers()
    assert "instruments.nifty.lot_size" in everything and "options.min_oi" in everything

    gold_only = raw_config.unset_blockers(["XAUUSD"])
    assert gold_only, "the raw config still has gold blockers"
    assert all(k.startswith("instruments.gold") or k == "data.gold_spread_max" for k in gold_only)
    assert not any("nifty" in k or "sensex" in k or k.startswith("options.") for k in gold_only)

    indian_only = raw_config.unset_blockers(["NIFTY50"])
    assert "options.min_oi" in indian_only and "instruments.nifty.lot_size" in indian_only
    assert not any("gold" in k for k in indian_only)


def test_an_xauusd_only_agent_never_builds_the_zerodha_adapter(cfg):
    from main import BeastRunner
    cfg.section("broker")["symbols"] = ["XAUUSD"]
    cfg.section("broker")["routing"]["XAUUSD"] = "paper"
    runner = BeastRunner.__new__(BeastRunner)
    runner.cfg = cfg
    runner.markets = ["XAUUSD"]
    built = runner._build_brokers()
    assert set(built) == {"paper"}, "only the adapters an active market routes to, plus paper"


def test_missing_routing_for_an_active_market_is_a_config_error(cfg):
    from main import BeastRunner
    cfg.section("broker")["symbols"] = ["XAUUSD"]
    del cfg.section("broker")["routing"]["XAUUSD"]
    runner = BeastRunner.__new__(BeastRunner)
    runner.cfg = cfg
    runner.markets = ["XAUUSD"]
    with pytest.raises(ConfigError):
        runner._build_brokers()


# -- D-80: session alerts and the reconnect loop ---------------------------------

class SessionAlerts:
    def __init__(self):
        self.lost, self.restored = [], []

    def api_lost(self, broker, detail=""): self.lost.append((broker, detail))
    def api_restored(self, broker, latency=None): self.restored.append(broker)


def session_runner(cfg, connected):
    from main import BeastRunner
    cfg.section("ops")["api_lost_reminder_minutes"] = 15
    state = {"connected": connected}
    runner = BeastRunner.__new__(BeastRunner)
    runner.cfg = cfg
    runner.brokers = {"mt5": SimpleNamespace(is_connected=lambda: state["connected"])}
    runner.alerts = SessionAlerts()
    runner._broker_connected = {}
    runner._broker_down_since = {}
    runner._broker_latency = lambda name: 3.0
    runner.logger = logging.getLogger("test")
    return runner, state


def test_a_drop_pages_once_then_only_at_the_reminder_interval(cfg):
    runner, state = session_runner(cfg, connected=True)
    runner._check_broker_sessions(T0)
    state["connected"] = False
    for seconds in (30, 60, 90, 120):                         # the old "every cycle" storm
        runner._check_broker_sessions(T0 + timedelta(seconds=seconds))
    assert len(runner.alerts.lost) == 1, "one page for one drop"

    runner._check_broker_sessions(T0 + timedelta(minutes=14))
    assert len(runner.alerts.lost) == 1
    runner._check_broker_sessions(T0 + timedelta(minutes=15, seconds=30))
    assert len(runner.alerts.lost) == 2 and "still down" in runner.alerts.lost[1][1]
    runner._check_broker_sessions(T0 + timedelta(minutes=31))
    assert len(runner.alerts.lost) == 3, "and again each interval while still down"


def test_a_return_is_announced_as_restored_and_resets_the_reminder(cfg):
    runner, state = session_runner(cfg, connected=True)
    runner._check_broker_sessions(T0)
    state["connected"] = False
    runner._check_broker_sessions(T0 + timedelta(seconds=30))
    state["connected"] = True
    runner._check_broker_sessions(T0 + timedelta(seconds=60))
    assert runner.alerts.restored == ["mt5"]
    assert "mt5" not in runner._broker_down_since


def test_a_broker_down_from_the_start_is_reminded_not_re_dropped(cfg):
    runner, state = session_runner(cfg, connected=False)
    runner._check_broker_sessions(T0)                          # startup already alerted this
    runner._check_broker_sessions(T0 + timedelta(minutes=1))
    assert runner.alerts.lost == []
    runner._check_broker_sessions(T0 + timedelta(minutes=16))
    assert len(runner.alerts.lost) == 1 and "still down" in runner.alerts.lost[0][1]


class FakeLink:
    """Just the surface MT5Adapter.maintain() touches."""

    def __init__(self, succeed_on: int | None = None, misconfigured: bool = False):
        from broker.mt5_connection import ConnectionState
        self.S = ConnectionState
        self.state = ConnectionState.HALTED
        self.misconfigured = misconfigured
        self.attempts = 0
        self.succeed_on = succeed_on
        self.waits = []

    def reconnect(self, wait_seconds=None):
        self.attempts += 1
        self.waits.append(wait_seconds)
        if self.succeed_on is not None and self.attempts >= self.succeed_on:
            self.state = self.S.READY
            return True
        self.state = self.S.HALTED
        return False


def adapter_with(cfg, link):
    from broker.mt5_adapter import MT5Adapter
    cfg.section("broker")["mt5"]["retry"].update(
        {"reconnect_base_seconds": 5, "reconnect_max_seconds": 40})
    adapter = MT5Adapter(config=cfg, connection=link)
    adapter._bars[("XAUUSD", "1M")] = "stale frame"
    return adapter


def test_reconnect_backs_off_exponentially_to_the_ceiling(cfg):
    link = FakeLink(succeed_on=None)
    adapter = adapter_with(cfg, link)
    clock = {"t": 1000.0}
    tick = lambda: clock["t"]

    assert adapter.maintain(clock=tick) is False and link.attempts == 0, "the first sight schedules"
    clock["t"] += 4; assert adapter.maintain(clock=tick) is False and link.attempts == 0
    clock["t"] += 1; adapter.maintain(clock=tick); assert link.attempts == 1        # t+5
    clock["t"] += 9; adapter.maintain(clock=tick); assert link.attempts == 1
    clock["t"] += 1; adapter.maintain(clock=tick); assert link.attempts == 2        # +10
    clock["t"] += 20; adapter.maintain(clock=tick); assert link.attempts == 3       # +20
    clock["t"] += 39; adapter.maintain(clock=tick); assert link.attempts == 3
    clock["t"] += 1; adapter.maintain(clock=tick); assert link.attempts == 4        # +40 (ceiling)
    clock["t"] += 40; adapter.maintain(clock=tick); assert link.attempts == 5       # stays at 40
    assert all(w == 0.0 for w in link.waits), "one handshake per attempt, never a blocking window"


def test_reconnect_success_resets_backoff_and_drops_the_bar_cache(cfg):
    link = FakeLink(succeed_on=2)
    adapter = adapter_with(cfg, link)
    clock = {"t": 0.0}
    tick = lambda: clock["t"]
    adapter.maintain(clock=tick)
    clock["t"] = 5; adapter.maintain(clock=tick)                 # attempt 1 fails
    clock["t"] = 15; assert adapter.maintain(clock=tick) is True # attempt 2 succeeds
    assert adapter.is_connected()
    assert adapter._bars == {}, "bars fetched before the gap may straddle it"
    assert adapter._reconnect_delay == 0.0 and adapter._reconnect_attempts == 0

    link.state = link.S.HALTED
    clock["t"] = 100; adapter.maintain(clock=tick)
    clock["t"] = 105; adapter.maintain(clock=tick)
    assert link.attempts == 3, "a later drop starts again from the base delay"


def test_a_misconfigured_login_waits_the_ceiling_not_the_base(cfg):
    link = FakeLink(succeed_on=None, misconfigured=True)
    adapter = adapter_with(cfg, link)
    clock = {"t": 0.0}
    tick = lambda: clock["t"]
    adapter.maintain(clock=tick)
    clock["t"] = 39; adapter.maintain(clock=tick); assert link.attempts == 0
    clock["t"] = 40; adapter.maintain(clock=tick); assert link.attempts == 1


def test_a_restore_revalidates_specs_reconciles_and_lifts_the_startup_pause(cfg, monkeypatch):
    import main as main_module
    from main import BeastRunner
    cfd_config(cfg)
    cfg.section("broker")["symbols"] = ["XAUUSD"]
    cfg.section("broker")["routing"]["XAUUSD"] = "mt5"
    runner = spec_runner(cfg, mt5_with(spec()))
    runner.positions = SimpleNamespace(open_markets=lambda: [])
    runner.journal = runner.risk_dummy = None
    runner._broker_connected = {"mt5": False}
    runner._entry_pause_reason = {"XAUUSD": "broker unreachable at startup: mt5 not connected"}
    runner._announced_blockers = None
    runner.brokers["mt5"].maintain = lambda: True

    seen = {}
    fake_report = SimpleNamespace(entries_to_pause={}, adopted={}, lines=lambda: ["clean"],
                                  repaired=False, disagreements=[], unreachable={}, safe_mode={},
                                  positions=[])
    def fake_reconcile(*a, **k):
        seen["markets"] = k.get("markets")
        return fake_report
    monkeypatch.setattr(main_module, "reconcile", fake_reconcile)

    runner._maintain_brokers(T0)

    assert seen["markets"] == ["XAUUSD"], "the venue is asked again for the routed markets"
    assert "XAUUSD" not in runner._entry_pause_reason, "the startup pause is lifted"
    assert runner._gold_specs_from_broker is not None, "specs were re-read"


# -- D-81: spec cache, drift, unresolved symbol ---------------------------------

def test_specs_are_cached_per_account_and_reapplied_when_unchanged(cfg, tmp_path):
    cache = tmp_path / "gold_specs_cache.json"
    cfg.section("ops")["gold_specs_cache_path"] = str(cache)
    section = gold_null(cfg)

    spec_runner(cfg, mt5_with(spec()))._fill_gold_specs()
    assert section["tick_value"] == 1.0 and section["symbol"] == "XAUUSD"
    written = json.loads(cache.read_text())
    assert written["login"] == 5055782131 and written["specs"]["volume_min"] == 0.01

    gold_null(cfg)                                              # a fresh boot: nulls again
    runner = spec_runner(cfg, mt5_with(spec()))
    runner._fill_gold_specs()
    assert section["tick_value"] == 1.0, "unchanged values are applied on the next boot"
    assert runner.alerts.drifts == []


def test_a_value_that_drifted_since_the_cache_is_alerted_and_not_applied(cfg, tmp_path):
    cache = tmp_path / "gold_specs_cache.json"
    cfg.section("ops")["gold_specs_cache_path"] = str(cache)
    section = gold_null(cfg)
    spec_runner(cfg, mt5_with(spec()))._fill_gold_specs()

    gold_null(cfg)
    runner = spec_runner(cfg, mt5_with(spec(contract_size=1000.0)))   # a 10x resize at the venue
    runner._fill_gold_specs()

    assert section["contract_multiplier"] is None, "nothing applied - not the old, not the new"
    assert runner.alerts.drifts == [("XAUUSD", {"contract_multiplier": (100.0, 1000.0)})]
    assert runner._entry_pause_reason["XAUUSD"].startswith("SPEC_DRIFT")
    assert json.loads(cache.read_text())["specs"]["contract_multiplier"] == 100.0, \
        "the cache keeps the last confirmed value until the operator clears it"


def test_a_cache_from_another_account_is_ignored_not_compared(cfg, tmp_path):
    cache = tmp_path / "gold_specs_cache.json"
    cfg.section("ops")["gold_specs_cache_path"] = str(cache)
    section = gold_null(cfg)
    spec_runner(cfg, mt5_with(spec(), login=1, server="Old-Demo"))._fill_gold_specs()

    gold_null(cfg)
    runner = spec_runner(cfg, mt5_with(spec(contract_size=1000.0), login=2, server="New-Demo"))
    runner._fill_gold_specs()
    assert runner.alerts.drifts == [] and section["contract_multiplier"] == 1000.0
    assert json.loads(cache.read_text())["login"] == 2, "the cache now belongs to the new account"


def test_a_connected_venue_with_no_symbol_is_symbol_unresolved(cfg):
    section = gold_null(cfg)
    runner = spec_runner(cfg, mt5_with(None))
    runner._fill_gold_specs()
    assert runner.alerts.unresolved and runner.alerts.unresolved[0][0] == "XAUUSD"
    assert "no usable symbol" in runner.alerts.unresolved[0][1]
    assert section["symbol"] is None


def test_blockers_are_alerted_after_discovery_and_only_on_change(cfg):
    section = gold_null(cfg)
    runner = spec_runner(cfg, mt5_with(spec()))
    cfg.section("broker")["symbols"] = ["XAUUSD"]

    first = runner._report_blockers()
    assert "instruments.gold.tick_value" in first and len(runner.alerts.blockers) == 1
    runner._report_blockers()
    assert len(runner.alerts.blockers) == 1, "the same board is not re-paged"

    runner._fill_gold_specs()
    assert runner._report_blockers() == []
    assert len(runner.alerts.blockers) == 1, "a cleared board is logged, not paged"


def test_preflight_excuses_gold_keys_discovery_will_fill(cfg, tmp_path):
    from tests.test_ops import preflight
    p = preflight(cfg, tmp_path)
    p.cfg.section("instruments")["gold"]["tick_value"] = None
    p.cfg.section("instruments")["gold"]["account_currency"] = None
    assert not p.check_blockers().ok, "without a resolved symbol nothing is excused"

    p._mt5.spec = lambda market: spec()
    p._mt5.facts.company = "MetaQuotes Ltd."
    assert p.check_bridge().ok and p.check_symbol().ok
    check = p.check_blockers()
    assert not check.ok and "account_currency" in check.detail, \
        "the operator's currency declaration is never a venue fact"
    assert "tick_value" in check.detail and "filled at startup" in check.detail

    p.cfg.section("instruments")["gold"]["account_currency"] = "USD"
    assert p.check_blockers().ok


# -- D-82: spread auto-calibration and the unreviewed banner ---------------------

def test_calibrator_takes_p90_times_the_multiplier_and_skips_bad_quotes():
    clock = {"t": 0.0}
    cal = SpreadCalibrator(window_seconds=60, multiplier=2.5, burst=5, burst_gap_seconds=0,
                           min_samples=10, clock=lambda: clock["t"], sleep=lambda s: None)
    spreads = iter([0.20, 0.22, float("inf"), 0.0, 0.21, 0.25, 0.23, 0.19, 0.30, 0.24,
                    0.22, 0.21, 0.26, 0.20, 0.23])
    quote = lambda: SimpleNamespace(spread=next(spreads))
    assert cal.sample(quote) is None
    assert cal.sample(quote) is None, "the window has not elapsed"
    clock["t"] = 61
    ceiling = cal.sample(quote)
    assert len(cal.samples) == 13, "inf and 0 are skipped, not recorded"
    assert cal.p90 == pytest.approx(0.26)
    assert ceiling == pytest.approx(0.65)
    assert "AUTO-CALIBRATED" in cal.describe()


def test_calibration_sets_the_ceiling_in_config_and_lists_it_as_unreviewed(cfg):
    from main import BeastRunner
    cfd_config(cfg)
    cfg.section("broker")["symbols"] = ["XAUUSD"]
    cfg.section("broker")["routing"]["XAUUSD"] = "mt5"
    data = cfg.section("data")
    data.update({"gold_spread_max": None, "gold_spread_auto_calibrate": True,
                 "gold_spread_calibration_seconds": 0, "gold_spread_safety_multiplier": 2.0,
                 "gold_spread_calibration_burst": 25})
    mt5 = mt5_with(spec())
    mt5.quote = lambda market: SimpleNamespace(spread=0.30)
    runner = spec_runner(cfg, mt5)
    runner.clocks = {"XAUUSD": SimpleNamespace(is_open=lambda now: True)}
    runner._spread_calibrator = None
    runner._auto_thresholds = {}

    assert runner.unreviewed_thresholds() == {
        "data.gold_spread_max": "AUTO-CALIBRATING (unset; p90 of the live spread x safety multiplier)"}
    runner._maybe_calibrate_spread(T0)
    runner._spread_calibrator.sleep = lambda s: None
    assert data["gold_spread_max"] == pytest.approx(0.60)
    assert "AUTO-CALIBRATED" in runner.unreviewed_thresholds()["data.gold_spread_max"]
    assert any(a[0] == "send" for a in runner.alerts.sent)


def test_an_operator_set_ceiling_is_never_calibrated_over(cfg):
    cfd_config(cfg)
    cfg.section("broker")["symbols"] = ["XAUUSD"]
    cfg.section("broker")["routing"]["XAUUSD"] = "mt5"
    cfg.section("data").update({"gold_spread_max": 0.35, "gold_spread_auto_calibrate": True})
    mt5 = mt5_with(spec())
    mt5.quote = lambda market: (_ for _ in ()).throw(AssertionError("must not sample"))
    runner = spec_runner(cfg, mt5)
    runner.clocks = {"XAUUSD": SimpleNamespace(is_open=lambda now: True)}
    runner._spread_calibrator = None
    runner._auto_thresholds = {}
    runner._maybe_calibrate_spread(T0)
    assert cfg.get("data.gold_spread_max") == 0.35
    assert runner.unreviewed_thresholds() == {}


def test_live_mode_is_refused_while_a_threshold_is_unreviewed(cfg):
    from main import BeastRunner
    cfg.section("broker")["symbols"] = ["XAUUSD"]
    cfg.section("data").update({"gold_spread_max": None, "gold_spread_auto_calibrate": True})
    runner = BeastRunner.__new__(BeastRunner)
    runner.cfg = cfg; runner.markets = ["XAUUSD"]; runner.brokers = {"paper": None}
    runner.logger = logging.getLogger("test"); runner._auto_thresholds = {}
    assert runner._step_config() is True, "paper trading proceeds on an auto ceiling"

    cfg.data["mode"] = "live"
    cfg.section("broker")["paper_trading"] = False
    assert runner._step_config() is False, "live on a placeholder is refused, not announced"

    cfg.section("data")["gold_spread_max"] = 0.35
    assert runner._step_config() is True


def test_placeholders_listed_by_the_operator_are_unreviewed_only_for_active_markets(cfg):
    from main import BeastRunner
    cfg.section("risk")["unreviewed_thresholds"] = ["options.min_oi", "data.gold_spread_max"]
    runner = BeastRunner.__new__(BeastRunner)
    runner.cfg = cfg; runner._auto_thresholds = {}
    runner.markets = ["XAUUSD"]
    assert list(runner.unreviewed_thresholds()) == ["data.gold_spread_max"]
    runner.markets = ["NIFTY50"]
    assert list(runner.unreviewed_thresholds()) == ["options.min_oi"]
