"""The monitoring package: log streams, alert triggers, dashboard rendering.

Monitoring is the layer nobody tests until an incident, at which point the
question is whether the log said what was true at the time. Three things are
worth pinning:

* Every record carries the runtime context, so a warning can be read against
  the equity and position state that produced it.
* Streams route correctly, and ``main`` keeps everything, so reconstructing a
  session never means merging four files by timestamp.
* Alerts are rate-limited per (kind, market) and critical kinds bypass it - a
  wide spread must not suppress a loss-limit pause on another market.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from monitoring.alerts import AlertKind, AlertManager
from monitoring.dashboard import CAUTION_AT, DANGER_AT, Dashboard, risk_bar
from monitoring.logger import (
    ALERT_EVENTS,
    BACKUP_COUNT,
    MAX_BYTES,
    REGIME_EVENTS,
    TRADE_EVENTS,
    get_runtime_context,
    log_alert,
    log_event,
    log_signal,
    log_vol_state,
    reset_runtime_context,
    set_runtime_context,
    setup_logging,
)


@pytest.fixture
def logdir(cfg, tmp_path):
    """A logger writing its four streams into a tmp directory."""
    cfg.section("monitoring")["log_dir"] = str(tmp_path)
    reset_runtime_context()
    logger = setup_logging(cfg, quiet_console=True)
    yield tmp_path
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    reset_runtime_context()


def read_stream(directory: Path, name: str) -> list[dict]:
    path = directory / name
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


class TestLogStreams:
    """Four streams, and `main` is a superset rather than a partition."""

    def test_all_four_files_are_created(self, logdir):
        log_event("signal", {"a": 1}, "a signal")
        for name in ("main.log", "trades.log", "alerts.log", "regime.log"):
            assert (logdir / name).exists(), f"{name} was not created"

    def test_a_trade_event_reaches_trades_and_main(self, logdir):
        log_event("trade", {"r_multiple": 1.4}, "trade closed")
        assert len(read_stream(logdir, "trades.log")) == 1
        assert len(read_stream(logdir, "main.log")) == 1
        assert read_stream(logdir, "alerts.log") == []

    def test_an_alert_reaches_alerts_and_main(self, logdir):
        log_alert("spread too wide", {"kind": "HIGH_SPREAD"})
        assert len(read_stream(logdir, "alerts.log")) == 1
        assert len(read_stream(logdir, "main.log")) == 1
        assert read_stream(logdir, "trades.log") == []

    def test_a_vol_state_reaches_regime_and_main(self, logdir):
        log_event("vol_state", {"vol_state": "CALM"}, "calm")
        assert len(read_stream(logdir, "regime.log")) == 1
        assert len(read_stream(logdir, "main.log")) == 1

    def test_main_keeps_everything_including_unrouted_events(self, logdir):
        """Reconstructing a session must not mean merging four files."""
        log_event("signal", {}, "s")
        log_alert("a")
        log_event("vol_state", {}, "v")
        log_event("something_new", {}, "unrouted")
        assert len(read_stream(logdir, "main.log")) == 4

    def test_the_stream_event_sets_are_disjoint(self):
        """An event landing in two streams would double-count it."""
        assert not TRADE_EVENTS & ALERT_EVENTS
        assert not TRADE_EVENTS & REGIME_EVENTS
        assert not ALERT_EVENTS & REGIME_EVENTS

    def test_records_are_one_json_object_per_line(self, logdir):
        log_signal({"signal_id": "s1"}, "NIFTY50 LONG setup 1")
        entry = read_stream(logdir, "main.log")[0]
        assert entry["event"] == "signal"
        assert entry["payload"]["signal_id"] == "s1"
        assert entry["message"] == "NIFTY50 LONG setup 1"
        assert "ts" in entry and "level" in entry

    def test_handlers_are_replaced_not_duplicated(self, cfg, tmp_path):
        """A leaked RotatingFileHandler holds a file Windows will not rename."""
        cfg.section("monitoring")["log_dir"] = str(tmp_path)
        first = setup_logging(cfg, quiet_console=True)
        count = len(first.handlers)
        second = setup_logging(cfg, quiet_console=True)
        assert len(second.handlers) == count
        for handler in list(second.handlers):
            second.removeHandler(handler)
            handler.close()

    def test_rotation_policy_is_bounded(self):
        assert MAX_BYTES == 10 * 1024 * 1024
        assert BACKUP_COUNT == 30


class TestRuntimeContext:
    """Every record answers "what was true when this happened?"."""

    def test_context_is_attached_to_every_record(self, logdir):
        set_runtime_context(
            equity=505_230.0,
            vol_state={"NIFTY50": "TURBULENT"},
            vol_probability={"NIFTY50": 0.72},
            open_positions=["NIFTY50"],
            daily_pnl={"indian": -12_500.0},
            daily_pnl_pct={"indian": -0.0247},
        )
        log_event("signal", {}, "a signal")
        log_alert("an alert")

        for name in ("main.log", "trades.log", "alerts.log"):
            for entry in read_stream(logdir, name):
                runtime = entry["runtime"]
                assert runtime["equity"] == 505_230.0
                assert runtime["vol_state"]["NIFTY50"] == "TURBULENT"
                assert runtime["open_positions"] == ["NIFTY50"]
                assert runtime["daily_pnl"]["indian"] == -12_500.0

    def test_the_six_required_fields_are_present(self, logdir):
        log_event("message", {}, "hello")
        runtime = read_stream(logdir, "main.log")[0]["runtime"]
        for field in ("vol_state", "vol_probability", "equity",
                      "open_positions", "daily_pnl", "mode"):
            assert field in runtime
        assert "ts" in read_stream(logdir, "main.log")[0]

    def test_unknown_fields_are_ignored_not_raised(self, logdir):
        """A caller passing a field this build lacks loses the field, not the line."""
        set_runtime_context(equity=1.0, not_a_real_field="x")
        log_event("message", {}, "still logged")
        assert len(read_stream(logdir, "main.log")) == 1

    def test_reset_clears_it(self, logdir):
        set_runtime_context(equity=999.0)
        reset_runtime_context()
        assert get_runtime_context().equity == 0.0

    def test_vol_state_helper_writes_the_full_record(self, logdir, cfg):
        from datetime import datetime as dt

        from core.regime.contracts import VolState

        state = VolState(
            market="NIFTY50", bar_ts=dt(2026, 9, 8, 10, 45), label="TURBULENT",
            bucket_source_state=2, probability=0.72, state_probabilities={2: 0.72},
            is_confirmed=True, consecutive_bars=14, flicker_rate=0.05,
            is_flickering=False, size_multiplier=0.6, veto=False,
            reason="TURBULENT confirmed", model_version="abc@2026-09-01",
        )
        log_vol_state("NIFTY50", state)
        entry = read_stream(logdir, "regime.log")[0]
        assert entry["payload"]["vol_state"] == "TURBULENT"
        assert entry["payload"]["vol_state_confirmed"] is True
        assert entry["payload"]["size_multiplier"] == 0.6


class TestAlertTriggers:
    """The seven operational triggers, and the rate limit around them."""

    @pytest.fixture
    def captured(self, cfg):
        sent = []
        return AlertManager(cfg, transport=sent.append), sent

    def test_every_requested_trigger_exists(self):
        for name in ("VOL_STATE_CHANGE", "REGIME_CHANGE", "CIRCUIT_BREAKER",
                     "LARGE_PNL", "FEED_DOWN", "API_LOST", "MODEL_RETRAINED",
                     "FLICKER_EXCEEDED"):
            assert hasattr(AlertKind, name)

    def test_vol_state_change_names_both_states(self, captured):
        manager, sent = captured
        manager.vol_state_change("NIFTY50", "CALM", "TURBULENT", 0.81, 4)
        assert "CALM -> TURBULENT" in sent[0].message
        assert "open positions are unaffected" in sent[0].message.lower()

    def test_circuit_breaker_alerts_on_both_edges(self, captured):
        manager, sent = captured
        manager.circuit_breaker("NIFTY50", "daily loss cap hit", tripped=True)
        manager.circuit_breaker("NIFTY50", "daily loss cap hit", tripped=False)
        assert "CIRCUIT BREAKER" in sent[0].message
        assert "resume" in sent[1].message

    def test_feed_down_is_more_urgent_with_a_position(self, captured):
        manager, sent = captured
        manager.feed_down("NIFTY50", 3, has_position=False)
        manager.feed_down("XAUUSD", 3, has_position=True)
        assert "No position open" in sent[0].message
        assert "OPEN POSITION IS AFFECTED" in sent[1].message

    def test_large_pnl_is_expressed_against_the_cap(self, captured):
        """"Down 45,000" means nothing without the cap it is measured against."""
        manager, sent = captured
        manager.large_pnl("indian", -45_000.0, 75_000.0, 0.6)
        assert "60%" in sent[0].message
        assert "30,000" in sent[0].message      # room left

    def test_model_retrained_flags_a_noise_margin(self, captured):
        manager, sent = captured
        manager.model_retrained("NIFTY50", "abc@2026-09-01", 3, 12_500, True)
        assert "BIC margin was noise" in sent[0].message

    def test_flicker_alert_explains_why_size_dropped(self, captured):
        manager, sent = captured
        manager.flicker_exceeded("NIFTY50", 6, 20, 4)
        assert "uncertainty mode" in sent[0].message

    def test_api_lost_and_restored(self, captured):
        manager, sent = captured
        manager.api_lost("zerodha", "token expired")
        manager.api_restored("zerodha", 23.0)
        assert "session lost" in sent[0].message
        assert "23ms" in sent[1].message


class TestRateLimiting:
    """One per event type per 15 minutes, per market."""

    def test_a_repeat_inside_the_window_is_suppressed(self, cfg):
        manager = AlertManager(cfg, transport=lambda _a: None)
        start = datetime(2026, 9, 8, 10, 0)
        assert manager.send(AlertKind.HIGH_SPREAD, "XAUUSD", "wide", start) is True
        assert manager.send(
            AlertKind.HIGH_SPREAD, "XAUUSD", "wide", start + timedelta(minutes=5)
        ) is False

    def test_it_fires_again_after_the_window(self, cfg):
        manager = AlertManager(cfg, transport=lambda _a: None)
        start = datetime(2026, 9, 8, 10, 0)
        window = int(cfg.get("monitoring.alert_rate_limit_minutes"))
        manager.send(AlertKind.HIGH_SPREAD, "XAUUSD", "wide", start)
        assert manager.send(
            AlertKind.HIGH_SPREAD, "XAUUSD", "wide",
            start + timedelta(minutes=window + 1),
        ) is True

    def test_the_limit_is_per_market(self, cfg):
        manager = AlertManager(cfg, transport=lambda _a: None)
        start = datetime(2026, 9, 8, 10, 0)
        manager.send(AlertKind.HIGH_SPREAD, "XAUUSD", "wide", start)
        assert manager.send(AlertKind.HIGH_SPREAD, "NIFTY50", "wide", start) is True

    def test_the_limit_is_per_kind(self, cfg):
        """A spread alert must never suppress a different condition."""
        manager = AlertManager(cfg, transport=lambda _a: None)
        start = datetime(2026, 9, 8, 10, 0)
        manager.send(AlertKind.HIGH_SPREAD, "XAUUSD", "wide", start)
        assert manager.send(
            AlertKind.NEWS_BLACKOUT, "XAUUSD", "blackout", start
        ) is True

    @pytest.mark.parametrize("kind", [
        AlertKind.CIRCUIT_BREAKER, AlertKind.FEED_DOWN, AlertKind.API_LOST,
        AlertKind.LOSS_LIMIT_PAUSE, AlertKind.ERROR,
    ])
    def test_critical_kinds_bypass_the_limit(self, cfg, kind):
        """The second occurrence matters: it means the condition did not clear."""
        manager = AlertManager(cfg, transport=lambda _a: None)
        start = datetime(2026, 9, 8, 10, 0)
        manager.send(kind, "NIFTY50", "condition", start)
        assert manager.send(
            kind, "NIFTY50", "condition", start + timedelta(seconds=30)
        ) is True

    def test_suppressed_alerts_are_still_kept_in_history(self, cfg):
        manager = AlertManager(cfg, transport=lambda _a: None)
        start = datetime(2026, 9, 8, 10, 0)
        manager.send(AlertKind.HIGH_SPREAD, "XAUUSD", "wide", start)
        manager.send(AlertKind.HIGH_SPREAD, "XAUUSD", "wide", start)
        assert len(manager.history) == 2


class TestRiskBar:
    """Colour-coded consumption bars."""

    def test_an_empty_bar_is_green(self):
        bar, colour = risk_bar(0.0, 100.0)
        assert bar == "-" * 10
        assert colour == "green"

    def test_a_full_bar_is_red(self):
        bar, colour = risk_bar(100.0, 100.0)
        assert bar == "#" * 10
        assert colour == "red"

    def test_colour_steps_at_the_documented_thresholds(self):
        assert risk_bar(CAUTION_AT * 100 - 1, 100.0)[1] == "green"
        assert risk_bar(CAUTION_AT * 100, 100.0)[1] == "yellow"
        assert risk_bar(DANGER_AT * 100, 100.0)[1] == "red"

    def test_an_unset_limit_is_not_a_full_one(self):
        """A null limit must not render as 100% consumed."""
        bar, colour = risk_bar(50.0, 0.0)
        assert "#" not in bar
        assert colour == "dim"

    def test_overshoot_is_clamped(self):
        bar, _ = risk_bar(500.0, 100.0)
        assert len(bar) == 10


class TestDashboardRendering:
    """The view must render whatever it is handed, including nothing."""

    def test_it_renders_with_no_state(self, cfg):
        assert Dashboard(cfg).render_plain()

    def test_it_shows_vol_state_and_regime_separately(self, cfg):
        dashboard = Dashboard(cfg)
        dashboard.update(markets={
            "NIFTY50": {
                "vol_label": "TURBULENT", "vol_probability": 0.72,
                "vol_confirmed": True, "vol_bars": 14, "regime": "TREND_UP",
                "size_multiplier": 0.6, "entries": "allowed",
            }
        })
        text = dashboard.render_plain()
        assert "vol_state=TURBULENT" in text
        assert "regime=TREND_UP" in text

    def test_no_directional_vocabulary_leaks_into_the_view(self, cfg):
        """`BULL` in a display is read as bias just as surely as in a log."""
        dashboard = Dashboard(cfg)
        dashboard.update(markets={
            "NIFTY50": {"vol_label": "CALM", "vol_probability": 0.9,
                        "vol_confirmed": True, "regime": "RANGE"}
        })
        text = dashboard.render_plain().upper()
        for word in ("BULL", "BEAR", "CRASH", "EUPHORIA", "LEVERAGE",
                     "ALLOCATION", "REBALANCE"):
            assert word not in text, f"{word} appeared in the dashboard"

    def test_risk_rows_render_a_bar_and_the_pause(self, cfg):
        dashboard = Dashboard(cfg)
        dashboard.update(risk={
            "NIFTY50": {"realised_pnl": -60_000.0, "daily_cap": 75_000.0,
                        "consecutive_losses": 2, "paused": True},
        })
        text = dashboard.render_plain()
        assert "RISK NIFTY50" in text
        assert "PAUSED" in text

    def test_the_indian_family_is_not_double_counted(self, cfg):
        """Nifty and Sensex share one MarketRiskState; summing rows would double it."""
        dashboard = Dashboard(cfg)
        shared = {"realised_pnl": -10_000.0, "daily_cap": 75_000.0,
                  "consecutive_losses": 1, "paused": False}
        dashboard.update(risk={"NIFTY50": dict(shared), "SENSEX": dict(shared)})
        assert "-10,000" in dashboard.render_plain()

    def test_positions_render_with_r_and_stop(self, cfg):
        dashboard = Dashboard(cfg)
        dashboard.update(positions=[{
            "market": "NIFTY50", "direction": "LONG", "entry": 24_180.0,
            "stop": 24_130.0, "unrealised_r": 1.2, "held": "3h05m",
            "risk_amount": 15_000.0,
        }])
        text = dashboard.render_plain()
        assert "POS NIFTY50 LONG" in text
        assert "+1.20" in text

    @pytest.mark.skipif(
        not __import__("monitoring.dashboard", fromlist=["RICH_AVAILABLE"]).RICH_AVAILABLE,
        reason="rich is not installed",
    )
    def test_the_rich_layout_builds(self, cfg):
        dashboard = Dashboard(cfg)
        dashboard.update(
            markets={"NIFTY50": {"vol_label": "CALM", "vol_probability": 0.8,
                                 "vol_confirmed": True, "regime": "RANGE"}},
            risk={"NIFTY50": {"realised_pnl": -1000.0, "daily_cap": 75_000.0,
                              "consecutive_losses": 0, "paused": False}},
            positions=[],
            system={"feeds_ok": True, "brokers": {"zerodha": {"connected": True,
                                                              "latency_ms": 23.0}}},
        )
        assert dashboard.render() is not None
