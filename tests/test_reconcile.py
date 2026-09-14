"""Startup reconcile against a fake venue - Part A, step 4.

The venue here is a recording fake that models MT5's semantics: the stop is
an attribute of the position, and ``set_position_stop`` is a re-read /
modify / re-read cycle. No Wine, no network.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from broker import BrokerOrder, BrokerPosition, OrderResult, OrderSide
from core.reconcile import reconcile
from core.schemas import Direction

NOW = datetime(2026, 9, 14, 12, 0)


class FakeVenue:
    name = "mt5"

    def __init__(self, positions=(), orders=(), connected=True, reachable=True):
        self._positions = list(positions)
        self._orders = list(orders)
        self._connected = connected
        self._reachable = reachable
        self.stop_writes: list[tuple[int, float, float | None]] = []
        self.cancelled: list[str] = []
        self.history_frame = None

    def is_connected(self):
        return self._connected

    def open_positions(self):
        return None if not self._reachable else list(self._positions)

    def pending_orders(self):
        return None if not self._reachable else list(self._orders)

    def set_position_stop(self, ticket, sl, tp=None):
        self.stop_writes.append((ticket, sl, tp))
        for p in self._positions:
            if p.ticket == ticket:
                p.sl = float(sl)
                if tp is not None:
                    p.tp = float(tp)
                return OrderResult(True, str(ticket), paper=False, message=f"confirmed sl={sl}")
        return OrderResult(False, message="not open")

    def cancel_order(self, order_id):
        self.cancelled.append(order_id)
        return True

    def history(self, market, timeframe, bars):
        if self.history_frame is None:
            raise RuntimeError("no feed")
        return self.history_frame


class FakeJournal:
    def __init__(self, plans=()):
        self.plans = list(plans)

    def open_signals_without_trades(self):
        return list(self.plans)


class FakeAlerts:
    def __init__(self):
        self.sent = []

    def api_lost(self, name, reason=""):
        self.sent.append(("api_lost", name, reason))

    def circuit_breaker(self, market, reason, tripped=True):
        self.sent.append(("circuit_breaker", market, reason))


def plan_row(signal_id="11111111-2222-3333-4444-555555555555", market="XAUUSD",
             direction="LONG", entry=2500.0, stop=2495.0, target=2510.0,
             when=NOW - timedelta(minutes=30)):
    payload = {"signal_id": signal_id, "market": market, "direction": direction,
               "entry_price": entry, "stop_price": stop, "target_price": target,
               "timestamp_ist": when.isoformat(), "setup_type": 1,
               "trail": {"activate_at": entry + 5, "activate_r": 1.0,
                         "method": "atr_chandelier", "mult": 1.5},
               "leg": {"type": "FUTURES", "_futures_only": {
                   "contract": "XAUUSD", "contract_multiplier": 100.0, "tick_size": 0.01,
                   "tick_value": 1.0, "expiry": None, "contracts": 1, "volume_lots": 0.1}}}
    return {"signal_id": signal_id, "market": market, "direction": direction,
            "entry_price": entry, "stop_price": stop, "target_price": target,
            "timestamp_ist": when.isoformat(), "payload": json.dumps(payload)}


def position(ticket=1, sl=0.0, tp=0.0, side=OrderSide.BUY, entry=2500.0, volume=0.1,
             comment="beast:XAUUSD:20260914:11111111", opened=NOW - timedelta(minutes=29)):
    return BrokerPosition(market="XAUUSD", symbol="XAUUSD", ticket=ticket, side=side,
                          volume=volume, entry_price=entry, sl=sl, tp=tp,
                          opened_at=opened, comment=comment, magic=20260910)


def _cfg(cfg):
    cfg.section("broker")["routing"]["XAUUSD"] = "mt5"
    cfg.section("broker")["symbols"] = ["XAUUSD"]
    cfg.data.setdefault("ops", {}).update({"reconcile_match_threshold": 0.7,
                                           "safe_mode_stop_atr_multiple": 2.0})
    gold = cfg.section("instruments")["gold"]
    gold.update({"point": 0.01, "stops_level_points": 100, "safe_mode_fallback_points": None})
    return cfg


def run(cfg, venue, plans=(), history=None):
    alerts = FakeAlerts()
    report = reconcile({"mt5": venue}, None, FakeJournal(plans), None, alerts, cfg, NOW,
                       markets=["XAUUSD"], history=history)
    return report, alerts


# -- the stop --------------------------------------------------------------------

def test_missing_stop_is_placed_at_the_plan_level(cfg):
    venue = FakeVenue([position(sl=0.0)])
    report, alerts = run(_cfg(cfg), venue, [plan_row()])

    assert venue.stop_writes and venue.stop_writes[0][1] == 2495.0
    assert any(a.kind == "stop_placed" and a.confirmed for a in report.actions)
    assert report.repaired
    assert "XAUUSD" in report.adopted and not report.safe_mode


def test_wider_broker_stop_is_left_alone_and_flags_safe_mode(cfg):
    venue = FakeVenue([position(sl=2490.0)])          # plan says 2495; venue is WIDER
    report, alerts = run(_cfg(cfg), venue, [plan_row()])

    assert venue.stop_writes == [], "a widening disagreement must never be 'corrected'"
    assert "XAUUSD" in report.safe_mode
    assert any("WIDER" in d for d in report.disagreements)
    assert any(a[0] == "circuit_breaker" for a in alerts.sent)


def test_tighter_broker_stop_is_kept_as_trail_state(cfg):
    venue = FakeVenue([position(sl=2498.0, tp=2510.0)])   # tighter than plan's 2495
    report, _ = run(_cfg(cfg), venue, [plan_row()])

    assert venue.stop_writes == []
    assert report.adopted["XAUUSD"]["current_stop"] == 2498.0
    assert report.adopted["XAUUSD"]["trail_activated"] is True
    assert not report.safe_mode


def test_undersized_stop_is_repaired(cfg):
    pos = position(sl=2495.0)
    pos.stop_volume = 0.05                              # covers half of 0.1
    venue = FakeVenue([pos])
    report, _ = run(_cfg(cfg), venue, [plan_row()])
    assert any(a.kind == "stop_repaired" for a in report.actions)


def test_missing_target_is_restored_from_plan(cfg):
    venue = FakeVenue([position(sl=2495.0, tp=0.0)])
    report, _ = run(_cfg(cfg), venue, [plan_row()])
    assert any(a.kind == "target_restored" and a.after == 2510.0 for a in report.actions)


# -- explaining the position ---------------------------------------------------

def test_position_with_no_plan_goes_to_safe_mode_and_pauses_entries(cfg):
    venue = FakeVenue([position(sl=2495.0, comment="manual")])
    report, alerts = run(_cfg(cfg), venue, plans=[])

    assert "XAUUSD" in report.safe_mode
    assert "XAUUSD" in report.entries_to_pause
    assert "XAUUSD" not in report.adopted
    assert any(a[0] == "circuit_breaker" and "SAFE MODE" in a[2] for a in alerts.sent)


def test_safe_mode_places_a_protective_stop_from_atr(cfg):
    import pandas as pd
    venue = FakeVenue([position(sl=0.0, comment="manual")])
    frame = pd.DataFrame({"high": [2503.0] * 60, "low": [2497.0] * 60, "close": [2500.0] * 60})
    report, _ = run(_cfg(cfg), venue, plans=[], history=lambda *a: frame)

    # ATR ~ 6 -> 2 x 6 = 12 below 2500
    assert venue.stop_writes and venue.stop_writes[0][1] == pytest.approx(2488.0, abs=0.5)
    assert any(a.kind == "safe_stop_placed" for a in report.actions)


def test_safe_mode_with_no_atr_and_no_fallback_pages_and_places_nothing(cfg):
    venue = FakeVenue([position(sl=0.0, comment="manual")])
    report, _ = run(_cfg(cfg), venue, plans=[])
    assert venue.stop_writes == [], "no distance may be invented"
    assert any("NO PROTECTIVE STOP" in d for d in report.disagreements)


def test_safe_mode_fallback_points_are_used_when_atr_is_unavailable(cfg):
    c = _cfg(cfg)
    c.section("instruments")["gold"]["safe_mode_fallback_points"] = 500   # 5.00
    venue = FakeVenue([position(sl=0.0, comment="manual")])
    run(c, venue, plans=[])
    assert venue.stop_writes[0][1] == pytest.approx(2495.0)


def test_safe_stop_never_further_than_an_existing_stop(cfg):
    c = _cfg(cfg)
    c.section("instruments")["gold"]["safe_mode_fallback_points"] = 500
    pos = position(sl=2497.0, comment="manual")
    pos.stop_volume = 0.05                  # present but undersized -> SAFE path recomputes
    venue = FakeVenue([pos])
    run(c, venue, plans=[])
    assert venue.stop_writes[0][1] == pytest.approx(2497.0), "clamped to the tighter existing stop"


# -- orders and reachability ---------------------------------------------------

def test_orphan_pending_order_is_cancelled(cfg):
    orphan = BrokerOrder(market="XAUUSD", symbol="XAUUSD", ticket=77, side=OrderSide.BUY,
                         volume=0.1, price=2480.0, comment="beast:XAUUSD:20260914:deadbeef",
                         magic=20260910)
    venue = FakeVenue([], [orphan])
    report, _ = run(_cfg(cfg), venue, plans=[])
    assert venue.cancelled == ["77"]
    assert any(a.kind == "order_cancelled" and a.ticket == 77 for a in report.actions)


def test_unreachable_broker_refuses_entries_and_assumes_nothing(cfg):
    venue = FakeVenue([position(sl=0.0)], reachable=False)
    report, alerts = run(_cfg(cfg), venue, [plan_row()])

    assert "XAUUSD" in report.unreachable
    assert "XAUUSD" in report.entries_to_pause
    assert report.positions == [], "nothing may be assumed about what the venue holds"
    assert venue.stop_writes == []
    assert any(a[0] == "api_lost" for a in alerts.sent)


def test_disconnected_broker_is_unreachable_not_flat(cfg):
    venue = FakeVenue([position(sl=0.0)], connected=False)
    report, _ = run(_cfg(cfg), venue, [plan_row()])
    assert "XAUUSD" in report.unreachable


# -- matching ------------------------------------------------------------------

def test_comment_tag_matches_with_full_confidence(cfg):
    venue = FakeVenue([position(sl=2495.0, tp=2510.0)])
    report, _ = run(_cfg(cfg), venue, [plan_row()])
    assert report.matches[0].confidence == 1.0 and report.matches[0].how == "comment tag"


def test_wrong_direction_plan_does_not_match(cfg):
    venue = FakeVenue([position(sl=2495.0, side=OrderSide.SELL, comment="x")])
    report, _ = run(_cfg(cfg), venue, [plan_row(direction="LONG")])
    assert report.matches[0].plan is None
    assert "XAUUSD" in report.safe_mode


# -- the runner uses the venue, not memory (the step 0 bug) ----------------------

def test_step_sync_positions_reports_what_the_venue_holds(cfg, tmp_path, monkeypatch):
    from main import BeastRunner
    import logging
    c = _cfg(cfg)
    venue = FakeVenue([position(sl=2495.0, tp=2510.0)])

    class Tracker:
        def __init__(self): self.adopted = []
        def open_markets(self): return [m for m, *_ in self.adopted]
        def adopt(self, signal, **kw): self.adopted.append((signal.market, kw)); return object()

    runner = BeastRunner.__new__(BeastRunner)
    runner.cfg = c; runner.brokers = {"mt5": venue}; runner.markets = ["XAUUSD"]
    runner.positions = Tracker(); runner.journal = FakeJournal([plan_row()])
    runner.risk = None; runner.alerts = FakeAlerts(); runner.logger = logging.getLogger("t")
    runner._entry_pause_reason = {}

    runner._step_sync_positions()

    assert runner._broker_positions == ["XAUUSD"], "must come from the venue, not memory"
    assert runner.positions.adopted and runner.positions.adopted[0][0] == "XAUUSD"
    assert runner.positions.adopted[0][1]["current_stop"] == 2495.0
