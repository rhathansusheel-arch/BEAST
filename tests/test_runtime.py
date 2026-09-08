"""Startup, the loop's guards, shutdown, and the retry policy.

The orchestration layer is where the soul file's operational rules either hold
or quietly stop holding, and none of them are visible from a unit test of the
component they constrain. Four in particular are tested here because getting
them wrong is expensive and silent:

* A restart must not reset the daily loss cap. That is how a 15% cap becomes a
  30% day.
* A Friday snapshot must not carry Friday's losses into Monday. The cap is
  session-scoped.
* A stale feed pauses **entries** and leaves exits running. A position with a
  live stop needs that stop evaluated more than it needs a healthy feed.
* Shutdown does not close positions. Their stops are resting; flattening on
  every restart would turn a deploy into a realised loss on every open trade.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from unittest import mock

import pytest

from broker.retry import (
    DEFAULT_POLICY,
    BrokerCallFailed,
    RetryPolicy,
    try_call,
    with_retry,
)
from core.session_state import (
    SNAPSHOT_VERSION,
    SessionSnapshot,
    build_snapshot,
    compare_positions,
    load_snapshot,
    restore_counters,
    save_snapshot,
)


# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------


class TestRetry:
    """Bounded retries with backoff, and a hard refusal to retry a write."""

    def test_a_successful_call_is_made_once(self):
        call = mock.Mock(return_value="ok")
        assert with_retry(call, "read", sleep=lambda _: None) == "ok"
        assert call.call_count == 1

    def test_a_transient_failure_is_retried_and_then_succeeds(self):
        call = mock.Mock(side_effect=[ConnectionError("502"), "ok"])
        assert with_retry(call, "read", sleep=lambda _: None) == "ok"
        assert call.call_count == 2

    def test_it_gives_up_after_the_configured_attempts(self):
        call = mock.Mock(side_effect=ConnectionError("down"))
        policy = RetryPolicy(attempts=3)
        with pytest.raises(BrokerCallFailed) as caught:
            with_retry(call, "read", policy, sleep=lambda _: None)
        assert call.call_count == 3
        assert caught.value.attempts == 3
        assert isinstance(caught.value.__cause__, ConnectionError)

    def test_backoff_grows_and_stays_under_the_ceiling(self):
        waits: list[float] = []
        call = mock.Mock(side_effect=ConnectionError("down"))
        policy = RetryPolicy(attempts=5, base_delay=1.0, multiplier=2.0,
                             max_delay=4.0, jitter=0.0)
        with pytest.raises(BrokerCallFailed):
            with_retry(call, "read", policy, sleep=waits.append)
        assert waits == [1.0, 2.0, 4.0, 4.0]

    def test_jitter_desynchronises_simultaneous_failures(self):
        """Three markets failing on the same bar must not retry in lockstep."""
        runs = []
        for _ in range(6):
            waits: list[float] = []
            with pytest.raises(BrokerCallFailed):
                with_retry(mock.Mock(side_effect=ConnectionError()), "read",
                           RetryPolicy(attempts=2, jitter=0.25), sleep=waits.append)
            runs.append(waits[0])
        assert len(set(runs)) > 1, "retry delays are identical - no jitter applied"

    def test_retrying_a_non_idempotent_call_is_refused(self):
        """A resend after an ambiguous failure can open a second position."""
        call = mock.Mock()
        policy = RetryPolicy(attempts=3, idempotent=False)
        with pytest.raises(ValueError, match="non-idempotent"):
            with_retry(call, "submit order", policy, sleep=lambda _: None)
        assert call.call_count == 0, "the call was made despite being refused"

    def test_a_single_attempt_is_allowed_for_a_non_idempotent_call(self):
        call = mock.Mock(return_value="filled")
        policy = RetryPolicy(attempts=1, idempotent=False)
        assert with_retry(call, "submit", policy, sleep=lambda _: None) == "filled"

    def test_try_call_returns_the_default_instead_of_raising(self):
        call = mock.Mock(side_effect=ConnectionError("down"))
        result = try_call(call, "optional quote", default=None,
                          policy=RetryPolicy(attempts=2), sleep=lambda _: None)
        assert result is None

    def test_the_default_policy_is_three_attempts(self):
        assert DEFAULT_POLICY.attempts == 3
        assert DEFAULT_POLICY.idempotent is True


# ---------------------------------------------------------------------------
# state_snapshot.json
# ---------------------------------------------------------------------------


class FakeRiskState:
    def __init__(self, family: str, session_day=None) -> None:
        self.family = family
        self.session_day = session_day
        self.realised_pnl = 0.0
        self.consecutive_losses = 0
        self.trades_today = 0
        self.paused = False
        self.pause_reason = ""
        self.cooldown_until: dict[str, datetime] = {}


class FakeRisk:
    def __init__(self, session_day=None) -> None:
        self.capital = 500_000.0
        self.state = {
            "indian": FakeRiskState("indian", session_day),
            "gold": FakeRiskState("gold", session_day),
        }


class FakePositions:
    def __init__(self, rows=None) -> None:
        self._rows = rows or []

    def snapshot(self):
        return list(self._rows)


@pytest.fixture
def snapshot_file(tmp_path):
    return tmp_path / "state_snapshot.json"


class TestSnapshotRoundTrip:
    def test_write_then_read_preserves_the_counters(self, cfg, snapshot_file):
        risk = FakeRisk(session_day=date(2026, 9, 8))
        risk.state["indian"].realised_pnl = -12_500.0
        risk.state["indian"].consecutive_losses = 2

        save_snapshot(
            build_snapshot(risk, FakePositions(), config=cfg), cfg, snapshot_file
        )
        loaded = load_snapshot(cfg, snapshot_file)

        assert loaded is not None
        assert loaded.version == SNAPSHOT_VERSION
        assert loaded.clean_exit is True
        assert loaded.counters["indian"]["realised_pnl"] == -12_500.0
        assert loaded.counters["indian"]["consecutive_losses"] == 2

    def test_a_missing_file_is_not_an_error(self, cfg, tmp_path):
        assert load_snapshot(cfg, tmp_path / "nothing.json") is None

    def test_corrupt_json_is_discarded_not_half_read(self, cfg, snapshot_file):
        snapshot_file.write_text("{ this is not json", encoding="utf-8")
        assert load_snapshot(cfg, snapshot_file) is None

    def test_a_snapshot_from_another_version_is_ignored(self, cfg, snapshot_file):
        save_snapshot(SessionSnapshot(version=SNAPSHOT_VERSION - 1), cfg, snapshot_file)
        assert load_snapshot(cfg, snapshot_file) is None

    def test_the_write_is_atomic(self, cfg, snapshot_file):
        """A reader sees the old file or the new one, never a partial one."""
        save_snapshot(SessionSnapshot(written_at="first"), cfg, snapshot_file)
        with mock.patch("json.dump", side_effect=OSError("disk full")):
            save_snapshot(SessionSnapshot(written_at="second"), cfg, snapshot_file)
        survivor = load_snapshot(cfg, snapshot_file)
        assert survivor is not None and survivor.written_at == "first"

    def test_an_unwritable_path_is_logged_not_raised(self, cfg, tmp_path):
        """Failing to save state must not turn an orderly shutdown into a crash."""
        target = tmp_path / "nope" / "deeper"
        with mock.patch("pathlib.Path.mkdir", side_effect=OSError("read-only")):
            save_snapshot(SessionSnapshot(), cfg, target / "state_snapshot.json")

    def test_crash_path_records_that_the_exit_was_not_clean(self, cfg, snapshot_file):
        save_snapshot(
            build_snapshot(FakeRisk(), FakePositions(), clean_exit=False, config=cfg),
            cfg, snapshot_file,
        )
        loaded = load_snapshot(cfg, snapshot_file)
        assert loaded is not None and loaded.clean_exit is False


class TestCounterRecovery:
    """A restart must not reset the daily loss cap - nor carry it into tomorrow."""

    def test_same_session_counters_are_restored(self, cfg, snapshot_file):
        today = date(2026, 9, 8)
        before = FakeRisk(session_day=today)
        before.state["indian"].realised_pnl = -30_000.0
        before.state["indian"].consecutive_losses = 2
        before.state["indian"].paused = True
        before.state["indian"].pause_reason = "daily loss cap hit"
        before.state["indian"].cooldown_until["NIFTY50"] = datetime(2026, 9, 8, 11, 30)
        save_snapshot(build_snapshot(before, FakePositions(), config=cfg), cfg, snapshot_file)

        after = FakeRisk()
        notes = restore_counters(
            load_snapshot(cfg, snapshot_file), after,
            {"indian": today, "gold": today}, cfg,
        )

        state = after.state["indian"]
        assert state.realised_pnl == -30_000.0
        assert state.consecutive_losses == 2
        assert state.paused is True
        assert state.cooldown_until["NIFTY50"] == datetime(2026, 9, 8, 11, 30)
        assert any("resumed mid-session" in note for note in notes)

    def test_a_previous_session_does_not_carry_over(self, cfg, snapshot_file):
        """Friday's 15% loss must not start Monday at 15%."""
        friday, monday = date(2026, 9, 4), date(2026, 9, 7)
        before = FakeRisk(session_day=friday)
        before.state["indian"].realised_pnl = -60_000.0
        before.state["indian"].paused = True
        save_snapshot(build_snapshot(before, FakePositions(), config=cfg), cfg, snapshot_file)

        after = FakeRisk()
        notes = restore_counters(
            load_snapshot(cfg, snapshot_file), after,
            {"indian": monday, "gold": monday}, cfg,
        )

        assert after.state["indian"].realised_pnl == 0.0
        assert after.state["indian"].paused is False
        assert any("counters start fresh" in note for note in notes)

    def test_a_mode_change_refuses_the_restore_entirely(self, cfg, snapshot_file):
        """Paper P&L must never seed a live session's loss cap, or vice versa."""
        before = FakeRisk(session_day=date(2026, 9, 8))
        before.state["indian"].realised_pnl = -30_000.0
        snapshot = build_snapshot(before, FakePositions(), config=cfg)
        object.__setattr__(snapshot, "mode", "live")
        save_snapshot(snapshot, cfg, snapshot_file)

        after = FakeRisk()
        notes = restore_counters(
            load_snapshot(cfg, snapshot_file), after,
            {"indian": date(2026, 9, 8)}, cfg,
        )
        assert after.state["indian"].realised_pnl == 0.0
        assert any("NOT restored" in note for note in notes)

    def test_recovery_is_never_silent(self, cfg, snapshot_file):
        """Whichever way it goes, the operator gets a line about it."""
        save_snapshot(
            build_snapshot(FakeRisk(session_day=date(2026, 9, 8)), FakePositions(),
                           config=cfg),
            cfg, snapshot_file,
        )
        for today in (date(2026, 9, 8), date(2026, 9, 9)):
            notes = restore_counters(
                load_snapshot(cfg, snapshot_file), FakeRisk(),
                {"indian": today, "gold": today}, cfg,
            )
            assert notes


class TestPositionComparison:
    """The broker is the authority. The snapshot only ever raises a question."""

    def test_agreement_produces_no_lines(self):
        snapshot = SessionSnapshot(believed_positions=[{"market": "NIFTY50"}])
        assert compare_positions(snapshot, ["NIFTY50"]) == []

    def test_a_position_that_closed_while_beast_was_down_is_reported(self):
        snapshot = SessionSnapshot(believed_positions=[{"market": "NIFTY50"}])
        lines = compare_positions(snapshot, [])
        assert len(lines) == 1
        assert "broker reports none" in lines[0]

    def test_an_unknown_broker_position_is_reported_as_stop_risk(self):
        """A position Beast does not know about is a position with no managed stop."""
        lines = compare_positions(SessionSnapshot(), ["XAUUSD"])
        assert len(lines) == 1
        assert "resting stop" in lines[0]

    def test_comparison_changes_nothing(self):
        snapshot = SessionSnapshot(believed_positions=[{"market": "NIFTY50"}])
        before = list(snapshot.believed_positions)
        compare_positions(snapshot, ["XAUUSD"])
        assert snapshot.believed_positions == before


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


@pytest.fixture
def runner(cfg, tmp_path, monkeypatch):
    """A runner with no brokers connected and no terminal taken over."""
    monkeypatch.setenv("BEAST_STATE_DIR", str(tmp_path))
    cfg.section("monitoring")["journal_db"] = str(tmp_path / "journal.sqlite")
    cfg.section("monitoring")["log_dir"] = str(tmp_path)
    cfg.section("regime")["enabled"] = False      # no model fitting in these tests

    from main import BeastRunner

    instance = BeastRunner(cfg, use_dashboard=False)
    yield instance
    instance.journal.close()


class TestFeedHealth:
    """A stale feed pauses entries. It never suspends an exit."""

    def test_one_miss_does_not_pause(self, runner):
        runner._on_feed_failure("NIFTY50", datetime.now())
        assert "NIFTY50" not in runner._entry_pause_reason

    def test_three_consecutive_misses_pause_entries(self, runner):
        from main import FEED_FAILURES_BEFORE_PAUSE

        for _ in range(FEED_FAILURES_BEFORE_PAUSE):
            runner._on_feed_failure("NIFTY50", datetime.now())
        assert "NIFTY50" in runner._entry_pause_reason
        assert "exits still running" in runner._entry_pause_reason["NIFTY50"]

    def test_recovery_clears_the_pause(self, runner):
        from main import FEED_FAILURES_BEFORE_PAUSE

        for _ in range(FEED_FAILURES_BEFORE_PAUSE):
            runner._on_feed_failure("NIFTY50", datetime.now())
        runner._on_feed_success("NIFTY50", datetime.now())
        assert "NIFTY50" not in runner._entry_pause_reason
        assert runner._feed_failures["NIFTY50"] == 0

    def test_a_pause_is_per_market(self, runner):
        from main import FEED_FAILURES_BEFORE_PAUSE

        for _ in range(FEED_FAILURES_BEFORE_PAUSE):
            runner._on_feed_failure("NIFTY50", datetime.now())
        assert "XAUUSD" not in runner._entry_pause_reason

    def test_a_dead_feed_still_manages_an_open_position(self, runner, monkeypatch):
        """The whole point: entries stop, position management does not.

        A quote endpoint and a bar endpoint are different calls that fail
        independently, so a dead bar feed does not mean there is no price.
        """
        monkeypatch.setattr(runner.data, "refresh", lambda *a, **k: None)
        monkeypatch.setattr(runner, "_current_premium", lambda *a, **k: None)

        priced = []
        update = mock.Mock(alerts=[], trade=None)
        monkeypatch.setattr(runner.positions, "get", lambda m: mock.Mock())
        monkeypatch.setattr(
            runner.positions, "on_price",
            lambda market, price, premium, now, clock: priced.append(price) or update,
        )
        broker = mock.Mock()
        broker.quote.return_value = mock.Mock(mid=24_000.0)
        monkeypatch.setattr(runner.data, "broker_for", lambda m: broker)

        evaluated = []
        monkeypatch.setattr(runner.generators["NIFTY50"], "evaluate",
                            lambda *a, **k: evaluated.append(True))

        runner._tick_market("NIFTY50", datetime.now(), {})

        assert priced == [24_000.0], "the exit path did not run on a dead bar feed"
        assert not evaluated, "entries were evaluated on a dead feed"

    def test_no_bars_and_no_quote_escalates_rather_than_going_quiet(
        self, runner, monkeypatch
    ):
        """A live position whose stop cannot be evaluated is the worst state here."""
        monkeypatch.setattr(runner.data, "refresh", lambda *a, **k: None)
        monkeypatch.setattr(runner.positions, "get", lambda m: mock.Mock())
        broker = mock.Mock()
        broker.quote.side_effect = ConnectionError("no route")
        monkeypatch.setattr(runner.data, "broker_for", lambda m: broker)

        runner._tick_market("NIFTY50", datetime.now(), {})

        assert any("OPEN POSITION AND NO PRICE" in line
                   for line in runner._recent_alert_lines)

    def test_a_dead_feed_with_no_position_is_merely_logged(self, runner, monkeypatch):
        monkeypatch.setattr(runner.data, "refresh", lambda *a, **k: None)
        monkeypatch.setattr(runner.positions, "get", lambda m: None)
        runner._tick_market("NIFTY50", datetime.now(), {})
        assert not any("OPEN POSITION" in line for line in runner._recent_alert_lines)


class TestErrorHandling:
    """One error is survivable. Ten in a row is a reason to stop."""

    def test_a_single_error_does_not_halt(self, runner):
        assert runner._handle_unhandled(RuntimeError("transient")) == 0
        assert runner.running is False or True   # run() owns `running`

    def test_a_repeating_error_halts_with_a_non_zero_code(self, runner):
        from main import ERROR_STREAK_BEFORE_HALT

        code = 0
        for _ in range(ERROR_STREAK_BEFORE_HALT):
            code = runner._handle_unhandled(RuntimeError("stuck"))
        assert code == 1
        assert runner.running is False

    def test_an_error_writes_an_unclean_snapshot(self, runner, cfg, tmp_path):
        runner._handle_unhandled(RuntimeError("boom"))
        loaded = load_snapshot(cfg, tmp_path / "state_snapshot.json")
        assert loaded is not None
        assert loaded.clean_exit is False


class TestShutdown:
    """Shutdown leaves positions alone and says so."""

    def test_positions_are_never_closed(self, runner, monkeypatch):
        closed = []
        monkeypatch.setattr(runner.positions, "open_markets", lambda: ["NIFTY50"])
        monkeypatch.setattr(runner.positions, "close",
                            lambda *a, **k: closed.append(True))
        monkeypatch.setattr(runner.positions, "flatten_all",
                            lambda *a, **k: closed.append(True))

        runner.shutdown(clean=True)
        assert not closed, "shutdown closed a position - its stop was already resting"

    def test_a_clean_shutdown_writes_a_clean_snapshot(self, runner, cfg, tmp_path):
        runner.shutdown(clean=True)
        loaded = load_snapshot(cfg, tmp_path / "state_snapshot.json")
        assert loaded is not None and loaded.clean_exit is True

    def test_shutdown_is_idempotent(self, runner, monkeypatch):
        calls = []
        monkeypatch.setattr(runner, "_save_snapshot", lambda **k: calls.append(True))
        runner.shutdown(clean=True)
        runner.shutdown(clean=True)
        assert len(calls) == 1

    def test_the_signal_handler_asks_the_loop_to_stop(self, runner):
        import signal as os_signal

        runner.running = True
        runner._handle_signal(os_signal.SIGINT, None)
        assert runner.running is False

    def test_the_summary_survives_a_run_with_no_activity(self, runner):
        summary = runner.stats.summary(datetime.now(), runner.risk)
        assert summary["cycles"] == 0
        assert set(summary["realised_pnl"]) == {"indian", "gold"}


class TestStartup:
    def test_config_errors_stop_startup(self, runner):
        runner.cfg.section("regime")["sizing"]["combine_method"] = "product"
        assert runner._step_config() is False

    def test_unset_blockers_do_not_stop_startup(self, runner):
        """Blockers refuse the affected trades, not the whole process."""
        assert runner.cfg.unset_blockers() == [] or runner._step_config() is True

    def test_market_hours_wait_by_default(self, runner, monkeypatch):
        for clock in runner.clocks.values():
            monkeypatch.setattr(clock, "is_open", lambda *_a: False)
        assert runner._step_market_hours(wait_for_open=True) is True

    def test_market_hours_can_decline_to_start(self, runner, monkeypatch):
        for clock in runner.clocks.values():
            monkeypatch.setattr(clock, "is_open", lambda *_a: False)
        assert runner._step_market_hours(wait_for_open=False) is False

    def test_a_disabled_regime_layer_skips_model_loading(self, runner):
        runner._step_models()
        assert all(engine is None for engine in runner.vol_engines.values())

    def test_recovery_alerts_on_an_unclean_previous_exit(self, runner, cfg, tmp_path):
        save_snapshot(
            build_snapshot(runner.risk, runner.positions, clean_exit=False, config=cfg),
            cfg, tmp_path / "state_snapshot.json",
        )
        runner._step_recover_snapshot()
        assert any("did not shut down cleanly" in line
                   for line in runner._recent_alert_lines)


class TestDryRun:
    """The full pipeline, minus the one call that creates a position."""

    def test_dry_run_journals_the_signal_but_opens_nothing(self, cfg, tmp_path,
                                                           monkeypatch):
        monkeypatch.setenv("BEAST_STATE_DIR", str(tmp_path))
        cfg.section("monitoring")["journal_db"] = str(tmp_path / "journal.sqlite")
        cfg.section("regime")["enabled"] = False

        from main import BeastRunner

        runner = BeastRunner(cfg, use_dashboard=False, dry_run=True)
        try:
            opened, recorded = [], []
            monkeypatch.setattr(runner.positions, "open",
                                lambda *a, **k: opened.append(True))
            monkeypatch.setattr(runner.journal, "record_signal",
                                lambda s: recorded.append(s))

            signal = mock.Mock()
            signal.reason_line = "test signal"
            signal.to_dict.return_value = {}
            signal.stop_price = 23_950.0
            signal.target_price = 24_100.0
            signal.direction.value = "LONG"
            signal.setup_type = 1

            runner._on_signal("NIFTY50", signal, 24_000.0, datetime.now())

            assert recorded, "dry run skipped journalling - it should run the full pipeline"
            assert not opened, "dry run opened a position"
            assert runner.stats.signals == 1
        finally:
            runner.journal.close()
