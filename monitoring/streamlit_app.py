"""Streamlit UI - a browser view of Beast's persisted state.

Run it with::

    streamlit run monitoring/streamlit_app.py

What it reads, and why that is all it reads
-------------------------------------------
Three durable artefacts a running Beast leaves behind:

* ``state_snapshot.json`` - equity, risk counters, the last ``VolState`` per
  market, and what Beast believed it held at its last write.
* ``logs/beast_journal.sqlite`` - Appendix B signals, Appendix C trades, and the
  per-gate rejection table section 9 reads.
* ``logs/*.log`` - the four JSON streams, for the alert and regime feeds.

It does **not** attach to the running process. Beast exposes no IPC or control
socket, and adding a network listener to the trading host so a browser tab can
watch it is not a trade worth making - the listener would be a live attack
surface on the machine holding the broker credentials, in exchange for a display
that is at most one cycle fresher than the files. What this shows is the same
information a cycle out of date, at no risk.

It is also strictly read-only. There are no buttons that place, cancel or close
anything. Soul file section 8 puts every manual deviation behind an explicit
confirmation step with full logging, and a web button is precisely the
frictionless override that section exists to prevent. Killing Beast is
``Ctrl-C`` on the process, deliberately.

Vocabulary
----------
``vol_state`` is ``CALM | NORMAL | TURBULENT | UNKNOWN`` - a volatility read, not
a direction. It is shown beside, never merged with, section 4.4's ``regime``
(``TREND_UP | TREND_DOWN | RANGE``). Risk is displayed against section 7's two
session-stopping conditions - the per-instrument daily loss cap and three
consecutive losses. There is no allocation, leverage or peak-drawdown figure,
because Beast has no such rules and displaying one would imply an enforcement
that does not exist.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from core.config import get_config
from core.session_state import load_snapshot, snapshot_path
from monitoring.logger import log_directory

REFRESH_SECONDS = 5

VOL_COLOURS = {
    "CALM": "#2e7d32",
    "NORMAL": "#0277bd",
    "TURBULENT": "#c62828",
    "UNKNOWN": "#f9a825",
}
REGIME_COLOURS = {
    "TREND_UP": "#2e7d32",
    "TREND_DOWN": "#c62828",
    "RANGE": "#f9a825",
}


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def read_journal(path: Path, query: str, params: tuple = ()) -> list[dict[str, Any]]:
    """Query the journal read-only.

    Opened with ``mode=ro`` so this process cannot write to, lock or corrupt the
    database a live Beast is writing to. A dashboard that can take a write lock
    on the trade journal is a dashboard that can stall the trading loop.
    """
    if not path.exists():
        return []
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error:
        return []
    try:
        connection.row_factory = sqlite3.Row
        with closing(connection.cursor()) as cursor:
            return [dict(row) for row in cursor.execute(query, params).fetchall()]
    except sqlite3.Error:
        return []
    finally:
        connection.close()


def read_log_stream(path: Path, limit: int = 200) -> list[dict[str, Any]]:
    """Read the tail of one JSON-lines stream, newest first.

    Malformed lines are skipped rather than raising: the writer may be mid-line
    when this reads, and one torn line is not a reason to show nothing.
    """
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    entries: list[dict[str, Any]] = []
    for line in reversed(lines):
        if len(entries) >= limit:
            break
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def header(cfg, snapshot) -> None:
    st.title("BEAST")
    if snapshot is None:
        st.warning(
            f"No state snapshot at `{snapshot_path(cfg)}`. Beast has not run "
            f"yet, or ran without writing one."
        )
        return

    written = _parse_ts(snapshot.written_at)
    age = datetime.now() - written if written else None

    columns = st.columns(4)
    columns[0].metric("Mode", snapshot.mode.upper())
    columns[1].metric("Equity", f"Rs {snapshot.capital:,.0f}")

    day_pnl = sum(
        float(row.get("realised_pnl", 0.0)) for row in _families(snapshot).values()
    )
    pct = day_pnl / snapshot.capital if snapshot.capital else 0.0
    columns[2].metric("Day P&L", f"Rs {day_pnl:+,.0f}", f"{pct:+.2%}")
    columns[3].metric(
        "Snapshot age",
        "unknown" if age is None else _humanise(age),
        delta="clean exit" if snapshot.clean_exit else "UNCLEAN EXIT",
        delta_color="normal" if snapshot.clean_exit else "inverse",
    )

    if not snapshot.clean_exit:
        st.error(
            "The run that wrote this snapshot did **not** shut down cleanly. "
            "Verify open positions and their resting stops at the broker before "
            "trusting anything below."
        )
    if age is not None and age > timedelta(minutes=10):
        st.info(
            f"This snapshot is {_humanise(age)} old. It is written at shutdown "
            f"and on the error path, so a long gap means Beast is running "
            f"normally and has not written since it started."
        )


def volatility_panel(snapshot, regime_entries: list[dict]) -> None:
    """vol_state and section 4.4 regime, side by side and never merged."""
    st.subheader("Volatility & regime")
    st.caption(
        "`vol_state` is a volatility read - CALM / NORMAL / TURBULENT / UNKNOWN. "
        "It is not a direction and never overrides section 4.4's regime, which "
        "governs which setups are permitted."
    )

    if snapshot is None or not snapshot.vol_states:
        st.write("_No volatility state recorded._")
        return

    latest_regime = {}
    for entry in regime_entries:
        payload = entry.get("payload") or {}
        if entry.get("event") == "regime" and payload.get("market") not in latest_regime:
            latest_regime[payload["market"]] = payload.get("to", "-")

    for market, state in snapshot.vol_states.items():
        label = str(state.get("vol_state", "UNKNOWN"))
        confirmed = bool(state.get("vol_state_confirmed"))
        colour = VOL_COLOURS.get(label, "#666")
        regime = latest_regime.get(market, "-")

        columns = st.columns([2, 1, 1, 1, 1])
        columns[0].markdown(
            f"**{market}** &nbsp; "
            f"<span style='background:{colour};color:white;padding:2px 10px;"
            f"border-radius:10px'>{label}</span>"
            + ("" if confirmed else " <em>unconfirmed</em>"),
            unsafe_allow_html=True,
        )
        columns[1].metric("Probability", f"{float(state.get('vol_state_probability') or 0):.0%}")
        columns[2].metric("Held", f"{int(state.get('vol_state_consecutive_bars') or 0)} bars")
        columns[3].metric(
            "Size multiplier", f"x{float(state.get('size_multiplier') or 1):.2f}"
        )
        columns[4].markdown(
            f"regime <span style='color:{REGIME_COLOURS.get(regime, '#666')}'>"
            f"<b>{regime}</b></span>",
            unsafe_allow_html=True,
        )

        notes = []
        if state.get("vol_state_flickering"):
            notes.append("flickering - sizing is in uncertainty mode")
        if state.get("vol_state_stale"):
            notes.append("stale - held through an inference failure")
        if state.get("data_delay_minutes"):
            notes.append(f"feed runs {state['data_delay_minutes']} min behind")
        if state.get("reason"):
            notes.append(str(state["reason"]))
        if notes:
            st.caption(" · ".join(notes))


def risk_panel(cfg, snapshot) -> None:
    """Section 7's two session-stopping conditions. Nothing else is a rule."""
    st.subheader("Risk status")
    st.caption(
        "Section 7 defines exactly two session-stopping conditions: the "
        "per-instrument daily loss cap and three consecutive losses. There is no "
        "drawdown-from-peak limit in the soul file, so none is shown."
    )
    if snapshot is None or not snapshot.counters:
        st.write("_No risk state recorded._")
        return

    trigger = int(cfg.get("risk.consecutive_loss_trigger"))
    for family, row in snapshot.counters.items():
        realised = float(row.get("realised_pnl", 0.0))
        losses = int(row.get("consecutive_losses", 0))
        cap_pct = _family_cap(cfg, family)
        cap_amount = snapshot.capital * cap_pct
        used = abs(min(0.0, realised))
        fraction = min(1.0, used / cap_amount) if cap_amount else 0.0

        st.markdown(f"**{family}**")
        columns = st.columns([3, 1, 1])
        columns[0].progress(
            fraction,
            text=f"day loss Rs {used:,.0f} of Rs {cap_amount:,.0f} "
                 f"({fraction * cap_pct:.2%} of {cap_pct:.0%})",
        )
        columns[1].metric("Consecutive losses", f"{losses}/{trigger}")
        columns[2].metric("Trades today", int(row.get("trades_today", 0)))

        if row.get("paused"):
            st.error(
                f"PAUSED for the session: {row.get('pause_reason', 'unknown')}. "
                f"No new entries. Open positions continue to be managed - their "
                f"stops are live."
            )
        elif fraction >= 0.8:
            st.warning("Within 20% of the daily cap.")


def positions_panel(snapshot) -> None:
    st.subheader("Positions")
    st.caption(
        "What Beast **believed** it held when the snapshot was written. The "
        "broker is the authority; a disagreement is reconciled at startup."
    )
    rows = snapshot.believed_positions if snapshot else []
    if not rows:
        st.write("_Flat._")
        return
    st.dataframe(
        [
            {
                "Market": row.get("market"),
                "Direction": row.get("direction"),
                "Entry": row.get("entry"),
                "Stop": row.get("stop"),
                "Target": row.get("target"),
                "Unrealised R": row.get("unrealised_r"),
                "Held": row.get("held"),
                "At risk": row.get("risk_amount"),
            }
            for row in rows
        ],
        width="stretch",
        hide_index=True,
    )


def signals_panel(journal_path: Path) -> None:
    st.subheader("Recent signals")
    rows = read_journal(
        journal_path,
        "SELECT timestamp_ist, market, direction, setup_type, regime, "
        "confluence_aligned, reason_line FROM signals "
        "ORDER BY timestamp_ist DESC LIMIT 20",
    )
    if not rows:
        st.write("_No signals recorded._")
        return
    st.dataframe(rows, width="stretch", hide_index=True)


def trades_panel(journal_path: Path) -> None:
    st.subheader("Closed trades")
    rows = read_journal(
        journal_path,
        "SELECT market, setup_type, direction, entry_time, exit_time, "
        "exit_reason, r_multiple, mae_r, mfe_r FROM trades "
        "WHERE exit_time IS NOT NULL ORDER BY entry_time DESC LIMIT 30",
    )
    if not rows:
        st.write("_No closed trades._")
        return

    total_r = sum(float(row.get("r_multiple") or 0.0) for row in rows)
    wins = sum(1 for row in rows if (row.get("r_multiple") or 0) > 0)
    columns = st.columns(3)
    columns[0].metric("Trades", len(rows))
    columns[1].metric("Win rate", f"{wins / len(rows):.0%}")
    columns[2].metric("Total", f"{total_r:+.2f}R")
    st.dataframe(rows, width="stretch", hide_index=True)


def gates_panel(journal_path: Path) -> None:
    """Section 9's "where do signals die" question."""
    st.subheader("Where signals die")
    st.caption(
        "Rejections by failing gate. A candidate is logged against the **first** "
        "gate it fails, which is what makes this histogram readable."
    )
    rows = read_journal(
        journal_path,
        "SELECT failed_gate, COUNT(*) AS count FROM rejections "
        "GROUP BY failed_gate ORDER BY count DESC",
    )
    if not rows:
        st.write("_No rejections recorded._")
        return
    st.bar_chart(
        {row["failed_gate"]: row["count"] for row in rows},
        horizontal=True,
    )


def alerts_panel(alerts_path: Path) -> None:
    st.subheader("Alerts")
    entries = read_log_stream(alerts_path, limit=40)
    if not entries:
        st.write("_No alerts._")
        return
    for entry in entries[:20]:
        payload = entry.get("payload") or {}
        kind = str(payload.get("kind", "ALERT"))
        message = str(entry.get("message", ""))
        stamp = str(entry.get("ts", ""))[11:19]
        if kind in ("CIRCUIT_BREAKER", "LOSS_LIMIT_PAUSE", "API_LOST",
                    "FEED_DOWN", "ERROR"):
            st.error(f"`{stamp}` {message}")
        elif kind in ("VOL_STATE_CHANGE", "REGIME_CHANGE", "MODEL_RETRAINED"):
            st.info(f"`{stamp}` {message}")
        else:
            st.warning(f"`{stamp}` {message}")


def system_panel(cfg, snapshot, main_entries: list[dict]) -> None:
    st.subheader("System")
    columns = st.columns(3)
    columns[0].metric("Mode", "PAPER" if cfg.is_paper else "LIVE")

    models = (snapshot.model_versions if snapshot else {}) or {}
    columns[1].metric(
        "Models",
        ", ".join(f"{market}" for market in models) if models else "none",
    )
    blockers = cfg.unset_blockers()
    columns[2].metric("Blocked config keys", len(blockers))

    if models:
        st.caption("Model versions: " + ", ".join(
            f"{market} `{version}`" for market, version in models.items()
        ))
    if blockers:
        st.warning(
            "Unset config values are refusing the affected trades. This is the "
            "soul file's designed refusal, not a bug - an unset threshold is "
            "treated as failing, not passing:\n\n"
            + "\n".join(f"- `{key}`" for key in blockers)
        )
    if snapshot and snapshot.session_summary:
        with st.expander("Last session summary"):
            st.json(snapshot.session_summary)
    if main_entries:
        with st.expander("Recent log lines"):
            for entry in main_entries[:30]:
                st.text(
                    f"{str(entry.get('ts', ''))[11:19]} "
                    f"{entry.get('level', ''):<7} {entry.get('message', '')}"
                )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_ts(text: str) -> datetime | None:
    try:
        return datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return None


def _humanise(delta: timedelta) -> str:
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    if seconds < 86400:
        return f"{seconds // 3600}h"
    return f"{seconds // 86400}d"


def _families(snapshot) -> dict[str, dict]:
    return snapshot.counters if snapshot else {}


def _family_cap(cfg, family: str) -> float:
    """Daily loss cap for a session family.

    v3.1's caps are per-instrument while the pause state is per-family, so the
    family's effective cap is the **tightest** of its instruments - that is the
    one that trips first and pauses the family.
    """
    caps = cfg.get("risk.daily_loss_cap")
    if family in caps:
        return float(caps[family])
    members = {"indian": ("nifty", "sensex"), "gold": ("gold",)}.get(family, ())
    values = [float(caps[name]) for name in members if name in caps]
    return min(values) if values else 0.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Beast", page_icon="=", layout="wide")
    cfg = get_config()

    with st.sidebar:
        st.header("Beast")
        st.caption(f"config `{cfg.path.name}`")
        auto = st.checkbox(f"Auto-refresh ({REFRESH_SECONDS}s)", value=True)
        if st.button("Refresh now"):
            st.rerun()
        st.divider()
        st.caption(
            "Read-only. This view has no controls that place, cancel or close "
            "anything. Section 8 puts every manual deviation behind an explicit "
            "confirmation step with full logging, and a web button is exactly "
            "the frictionless override that section exists to prevent."
        )
        st.caption(
            "It reads the files a running Beast leaves behind, and does not "
            "attach to the process - there is no control socket to attach to."
        )

    @st.fragment(run_every=REFRESH_SECONDS if auto else None)
    def live_view() -> None:
        """The values that change while Beast runs.

        Wrapped in a fragment so auto-refresh reruns and redraws only this
        block - a Streamlit-level diff of the metrics, tables and panels below
        - rather than reloading the whole browser page every cycle, which
        used to reset scroll position and re-fetch the sidebar for no reason.
        """
        snapshot = load_snapshot(cfg)
        directory = log_directory(cfg)
        journal_path = Path(str(cfg.get("monitoring.journal_db")))
        if not journal_path.is_absolute():
            journal_path = PROJECT_ROOT / journal_path

        header(cfg, snapshot)
        st.divider()

        left, right = st.columns([3, 2])
        with left:
            volatility_panel(snapshot, read_log_stream(directory / "regime.log"))
            st.divider()
            positions_panel(snapshot)
            st.divider()
            risk_panel(cfg, snapshot)
        with right:
            alerts_panel(directory / "alerts.log")
            st.divider()
            gates_panel(journal_path)

        st.divider()
        signals_panel(journal_path)
        trades_panel(journal_path)
        st.divider()
        system_panel(cfg, snapshot, read_log_stream(directory / "main.log", limit=60))

    live_view()


main()
