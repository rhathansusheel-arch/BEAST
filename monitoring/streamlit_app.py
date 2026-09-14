"""Streamlit UI - a browser view of Beast's live state and its journal.

Run it with::

    streamlit run monitoring/streamlit_app.py

Two sources, two cadences (D-74)
--------------------------------
* ``heartbeat.json`` - the live state Beast writes every cycle (D-72). Small,
  read fresh on every refresh, never cached. Everything on the page that can
  change between two cycles - the status pill, positions, risk, regime,
  system - comes from here.
* ``logs/beast_journal.sqlite`` and the log tails - signals, rejections,
  closed trades, the equity curve. These queries are the expensive part and
  are cached for ``monitoring.dashboard_journal_ttl_seconds`` on the session
  day, so a five-second refresh does not re-scan the journal five times a
  minute.

It does **not** attach to the running process (D-37, D-53). Beast exposes no
IPC or control socket, and adding a listener to the host holding the broker
credentials so a browser tab can be one cycle fresher is not a trade worth
making. What this shows is the file Beast already writes, one cycle old.

It is strictly read-only (D-54). There are no buttons that place, cancel,
close, resize or override anything. Section 8 puts every manual deviation
behind an explicit typed confirmation with full logging; a web button is
precisely the frictionless override that section exists to prevent.
Stopping Beast is ``Ctrl-C`` on the process or ``python -m ops.killswitch``
over SSH, deliberately.

The status pill is the most important element on the page
----------------------------------------------------------
"Beast is quiet" and "Beast is dead" look identical on a naive dashboard.
Here the age of the live file against the loop interval decides LIVE /
LAGGING / STALE / DOWN, a KILL flag shows HALTED and a clean shutdown shows
STOPPED, and when the file is stale or missing every number below is dimmed
or withheld so a glance at a phone cannot read a three-hour-old P&L as now.

Vocabulary
----------
``vol_state`` (CALM / NORMAL / TURBULENT / UNKNOWN) is a volatility read, not
a direction, and is shown beside - never merged with - section 4.4's
``regime`` (TREND_UP / TREND_DOWN / RANGE). There is no allocation, leverage,
drawdown-from-peak or BULL/BEAR anywhere on the page (D-40 to D-43).
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from core.config import get_config
from monitoring.logger import log_directory
from ops.heartbeat import (
    DOWN, HALTED, LAGGING, LIVE, STALE, STOPPED, age_seconds, classify, read_live_state,
)

VOL_COLOURS = {"CALM": "#2e7d32", "NORMAL": "#0277bd", "TURBULENT": "#c62828", "UNKNOWN": "#f9a825"}
REGIME_COLOURS = {"TREND_UP": "#2e7d32", "TREND_DOWN": "#c62828", "RANGE": "#f9a825"}
PILL = {
    LIVE: ("#2e7d32", "LIVE"),
    LAGGING: ("#f9a825", "LAGGING"),
    STALE: ("#c62828", "STALE"),
    DOWN: ("#c62828", "DOWN"),
    HALTED: ("#6a1b9a", "HALTED"),
    STOPPED: ("#616161", "STOPPED (clean)"),
}
NUMBERS_OK = {LIVE, LAGGING, HALTED}      # states in which the numbers may be shown undimmed


# ---------------------------------------------------------------------------
# Reading - the live file (never cached) and the journal (cached, read-only)
# ---------------------------------------------------------------------------


def live_state_path(cfg) -> Path:
    raw = Path(str(cfg.get("ops.heartbeat_path", "./heartbeat.json")))
    return raw if raw.is_absolute() else PROJECT_ROOT / raw


def journal_path(cfg) -> Path:
    raw = Path(str(cfg.get("monitoring.journal_db")))
    return raw if raw.is_absolute() else PROJECT_ROOT / raw


def read_journal(path: Path, query: str, params: tuple = ()) -> tuple[list[dict[str, Any]], str | None]:
    """Query the journal read-only.

    ``mode=ro`` plus ``PRAGMA query_only`` so this process can never take a
    write lock on a database the trading loop is writing. A lock held by the
    writer is reported, not raised - the page keeps its last good values.
    """
    if not path.exists():
        return [], None
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=1.0)
    except sqlite3.Error as error:
        return [], f"journal unavailable: {error}"
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        with closing(connection.cursor()) as cursor:
            return [dict(row) for row in cursor.execute(query, params).fetchall()], None
    except sqlite3.OperationalError as error:
        return [], f"journal busy ({error}); showing last good values"
    except sqlite3.Error as error:
        return [], f"journal error: {error}"
    finally:
        connection.close()


def tail_log_stream(path: Path, max_kb: int = 256, limit: int = 200) -> list[dict[str, Any]]:
    """Newest-first records from the last ``max_kb`` of a JSON-lines stream.

    The streams rotate at 10 MB; reading a whole one every few seconds would
    peg a small VPS, so this seeks to the end and reads only the tail, then
    drops the first line, which is almost certainly a partial one.
    """
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            start = max(0, size - max_kb * 1024)
            handle.seek(start)
            chunk = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    lines = chunk.splitlines()
    if start > 0 and lines:
        lines = lines[1:]                    # partial first line
    for line in reversed(lines):
        if len(entries) >= limit:
            break
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def _journal_cached(ttl: int):
    """Journal aggregates keyed on the session day, cached for ``ttl`` seconds."""

    @st.cache_data(ttl=ttl, show_spinner=False)
    def load(path_str: str, day: str) -> dict[str, Any]:
        path = Path(path_str)
        out: dict[str, Any] = {"notes": []}
        rows, note = read_journal(
            path, "SELECT timestamp_ist, market, direction, setup_type, regime, "
                  "confluence_aligned, reason_line FROM signals ORDER BY timestamp_ist DESC LIMIT 20")
        out["signals"] = rows
        if note:
            out["notes"].append(note)
        rows, note = read_journal(
            path, "SELECT timestamp, instrument, direction, failed_gate, gate_detail "
                  "FROM rejections WHERE timestamp >= ? ORDER BY id DESC LIMIT 50", (day,))
        out["rejections"] = rows
        if note:
            out["notes"].append(note)
        rows, note = read_journal(
            path, "SELECT failed_gate, COUNT(*) AS count FROM rejections WHERE timestamp >= ? "
                  "GROUP BY failed_gate ORDER BY count DESC", (day,))
        out["gates"] = rows
        rows, note = read_journal(
            path, "SELECT market, setup_type, direction, entry_time, exit_time, exit_reason, "
                  "r_multiple, mae_r, mfe_r FROM trades WHERE exit_time IS NOT NULL "
                  "ORDER BY exit_time DESC LIMIT 50")
        out["trades"] = rows
        rows, note = read_journal(
            path, "SELECT exit_time, r_multiple FROM trades WHERE exit_time IS NOT NULL "
                  "AND exit_time >= ? ORDER BY exit_time", (day,))
        out["curve"] = rows
        return out

    return load


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _pill(colour: str, text: str) -> str:
    return (f"<span style='background:{colour};color:white;padding:4px 14px;"
            f"border-radius:14px;font-weight:600;font-size:1.05em'>{text}</span>")


def _dim(active: bool) -> str:
    return "" if active else "opacity:0.35;"


def _fmt_age(seconds: float | None) -> str:
    if seconds is None:
        return "no file"
    seconds = int(seconds)
    if seconds < 90:
        return f"{seconds}s"
    if seconds < 5400:
        return f"{seconds // 60}m {seconds % 60}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60}m"


def _parse(stamp) -> datetime | None:
    try:
        value = datetime.fromisoformat(str(stamp))
    except (TypeError, ValueError):
        return None
    return value


# ---------------------------------------------------------------------------
# Panels - live (from the file)
# ---------------------------------------------------------------------------


def header(state: dict | None, verdict: str, age: float | None, cfg) -> None:
    colour, label = PILL[verdict]
    written = state.get("written_at") if state else None
    detail = f"written {written}" if written else f"expected at `{live_state_path(cfg)}`"
    if verdict == HALTED and state:
        kill = (state.get("system") or {}).get("kill_flag") or {}
        label = f"HALTED ({kill.get('mode', '?')})"
        detail += f" · {kill.get('reason', '')} — set by {kill.get('set_by', '?')} at {kill.get('set_at', '?')}"
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:16px;flex-wrap:wrap'>"
        f"<span style='font-size:2em;font-weight:700'>BEAST</span>{_pill(colour, label)}"
        f"<span style='opacity:0.8'>age {_fmt_age(age)} · {detail}</span></div>",
        unsafe_allow_html=True,
    )

    if verdict == DOWN:
        st.error("**No live state.** The file is missing or unreadable. Beast is not writing - "
                 "it is not running, or it cannot write to its directory. Nothing below is current.")
        return
    if verdict == STALE:
        st.error(f"**Beast stopped writing {_fmt_age(age)} ago.** The loop is not running. "
                 f"Every number below is from that moment and is dimmed. Check `heartbeat.json`, "
                 f"the watchdog, and the resting stops at the broker.")
    elif verdict == LAGGING:
        st.warning(f"Live file is {_fmt_age(age)} old - more than two loop intervals. "
                   f"A slow cycle, or Beast is about to be declared stale.")
    elif verdict == STOPPED:
        st.info("Beast shut down cleanly. Numbers are from its last cycle.")

    account = (state or {}).get("account") or {}
    columns = st.columns(6)
    columns[0].metric("Mode", str(state.get("mode", "?")).upper())
    mode_text = str(account.get("account_mode", "?")).upper()
    if not account.get("account_verified"):
        mode_text = f"UNVERIFIED (cfg: {str(account.get('account_mode_configured', '?')).upper()})"
    columns[1].metric("Account", mode_text,
                      help="Verified by the MT5 handshake only. UNVERIFIED means the bridge "
                           "has not completed a handshake this session.")
    columns[2].metric("Server", account.get("server") or "—")
    columns[3].metric("Currency", account.get("currency") or "—")
    equity = account.get("equity")
    columns[4].metric("Equity" if equity is not None else "Session capital",
                      f"{equity:,.2f}" if equity is not None
                      else f"{float(account.get('session_capital') or 0):,.0f}",
                      help=f"source: {account.get('capital_source', '?')}")
    drift = None
    server_time = _parse(state.get("broker_server_time"))
    local_time = _parse(state.get("written_at_local"))
    if server_time is not None and local_time is not None:
        utc_written = _parse(state.get("written_at"))
        if utc_written is not None:
            expected = utc_written.astimezone(timezone.utc)
            drift = (server_time.astimezone(timezone.utc) - expected).total_seconds()
    columns[5].metric("Clock drift", f"{drift:+.0f}s" if drift is not None else "n/a",
                      help="MT5 server clock minus this host's clock at write time. Session "
                           "windows depend on this; drift fails silently.")
    if drift is not None and abs(drift) > 30:
        st.error(f"Clock drift {drift:+.0f}s between this host and the MT5 server. "
                 f"Session windows, the last-entry cutoff and the flatten time are all wrong "
                 f"by that much.")


def markets_panel(state: dict, active: bool) -> None:
    """Section 4.4 regime and vol_state, side by side, never merged (D-40)."""
    st.subheader("Markets · regime & volatility")
    st.caption("`regime` is section 4.4 (ADX/DI on the bias TF) and decides which setups are "
               "permitted. `vol_state` is the HMM's volatility read and only ever shrinks size. "
               "They are different things and are shown on different lines.")
    for row in state.get("markets") or []:
        market = row["market"]
        is_open = bool(row.get("is_open"))
        status = "market open" if is_open else "market closed"
        pause = row.get("entry_pause_reason")
        regime = row.get("regime") or {}
        vol = row.get("vol_state") or {}
        label = str(vol.get("label") or "UNKNOWN")
        st.markdown(
            f"<div style='{_dim(active)}'><b>{market}</b> &nbsp; "
            f"<span style='opacity:0.8'>{status} · session {row.get('session_day')} · "
            f"phase {row.get('phase') or '—'}</span></div>",
            unsafe_allow_html=True,
        )
        cols = st.columns([2, 2, 1, 1, 1])
        adx = regime.get("adx")
        adx_text = f" · ADX {float(adx):.1f}" if adx is not None else ""
        cols[0].markdown(
            f"<div style='{_dim(active)}'>regime: "
            f"<span style='color:{REGIME_COLOURS.get(regime.get('label'), '#666')}'>"
            f"<b>{regime.get('label') or '—'}</b></span>{adx_text}</div>",
            unsafe_allow_html=True,
        )
        cols[1].markdown(
            f"<div style='{_dim(active)}'>vol_state: "
            f"<span style='background:{VOL_COLOURS.get(label, '#666')};color:white;"
            f"padding:2px 10px;border-radius:10px'>{label}</span>"
            f"{'' if vol.get('is_confirmed') else ' <em>unconfirmed</em>'}</div>",
            unsafe_allow_html=True,
        )
        cols[2].metric("Probability", f"{float(vol.get('probability') or 0):.0%}")
        cols[3].metric("Size ×", f"{float(vol.get('size_multiplier') or 1):.2f}")
        cols[4].metric("Feed", "ok" if row.get("feed_ok") else f"down ({row.get('feed_failure_count', 0)})")
        notes = []
        if pause:
            notes.append(f"entries paused: {pause}")
        if vol.get("is_flickering"):
            notes.append("flickering - sizing in uncertainty mode")
        if vol.get("data_delay_minutes"):
            notes.append(f"feed runs {vol['data_delay_minutes']} min behind")
        if row.get("spread") is not None:
            notes.append(f"spread {float(row['spread']):.3f}")
        if notes:
            st.caption(" · ".join(notes))


def positions_panel(state: dict, active: bool) -> None:
    st.subheader("Open positions")
    rows = state.get("positions") or []
    if not rows:
        st.write("_Flat._")
        return
    st.caption("`stop at broker` is the reconcile's answer, not an assumption. Red means the "
               "position is running on nothing - fix it at the venue now.")
    table = []
    for p in rows:
        table.append({
            "market": p.get("market"),
            "dir": p.get("direction"),
            "volume": p.get("volume"),
            "entry": p.get("entry_price"),
            "now": p.get("current_price"),
            "unrealised": p.get("unrealised_pnl"),
            "R": p.get("unrealised_r"),
            "stop": p.get("stop_price"),
            "target": p.get("target_price"),
            "stop at broker": "🟢 yes" if p.get("stop_present_at_broker") else "🔴 NO",
            "trail": ("armed @ " + f"{p.get('trail_level')}") if p.get("trail_armed") else "—",
            "in trade": _fmt_age(p.get("time_in_trade_seconds")),
            "SAFE": "⚠ yes" if p.get("safe_mode") else "",
        })
    st.dataframe(table, width="stretch", hide_index=True)
    for p in rows:
        if not p.get("stop_present_at_broker"):
            st.error(f"{p.get('market')}: NO STOP AT THE BROKER.")
        if p.get("safe_mode"):
            st.warning(f"{p.get('market')}: SAFE mode - Beast is protecting, not managing, this "
                       f"position. Entries for the market are paused.")


def risk_panel(state: dict, active: bool) -> None:
    """Section 7's two session-stopping conditions, plus open risk. Nothing else is a rule."""
    st.subheader("Risk")
    st.caption("The daily loss cap and three consecutive losses are the only two conditions that "
               "stop a session (section 7). Open risk is what every open position loses if every "
               "stop fills. No leverage, no allocation, no drawdown-from-peak - Beast has no "
               "such rules and showing one would imply an enforcement that does not exist.")
    risk = state.get("risk") or {}
    account = state.get("account") or {}
    capital = float(account.get("session_capital") or 0.0)
    realised = float(risk.get("realised_pnl_today") or 0.0)
    cap = float(risk.get("daily_loss_cap_amount") or 0.0)
    used = float(risk.get("daily_loss_cap_pct_used") or 0.0)
    cols = st.columns([3, 1, 1, 1])
    cols[0].progress(min(1.0, used),
                     text=f"day P&L {realised:+,.2f} · loss cap {cap:,.0f} · {used:.0%} of cap used")
    cols[1].metric("Consecutive losses",
                   f"{risk.get('consecutive_losses', 0)}/{risk.get('consecutive_loss_trigger', 3)}")
    open_risk = float(risk.get("open_risk_amount") or 0.0)
    cols[2].metric("Open risk", f"{open_risk:,.2f}",
                   f"{float(risk.get('open_risk_pct') or 0):.2%} of capital" if capital else None,
                   delta_color="off")
    cols[3].metric("Concurrent", f"{risk.get('concurrent_open', 0)}/{risk.get('concurrent_max', 0)}")
    if risk.get("session_paused"):
        st.error(f"SESSION PAUSED: {risk.get('pause_reason') or 'section 7'}. No new entries; "
                 f"open positions keep their stops.")


def system_panel(state: dict, active: bool) -> None:
    st.subheader("System")
    system = state.get("system") or {}
    cols = st.columns(5)
    cols[0].metric("Uptime", _fmt_age(state.get("uptime_seconds")))
    cols[1].metric("Cycles", f"{int(state.get('cycle_count') or 0):,}")
    cols[2].metric("Loop", f"{float(state.get('last_cycle_duration_ms') or 0):.0f} ms",
                   help=f"interval {state.get('loop_interval_seconds')}s · "
                        f"live-state write {state.get('heartbeat_write_ms')} ms")
    cols[3].metric("Bridge", "ok" if system.get("bridge_ok") else "down")
    free = system.get("log_dir_free_mb")
    cols[4].metric("Free disk", f"{float(free):,.0f} MB" if free is not None else "n/a")

    brokers = system.get("brokers") or []
    st.caption("brokers: " + " · ".join(
        f"{b.get('name')} {'connected' if b.get('connected') else 'DOWN'}"
        f"{' (' + str(b.get('state')) + ')' if b.get('state') else ''}" for b in brokers) or "none")
    breakers = system.get("circuit_breakers") or {}
    lines = []
    if breakers.get("session_pause"):
        lines.append("session pause TRIPPED")
    for market, reason in (breakers.get("feed_pause") or {}).items():
        lines.append(f"feed pause {market}: {reason}")
    if int(breakers.get("error_streak") or 0):
        lines.append(f"error streak {breakers['error_streak']}")
    if lines:
        st.warning("circuit breakers: " + "; ".join(lines))
    error = system.get("last_error")
    if error:
        st.error(f"last error at {error.get('at')}: {error.get('message')}")
    if int(system.get("heartbeat_write_failures") or 0):
        st.warning(f"{system['heartbeat_write_failures']} live-state write failure(s) this "
                   f"session - trading unaffected, this page may lag.")


# ---------------------------------------------------------------------------
# Panels - journal (cached)
# ---------------------------------------------------------------------------


def journal_panels(data: dict, active: bool) -> None:
    for note in data.get("notes") or []:
        st.caption(f"⚠ {note}")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Recent signals")
        if data.get("signals"):
            st.dataframe(data["signals"], width="stretch", hide_index=True)
        else:
            st.write("_No signals recorded._")
    with right:
        st.subheader("Where signals die today")
        st.caption("Rejections by the **first** gate a candidate failed - the histogram section 9 "
                   "reads. If 90% die at one gate, that gate is the story.")
        gates = data.get("gates") or []
        if gates:
            st.bar_chart({row["failed_gate"]: row["count"] for row in gates}, horizontal=True)
        else:
            st.write("_No rejections today._")

    st.subheader("Recent rejections")
    if data.get("rejections"):
        st.dataframe([{"at": r.get("timestamp"), "market": r.get("instrument"),
                       "dir": r.get("direction"), "gate": r.get("failed_gate"),
                       "reason": r.get("gate_detail")} for r in data["rejections"]],
                     width="stretch", hide_index=True)
    else:
        st.write("_None._")

    st.subheader("Closed trades")
    trades = data.get("trades") or []
    if trades:
        total_r = sum(float(t.get("r_multiple") or 0.0) for t in trades)
        wins = sum(1 for t in trades if (t.get("r_multiple") or 0) > 0)
        cols = st.columns(3)
        cols[0].metric("Trades", len(trades))
        cols[1].metric("Win rate", f"{wins / len(trades):.0%}")
        cols[2].metric("Total", f"{total_r:+.2f}R")
        st.dataframe(trades, width="stretch", hide_index=True)
    else:
        st.write("_No closed trades._")

    st.subheader("Equity curve today · in R")
    st.caption("Cumulative R, not currency - consistent with `--compare` (D-38). One R is the "
               "risk each trade was entered with.")
    curve = data.get("curve") or []
    if curve:
        running, points = 0.0, {}
        for row in curve:
            running += float(row.get("r_multiple") or 0.0)
            points[str(row.get("exit_time"))] = running
        st.line_chart(points)
    else:
        st.write("_No closed trades today._")


def alerts_panel(cfg, active: bool) -> None:
    st.subheader("Alerts")
    directory = log_directory(cfg)
    kb = int(cfg.get("monitoring.dashboard_log_tail_kb", 256))
    entries = tail_log_stream(directory / "alerts.log", max_kb=kb, limit=40)
    if not entries:
        st.write("_No alerts._")
        return
    for entry in entries[:20]:
        payload = entry.get("payload") or {}
        kind = str(payload.get("kind", "ALERT"))
        message = str(entry.get("message", ""))
        stamp = str(entry.get("ts", ""))[11:19]
        if kind in ("CIRCUIT_BREAKER", "LOSS_LIMIT_PAUSE", "API_LOST", "FEED_DOWN", "ERROR"):
            st.error(f"`{stamp}` {message}")
        elif kind in ("VOL_STATE_CHANGE", "REGIME_CHANGE", "MODEL_RETRAINED"):
            st.info(f"`{stamp}` {message}")
        else:
            st.warning(f"`{stamp}` {message}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Beast", page_icon="=", layout="wide")
    cfg = get_config()
    refresh = int(cfg.get("monitoring.dashboard_refresh_seconds", 5))
    ttl = int(cfg.get("monitoring.dashboard_journal_ttl_seconds", 30))

    with st.sidebar:
        st.header("Beast")
        st.caption(f"config `{cfg.path.name}` · live file `{live_state_path(cfg).name}`")
        auto = st.checkbox(f"Auto-refresh ({refresh}s)", value=True)
        st.divider()
        st.caption("Read-only. Nothing here places, cancels, closes, resizes or overrides "
                   "anything. Section 8 puts every manual deviation behind a typed "
                   "confirmation with full logging; a web button is exactly the "
                   "frictionless override that section exists to prevent.")
        st.caption("It reads the file Beast writes every cycle and does not attach to the "
                   "process - there is no control socket to attach to.")
        blockers = cfg.unset_blockers()
        if blockers:
            st.warning("Unset config values are refusing trades:\n\n"
                       + "\n".join(f"- `{k}`" for k in blockers))

    @st.fragment(run_every=refresh if auto else None)
    def live_view() -> None:
        """Everything that can change between two cycles. Reads the file fresh; no cache."""
        state = read_live_state(live_state_path(cfg))
        interval = float((state or {}).get("loop_interval_seconds") or refresh)
        verdict, age = classify(state, interval)
        header(state, verdict, age, cfg)
        if state is None:
            return
        active = verdict in NUMBERS_OK
        st.divider()
        positions_panel(state, active)
        st.divider()
        risk_panel(state, active)
        st.divider()
        markets_panel(state, active)
        st.divider()
        system_panel(state, active)

    @st.fragment(run_every=refresh if auto else None)
    def journal_view() -> None:
        """The expensive part, cached on the session day for ``ttl`` seconds."""
        state = read_live_state(live_state_path(cfg))
        verdict, _ = classify(state, float((state or {}).get("loop_interval_seconds") or refresh))
        active = verdict in NUMBERS_OK
        day = datetime.now().strftime("%Y-%m-%d")
        data = _journal_cached(ttl)(str(journal_path(cfg)), day)
        st.divider()
        alerts_panel(cfg, active)
        st.divider()
        journal_panels(data, active)

    live_view()
    journal_view()


main()
