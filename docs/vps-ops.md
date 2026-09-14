# Running Beast unattended on the VPS

The ops layer: what supervises what, how to stop it, and what the exit codes
mean. The bridge itself (Wine, RPyC, the terminal) is in `mt5-bridge.md`.

## The units

```
xvfb.service                 virtual display
  └─ beast-mt5.service         MT5 terminal under Wine            Restart=always
       └─ beast-rpyc.service    RPyC bridge, 127.0.0.1:18812 ONLY  Restart=on-failure, PartOf beast-mt5
            └─ beast.service     Beast                              ExecStartPre=preflight; Restart=on-failure,
                                                                    RestartPreventExitStatus=0 1 2 3
beast-dashboard.service      Streamlit on 127.0.0.1:8501            Restart=always  (nginx :80 in front)
beast-watchdog.service       heartbeat / terminal / bridge / dashboard   Restart=always
```

Install after a deploy:

```bash
cd /home/beast-agent && git pull
cp ops/systemd/*.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable beast-mt5 beast-rpyc beast beast-dashboard beast-watchdog
systemctl restart beast-mt5     # the bridge follows (PartOf)
sleep 60
systemctl restart beast beast-watchdog
```

`loginctl enable-linger root` must be set once (Wine's socket dir lives in
`/run/user/0`). `ufw` allows 22 and 80 only; 18812 and 8501 are denied.

## Exit codes (`main.py`)

| Code | Meaning | systemd | watchdog |
|---|---|---|---|
| 0 | clean shutdown (SIGTERM/SIGINT) | no restart | no restart (`clean_shutdown: true` in heartbeat) |
| 1 | config or startup failure | no restart | pages, no restart |
| 2 | deliberate halt: 10 consecutive unhandled errors | no restart | pages, no restart |
| 3 | KILL flag (`flatten` completed, or flag present at start) | no restart | no restart |
| 10 | uncaught crash | restart after 10 s | restart if the heartbeat goes stale |

## Heartbeat

`heartbeat.json` (path `ops.heartbeat_path`) is written atomically at the end
of **every** loop cycle, including one that raised - `last_error` is set on
that cycle. Fields: `written_at, pid, mode, markets, loop_cycle_count,
open_positions[{market,direction,volume,sl_present,stop}], feed_ok,
entry_pause_reason, last_error, session_day, capital, realised_pnl_today,
clean_shutdown, mt5_state`.

`state_snapshot.json` is also written every `ops.snapshot_every_seconds` with
`in_progress: true`, so the dashboard shows the live session. The header
shows **LIVE** for such a snapshot and turns red if it is more than two
minutes old while marked live - that means the loop stopped writing.

## Watchdog

Every `ops.watchdog_interval_seconds` it checks the four things in the table
above. A heartbeat older than `ops.heartbeat_stale_seconds` is **re-read
once** on the next pass before anything happens - a slow cycle is not a dead
process. Then `systemctl restart beast.service`, unless:

- the last heartbeat has `clean_shutdown: true`;
- the KILL flag is set;
- `beast.service` exited with code 0-3 (deliberate) - it pages instead;
- more than `ops.max_restarts` happened inside `ops.restart_window_minutes`.
  Then it trips the **crash-loop guard**: stops restarting, raises CRITICAL,
  stays up reporting. Clear with `systemctl restart beast-watchdog` once the
  cause is fixed.

If the last heartbeat shows an open position, the restart alert is CRITICAL
and names the position and whether its stop was present: until Beast is back
and the reconcile has run, that position is on its venue-side stop alone.

The watchdog never places, modifies or cancels an order. It has no broker
handle. `python -m ops.watchdog --once` prints one pass and exits 0 if all
four are healthy.

## Kill switch

```
python -m ops.killswitch halt     --reason "..."   # entries off; exits keep running   (DEFAULT, safe)
python -m ops.killswitch flatten  --reason "..."   # close everything against plan, then halt
python -m ops.killswitch clear                     # resume
python -m ops.killswitch status
```

`halt` pauses entries only. Every open position keeps its resting stop and
the exit path keeps evaluating it - the same behaviour sections 4.6 / 5.6
already give a stale feed. Clearing the flag lifts the pause on the next
cycle; Beast keeps running throughout.

`flatten` is a close against plan and section 8 governs it. The command
demands the exact phrase `CONFIRM OVERRIDE: closing against plan` before it
writes anything, the flag carries that phrase, and the loop routes each close
through `core/override.py` again with it - the same friction step, not a
second path. Each close is logged with the R the plan would have produced.
Beast then exits 3; nothing restarts it until the flag is cleared.

Beast reads the flag at the top of every cycle, before any market is
evaluated. Preflight refuses to start with a flag present, and the watchdog
refuses to restart through one.

## Preflight

`python -m ops.preflight` - also `ExecStartPre` for `beast.service`, so a
failing box does not start. Eight rows, any FAIL blocks:

1. no config blockers for the markets in `broker.symbols`
2. `MT5_GOLD_*` present; `BEAST_ALLOW_LIVE` not set unless `account_mode: live`
3. bridge reachable, terminal and account answer, account is DEMO when `mode: paper`
4. the gold symbol resolves and selects
5. NTP synchronised (`timedatectl`) and local time within
   `ops.max_clock_drift_seconds` of the venue's tick clock
6. free disk >= `ops.min_free_disk_mb` under the log directory
7. no stale KILL flag
8. journal opens read-write and is not locked

## The reconcile, on the way back up

Startup step 6 (`core/reconcile.py`) asks each connected venue what it holds
under Beast's magic number - never the tracker, never the snapshot - and:

- matches each position to its plan (`signals` row with no `trades` row) by
  comment tag `beast:<market>:<day>:<plan_id>`, then direction + entry time;
- places a stop at the plan level when none is resting, **before any other
  startup step completes**; repairs one that covers less than the volume;
- leaves a **tighter** stop alone (6.3 ratchet) and a **wider** one
  untouched but flagged SAFE (section 8 - no widening, and no silent
  "correction" either);
- restores an absent target; cancels orphan pending orders;
- puts anything it cannot explain into SAFE mode: protective stop at
  `entry ± ops.safe_mode_stop_atr_multiple × ATR`, clamped inside any existing
  stop and outside `stops_level`, falling back to
  `instruments.gold.safe_mode_fallback_points`, and if that is unset it pages
  CRITICAL and places nothing; entries paused for that market; operator paged;
- refuses entries on any market whose venue could not be asked. Nothing is
  ever assumed flat.

The whole report is logged at ERROR when anything was repaired or disagreed.
`python -m ops.watchdog --once` and `journalctl -u beast -n 60` are the two
commands to read after any restart.

## The dashboard: live, and reachable without opening a port

Beast writes `heartbeat.json` at the end of **every** cycle (D-72) - open
positions, equity, risk headroom, regime and `vol_state`, the system state,
and short tails of signals, rejections and today's trades. The Streamlit
app reads that file fresh every `monitoring.dashboard_refresh_seconds` and
caches only the journal aggregates (`dashboard_journal_ttl_seconds`). It
does not attach to the process (D-37, D-53) and has no button that touches
an order (D-54).

**The status pill at the top is the point of the page.** Against the loop
interval: under 2x is **LIVE** (green), 2-6x **LAGGING** (amber, age shown),
beyond **STALE** (red banner, every number dimmed), no file or unparseable
**DOWN** (red banner, no numbers at all), a KILL flag **HALTED** (purple,
who and why), a clean shutdown **STOPPED** (grey). A closed market with a
fresh file is "LIVE / market closed"; a closed market with a stale file is
DOWN. The absolute `written_at` sits next to the age, always, and the MT5
server clock's drift from the host is shown in the header.

### Reaching it - path A, an SSH tunnel (recommended)

Nothing new listens on the public interface. In `~/.ssh/config` on the laptop:

```
Host beast
    HostName 200.141.7.185
    User root
    LocalForward 8501 127.0.0.1:8501
    ServerAliveInterval 30
    ServerAliveCountMax 3
```

Then `ssh -N beast` and open <http://localhost:8501>. On a phone, an SSH
client with port forwarding (Termius, a-Shell, Blink) does the same.

### Path B - nginx on :80 with basic auth (already in place)

`/etc/nginx/sites-available/beast-dashboard` proxies :80 to 127.0.0.1:8501
with the WebSocket upgrade headers (without them Streamlit sits on "Please
wait..." forever) and `auth_basic` from `/etc/nginx/.htpasswd`. Change the
password with `htpasswd /etc/nginx/.htpasswd beast`. TLS needs a hostname
for certbot; this box has only an IP, so :80 stays plain HTTP - use the
tunnel for anything you would not say over the air. `fail2ban` watches the
nginx auth log.

### The unit

`beast-dashboard.service` runs `streamlit run monitoring/streamlit_app.py`
as user **`beastview`**, a member of group `beast` with read access to the
checkout, the journal and `heartbeat.json`, and write access only to its own
home. `.streamlit/config.toml` binds `127.0.0.1`; the command line passes
no address so nothing can override it. `ss -tlnp | grep 8501` must show
`127.0.0.1:8501` and nothing on `0.0.0.0`.

```bash
systemctl status beast-dashboard          # state
journalctl -u beast-dashboard -n 50       # logs
systemctl restart beast-dashboard         # after a deploy
```

## On the local Windows box

The same `ops/` package runs there for development; only `systemctl` and
`timedatectl` are absent, so the watchdog's restart action and preflight's
clock check report as unavailable rather than passing.
