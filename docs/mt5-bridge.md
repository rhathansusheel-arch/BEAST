# The MT5 bridge on the VPS

How Beast reaches MetaTrader 5 from a Linux host, what runs where, and what to
do when it breaks. Everything here was set up on `srv1960318.hstgr.cloud`
(Hostinger KVM, Ubuntu 22.04.5, 2 vCPU, 7.8 GB) on 2026-09-11.

## Topology

```
MT5 terminal (Wine)  <->  mt5linux server (Wine-side Python 3.12)  <->  RPyC on 127.0.0.1:18812
                                                                     <->  Beast (native Python 3.12 venv)
```

Four links. `broker/mt5_connection.py` names which one broke: server
unreachable, terminal lost its broker, Algo Trading off, or a call hung inside
Wine.

## Pinned versions

| Component | Version | Why it matters |
|---|---|---|
| Ubuntu | 22.04.5 LTS | stock Python is 3.10 - **not enough**, see below |
| Native Python (Beast venv) | 3.12.13 (deadsnakes) | `mt5linux 1.1.1` uses 3.12-only f-string syntax; 3.10/3.11 fail at `import mt5linux` |
| Wine | 6.0.3 (Ubuntu repack) | prefix must identify as **Windows 10**: Python 3.11+'s installer refuses Wine's default Win7 |
| Windows Python (in Wine) | 3.12.10 at `C:\Python312` | same 3.12 requirement on the server side |
| MetaTrader5 (Wine-side pip) | 5.0.6180 | the only thing that talks to the terminal |
| mt5linux | 1.1.1 (both sides) | client and server must match |
| rpyc | 6.0.2 | transport |
| MT5 terminal | build from `mt5setup.exe /auto` | at `C:\Program Files\MetaTrader 5`, run with `/portable` |

## Paths

| What | Where |
|---|---|
| Beast checkout | `/home/beast-agent` (branch `main`) |
| Beast venv | `/home/beast-agent/venv` |
| Secrets | `/home/beast-agent/.env` (mode 600, gitignored) |
| Wine prefix | `/root/.wine` (`WINEARCH=win64`) |
| Terminal | `/root/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe` |
| Terminal data (portable) | same directory: `logs/`, `config/`, `MQL5/` |
| Wine-side Python | `/root/.wine/drive_c/Python312/python.exe` |
| Virtual display | Xvfb `:1`, 1024x768x16 |

## systemd units, in start order

```
xvfb.service          Xvfb :1
  └─ mt5-terminal.service   wine terminal64.exe /portable        (Requires xvfb)
       └─ mt5linux.service  wine python.exe -m mt5linux --host 127.0.0.1 -p 18812   (Requires mt5-terminal)
            └─ beast.service    venv/bin/python main.py --no-dashboard             (Wants mt5linux)
```

All four are enabled at boot. `beast.service` is `Restart=always`; the Wine
units are `Restart=on-failure` with a 5-in-300s limit so a broken terminal
does not restart-loop forever.

**Two systemd traps that cost time here:**

1. `ExecStart` treats `\t` as a TAB. `C:\Program Files\MetaTrader 5\terminal64.exe`
   becomes `MetaTrader 5<TAB>erminal64.exe`. Use Unix paths in units - Wine
   accepts them.
2. `ExecStopPost=/usr/bin/wineserver -k` exits 1 when nothing is running. Prefix
   it with `-` or the unit reports failure on a clean stop.

## Security boundary

RPyC classic mode is remote code execution for anyone who can reach the port.
The server binds `--host 127.0.0.1` and `ufw` additionally denies 18812. Only
22/tcp is open. **Never widen either.** `scripts/mt5_check.py` fails hard if the
configured endpoint is not loopback.

## Credentials

Environment only. `beast.service` loads `/home/beast-agent/.env` via
`EnvironmentFile`; `core/config.py` loads the same file for interactive runs.

```
MT5_GOLD_LOGIN=      numeric account number - NOT an MQL5.community username
MT5_GOLD_PASSWORD=
MT5_GOLD_SERVER=     exactly as the terminal shows it, e.g. MetaQuotes-Demo
BEAST_ALLOW_LIVE=    leave empty on a demo account
```

The Wine terminal must ALSO be logged into the same account, once, by hand:
the Python API attaches to a logged-in terminal, it does not perform the login
itself. See "First login" below.

## Bring-up checklist

```bash
systemctl status xvfb mt5-terminal mt5linux beast --no-pager
ss -tlnp | grep 18812                      # must show 127.0.0.1:18812 only
cd /home/beast-agent && venv/bin/python scripts/mt5_check.py
```

`mt5_check.py` exits 0 only when the handshake, account-mode guard, symbol
spec, live quote and history all pass. It prints the server clock offset and
latency p50/p95 - those are the numbers to record after any change.

## First login (manual, once per account)

The terminal under Xvfb has no visible window. To log a new account in:

```bash
apt install -y x11vnc
x11vnc -display :1 -localhost -once -nopw &
ssh -L 5900:127.0.0.1:5900 root@200.141.7.185     # from your PC, then VNC to localhost:5900
```

In the terminal: **File > Login to Trade Account**, enter login / password /
server, tick "Save password". Then **Tools > Options > Expert Advisors > Allow
algorithmic trading** (or Ctrl+E). Close VNC; the terminal keeps running.

Because the terminal runs `/portable`, the saved login lives under the
terminal directory and survives restarts.

## Recovery

| Symptom | Layer | Action |
|---|---|---|
| `mt5_check.py`: "mt5linux server ... unreachable" | RPyC server | `systemctl restart mt5linux` |
| "terminal is running but has no broker connection" | terminal <-> broker | check the account still exists (MetaQuotes demo accounts get purged); re-login via VNC |
| "Algo Trading is disabled" | terminal setting | Ctrl+E via VNC |
| "exceeded Ns - the terminal or Wine is hung" | Wine | `systemctl restart mt5-terminal mt5linux` (order matters) |
| Beast log: `MT5 bridge ... -> HALTED` | any | Beast has stopped opening entries; exits still run. Fix the layer named in the log, then `systemctl restart beast` |
| `Invalid account` in the terminal log | account deleted | open a new demo account; update `.env`; re-login via VNC |

Terminal log: `/root/.wine/drive_c/Program Files/MetaTrader 5/logs/YYYYMMDD.log`.
Beast logs: `/home/beast-agent/logs/{main,trades,alerts,regime}.log` and
`journalctl -u beast`.

## When a terminal update breaks Wine

MetaQuotes pushes terminal builds automatically. If a new build fails under
Wine 6.0.3 (symptom: `mt5-terminal` restart-loops, or hangs at startup):

1. Stop the units: `systemctl stop beast mt5linux mt5-terminal`.
2. The previous build is at `.../MetaTrader 5/terminal64.exe.bak` if the
   updater kept one; otherwise reinstall from a known-good `mt5setup.exe`.
3. Block auto-update: in the terminal, **Tools > Options > Server**, untick
   "Enable news" and set the update policy off (via VNC).
4. Longer term, Wine 8+ from WineHQ's repo is the fix; that is an OS-level
   change and should be tested on a snapshot first.

## Full rebuild from scratch

The scripts under `scripts/vps/` (if present) reproduce every step above.
Without them, the order is: apt base packages -> deadsnakes Python 3.12 ->
clone + venv -> Wine + Xvfb + `winetricks win10` -> Windows Python 3.12 in
Wine -> `pip install MetaTrader5 mt5linux` in Wine -> `mt5setup.exe /auto` ->
the four systemd units -> `.env` -> `ufw` -> manual first login.
