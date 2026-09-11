# The MT5 bridge on the VPS

How Beast reaches MetaTrader 5 from a Linux host, what runs where, and what to
do when it breaks. Set up on `srv1960318.hstgr.cloud` (Hostinger KVM, Ubuntu
22.04.5, 2 vCPU, 7.8 GB) on 2026-09-11/12. Everything below was verified
end-to-end: the MetaQuotes-Demo server answered a login attempt through every
layer, in 24.5 s.

## Topology

```
MT5 terminal (Wine 11)  <->  RPyC classic server (Wine-side Python 3.11)  <->  127.0.0.1:18812
                                                                          <->  Beast (native Python 3.12 venv)
```

Beast's `broker/mt5_connection.py` binds to `conn.modules.MetaTrader5` on a
plain RPyC classic connection. The `mt5linux` client class is not used: the
server it ships is this same bare classic server, and its 1.1.1 client insists
on managing a Docker container. Only `rpyc` is needed on the Linux side.

## Pinned versions - and why each one

| Component | Version | Why it matters |
|---|---|---|
| Wine | **11.0** (WineHQ stable, `/opt/wine-stable`) | terminal build 6191 refuses anything below 10.0: *"unstable and unsupported Wine 6.0.3, please upgrade to Wine 10.0 or later"* in its own log |
| Native Python (Beast venv) | 3.12.13 (deadsnakes) | stock 3.10 is fine now that mt5linux is gone; 3.12 stays because the venv was built on it |
| Windows Python (in Wine) | 3.11.9 at `C:\Python311` | 3.12's pip aborts on Wine (`propsys.dll.VariantToString`); 3.11 works. Relocatable - copied into the fresh prefix |
| MetaTrader5 (Wine-side pip) | 5.0.6180 | the only thing that talks to the terminal |
| numpy (Wine-side) | **1.23.5** | every release from 1.24 calls `fetestexcept`, which Wine 6's apiset lacked. Harmless on Wine 11 but proven, so it stays |
| rpyc | 6.0.2 both sides | transport; must be protocol-compatible across the bridge |
| MT5 terminal | build 6191 | at `C:\Program Files\MetaTrader 5`, run with `/portable` |

Wine-side packages are recorded in `C:\mt5libs\requirements-wine.txt`.

## Paths

| What | Where |
|---|---|
| Beast checkout | `/home/beast-agent` (branch `main`) |
| Beast venv | `/home/beast-agent/venv` |
| Secrets | `/home/beast-agent/.env` (mode 600, gitignored) - loaded by `core.config`, **not** by systemd |
| Wine prefix | `/root/.wine` (`WINEARCH=win64`, Windows 10 identity) |
| Previous Wine 6 prefix | `/root/.wine-v6` (kept; safe to delete) |
| Terminal | `/root/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe` |
| Terminal data (portable) | same directory: `logs/`, `config/`, `MQL5/` |
| RPyC server script | `/root/.wine/drive_c/mt5libs/mt5server.py` (= `mt5linux/__main__.py`, one line) |
| Wine-side Python | `/root/.wine/drive_c/Python311/python.exe` |
| Virtual display | Xvfb `:1`, 1280x800x24 |
| Dashboard | `beast-dashboard.service` on `127.0.0.1:8501`, behind nginx basic auth on :80 |

## systemd units, in start order

```
xvfb.service               Xvfb :1
  └─ mt5-terminal.service    wine terminal64.exe /portable       Requires xvfb; owns the Wine session
       └─ mt5linux.service   wine python.exe mt5server.py --host 127.0.0.1 -p 18812   Requires+PartOf mt5-terminal
            └─ beast.service   venv/bin/python main.py --no-dashboard   Wants mt5linux; Restart=always
beast-dashboard.service    streamlit on 127.0.0.1:8501 (independent)
nginx.service              :80 -> 8501, basic auth
```

Both Wine units carry `Environment=WINEDLLOVERRIDES=mscoree=d;mshtml=d;winemenubuilder.exe=d`
and `XDG_RUNTIME_DIR=/run/user/0`. `loginctl enable-linger root` is set.

## Security boundary

RPyC classic mode is remote code execution for anyone who can reach the port.
The server binds `--host 127.0.0.1`; `ufw` allows only 22 and 80 and
explicitly denies 18812 and 8501. `scripts/mt5_check.py` fails hard if the
configured endpoint is not loopback. **Never widen any of this.**

The terminal's built-in MCP servers (`config/assistant.ini`, `[MCP.MetaTrader]`
and `[MCP.MetaEditor]`) are set to `Enable=0`. They are LLM-tool listeners on
loopback that Beast does not use; fewer listeners on a trading host is the
right default. One line each to re-enable.

## Credentials

Environment only, read by `core.config` from `.env`:

```
MT5_GOLD_LOGIN=           numeric account number - NOT an MQL5.community username
MT5_GOLD_PASSWORD=
MT5_GOLD_SERVER=          exactly as the terminal shows it, e.g. MetaQuotes-Demo
MT5_GOLD_TERMINAL_PATH=C:\Program Files\MetaTrader 5\terminal64.exe
MT5_GOLD_PORTABLE=1       the units start the terminal with /portable; the IPC
                          endpoint name derives from that, so the client must know
BEAST_ALLOW_LIVE=         leave empty on a demo account
```

`initialize(login, password, server, path, portable)` performs the login
itself; the terminal does **not** need to be logged in first. There is no VNC
step. Algo Trading is enabled through `config/common.ini` (`[Experts]
Enabled=1`, `AllowLiveTrading=1`).

## Bring-up checklist

```bash
systemctl status xvfb mt5-terminal mt5linux beast beast-dashboard nginx --no-pager
ss -tlnp | grep -E '18812|8501'          # both 127.0.0.1 only
pgrep -x wineserver | wc -l              # exactly 1
cd /home/beast-agent && venv/bin/python scripts/mt5_check.py
```

`mt5_check.py` exits 0 only when the handshake, account-mode guard, symbol
spec, live quote and history all pass, and prints the server clock offset and
latency p50/p95.

## Reading the terminal's own log

It is UTF-16LE:

```bash
iconv -f UTF-16LE -t UTF-8 "/root/.wine/drive_c/Program Files/MetaTrader 5/logs/$(date +%Y%m%d).log" | tail
```

## Recovery

| Symptom | Layer | Action |
|---|---|---|
| `RPyC bridge at 127.0.0.1:18812 is unreachable` | RPyC server | `systemctl restart mt5linux` |
| `(-6, 'Terminal: Authorization failed')` | broker | the account is wrong or deleted. MetaQuotes-Demo accounts are purged without notice. Open a new one, update `.env`, `systemctl restart beast`. Not retried - it halts immediately |
| `(-10005, 'IPC timeout')` | terminal | the terminal is up but has nothing to connect to: no account, or `.env` credentials never reached it. Check `MT5_GOLD_*`, `_PORTABLE=1`, and that `initialize` was called with them |
| `(-10003, 'IPC initialize failed, Process create failed')` | path | the path in `.env` is wrong, or its backslashes were stripped - never load `.env` through systemd `EnvironmentFile=` |
| `Algo Trading is disabled` | terminal setting | check `config/common.ini` `[Experts]`; restart `mt5-terminal` |
| `exceeded Ns - the terminal or Wine is hung` | Wine | `systemctl restart mt5-terminal` (the bridge follows via PartOf) |
| Beast log `MT5 bridge ... -> HALTED` | any | entries stop, exits keep running. Fix the named layer, `systemctl restart beast` |
| Two `wineserver` processes | Wine session | something ran `wine` from a shell against the service's prefix. Never do that; use `systemd-run -p Environment=... ` in system.slice, or go through the bridge |

## The eight traps, so nobody re-learns them

1. **`mt5linux 1.1.1` client needs Python 3.12** (3.12-only f-strings) and
   **demands Docker**. Not used. RPyC directly.
2. **Python 3.12's pip aborts on Wine** (`propsys.dll.VariantToString`). The
   Wine side is 3.11.
3. **numpy ≥ 1.24 calls `fetestexcept`**; on Wine 6 that was a hard abort, and
   apiset DLLs ignore `DllOverrides`. Pinned to 1.23.5.
4. **Wine's socket dir lives in `/run/user/0`**, which logind deletes when the
   last SSH session closes - orphaning the terminal's wineserver. Fixed with
   `loginctl enable-linger root`. (Wine ignores `XDG_RUNTIME_DIR` overrides.)
5. **`wineboot` hangs forever headless** on the "install Wine Mono?" dialog.
   `WINEDLLOVERRIDES=mscoree=d;mshtml=d` before any `wineboot --init`.
6. **Upgrading Wine in place half-migrates the prefix** (`syswow64` never
   regenerated, because trap 5 hung the migration). Build a fresh prefix and
   copy the terminal dir, `mt5libs`, and `Python311` across - 13 s plus copies.
7. **systemd `EnvironmentFile=` strips backslashes**, turning
   `C:\Program Files\...` into `C:Program Files...`. Beast reads `.env` itself;
   the units must not.
8. **`initialize()` on a terminal with no account returns `-10005 IPC
   timeout`**, which looks like a broken bridge and is not. It polls the
   terminal for "connected" and never gets it. Pass the credentials to
   `initialize()` and the answer becomes definitive within seconds.

## Fresh prefix recipe (Wine 11)

```bash
export DISPLAY=:1 WINEARCH=win64 WINEPREFIX=/root/.wine XDG_RUNTIME_DIR=/run/user/0
export WINEDLLOVERRIDES="mscoree=d;mshtml=d;winemenubuilder.exe=d"
systemctl stop beast mt5linux mt5-terminal; pkill -9 -x wineserver
mv /root/.wine /root/.wine-old && timeout 240 wineboot --init && wineserver -w
cp -a "/root/.wine-old/drive_c/Program Files/MetaTrader 5" "/root/.wine/drive_c/Program Files/"
cp -a /root/.wine-old/drive_c/{mt5libs,Python311} /root/.wine/drive_c/
wine reg add 'HKCU\Software\Wine\DllOverrides' /v mscoree /t REG_SZ /d "" /f
wine reg add 'HKCU\Software\Wine\DllOverrides' /v mshtml  /t REG_SZ /d "" /f
wineserver -k; systemctl start mt5-terminal; sleep 40; systemctl start mt5linux
```

## When a terminal update breaks Wine

MetaQuotes pushes builds automatically. If a new build restart-loops or hangs:
stop the three units, restore `terminal64.exe` from a known-good copy (keep
one), and block auto-update in the terminal's options via a VNC session
(`apt install x11vnc; x11vnc -display :1 -localhost -once -nopw`, tunnel 5900).
Longer term, track WineHQ stable; the terminal names the minimum version it
accepts in its own log.
