"""The operator's stop, over SSH.

    python -m ops.killswitch halt     [--reason "..."]   # entries off; positions stay on their stops
    python -m ops.killswitch flatten  [--reason "..."]   # close everything against plan, then halt
    python -m ops.killswitch clear                       # resume
    python -m ops.killswitch status

Writes a ``KILL`` file at ``ops.kill_flag_path``. ``BeastRunner.tick()`` reads
it every cycle before anything else; the watchdog reads it and refuses to
restart through it.

``halt`` is the default and the safe one (D-66). It pauses entries only. The
exit path keeps running, which is exactly what sections 4.6 and 5.6 already
do for a stale feed: a position's stop does not stop being a stop because the
operator wants no *new* risk.

``flatten`` is a close against plan, which section 8 governs. It is not a
second, frictionless path around that section: the flag records the request,
and Beast's loop routes each close through ``core/override.py`` with the exact
typed confirmation this command demands up front. Without the phrase, nothing
is written.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.override import OverrideAction  # noqa: E402

MODES = ("halt", "flatten")
FLATTEN_PHRASE = OverrideAction.CLOSE_EARLY.confirmation_phrase


def flag_path(cfg=None) -> Path:
    if cfg is None:
        from core.config import get_config
        cfg = get_config()
    raw = Path(str(cfg.get("ops.kill_flag_path", "./KILL")))
    return raw if raw.is_absolute() else PROJECT_ROOT / raw


def read_flag(path: str | Path) -> dict[str, Any] | None:
    """The KILL flag's contents, or None when it is not set."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def set_flag(path: str | Path, mode: str, reason: str, confirmation: str = "",
             who: str | None = None) -> dict[str, Any]:
    """Write the flag. Refuses ``flatten`` without the section 8 phrase."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if mode == "flatten" and confirmation != FLATTEN_PHRASE:
        raise PermissionError(
            f"flatten closes positions against their plan (section 8). Type exactly: "
            f"{FLATTEN_PHRASE!r}"
        )
    payload = {
        "mode": mode,
        "reason": reason or "(none given)",
        "set_by": who or f"{getpass.getuser()}@{socket.gethostname()}",
        "set_at": datetime.now().isoformat(timespec="seconds"),
        "confirmation": confirmation if mode == "flatten" else "",
    }
    target = Path(path)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, target)
    return payload


def clear_flag(path: str | Path) -> bool:
    try:
        Path(path).unlink()
        return True
    except FileNotFoundError:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Beast kill switch")
    parser.add_argument("command", choices=("halt", "flatten", "clear", "status"))
    parser.add_argument("--reason", default="")
    parser.add_argument("--confirm", default="",
                        help=f"required for flatten: {FLATTEN_PHRASE!r}")
    parser.add_argument("--path", default=None, help="override ops.kill_flag_path")
    args = parser.parse_args(argv)

    path = Path(args.path) if args.path else flag_path()

    if args.command == "status":
        flag = read_flag(path)
        if flag is None:
            print(f"KILL flag: not set ({path})")
        else:
            print(f"KILL flag: {flag['mode'].upper()} set by {flag['set_by']} at {flag['set_at']}")
            print(f"  reason: {flag['reason']}")
        return 0

    if args.command == "clear":
        print("cleared" if clear_flag(path) else "no flag was set")
        return 0

    confirmation = args.confirm
    if args.command == "flatten" and not confirmation and sys.stdin.isatty():
        print("flatten closes every position against its plan. Section 8 requires the")
        print(f"exact phrase: {FLATTEN_PHRASE}")
        confirmation = input("> ").strip()
    try:
        flag = set_flag(path, args.command, args.reason, confirmation)
    except PermissionError as error:
        print(f"refused: {error}")
        return 2
    print(f"{flag['mode'].upper()} set at {path}")
    if flag["mode"] == "halt":
        print("entries are off; open positions keep their stops and exits keep running")
    else:
        print("Beast will close every position at market through the section 8 override, "
              "log the outcome the plan would have produced, then halt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
