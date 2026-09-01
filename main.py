"""Beast - entry point.

Beast is governed by ``beast/soul/BEAST_SOUL_v3.md``. Where any instruction conflicts with
that file, the Soul File wins (PRECEDENCE.md). This CLI does not add behaviour; it only
runs the engine and prints what it decided.

Usage::

    python main.py --check                       # what governs Beast, and what blocks it
    python main.py --market NIFTY --replay bars.csv
    python main.py --market GOLD --replay xau_1m.csv --from "2026-09-01 05:00"
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from beast.config import Config
from beast.constants import Market
from beast.engine import Beast
from beast.feeds.csv_feed import build_feed, load_bars, resample
from beast.analysis.indicators import tf_minutes


def _use_utf8_output() -> None:
    """Section 11's lines carry ``Δ`` and ``⚠``; a cp1252 console would crash on them."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):  # already redirected, or not reconfigurable
            pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Beast - rules-bound intraday trading agent")
    parser.add_argument("--config", default=None, help="path to config/beast.yaml")
    parser.add_argument("--check", action="store_true", help="print the integrity report and exit")
    parser.add_argument("--market", choices=[m.value for m in Market], help="market to evaluate")
    parser.add_argument("--replay", help="base-resolution OHLCV CSV of the underlying")
    parser.add_argument("--from", dest="start", help="replay from this timestamp")
    parser.add_argument("--to", dest="end", help="replay up to this timestamp")
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    return parser.parse_args(argv)


def check(cfg) -> int:
    """Report what governs Beast and what is currently stopping it from trading."""
    beast = Beast(cfg)
    report = beast.integrity()
    print(f"Soul File   : {report['soul_path']}")
    print(f"Version     : {report['soul_version']}")
    print(f"SHA-256     : {report['soul_sha256']}")
    print(f"Config      : {report['config_path']}")
    print(f"Mode        : {report['mode']} (Section 10)")
    print("Precedence  : " + " > ".join(report["precedence"]))
    blockers = report["unresolved_blockers"]
    if blockers:
        print("\nUnresolved Appendix A blockers - Beast will refuse to trade these paths:")
        for item in blockers:
            print(f"  - {item}")
        print("\nAn unset blocker is treated as FAILING, not as 'no limit' (5.7.3, 3.1).")
    else:
        print("\nNo unresolved blockers.")
    return 0


def replay(cfg, market: Market, path: str, start: str | None, end: str | None, as_json: bool) -> int:
    """Run the engine bar by bar over a CSV, exactly as it would run live (Section 10)."""
    bars = load_bars(path)
    beast = Beast(cfg)
    tfs = cfg.timeframes(market)
    trigger = resample(bars, tfs["trigger"])
    step = tf_minutes(tfs["trigger"])

    # --from/--to select the *evaluation* window, never the history behind it. Truncating
    # the history would starve the bias-TF indicators and put Beast in RANGE on NaN.
    if start:
        trigger = trigger[trigger.index >= start]
    if end:
        trigger = trigger[trigger.index <= end]

    emitted = 0
    for stamp in trigger.index:
        now = (stamp + __import__("pandas").Timedelta(minutes=step)).to_pydatetime()
        feed = build_feed(market, bars, cfg, now)
        if feed.setup.empty or feed.bias.empty or feed.trigger.empty:
            continue
        result = beast.on_trigger_close(feed, now)
        for line in result.lines():
            emitted += 1
            print(f"{now:%Y-%m-%d %H:%M}  {line}")

    summary = beast.weekly_report()
    if as_json:
        print(json.dumps(summary, indent=2, default=str))
    else:
        print(f"\n{emitted} line(s) emitted. Gate rejections: {beast.learning.gate_report()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    _use_utf8_output()
    args = parse_args(argv)
    cfg = Config.load(args.config)

    if args.check or not args.market:
        return check(cfg)

    market = Market(args.market)
    if not args.replay:
        print(
            "No data source given. Beast analyses the underlying only (Soul File 3.1), so the "
            "data layer must supply underlying OHLCV - pass --replay <csv>, or wire a live feed "
            "into beast.analysis.context.MarketFeed.",
            file=sys.stderr,
        )
        return 2
    if not Path(args.replay).exists():
        print(f"no such file: {args.replay}", file=sys.stderr)
        return 2
    return replay(cfg, market, args.replay, args.start, args.end, args.json)


if __name__ == "__main__":
    raise SystemExit(main())
