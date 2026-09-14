"""Discover the gold CFD specs from the live MT5 symbol. Read-only.

    python scripts/mt5_symbol_specs.py                 # 5-minute spread sample
    python scripts/mt5_symbol_specs.py --minutes 1     # quicker look
    python scripts/mt5_symbol_specs.py --minutes 0     # specs only, no sample

Prints which candidate symbol names the broker actually lists, the account
context, every symbol field the CFD sizing path depends on, a spread sample
with min / median / p90 / max in both points and price, and a ready-to-paste
``instruments.gold`` block annotated with where each number came from.

It goes through :class:`broker.mt5_connection.MT5Connection` - the same
serialised, bounded, loopback-only path Beast uses - and never calls
``order_send``. Nothing here can place, modify or close anything.

Why the sample matters: ``data.gold_spread_max`` is compared to ``ask - bid``
in **price units** at gate G1. Set from a guess it either gates gold out
permanently or lets the daily-rollover blowout through. Set from the p90 of a
sample taken in the hours Beast actually trades, it does neither. Note the
hours the sample covered next to the value in config.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from broker.mt5_connection import BridgeError, MT5Connection
from core.config import get_config

CANDIDATES = ("XAUUSD", "XAUUSD.m", "XAUUSD.raw", "XAUUSDm", "GOLD", "GOLD.m")

SPEC_FIELDS = (
    ("name", "exact symbol string Beast must use"),
    ("trade_contract_size", "ounces per lot"),
    ("digits", "price decimals"),
    ("point", "price precision unit"),
    ("trade_tick_size", "price units per tick"),
    ("trade_tick_value", "ACCOUNT CURRENCY per tick per lot"),
    ("volume_min", "smallest legal volume"),
    ("volume_step", "volume granularity"),
    ("volume_max", "largest legal volume"),
    ("trade_stops_level", "min stop distance, POINTS - closer is retcode 10016"),
    ("trade_freeze_level", "no modification inside this, POINTS"),
    ("spread", "current spread, POINTS"),
    ("spread_float", "spread is floating"),
    ("filling_mode", "bitmask FOK=1 IOC=2 BOC=4"),
    ("currency_base", "base currency"),
    ("currency_profit", "P&L currency"),
    ("currency_margin", "margin currency"),
    ("trade_mode", "0=disabled 1=long only 2=short only 3=close only 4=full"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Read the gold CFD specs from MT5.")
    parser.add_argument("--minutes", type=float, default=5.0,
                        help="length of the spread sample (0 to skip)")
    parser.add_argument("--interval", type=float, default=2.0, help="seconds between ticks")
    args = parser.parse_args()

    cfg = get_config()
    link = MT5Connection(profile=str(cfg.get("broker.mt5.active_profile", "gold")), config=cfg)
    print("Beast - MT5 gold symbol discovery (read-only)\n")

    if not link.connect(wait_seconds=30):
        print("could not connect - the log line above names the layer. Nothing read.")
        return 1

    try:
        return _discover(link, cfg, args)
    finally:
        link.disconnect()


def _discover(link: MT5Connection, cfg, args) -> int:
    # -- account context ------------------------------------------------------
    account = link.call("account_info", call_class="connect")
    print("ACCOUNT")
    for field in ("company", "server", "login", "currency", "trade_mode", "balance",
                  "equity", "leverage", "margin_mode", "trade_allowed", "trade_expert"):
        value = getattr(account, field, "?")
        note = ""
        if field == "trade_mode":
            note = "  (0 = DEMO, 1 = contest, 2 = REAL)"
        if field == "login":
            value = _mask(value)
        print(f"  {field:<14} {value}{note}")
    currency = str(getattr(account, "currency", ""))
    print()

    # -- symbol resolution ---------------------------------------------------
    print("SYMBOL RESOLUTION")
    listed = set()
    for group in ("*XAU*", "*GOLD*"):
        for item in link.call("symbols_get", group=group, allow_none=True) or ():
            listed.add(str(getattr(item, "name", "")))
    found = [name for name in CANDIDATES if name in listed]
    for name in CANDIDATES:
        print(f"  {name:<12} {'listed' if name in listed else '-'}")
    extra = sorted(listed - set(CANDIDATES))
    if extra:
        print(f"  also listed : {', '.join(extra[:12])}{' ...' if len(extra) > 12 else ''}")
    configured = cfg.get("instruments.gold.symbol", None)
    chosen = configured or (found[0] if found else None)
    if chosen is None:
        print("\n  no gold symbol found - nothing to read. Set instruments.gold.symbol by hand "
              "from the 'also listed' names above if one of them is gold.")
        return 1
    print(f"  using       : {chosen}{'  (from config)' if configured else '  (first candidate listed)'}")
    print()

    # -- symbol specs --------------------------------------------------------
    if not link.call("symbol_select", chosen, True):
        print(f"  symbol_select({chosen}) failed - an unselected symbol reports empty fields")
        return 1
    info = link.call("symbol_info", chosen)
    print(f"SYMBOL {chosen}")
    for field, why in SPEC_FIELDS:
        print(f"  {field:<20} {getattr(info, field, '?')!s:<14} {why}")
    point = float(getattr(info, "point", 0.0) or 0.0)
    print()

    # -- spread sample -------------------------------------------------------
    spreads_pts: list[float] = []
    started = datetime.now()
    if args.minutes > 0 and point > 0:
        n = max(1, int(args.minutes * 60 / args.interval))
        print(f"SPREAD SAMPLE  {n} ticks over {args.minutes:g} min, from {started:%Y-%m-%d %H:%M} local")
        for i in range(n):
            try:
                tick = link.call("symbol_info_tick", chosen)
                bid, ask = float(tick.bid), float(tick.ask)
                if bid > 0 and ask > 0:
                    spreads_pts.append(round((ask - bid) / point))
            except BridgeError as error:
                print(f"  tick {i}: {error}")
            if i < n - 1:
                time.sleep(args.interval)
        if spreads_pts:
            s = sorted(spreads_pts)
            p90 = s[min(len(s) - 1, int(len(s) * 0.9))]
            rows = (("min", s[0]), ("median", statistics.median(s)), ("p90", p90), ("max", s[-1]))
            print(f"  {'':<8}{'points':>8}{'price':>12}")
            for label, value in rows:
                print(f"  {label:<8}{value:>8.0f}{value * point:>12.{int(getattr(info, 'digits', 2))}f}")
            print(f"  sampled {len(s)} ticks, {started:%H:%M}-{datetime.now():%H:%M} local")
        else:
            print("  no usable ticks - market closed?")
        print()

    # -- YAML block ----------------------------------------------------------
    venue = f"{getattr(account, 'company', '')} / {getattr(account, 'server', '')}".strip(" /")
    digits = int(getattr(info, "digits", 2))
    p90_price = None
    if spreads_pts:
        s = sorted(spreads_pts)
        p90_price = s[min(len(s) - 1, int(len(s) * 0.9))] * point
    print("READY TO PASTE  ->  config/beast_config.yaml")
    print("instruments:")
    print("  gold:")
    print("    trade: cfd")
    print(f"    venue: {_yq(venue)}{'':<14}# account_info().company / .server")
    print(f"    symbol: {_yq(chosen)}{'':<12}# symbols_get + symbol_select")
    print(f"    contract_multiplier: {getattr(info, 'trade_contract_size', None)}   # trade_contract_size")
    print(f"    tick_size: {getattr(info, 'trade_tick_size', None)}             # trade_tick_size")
    print(f"    tick_value: {getattr(info, 'trade_tick_value', None)}            # trade_tick_value, {currency} per tick per lot")
    print(f"    point: {getattr(info, 'point', None)}                 # point")
    print(f"    volume_min: {getattr(info, 'volume_min', None)}            # volume_min")
    print(f"    volume_step: {getattr(info, 'volume_step', None)}           # volume_step")
    print(f"    volume_max: {getattr(info, 'volume_max', None)}           # volume_max")
    print(f"    stops_level_points: {getattr(info, 'trade_stops_level', None)}   # trade_stops_level")
    print(f"    filling_mode: {getattr(info, 'filling_mode', None)}            # filling_mode bitmask")
    print(f"    account_currency: {currency}      # account_info().currency - MUST match risk.capital's currency")
    print("data:")
    if p90_price is not None:
        print(f"    gold_spread_max: {p90_price:.{digits}f}       # PRICE units; p90 of {len(spreads_pts)} ticks "
              f"{started:%H:%M}-{datetime.now():%H:%M} local on {started:%Y-%m-%d}")
    else:
        print("    gold_spread_max: null        # no sample taken - run again with --minutes 5 during trading hours")
    print()
    if currency and currency.upper() != "INR":
        print(f"NOTE  the account is {currency}. risk.capital is a rupee figure (D-41); sizing {currency} "
              f"tick values against INR risk is wrong by the exchange rate. Gold stays refused until "
              f"instruments.gold.account_currency and risk.capital agree - that is a separate decision.")
    return 0


def _mask(value) -> str:
    text = str(value)
    return text if len(text) <= 4 else "*" * (len(text) - 4) + text[-4:]


def _yq(text: str) -> str:
    return '"' + str(text).replace('"', '\\"') + '"'


if __name__ == "__main__":
    raise SystemExit(main())
