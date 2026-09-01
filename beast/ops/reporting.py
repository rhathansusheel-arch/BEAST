"""Section 11 - communication style.

Tone: strict risk manager. Calm, factual, unemotional. No hype on wins, no
self-flagellation on losses. Output level is signal + basic reasoning only.

For an options signal the **underlying plan comes first and the leg second**, always: "the
operator should always be able to see what Beast thinks price will do, separately from what
it bought to express that."
"""

from __future__ import annotations

from beast.constants import Direction, Flag, Market, SETUP_SHORT

INDICATOR_LABELS = {
    "adx": "ADX+DI",
    "stoch": "Stoch",
    "macd": "MACD",
    "rsi": "RSI",
    "bb": "BB",
    "vwap": "VWAP",
}

SENSEX_SUFFIX = "⚠ Sensex feed delay up to 15 min - price may be stale."


def _labels(names: list[str]) -> str:
    return ", ".join(INDICATOR_LABELS[n] for n in names)


def _confluence_phrase(tally, signal) -> str:
    side = "bullish" if signal.direction is Direction.LONG else "bearish"
    mode = "reversal-mode" if tally.mode.value == "REVERSAL" else ""
    head = f"{tally.aligned}/6 {side}{(' ' + mode) if mode else ''}: {_labels(tally.aligned_names())}"
    notes = []
    if tally.neutral_names():
        notes.append(f"{_labels(tally.neutral_names())} neutral")
    if tally.opposing_names():
        notes.append(f"{_labels(tally.opposing_names())} opposing")
    return head + (f" ({'; '.join(notes)})" if notes else "")


def reason_line(signal, tally, setup, leg, leg_detail: str, ctx) -> str:
    """The one-line signal per Section 11, in the market's own format."""
    tf = signal.timeframes["setup"]
    name = SETUP_SHORT[signal.setup_type]
    setup_text = f"Setup {signal.setup_type} {name} {tf}"
    if setup.zone is not None:
        setup_text = (
            f"Setup {signal.setup_type} {name} at "
            f"{setup.zone.center:,.0f} {setup.zone.kind.value.lower()} {tf}"
        )

    # Index levels are quoted in whole points and gold to the tick, matching the two
    # worked examples in Section 11.
    px = "{:,.0f}" if signal.market.is_option_market else "{:,.2f}"

    if signal.market is Market.GOLD:
        parts = [
            f"{signal.market.value} {signal.direction.value} (futures)",
            setup_text,
            f"entry {px.format(signal.entry_price)}",
            f"SL {px.format(signal.stop_price)} ({signal.stop_source})",
            f"TP {px.format(signal.target_price)} ({signal.target_r}R)",
            f"trail arms at {px.format(signal.trail.activate_at)}",
            _confluence_phrase(tally, signal),
        ]
        return " | ".join(parts)

    option = signal.leg["_option_only"]
    parts = [
        f"{signal.market.value} {signal.direction.value}",
        setup_text,
        f"underlying entry {px.format(signal.entry_price)}",
        f"SL {px.format(signal.stop_price)}",
        f"TP {px.format(signal.target_price)} ({signal.target_r}R)",
        f"BUY {option['strike']:g} {option['option_type']} {option['expiry']} @ {option['mid_premium']:,.2f}",
        f"Δ{option['delta']:.2f}",
        f"{option['lots']} lot(s) ({option['binding_cap'].replace('_', ' ')} cap binding)",
        f"premium stop {option['premium_stop']:,.2f}",
        _confluence_phrase(tally, signal),
    ]
    if signal.chain_context and signal.chain_context.max_put_oi_strike and signal.direction is Direction.LONG:
        parts.append(f"max put OI {signal.chain_context.max_put_oi_strike:,.0f} supports")
    if signal.chain_context and signal.chain_context.max_call_oi_strike and signal.direction is Direction.SHORT:
        parts.append(f"max call OI {signal.chain_context.max_call_oi_strike:,.0f} caps")

    line = " | ".join(parts)
    if signal.market is Market.SENSEX:
        line += f"\n{SENSEX_SUFFIX}"
    return line


def alert(kind: str, market: Market, detail: str) -> str:
    """Alerts are delivered the same way as signals - short, factual, no padding (11)."""
    return f"[{kind}] {market.value}: {detail}"


def exit_line(record) -> str:
    """A closed trade, reported without hype or self-flagellation (11)."""
    r = record.r_multiple if record.r_multiple is not None else 0.0
    parts = [
        f"{record.market.value} {record.signal.direction.value} closed",
        f"reason {record.exit_reason.value}",
        f"{r:+.2f}R",
        f"MAE {record.mae_r:+.2f}R / MFE {record.mfe_r:+.2f}R",
        f"{record.bars_held} bars",
    ]
    if record.market.is_option_market and record.underlying_r_multiple is not None:
        parts.append(f"underlying {record.underlying_r_multiple:+.2f}R")
    return " | ".join(parts)


def flag_alerts(market: Market, flags: list[str]) -> list[str]:
    """Turn context flags into the Section 11 alert lines the operator sees."""
    messages = {
        Flag.HIGH_SPREAD.value: "HIGH SPREAD ALERT - no entry until spread normalises",
        Flag.STALE_FEED.value: "feed STALE - entries suppressed until fresh data resumes",
        Flag.STALE_CHAIN.value: "option chain snapshot stale - OI levels dropped this cycle",
        Flag.NEWS_NEAR.value: "news blackout window active - no new entries",
        Flag.IV_ELEVATED.value: "IV percentile elevated - long premium exposed to crush",
        Flag.EXPIRY_DAY.value: "expiry day - ATM/ITM only, earlier cutoff, faster trail",
        Flag.SENSEX_DELAY.value: SENSEX_SUFFIX,
        Flag.GAP_OPEN.value: "session gapped - levels recomputed before the first entry",
    }
    return [alert("ALERT", market, messages[f]) for f in flags if f in messages]
