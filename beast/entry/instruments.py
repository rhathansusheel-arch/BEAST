"""Section 5.7 - G8, instrument selection.

This is the execution layer. It runs **only after** a valid underlying signal with a
complete price plan exists (5.7), and it never feeds anything back into the analysis layer
- Immutable Rule 9.

"If no strike passes every filter below, the signal is rejected at G8 and logged - Beast
does not buy a worse strike to force a trade it already qualified for."
"""

from __future__ import annotations

from datetime import datetime, time
from typing import Optional

from beast.analysis.option_chain import ChainSnapshot, StrikeQuote
from beast.config import ConfigUnset
from beast.constants import Direction, Market
from beast.ops.immutable import assert_long_option_only
from beast.schemas import FuturesLeg, OptionLeg


def select_option_leg(plan, ctx) -> tuple[Optional[OptionLeg], str]:
    """Build the option leg for a qualified underlying signal (5.7.1 - 5.7.5).

    Expiry (5.7.1): nearest weekly; on expiry day past the cutoff, roll to the next weekly
    rather than skipping the trade. Strike (5.7.2): bullish buys CE, bearish buys PE -
    never sold, never spread - targeting the strike whose delta is closest to 0.55 inside
    the band. Liquidity (5.7.3): every filter must pass, and an unset threshold *fails*.
    """
    cfg = ctx.cfg
    chain: Optional[ChainSnapshot] = ctx.chain
    if chain is None:
        return None, "no option chain snapshot available"

    instrument = cfg.instrument(ctx.market)
    assert_long_option_only("BUY", str(instrument.get("option_side")))

    dte = chain.dte(ctx.now)
    expiry_day = dte == 0 and bool(cfg.get("options.expiry_day.enabled"))
    if dte == 0:
        cutoff = time.fromisoformat(str(cfg.get("options.expiry_day.last_entry")))
        if ctx.now.time() >= cutoff:
            if ctx.next_chain is None:
                return None, (
                    f"expiry day past {cutoff.isoformat(timespec='minutes')} and no next-weekly "
                    "chain available to roll to (5.7.1)"
                )
            chain = ctx.next_chain
            dte = chain.dte(ctx.now)
            expiry_day = False

    option_type = "CE" if plan_direction(plan, ctx) is Direction.LONG else "PE"
    candidates = [q for q in chain.quotes if q.option_type == option_type and q.delta]

    if expiry_day:
        low, high = (float(x) for x in cfg.get("options.expiry_day.delta_band"))
        if not cfg.get("options.expiry_day.otm_allowed"):
            # 5.7.4 - ATM or ITM only. OTM strikes go to zero in minutes on expiry day.
            atm = chain.atm_strike()
            candidates = [
                q
                for q in candidates
                if (q.strike <= atm if option_type == "CE" else q.strike >= atm)
            ]
    else:
        low, high = (float(x) for x in cfg.get("options.delta_band"))

    target_delta = float(cfg.get("options.target_delta"))
    in_band = [q for q in candidates if low <= abs(q.delta) <= high]
    pool = in_band or _atm_fallback(chain, candidates)
    if not pool:
        return None, f"no {option_type} strike inside delta band {low}-{high} and no ATM fallback"

    pool = sorted(pool, key=lambda q: abs(abs(q.delta) - target_delta))
    failures: list[str] = []
    for quote in pool:
        ok, why = passes_liquidity(quote, cfg)
        if ok:
            return _build_leg(quote, chain, dte, cfg), (
                f"{quote.strike:g} {option_type} delta {abs(quote.delta):.2f}"
                + ("" if in_band else " (ATM fallback - no strike in the delta band)")
            )
        failures.append(f"{quote.strike:g}: {why}")
    return None, "no strike passed the 5.7.3 liquidity filters - " + "; ".join(failures[:3])


def plan_direction(plan, ctx) -> Direction:
    """The underlying signal's direction, carried through to the leg (5.7.2)."""
    return ctx.pending_direction


def _atm_fallback(chain: ChainSnapshot, candidates: list[StrikeQuote]) -> list[StrikeQuote]:
    """5.7.2 - fall back to the nearest ATM strike, and only if it passes liquidity too."""
    if not candidates:
        return []
    atm = chain.atm_strike()
    nearest = min(candidates, key=lambda q: abs(q.strike - atm))
    return [nearest]


def passes_liquidity(quote: StrikeQuote, cfg) -> tuple[bool, str]:
    """5.7.3 - open interest, volume, spread, two-sided quotes, premium floor.

    An unset threshold is treated as **failing**, not passing: "Beast will not trade
    Nifty/Sensex options until they carry real numbers."
    """
    try:
        min_oi = int(cfg.require("options.min_oi", "5.7.3 - thin strikes cannot be exited fairly"))
        min_volume = int(cfg.require("options.min_volume", "5.7.3 - OI can be stale, volume proves it is live"))
        min_premium = float(cfg.require("options.min_premium", "5.7.3 - sub-floor options have brutal relative spreads"))
    except ConfigUnset as exc:
        return False, str(exc)

    if quote.bid <= 0 or quote.ask <= 0:
        return False, "one-sided book"
    if quote.oi < min_oi:
        return False, f"OI {quote.oi} below {min_oi}"
    if quote.volume < min_volume:
        return False, f"volume {quote.volume} below {min_volume}"
    if quote.mid < min_premium:
        return False, f"premium {quote.mid:.2f} below floor {min_premium}"

    max_spread = max(
        float(cfg.get("options.max_spread_abs")),
        float(cfg.get("options.max_spread_pct")) * quote.mid,
    )
    if quote.spread > max_spread:
        return False, f"spread {quote.spread:.2f} above {max_spread:.2f}"
    return True, "liquidity filters passed"


def _build_leg(quote: StrikeQuote, chain: ChainSnapshot, dte: int, cfg) -> OptionLeg:
    """5.7.5 - the leg object execution acts on, including its 6.10 premium stop."""
    stop_pct = float(cfg.get("options.premium_stop_pct"))
    return OptionLeg(
        expiry=chain.expiry.isoformat(),
        dte=dte,
        strike=quote.strike,
        option_type=quote.option_type,
        delta=abs(quote.delta),
        iv=quote.iv,
        mid_premium=quote.mid,
        bid=quote.bid,
        ask=quote.ask,
        oi=quote.oi,
        volume=quote.volume,
        premium_stop=round(quote.mid * (1.0 - stop_pct), 2),
    )


def select_futures_contract(ctx) -> tuple[Optional[FuturesLeg], str]:
    """5.7.6 - front-month Gold futures, unless inside the rollover window.

    "No trades are opened during a rollover window." Contract multiplier, tick size and
    tick value come from Appendix A and must be populated before Beast will size a trade.
    """
    cfg = ctx.cfg
    try:
        multiplier = float(
            cfg.require(
                "instruments.gold.contract_multiplier",
                "3.1 / Open Item 17 - Gold venue and contract specs are a blocker",
            )
        )
        cfg.require("instruments.gold.tick_size", "3.1 / Open Item 17")
        cfg.require("instruments.gold.tick_value", "3.1 / Open Item 17")
        cfg.require("instruments.gold.venue", "3.1 / Open Item 17")
    except ConfigUnset as exc:
        return None, str(exc)

    contract = ctx.futures_contract
    if contract is None:
        return None, "no front-month contract supplied by the data layer"

    buffer_days = int(cfg.get("instruments.gold.rollover_buffer_days"))
    if contract.get("days_to_expiry") is not None:
        if int(contract["days_to_expiry"]) <= buffer_days:
            if contract.get("next_symbol"):
                return (
                    FuturesLeg(contract=str(contract["next_symbol"]), contract_multiplier=multiplier),
                    f"front month inside the {buffer_days}-day rollover buffer - using next month",
                )
            return None, f"inside the {buffer_days}-day rollover window and no next contract supplied"
    return (
        FuturesLeg(contract=str(contract["symbol"]), contract_multiplier=multiplier),
        f"front-month {contract['symbol']}",
    )


def select(plan, ctx) -> tuple[Optional[object], str]:
    """G8 dispatch: an option leg for Nifty/Sensex, a futures contract for Gold (3.1)."""
    if ctx.market is Market.GOLD:
        return select_futures_contract(ctx)
    return select_option_leg(plan, ctx)
