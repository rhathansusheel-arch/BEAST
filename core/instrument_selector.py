"""Gate G8 - instrument selection (soul file 5.7).

    Runs **only after** a valid underlying signal with a complete price plan
    exists. If no strike passes every filter below, the signal is rejected at G8
    and logged - Beast does not buy a worse strike to force a trade it already
    qualified for.

This is the execution layer of the two-layer separation in 3.1. It takes a price
plan expressed in underlying points and turns it into something tradable: a long
option leg for Nifty/Sensex, a futures contract for Gold. It never revisits the
analysis - if the plan says long, the only question here is *which* instrument
expresses that long most efficiently.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

from core.config import Config, ConfigBlockerError, get_config
from core.option_chain import ChainAnalyzer, ChainSnapshot, OptionQuote
from core.schemas import Direction, FuturesLeg, OptionLeg, TradePlan


@dataclass
class LegSelection:
    """Result of G8.

    Attributes:
        ok: True when a tradable instrument was found.
        option_leg / futures_leg: Whichever applies.
        reason: Why it passed, or precisely which filter rejected it. The reason
            is written to the rejection log so section 9 can see whether G8 is
            failing on liquidity, on delta, or on unset config.
        rejected_candidates: Per-strike rejection reasons, for diagnostics.
    """

    ok: bool
    reason: str
    option_leg: OptionLeg | None = None
    futures_leg: FuturesLeg | None = None
    rejected_candidates: dict[float, str] | None = None


class OptionLegSelector:
    """Selects a long CE/PE leg for a Nifty or Sensex signal (5.7.1 - 5.7.5)."""

    def __init__(self, config: Config | None = None) -> None:
        self.cfg = config or get_config()
        self.analyzer = ChainAnalyzer(self.cfg)

    # -- expiry --------------------------------------------------------------

    def choose_expiry(self, expiries: list[date], now: datetime) -> tuple[date | None, str]:
        """Pick the expiry to trade (5.7.1).

        Nearest weekly by default. On expiry day, past the expiry-day cutoff,
        roll to the next weekly rather than skipping the trade.

        Args:
            expiries: Available expiries, any order.
            now: Current IST time.

        Returns:
            ``(expiry, reason)``; ``expiry`` is ``None`` when nothing is tradable.
        """
        upcoming = sorted(expiry for expiry in expiries if expiry >= now.date())
        if not upcoming:
            return None, "no expiry at or after today in the contract master"

        nearest = upcoming[0]
        if nearest > now.date():
            return nearest, f"nearest weekly expiry {nearest.isoformat()}"

        # DTE = 0.
        expiry_cfg = self.cfg.get("options.expiry_day")
        if not bool(expiry_cfg.get("enabled", True)):
            if len(upcoming) > 1:
                return upcoming[1], "expiry-day trading disabled; rolled to next weekly"
            return None, "expiry-day trading disabled and no later expiry available"

        cutoff = _parse_time(str(expiry_cfg["last_entry"]))
        if now.time() > cutoff:
            if len(upcoming) > 1:
                return (
                    upcoming[1],
                    f"past the {cutoff.strftime('%H:%M')} expiry-day cutoff; rolled to next weekly",
                )
            return None, f"past the {cutoff.strftime('%H:%M')} expiry-day cutoff, no later expiry"
        return nearest, f"expiry day {nearest.isoformat()}, before the cutoff"

    # -- selection -----------------------------------------------------------

    def select(self, plan: TradePlan, snapshot: ChainSnapshot, market: str,
               now: datetime) -> LegSelection:
        """Choose the option leg for ``plan``, or reject with a reason.

        The order of operations matters and follows 5.7: side from direction,
        then the delta band (widened and ITM-biased on expiry day), then the
        liquidity filters, then the ATM fallback - which must itself pass
        liquidity. An unset liquidity floor fails; it never passes by default.
        """
        instrument = self.cfg.instrument_key(market)
        try:
            lot_size = int(
                self.cfg.require(
                    f"instruments.{instrument}.lot_size",
                    "Supply the exchange lot size, or read it from the broker contract master.",
                )
            )
        except ConfigBlockerError as error:
            return LegSelection(False, str(error))

        dte = snapshot.dte(now.date())
        is_expiry_day = dte == 0
        expiry_cfg = self.cfg.get("options.expiry_day")

        if is_expiry_day:
            band = [float(value) for value in expiry_cfg["delta_band"]]
            otm_allowed = bool(expiry_cfg.get("otm_allowed", False))
        else:
            band = [float(value) for value in self.cfg.get("options.delta_band")]
            otm_allowed = True
        target_delta = float(self.cfg.get("options.target_delta"))
        if is_expiry_day:
            # Bias toward ITM: aim at the middle of the widened band.
            target_delta = (band[0] + band[1]) / 2.0

        candidates = self.analyzer.candidate_quotes(snapshot, plan.direction)
        if not candidates:
            side = "CE" if plan.direction is Direction.LONG else "PE"
            return LegSelection(False, f"no {side} quotes in the snapshot")

        rejections: dict[float, str] = {}

        # Expiry day: ATM or ITM only, regardless of the delta band (5.7.4).
        if is_expiry_day and not otm_allowed:
            filtered = []
            for quote in candidates:
                if self.analyzer.is_otm(quote, snapshot.spot):
                    rejections[quote.strike] = "OTM not permitted on expiry day"
                else:
                    filtered.append(quote)
            candidates = filtered
            if not candidates:
                return LegSelection(
                    False, "expiry day: no ATM or ITM strike available",
                    rejected_candidates=rejections,
                )

        in_band = []
        for quote in candidates:
            if band[0] <= quote.abs_delta <= band[1]:
                in_band.append(quote)
            elif quote.strike not in rejections:
                rejections[quote.strike] = (
                    f"delta {quote.abs_delta:.2f} outside band {band[0]:.2f}-{band[1]:.2f}"
                )

        ordered = sorted(in_band, key=lambda quote: abs(quote.abs_delta - target_delta))
        fallback_used = False

        if not ordered:
            # 5.7.2 fallback: the nearest ATM strike, and only if it also passes
            # the liquidity filters below.
            atm = snapshot.atm_strike()
            if atm is None:
                return LegSelection(
                    False, "no strike inside the delta band and no ATM strike",
                    rejected_candidates=rejections,
                )
            ordered = [quote for quote in candidates if quote.strike == atm]
            fallback_used = True
            if not ordered:
                return LegSelection(
                    False, "no strike inside the delta band; ATM fallback unavailable",
                    rejected_candidates=rejections,
                )

        for quote in ordered:
            ok, why = self.passes_liquidity(quote)
            if not ok:
                rejections[quote.strike] = why
                continue

            premium_stop = round(
                quote.mid * (1.0 - float(self.cfg.get("options.premium_stop_pct"))), 2
            )
            leg = OptionLeg(
                expiry=snapshot.expiry,
                dte=dte,
                strike=float(quote.strike),
                option_type=quote.option_type,
                delta=float(quote.abs_delta),
                iv=float(quote.iv),
                mid_premium=float(quote.mid),
                bid=float(quote.bid),
                ask=float(quote.ask),
                oi=int(quote.oi),
                volume=int(quote.volume),
                premium_stop=premium_stop,
                lot_size=lot_size,
                tradingsymbol=quote.tradingsymbol,
            )
            reason = (
                f"{quote.option_type} {quote.strike:g} delta {quote.abs_delta:.2f}"
                f"{' (ATM fallback)' if fallback_used else ''}"
            )
            return LegSelection(True, reason, option_leg=leg, rejected_candidates=rejections)

        return LegSelection(
            False,
            "no strike passed the 5.7.3 liquidity filters",
            rejected_candidates=rejections,
        )

    def passes_liquidity(self, quote: OptionQuote) -> tuple[bool, str]:
        """Apply the 5.7.3 filters. All must pass.

        An unset threshold is treated as *failing*, not passing (5.7.3): Beast
        will not trade Nifty/Sensex options until the floors carry real numbers.
        """
        try:
            min_oi = int(self.cfg.require("options.min_oi", "Set a contract floor, e.g. 100000."))
            min_volume = int(
                self.cfg.require("options.min_volume", "Set a session-volume floor.")
            )
            min_premium = float(
                self.cfg.require("options.min_premium", "Set a premium floor, e.g. 5.")
            )
        except ConfigBlockerError as error:
            return False, str(error)

        if quote.bid <= 0 or quote.ask <= 0:
            return False, "one-sided book (bid or ask is zero)"
        if quote.oi < min_oi:
            return False, f"OI {quote.oi} below {min_oi}"
        if quote.volume < min_volume:
            return False, f"volume {quote.volume} below {min_volume}"
        if quote.mid < min_premium:
            return False, f"mid premium {quote.mid:.2f} below {min_premium:.2f}"

        max_spread = max(
            float(self.cfg.get("options.max_spread_abs")),
            float(self.cfg.get("options.max_spread_pct")) * quote.mid,
        )
        if quote.spread > max_spread:
            return False, f"spread {quote.spread:.2f} above {max_spread:.2f}"
        return True, "liquidity filters passed"


class FuturesContractSelector:
    """Selects the Gold futures contract (soul file 5.7.6).

    Simpler by design: the front month, unless we are within
    ``rollover_buffer_days`` of its expiry, in which case the next month. No
    trades are opened during a rollover window.
    """

    def __init__(self, config: Config | None = None) -> None:
        self.cfg = config or get_config()

    def select(self, contracts: list[tuple[str, date]], now: datetime,
               market: str = "GOLD") -> LegSelection:
        """Pick the contract to trade.

        Args:
            contracts: ``(symbol, expiry)`` pairs from the broker contract master.
            now: Current IST time.
            market: Market key, for the config lookup.

        Returns:
            A :class:`LegSelection`. Beast refuses to proceed when the contract
            specs are unset - it cannot size a trade without a multiplier, and
            guessing one would silently break every cap in section 7.
        """
        instrument = self.cfg.instrument_key(market)
        try:
            multiplier = float(
                self.cfg.require(
                    f"instruments.{instrument}.contract_multiplier",
                    "Populate the Gold contract specs (venue, multiplier, tick size, tick value).",
                )
            )
            tick_size = float(self.cfg.require(f"instruments.{instrument}.tick_size"))
            tick_value = float(self.cfg.require(f"instruments.{instrument}.tick_value"))
        except ConfigBlockerError as error:
            return LegSelection(False, str(error))

        if not contracts:
            return LegSelection(False, "no futures contracts in the contract master")

        buffer_days = int(self.cfg.get(f"instruments.{instrument}.rollover_buffer_days"))
        upcoming = sorted(
            (item for item in contracts if item[1] >= now.date()),
            key=lambda item: item[1],
        )
        if not upcoming:
            return LegSelection(False, "every listed contract has expired")

        symbol, expiry = upcoming[0]
        days_left = (expiry - now.date()).days
        if days_left <= buffer_days:
            if len(upcoming) < 2:
                return LegSelection(
                    False,
                    f"front month expires in {days_left}d (rollover window) and no next month listed",
                )
            symbol, expiry = upcoming[1]
            reason = f"front month inside the {buffer_days}d rollover window; using {symbol}"
        else:
            reason = f"front month {symbol}, {days_left}d to expiry"

        return LegSelection(
            True,
            reason,
            futures_leg=FuturesLeg(
                contract=symbol,
                contract_multiplier=multiplier,
                tick_size=tick_size,
                tick_value=tick_value,
                expiry=expiry,
            ),
        )


class InstrumentSelector:
    """Facade over the two selectors - the single entry point for gate G8."""

    def __init__(self, config: Config | None = None) -> None:
        self.cfg = config or get_config()
        self.options = OptionLegSelector(self.cfg)
        self.futures = FuturesContractSelector(self.cfg)

    def select(self, plan: TradePlan, market: str, now: datetime,
               snapshot: ChainSnapshot | None = None,
               contracts: list[tuple[str, date]] | None = None) -> LegSelection:
        """Route to the option or futures path based on the instrument config."""
        instrument = self.cfg.instrument_key(market)
        traded_as = str(self.cfg.get(f"instruments.{instrument}.trade"))

        if traded_as == "options":
            if snapshot is None:
                return LegSelection(False, "no option chain snapshot available")
            return self.options.select(plan, snapshot, market, now)
        if traded_as == "futures":
            return self.futures.select(contracts or [], now, market)
        return LegSelection(False, f"unsupported trade mode '{traded_as}' for {market}")


def _parse_time(text: str) -> time:
    """Parse ``"HH:MM"`` into a :class:`datetime.time`."""
    hour, minute = (int(part) for part in text.split(":"))
    return time(hour=hour, minute=minute)
