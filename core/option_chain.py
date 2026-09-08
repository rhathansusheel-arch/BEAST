"""Option chain analysis - soul file 4.7 (Nifty / Sensex only).

    The option chain is a **context and level source**, not a signal source. It
    never adds to or subtracts from the 4-of-6 confluence count in 5.3.

What the chain does is exactly three things:

* contribute Tier A levels to the same pool as price structure (4.7.1),
* inform the target-feasibility check at G7,
* constrain which instrument is bought at G8 (5.7).

Nothing in this module touches the indicator engine. Option premium never enters
the analysis layer - that is soul file rule 13.9 and it is the reason this file
exists separately from ``core/levels.py`` rather than being folded into it.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Iterable, Sequence

import numpy as np

from core.config import Config, get_config
from core.schemas import (
    ChainContext,
    Direction,
    Flag,
    LevelKind,
    LevelTier,
    OITag,
    Zone,
)


@dataclass
class OptionQuote:
    """One strike on one side of the chain.

    Attributes:
        strike: Strike price in underlying points.
        option_type: ``"CE"`` or ``"PE"``.
        bid / ask: Top-of-book quotes. Zero means unquoted, which fails 5.7.3.
        oi: Open interest, in contracts.
        oi_change: Change in open interest since session open.
        volume: Contracts traded today.
        iv: Implied volatility, in percent.
        delta: Signed delta as reported by the feed; Beast uses ``abs(delta)``
            for the band test since a long PE has negative delta but the same
            conversion efficiency as the mirror-image CE.
        tradingsymbol: Broker symbol, carried through to the order layer.
    """

    strike: float
    option_type: str
    bid: float = 0.0
    ask: float = 0.0
    oi: int = 0
    oi_change: int = 0
    volume: int = 0
    iv: float = 0.0
    delta: float = 0.0
    tradingsymbol: str = ""

    @property
    def mid(self) -> float:
        """Mid premium. Falls back to whichever side is quoted."""
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return max(self.bid, self.ask)

    @property
    def spread(self) -> float:
        """Absolute bid-ask spread; ``inf`` when the book is one-sided."""
        if self.bid <= 0 or self.ask <= 0:
            return float("inf")
        return self.ask - self.bid

    @property
    def abs_delta(self) -> float:
        return abs(self.delta)


@dataclass
class ChainSnapshot:
    """An ATM +/- N strike slice of one expiry, taken on a setup-TF close."""

    underlying: str
    spot: float
    expiry: date
    taken_at: datetime
    quotes: list[OptionQuote] = field(default_factory=list)
    session_open_spot: float | None = None

    def age_seconds(self, now: datetime) -> float:
        """Seconds since the snapshot was taken."""
        return max(0.0, (now - self.taken_at).total_seconds())

    def calls(self) -> list[OptionQuote]:
        return [quote for quote in self.quotes if quote.option_type == "CE"]

    def puts(self) -> list[OptionQuote]:
        return [quote for quote in self.quotes if quote.option_type == "PE"]

    def dte(self, today: date) -> int:
        """Days to expiry. ``0`` on expiry day."""
        return (self.expiry - today).days

    def atm_strike(self) -> float | None:
        """The strike closest to spot."""
        strikes = sorted({quote.strike for quote in self.quotes})
        if not strikes:
            return None
        return min(strikes, key=lambda strike: abs(strike - self.spot))


class ChainAnalyzer:
    """Derives Tier A levels and context tags from a chain snapshot."""

    def __init__(self, config: Config | None = None) -> None:
        self.cfg = config or get_config()

    # -- top level -----------------------------------------------------------

    def analyse(self, snapshot: ChainSnapshot, now: datetime, atr_value: float,
                market: str, iv_history: Sequence[float] = (),
                price_change: float | None = None) -> tuple[ChainContext, list[Zone], list[Flag]]:
        """Produce the chain context, its Tier A levels and any flags.

        Args:
            snapshot: The chain slice.
            now: Current IST time, for the freshness test.
            atr_value: Setup-TF ATR, used only for the convergence bonus.
            market: ``"NIFTY50"`` or ``"SENSEX"``.
            iv_history: ATM IV from the trailing sessions, for the percentile.
            price_change: Underlying change since session open, for the OI tag.

        Returns:
            ``(context, zones, flags)``. When the snapshot is stale the zones
            list is empty and ``STALE_CHAIN`` is flagged - a stale chain degrades
            context, it does not halt trading (4.6).
        """
        flags: list[Flag] = []
        max_age = float(self.cfg.get("options.chain_max_age_sec"))
        age = snapshot.age_seconds(now)
        stale = age > max_age

        context = ChainContext(snapshot_age_sec=age, stale=stale)
        if not snapshot.quotes:
            return context, [], [Flag.STALE_CHAIN]

        calls, puts = snapshot.calls(), snapshot.puts()

        context.max_call_oi_strike = self._top_strike(calls, "oi", 0)
        context.second_max_call_oi_strike = self._top_strike(calls, "oi", 1)
        context.max_put_oi_strike = self._top_strike(puts, "oi", 0)
        context.second_max_put_oi_strike = self._top_strike(puts, "oi", 1)

        max_change_call = self._top_strike(calls, "oi_change", 0)
        max_change_put = self._top_strike(puts, "oi_change", 0)
        context.max_oi_change_strike = self._dominant_change_strike(calls, puts)

        context.max_pain = self.max_pain(snapshot)
        context.pcr = self.pcr(snapshot)
        context.atm_iv = self.atm_iv(snapshot)
        context.iv_percentile = self.iv_percentile(context.atm_iv, iv_history)
        context.oi_tag = self.oi_tag(snapshot, price_change)

        if context.pcr is not None:
            extremes = self.cfg.get("options.pcr_extreme")
            if context.pcr > float(extremes["high"]) or context.pcr < float(extremes["low"]):
                flags.append(Flag.PCR_EXTREME)

        if context.iv_percentile is not None:
            if context.iv_percentile > float(self.cfg.get("options.iv_elevated_percentile")):
                flags.append(Flag.IV_ELEVATED)

        if stale:
            flags.append(Flag.STALE_CHAIN)
            return context, [], flags

        zones = self.oi_zones(
            market,
            context,
            max_change_call=max_change_call,
            max_change_put=max_change_put,
            created_at=snapshot.taken_at,
        )
        return context, zones, flags

    # -- level construction --------------------------------------------------

    def oi_zones(self, market: str, context: ChainContext,
                 max_change_call: float | None, max_change_put: float | None,
                 created_at: datetime) -> list[Zone]:
        """Build the Tier A zones described in 4.7.1.

        Zone width is ``+/- 0.25 x strike_interval`` rather than ATR-based,
        because these are fixed price ladders, not swing clusters.

        Raises:
            ConfigBlockerError: ``strike_interval`` is unset for this market.
                The caller catches this at G7/G8 and logs a gate rejection - it
                is the designed refusal, not a crash.
        """
        instrument = self.cfg.instrument_key(market)
        interval = float(
            self.cfg.require(
                f"instruments.{instrument}.strike_interval",
                "Set the strike ladder (e.g. 50 for Nifty) before OI levels can be built.",
            )
        )
        half_width = float(self.cfg.get("options.oi_zone_strike_fraction")) * interval

        plan: list[tuple[LevelKind, float | None, bool]] = [
            (LevelKind.MAX_CALL_OI, context.max_call_oi_strike, False),
            (LevelKind.MAX_PUT_OI, context.max_put_oi_strike, True),
            (LevelKind.SECOND_MAX_CALL_OI, context.second_max_call_oi_strike, False),
            (LevelKind.SECOND_MAX_PUT_OI, context.second_max_put_oi_strike, True),
            (LevelKind.MAX_OI_CHANGE_CALL, max_change_call, False),
            (LevelKind.MAX_OI_CHANGE_PUT, max_change_put, True),
        ]

        zones: list[Zone] = []
        for kind, strike, is_support in plan:
            if strike is None:
                continue
            zones.append(
                Zone(
                    zone_id=f"{kind.value}-{uuid.uuid4().hex[:8]}",
                    kind=kind,
                    tier=LevelTier.A,
                    low=strike - half_width,
                    high=strike + half_width,
                    centre=float(strike),
                    # Strength 2: an OI wall is a defended level, above a single
                    # swing touch but below a multi-touch structural zone. The
                    # 4.7 convergence bonus is what lifts it above both.
                    strength=2.0,
                    is_support=is_support,
                    touches=1,
                    created_at=created_at,
                )
            )
        return zones

    # -- individual metrics --------------------------------------------------

    @staticmethod
    def _top_strike(quotes: Sequence[OptionQuote], attribute: str,
                    rank: int) -> float | None:
        """Strike with the ``rank``-th highest value of ``attribute``."""
        ranked = sorted(quotes, key=lambda quote: getattr(quote, attribute), reverse=True)
        ranked = [quote for quote in ranked if getattr(quote, attribute) > 0]
        if len(ranked) <= rank:
            return None
        return float(ranked[rank].strike)

    @staticmethod
    def _dominant_change_strike(calls: Sequence[OptionQuote],
                                puts: Sequence[OptionQuote]) -> float | None:
        """The strike with the single largest OI addition, either side.

        4.7.1 describes the largest OI addition on the call side and on the put
        side; this reports whichever of the two is larger as the day's operative
        boundary, which is what the signal records.
        """
        candidates = list(calls) + list(puts)
        additions = [quote for quote in candidates if quote.oi_change > 0]
        if not additions:
            return None
        return float(max(additions, key=lambda quote: quote.oi_change).strike)

    @staticmethod
    def pcr(snapshot: ChainSnapshot) -> float | None:
        """Total put OI divided by total call OI across the snapshot range."""
        call_oi = sum(quote.oi for quote in snapshot.calls())
        put_oi = sum(quote.oi for quote in snapshot.puts())
        if call_oi <= 0:
            return None
        return round(put_oi / call_oi, 4)

    @staticmethod
    def atm_iv(snapshot: ChainSnapshot) -> float | None:
        """Average of the CE and PE implied volatility at the ATM strike."""
        atm = snapshot.atm_strike()
        if atm is None:
            return None
        values = [
            quote.iv
            for quote in snapshot.quotes
            if quote.strike == atm and quote.iv > 0
        ]
        if not values:
            return None
        return round(float(np.mean(values)), 4)

    @staticmethod
    def iv_percentile(current: float | None,
                      history: Sequence[float]) -> float | None:
        """Rank of ``current`` within the trailing session history, 0-100.

        Returns ``None`` when there is no history: an unknown percentile must
        not read as a low one, or the ``IV_ELEVATED`` guard silently disables
        itself on the days it matters most.
        """
        if current is None or not history:
            return None
        values = [value for value in history if value is not None and value > 0]
        if not values:
            return None
        below = sum(1 for value in values if value <= current)
        return round(100.0 * below / len(values), 2)

    @staticmethod
    def max_pain(snapshot: ChainSnapshot) -> float | None:
        """The strike at which total option-holder value is minimised.

        Context and magnet only - never a trade trigger (4.7.1).
        """
        strikes = sorted({quote.strike for quote in snapshot.quotes})
        if not strikes:
            return None
        calls, puts = snapshot.calls(), snapshot.puts()

        best_strike, best_pain = None, float("inf")
        for settle in strikes:
            pain = 0.0
            for quote in calls:
                if settle > quote.strike:
                    pain += (settle - quote.strike) * quote.oi
            for quote in puts:
                if settle < quote.strike:
                    pain += (quote.strike - settle) * quote.oi
            if pain < best_pain:
                best_pain, best_strike = pain, settle
        return float(best_strike) if best_strike is not None else None

    @staticmethod
    def oi_tag(snapshot: ChainSnapshot, price_change: float | None) -> OITag:
        """Classify price/OI change per the 4.7.2 table.

        Recorded on every Nifty/Sensex signal and consumed by section 9. It is
        not a gate by default (``options.oi_tag_as_gate: false``) - it needs
        trade history before that decision is worth making.
        """
        if price_change is None:
            if snapshot.session_open_spot is None:
                return OITag.UNKNOWN
            price_change = snapshot.spot - snapshot.session_open_spot

        oi_change = sum(quote.oi_change for quote in snapshot.quotes)
        if price_change == 0 or oi_change == 0:
            return OITag.UNKNOWN

        price_up = price_change > 0
        oi_up = oi_change > 0
        if price_up and oi_up:
            return OITag.LONG_BUILDUP
        if not price_up and oi_up:
            return OITag.SHORT_BUILDUP
        if price_up and not oi_up:
            return OITag.SHORT_COVERING
        return OITag.LONG_UNWINDING

    # -- strike shortlist for G8 --------------------------------------------

    def candidate_quotes(self, snapshot: ChainSnapshot,
                         direction: Direction) -> list[OptionQuote]:
        """Quotes on the correct side for ``direction``.

        Bullish underlying signal buys CE, bearish buys PE. Never sold, never
        spread (soul file 5.7.2, rule 13.10).
        """
        wanted = "CE" if direction is Direction.LONG else "PE"
        return [quote for quote in snapshot.quotes if quote.option_type == wanted]

    def is_otm(self, quote: OptionQuote, spot: float) -> bool:
        """True when ``quote`` is out of the money relative to ``spot``."""
        if quote.option_type == "CE":
            return quote.strike > spot
        return quote.strike < spot


def slice_around_atm(quotes: Iterable[OptionQuote], spot: float, strike_interval: float,
                     n_strikes: int) -> list[OptionQuote]:
    """Trim a full chain to ATM +/- ``n_strikes`` (soul file 4.7).

    Keeping the snapshot narrow matters: PCR and max pain computed over the whole
    ladder are dominated by far-dated, illiquid strikes and stop describing what
    is happening around price.
    """
    if strike_interval <= 0:
        return list(quotes)
    span = n_strikes * strike_interval
    return [quote for quote in quotes if abs(quote.strike - spot) <= span + 1e-9]
