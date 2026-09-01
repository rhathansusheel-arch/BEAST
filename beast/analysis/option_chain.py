"""Section 4.7 - option chain analysis (Nifty / Sensex only).

The chain is a **context and level source, not a signal source**. It never adds to or
subtracts from the 4-of-6 confluence count in 5.3 - that engine stays on six indicators, on
the underlying. What the chain does is:

* (a) contribute Tier A levels to the same pool as price structure (4.7.1),
* (b) inform the target-feasibility check at G7,
* (c) constrain which instrument is bought at G8 (5.7).

Everything here is recorded on the signal (Appendix B ``chain_context``) so Section 9 can
learn from it before any of it is ever promoted to a gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

from beast.constants import Flag, OITag, Tier, ZoneKind
from beast.analysis.levels import Zone


@dataclass
class StrikeQuote:
    """One strike's option data on one side of the chain."""

    strike: float
    option_type: str  # CE | PE
    bid: float
    ask: float
    last: float
    oi: int
    oi_change: int
    volume: int
    iv: float
    delta: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0 if self.bid > 0 and self.ask > 0 else self.last

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass
class ChainSnapshot:
    """A chain snapshot for the nearest weekly expiry, ATM +/- N strikes (4.7).

    Pulled on each setup-TF close. ``taken_at`` drives the 4.6 freshness check - a snapshot
    older than ``options.chain_max_age_sec`` degrades context but does not halt trading.
    """

    underlying: str
    spot: float
    expiry: date
    taken_at: datetime
    strike_interval: float
    quotes: list[StrikeQuote] = field(default_factory=list)
    iv_percentile: Optional[float] = None
    expiries: list[date] = field(default_factory=list)

    # -- basics ---------------------------------------------------------------

    def calls(self) -> list[StrikeQuote]:
        return [q for q in self.quotes if q.option_type == "CE"]

    def puts(self) -> list[StrikeQuote]:
        return [q for q in self.quotes if q.option_type == "PE"]

    def age_seconds(self, now: datetime) -> float:
        return (now - self.taken_at).total_seconds()

    def dte(self, now: datetime) -> int:
        """4.7.5 - days to expiry, driving strike selection in 5.7."""
        return (self.expiry - now.date()).days

    def atm_strike(self) -> float:
        return min((q.strike for q in self.quotes), key=lambda s: abs(s - self.spot))

    # -- 4.7.1 OI-derived levels ---------------------------------------------

    def max_call_oi(self) -> Optional[StrikeQuote]:
        return max(self.calls(), key=lambda q: q.oi, default=None)

    def max_put_oi(self) -> Optional[StrikeQuote]:
        return max(self.puts(), key=lambda q: q.oi, default=None)

    def second_max_call_oi(self) -> Optional[StrikeQuote]:
        ranked = sorted(self.calls(), key=lambda q: q.oi, reverse=True)
        return ranked[1] if len(ranked) > 1 else None

    def second_max_put_oi(self) -> Optional[StrikeQuote]:
        ranked = sorted(self.puts(), key=lambda q: q.oi, reverse=True)
        return ranked[1] if len(ranked) > 1 else None

    def max_oi_change(self, option_type: str) -> Optional[StrikeQuote]:
        """Largest OI *addition* since session open on one side (4.7.1)."""
        side = [q for q in self.quotes if q.option_type == option_type and q.oi_change > 0]
        return max(side, key=lambda q: q.oi_change, default=None)

    def max_pain(self) -> Optional[float]:
        """Strike minimising total option-holder value.

        Magnet/context only - 4.7.1 is explicit that this is never a trade trigger.
        """
        strikes = sorted({q.strike for q in self.quotes})
        if not strikes:
            return None
        pains = []
        for expiry_price in strikes:
            total = 0.0
            for q in self.quotes:
                if q.option_type == "CE":
                    total += max(expiry_price - q.strike, 0.0) * q.oi
                else:
                    total += max(q.strike - expiry_price, 0.0) * q.oi
            pains.append((total, expiry_price))
        return min(pains)[1]

    # -- 4.7.3 / 4.7.4 context ------------------------------------------------

    def pcr(self) -> Optional[float]:
        """Total put OI / total call OI across the snapshot range (4.7.3)."""
        call_oi = sum(q.oi for q in self.calls())
        put_oi = sum(q.oi for q in self.puts())
        return put_oi / call_oi if call_oi else None

    def atm_iv(self) -> Optional[float]:
        atm = self.atm_strike()
        ivs = [q.iv for q in self.quotes if q.strike == atm and q.iv]
        return sum(ivs) / len(ivs) if ivs else None


def oi_zones(snapshot: ChainSnapshot, cfg) -> list[Zone]:
    """Build Tier A zones from OI levels (4.7.1).

    Zone width is ``+/- 0.25 x strike_interval`` rather than ATR-based, because these are
    fixed price ladders rather than swing clusters.
    """
    frac = float(cfg.get("options.oi_level_zone_strike_frac"))
    half = frac * snapshot.strike_interval
    zones: list[Zone] = []

    def add(quote: Optional[StrikeQuote], kind: ZoneKind, label: str, strength: int) -> None:
        if quote is None:
            return
        zones.append(
            Zone(
                kind=kind,
                low=quote.strike - half,
                high=quote.strike + half,
                tier=Tier.A,
                source=f"oi:{label}",
                touches=1,
                strength=strength,
                created_ts=snapshot.taken_at,
            )
        )

    add(snapshot.max_call_oi(), ZoneKind.RESISTANCE, "max_call_oi", 3)
    add(snapshot.max_put_oi(), ZoneKind.SUPPORT, "max_put_oi", 3)
    add(snapshot.second_max_call_oi(), ZoneKind.RESISTANCE, "second_call_oi", 2)
    add(snapshot.second_max_put_oi(), ZoneKind.SUPPORT, "second_put_oi", 2)
    add(snapshot.max_oi_change("CE"), ZoneKind.RESISTANCE, "max_oi_change_call", 2)
    add(snapshot.max_oi_change("PE"), ZoneKind.SUPPORT, "max_oi_change_put", 2)
    return zones


def oi_tag(price_change: float, oi_change: float) -> Optional[OITag]:
    """4.7.2 - the price/OI quadrant, recorded on every Nifty/Sensex signal.

    Context only by default (``options.oi_tag_as_gate: false``): it is the natural first
    candidate for a chain-based filter, but that decision needs trade history first.
    """
    if price_change == 0 or oi_change == 0:
        return None
    if price_change > 0:
        return OITag.LONG_BUILDUP if oi_change > 0 else OITag.SHORT_COVERING
    return OITag.SHORT_BUILDUP if oi_change > 0 else OITag.LONG_UNWINDING


def chain_flags(snapshot: ChainSnapshot, cfg, now: datetime, news_in_window: bool = False) -> list[Flag]:
    """4.7.3 / 4.7.4 / 4.7.5 context flags."""
    flags: list[Flag] = []
    pcr = snapshot.pcr()
    if pcr is not None:
        high = float(cfg.get("options.pcr_extreme.high"))
        low = float(cfg.get("options.pcr_extreme.low"))
        if pcr > high or pcr < low:
            flags.append(Flag.PCR_EXTREME)
    if snapshot.iv_percentile is not None:
        if snapshot.iv_percentile > float(cfg.get("options.iv_elevated_percentile")):
            flags.append(Flag.IV_ELEVATED)
    if news_in_window:
        flags.append(Flag.IV_CRUSH_RISK)
    if snapshot.dte(now) == 0:
        flags.append(Flag.EXPIRY_DAY)
    return flags
