"""Section 7 position sizing, and 7.1 delta-based sizing for option legs.

The risk-per-trade % is a **ceiling, not a target** (Section 7). ``vol_factor`` is clamped
at 1.0 so volatility adjustment can only ever reduce exposure.

7.1 exists because options break the linear formula: the loss at the stop is not the
premium paid, it is the *premium decline* when the underlying reaches the stop. Sizing off
premium paid systematically undersizes; sizing off the raw point distance systematically
oversizes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from beast.config import ConfigUnset
from beast.constants import Market
from beast.ops.immutable import ImmutableRuleViolation, assert_whole_lots


@dataclass
class SizedPosition:
    """The G9 output: how many lots/contracts, and which cap decided it."""

    permitted: bool
    reason: str
    quantity: int = 0
    lot_size: int = 0
    vol_factor: float = 1.0
    risk_amount: float = 0.0
    risk_pct: float = 0.0
    total_premium_outlay: float = 0.0
    binding_cap: str = ""


def vol_factor(atr_current: float, atr_median: Optional[float], cfg) -> float:
    """``clamp(ATR_median_20d / ATR_current, floor, 1.0)`` (Section 7).

    Sizing shrinks in choppy conditions and never expands past 1.0. With no median
    available yet the factor is 1.0 - the risk cap alone then governs.
    """
    if not atr_median or not atr_current:
        return 1.0
    floor = float(cfg.get("risk.vol_factor_floor"))
    return max(floor, min(1.0, atr_median / atr_current))


def size_futures(
    plan, cfg, market: Market, atr_current: float, atr_median: Optional[float] = None
) -> SizedPosition:
    """Linear sizing for Gold futures (Section 7).

    Refuses to size at all until the contract specs are populated - Section 3.1: "Beast
    will refuse to size a Gold trade until they are populated."
    """
    try:
        multiplier = float(
            cfg.require(
                f"instruments.{market.instrument_key}.contract_multiplier",
                "3.1 - contract specs must be populated before Beast will size a Gold trade",
            )
        )
        capital = cfg.capital()
    except ConfigUnset as exc:
        return SizedPosition(False, str(exc))

    risk_pct = cfg.risk_per_trade(market)
    risk_amount = capital * risk_pct
    distance = abs(plan.entry_price - plan.stop_price)
    if distance <= 0:
        return SizedPosition(False, "zero stop distance")

    factor = vol_factor(atr_current, atr_median, cfg)
    raw = risk_amount / (distance * multiplier)
    contracts = math.floor(raw * factor)
    if contracts < 1:
        return SizedPosition(
            False,
            f"sized below one contract ({raw * factor:.3f}) - rejected at G9, never rounded up",
            vol_factor=factor,
            risk_amount=risk_amount,
            risk_pct=risk_pct,
        )
    return SizedPosition(
        True,
        f"{contracts} contract(s) at {risk_pct:.1%} risk, vol factor {factor:.2f}",
        quantity=contracts,
        lot_size=1,
        vol_factor=factor,
        risk_amount=risk_amount,
        risk_pct=risk_pct,
        binding_cap="risk",
    )


def size_option(
    plan, leg, cfg, market: Market, atr_current: float, atr_median: Optional[float] = None
) -> SizedPosition:
    """Delta-based sizing for a long option leg (7.1).

    ::

        premium_loss_per_unit = underlying_stop_distance x delta
        raw_lots              = risk_amount / (premium_loss_per_unit x lot_size)
        lots                  = floor(raw_lots x vol_factor)

    Three caps sit on top, and the binding one wins:

    1. **Premium outlay cap** - total premium paid <= 10% of capital. Premium paid is the
       theoretical maximum loss and must stay survivable even if the position goes to zero.
    2. **Whole-lot floor** - ``lots < 1`` is rejected at G9, never rounded up (Rule 11).
    3. **Delta drift** - delta is read at entry and never recomputed for sizing; Beast
       never adds to a position mid-trade.
    """
    try:
        capital = cfg.capital()
        lot_size = int(
            cfg.require(
                f"instruments.{market.instrument_key}.lot_size",
                "Open Item 18 - lot size must be supplied or read from the broker API",
            )
        )
    except ConfigUnset as exc:
        return SizedPosition(False, str(exc))

    risk_pct = cfg.risk_per_trade(market)
    risk_amount = capital * risk_pct
    distance = abs(plan.entry_price - plan.stop_price)
    if distance <= 0 or leg.delta <= 0:
        return SizedPosition(False, "zero stop distance or delta")

    premium_loss_per_unit = distance * leg.delta
    factor = vol_factor(atr_current, atr_median, cfg)
    raw_lots = risk_amount / (premium_loss_per_unit * lot_size)
    lots_by_risk = math.floor(raw_lots * factor)

    outlay_cap = capital * float(cfg.get("options.max_premium_outlay_pct"))
    lots_by_outlay = math.floor(outlay_cap / (lot_size * leg.mid_premium)) if leg.mid_premium else 0

    lots = min(lots_by_risk, lots_by_outlay)
    binding = "premium_outlay" if lots_by_outlay < lots_by_risk else "risk"

    try:
        assert_whole_lots(lots, max(raw_lots * factor, 0.0))
    except ImmutableRuleViolation as exc:
        return SizedPosition(
            False,
            f"{exc} (risk-capped {lots_by_risk}, outlay-capped {lots_by_outlay})",
            vol_factor=factor,
            risk_amount=risk_amount,
            risk_pct=risk_pct,
            binding_cap=binding,
        )

    outlay = lots * lot_size * leg.mid_premium
    return SizedPosition(
        True,
        f"{lots} lot(s), {binding} cap binding, vol factor {factor:.2f}",
        quantity=lots,
        lot_size=lot_size,
        vol_factor=factor,
        risk_amount=risk_amount,
        risk_pct=risk_pct,
        total_premium_outlay=outlay,
        binding_cap=binding,
    )
