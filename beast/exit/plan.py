"""Sections 6.1 and 6.2 - stop placement, the fixed R:R target, and the G7 viability check.

"Every trade is entered with a complete exit plan - stop, target, and trail parameters -
computed **before** the entry signal is emitted. An entry whose exit plan cannot be
constructed is not an entry. Nothing in this section is decided after the fact."

All prices here are **underlying** points (index points / gold price). Section 6.10 handles
how an underlying-level exit is executed on an option premium; nothing in this module knows
what a premium is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from beast.constants import Direction, Tier, ZoneKind
from beast.schemas import TrailSpec


@dataclass
class TradePlan:
    """A complete exit plan in underlying points, or a reason it could not be built."""

    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    stop_source: str = ""
    target_price: Optional[float] = None
    target_r: float = 0.0
    trail: Optional[TrailSpec] = None
    r_points: float = 0.0
    viable: bool = False
    reject_reason: str = ""

    def r_multiple(self, price: float, direction: Direction) -> float:
        """Where ``price`` sits in R, signed in the trade's favour."""
        if not self.r_points:
            return 0.0
        return direction.sign * (price - self.entry_price) / self.r_points


def _opposing_kind(direction: Direction) -> ZoneKind:
    return ZoneKind.RESISTANCE if direction is Direction.LONG else ZoneKind.SUPPORT


def next_opposing_level(zones, entry: float, direction: Direction, tier: Optional[Tier] = Tier.A):
    """The nearest opposing level ahead of the trade (6.2, 5.2).

    For Nifty/Sensex this pool includes OI-derived Tier A levels (4.7.1), which is why a
    heavy call wall above can reject a long at G7.
    """
    kind = _opposing_kind(direction)
    ahead = []
    for zone in zones:
        if zone.kind is not kind:
            continue
        if tier is not None and zone.tier is not tier:
            continue
        edge = zone.low if direction is Direction.LONG else zone.high
        if direction is Direction.LONG and edge > entry:
            ahead.append((edge - entry, zone, edge))
        elif direction is Direction.SHORT and edge < entry:
            ahead.append((entry - edge, zone, edge))
    if not ahead:
        return None, None
    _, zone, edge = min(ahead, key=lambda t: t[0])
    return zone, edge


def build(setup, ctx, entry_price: float) -> TradePlan:
    """Construct the full exit plan, or return a non-viable plan with the G7 reason.

    Stop (6.1): structural point from the setup, plus a ``0.25 x ATR`` buffer, then clamped
    to a minimum of ``0.5 x ATR``. A structure demanding more than ``2.5 x ATR`` is
    **rejected**, not sized down - "Beast does not size down to accommodate a broken
    structure".

    Target (6.2): fixed R:R at ``target_r``. If an opposing Tier A level sits between entry
    and that target, the configured ``target_infeasible_policy`` decides between rejecting
    the trade (default) and targeting the level with a 1.5R floor.
    """
    cfg, atr_value = ctx.cfg, ctx.atr
    direction = setup.direction
    stop = setup.stop_price(atr_value, cfg)
    distance = abs(entry_price - stop)

    min_dist = float(cfg.get("exit.stop_min_atr")) * atr_value
    max_dist = float(cfg.get("exit.stop_max_atr")) * atr_value
    stop_source = setup.stop_source

    if distance > max_dist:
        return TradePlan(
            reject_reason=(
                f"structural stop {distance:.2f} exceeds max {max_dist:.2f} "
                f"({cfg.get('exit.stop_max_atr')} ATR) - rejected, never sized down"
            )
        )
    if distance < min_dist:
        distance = min_dist
        stop = entry_price - direction.sign * distance
        stop_source = f"{setup.stop_source} widened to min {cfg.get('exit.stop_min_atr')}ATR"

    target_r = float(cfg.get("exit.target_r"))
    target = entry_price + direction.sign * target_r * distance

    # Setup 1 disqualifier (5.2): no room to target.
    if setup.setup_type == 1:
        _, edge = next_opposing_level(ctx.zones, entry_price, direction)
        if edge is not None and abs(edge - entry_price) < distance:
            return TradePlan(
                reject_reason=f"opposing Tier A level at {edge:.2f} is closer than 1.0R - no room to target"
            )

    zone, edge = next_opposing_level(ctx.zones, entry_price, direction)
    if edge is not None and abs(edge - entry_price) < abs(target - entry_price):
        policy = str(cfg.get("exit.target_infeasible_policy"))
        if policy == "reject":
            return TradePlan(
                reject_reason=(
                    f"Tier A {zone.source} at {edge:.2f} blocks the {target_r}R target - "
                    "Beast does not shrink R:R to fit a marginal setup"
                )
            )
        floor = float(cfg.get("exit.target_level_min_r"))
        level_r = abs(edge - entry_price) / distance
        if level_r < floor:
            return TradePlan(
                reject_reason=f"level target only {level_r:.2f}R, below the {floor}R floor"
            )
        target, target_r = edge, level_r

    activate_r = _trail_activation_r(ctx, cfg)
    trail = TrailSpec(
        activate_at=entry_price + direction.sign * activate_r * distance,
        method=str(cfg.get("exit.trail_method")),
        mult=float(cfg.get("exit.trail_atr_mult")),
    )
    return TradePlan(
        entry_price=entry_price,
        stop_price=stop,
        stop_source=stop_source,
        target_price=target,
        target_r=target_r,
        trail=trail,
        r_points=distance,
        viable=True,
    )


def _trail_activation_r(ctx, cfg) -> float:
    """+1.0R normally; +0.7R on option expiry day (5.7.4) - decay works against the trade."""
    base = float(cfg.get("exit.trail_activate_r"))
    if ctx.market.is_option_market and ctx.chain is not None and cfg.get("options.expiry_day.enabled"):
        if ctx.chain.dte(ctx.now) == 0:
            return float(cfg.get("options.expiry_day.trail_activate_r"))
    return base
