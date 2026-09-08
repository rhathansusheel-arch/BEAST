"""Open positions, P&L, and the trade record each exit produces.

This sits between :mod:`core.exit_manager`, which decides *whether* to exit, and
:mod:`broker.order_executor`, which *does* it. It owns the bookkeeping the soul
file requires on every exit (6.9):

    ``exit_time``, ``exit_price``, ``exit_reason``, ``r_multiple`` (actual,
    including slippage), ``mae``, ``mfe``, ``bars_held``, ``trail_activated``,
    and ``hypothetical_r_if_held_to_target``.

For options, both the **underlying** and **premium** values are recorded for
entry, exit, MAE and MFE. The R-multiple of record is the premium-based one -
that is the actual money - with ``underlying_r_multiple`` stored alongside it.
The gap between the two is the strike-selection diagnostic in 6.9, and section 9
reads it to decide whether edge is leaking in the analysis or in 5.7.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from broker.order_executor import Fill, OrderExecutor
from core.config import Config, get_config
from core.exit_manager import ExitDecision, ExitManager, ManagedPosition
from core.risk_manager import RiskManager
from core.schemas import (
    Direction,
    ExitReason,
    Flag,
    OverrideRecord,
    Signal,
    TradeRecord,
)
from core.session import SessionClock

logger = logging.getLogger("beast.positions")


@dataclass
class PositionUpdate:
    """What one price update produced."""

    closed: bool = False
    trade: TradeRecord | None = None
    note: str | None = None
    alerts: list[str] = field(default_factory=list)


class PositionTracker:
    """Tracks live positions and turns exits into Appendix C records.

    Args:
        executor: The order executor.
        risk: Shared risk manager - it needs to hear about every close so the
            daily loss cap and consecutive-loss trigger stay accurate.
        config: Injected for tests.
    """

    def __init__(self, executor: OrderExecutor, risk: RiskManager,
                 config: Config | None = None) -> None:
        self.cfg = config or get_config()
        self.executor = executor
        self.risk = risk
        self.exits = ExitManager(self.cfg)
        self.positions: dict[str, ManagedPosition] = {}
        self.entry_fills: dict[str, Fill] = {}

    # -- opening -------------------------------------------------------------

    def open(self, signal: Signal, underlying_price: float, now: datetime,
             premium: float | None = None) -> ManagedPosition | None:
        """Place the entry and start tracking the position.

        Returns:
            The tracked position, or ``None`` when the entry was refused.
        """
        if signal.market in self.positions:
            logger.warning("Refusing a second position on %s", signal.market)
            return None

        fill = self.executor.enter(signal, underlying_price, now)
        if fill is None:
            return None

        if signal.leg_type == "OPTION":
            entry_premium = fill.price
            entry_underlying = fill.underlying_price
        else:
            entry_premium = None
            entry_underlying = fill.price

        position = ManagedPosition(
            signal=signal,
            plan=_plan_from_signal(signal),
            entry_time=now,
            entry_underlying=entry_underlying,
            entry_premium=entry_premium,
        )
        self.positions[signal.market] = position
        self.entry_fills[signal.market] = fill

        risk_amount = self.risk.capital * signal.risk_pct
        self.risk.register_open(signal.market, signal.direction, risk_amount, now)
        logger.info(
            "OPEN %s %s at %.2f (underlying %.2f)%s",
            signal.market, signal.direction.value, fill.price, entry_underlying,
            " [paper]" if fill.paper else "",
        )
        return position

    # -- updating ------------------------------------------------------------

    def on_price(self, market: str, underlying_price: float, premium: float | None,
                 now: datetime, clock: SessionClock, high: float | None = None,
                 low: float | None = None, bar_open: float | None = None) -> PositionUpdate:
        """Evaluate exits against a live price update."""
        position = self.positions.get(market)
        if position is None:
            return PositionUpdate()

        decision = self.exits.on_price(
            position, underlying_price, premium, now, clock,
            high=high, low=low, bar_open=bar_open,
        )
        if not decision.should_exit:
            return PositionUpdate()

        trade, alerts = self.close(market, decision, now)
        return PositionUpdate(closed=True, trade=trade, alerts=alerts)

    def on_trigger_close(self, market: str, trigger_df: pd.DataFrame, atr_value: float,
                         now: datetime, clock: SessionClock) -> str | None:
        """Recompute the trailing stop on a trigger-TF close (6.3)."""
        position = self.positions.get(market)
        if position is None:
            return None
        return self.exits.on_trigger_close(position, trigger_df, atr_value, now, clock)

    # -- closing -------------------------------------------------------------

    def close(self, market: str, decision: ExitDecision, now: datetime,
              override: OverrideRecord | None = None) -> tuple[TradeRecord | None, list[str]]:
        """Square off and build the Appendix C trade record.

        Args:
            market: The market to close.
            decision: The exit decision, carrying reason and price.
            now: Exit time.
            override: Set only when the close came through the section 8
                friction step; it is stored on the record as a rule deviation.

        Returns:
            ``(trade_record, alerts)``.
        """
        position = self.positions.get(market)
        if position is None:
            return None, []

        signal = position.signal
        quantity = (
            signal.option_leg.lots * signal.option_leg.lot_size
            if signal.leg_type == "OPTION" and signal.option_leg
            else (signal.futures_leg.contracts if signal.futures_leg else 1)
        )
        reason = decision.reason or ExitReason.SESSION
        fill = self.executor.exit(
            signal, quantity, reason, decision.exit_price, decision.exit_premium, now
        )

        exit_premium = fill.price if (fill and signal.leg_type == "OPTION") else decision.exit_premium
        exit_underlying = decision.exit_price
        entry_fill = self.entry_fills.get(market)

        trade = self._build_record(
            position, reason, exit_underlying, exit_premium, now, entry_fill, fill, override
        )

        pnl = self._realised_pnl(position, exit_underlying, exit_premium, quantity, trade)
        alerts = self.risk.register_close(market, pnl, now)

        self.positions.pop(market, None)
        self.entry_fills.pop(market, None)
        logger.info(
            "CLOSE %s %s at %.2f - %s | %s R %.2f",
            market, signal.direction.value, exit_underlying, reason.value,
            "premium" if signal.leg_type == "OPTION" else "underlying",
            trade.r_multiple if trade.r_multiple is not None else 0.0,
        )
        return trade, alerts

    def flatten_all(self, now: datetime, clocks: dict[str, SessionClock],
                    prices: dict[str, float],
                    premiums: dict[str, float | None] | None = None) -> list[TradeRecord]:
        """Hard-flat every open position (6.7 step 3).

        Positions are never carried past the hard-flat time in either market.
        """
        records: list[TradeRecord] = []
        for market in list(self.positions):
            clock = clocks.get(market)
            if clock is None or not clock.must_flatten(now):
                continue
            price = prices.get(market)
            if price is None:
                continue
            decision = ExitDecision(
                True, ExitReason.SESSION, price,
                (premiums or {}).get(market), "hard flat",
            )
            trade, _ = self.close(market, decision, now)
            if trade:
                records.append(trade)
        return records

    # -- record construction --------------------------------------------------

    def _build_record(self, position: ManagedPosition, reason: ExitReason,
                      exit_underlying: float, exit_premium: float | None,
                      now: datetime, entry_fill: Fill | None, exit_fill: Fill | None,
                      override: OverrideRecord | None) -> TradeRecord:
        """Assemble the Appendix C record, including both R-multiples."""
        signal = position.signal
        underlying_r = position.r_at(exit_underlying)

        premium_r: float | None = None
        if signal.leg_type == "OPTION" and exit_premium is not None:
            premium_r = position.premium_r_at(exit_premium)

        # The R of record: premium-based for options, underlying for futures.
        r_of_record = premium_r if premium_r is not None else underlying_r

        mae_premium_r = mfe_premium_r = None
        if position.mae_premium is not None and position.entry_premium:
            mae_premium_r = position.premium_r_at(position.mae_premium)
            mfe_premium_r = position.premium_r_at(position.mfe_premium or position.entry_premium)

        flags = list(signal.flags)
        for flag in position.flags:
            if flag not in flags:
                flags.append(flag)
        signal.flags = flags

        leg = signal.option_leg
        return TradeRecord(
            signal=signal,
            entry_time=position.entry_time,
            entry_fill_price=position.entry_underlying,
            exit_time=now,
            exit_price=round(exit_underlying, 4),
            exit_reason=reason,
            r_multiple=round(r_of_record, 4),
            underlying_r_multiple=round(underlying_r, 4),
            mae_r=round(position.mae_r, 4),
            mfe_r=round(position.mfe_r, 4),
            bars_held=position.bars_held,
            trail_activated=position.trail_activated,
            hypothetical_r_if_held_to_target=self.exits.hypothetical_r_if_held(position),
            override=override,
            slippage=(entry_fill.slippage if entry_fill else 0.0)
            + (exit_fill.slippage if exit_fill else 0.0),
            costs=(entry_fill.costs if entry_fill else 0.0)
            + (exit_fill.costs if exit_fill else 0.0),
            entry_premium=position.entry_premium,
            exit_premium=exit_premium,
            entry_underlying=position.entry_underlying,
            exit_underlying=round(exit_underlying, 4),
            premium_r_multiple=round(premium_r, 4) if premium_r is not None else None,
            mae_premium=position.mae_premium,
            mfe_premium=position.mfe_premium,
            delta_at_entry=leg.delta if leg else None,
            iv_at_entry=leg.iv if leg else None,
            iv_at_exit=None,
            dte=leg.dte if leg else None,
            theta_cost_estimate=self._theta_estimate(position, exit_premium),
            slippage_premium=(exit_fill.slippage if exit_fill and leg else 0.0),
        )

    def _theta_estimate(self, position: ManagedPosition,
                        exit_premium: float | None) -> float | None:
        """Rough theta cost: premium change unexplained by the underlying move.

        ``delta x underlying_move`` is what the position *should* have made. The
        shortfall against the actual premium change is theta plus IV drift, and
        6.10's theta guard exists precisely because that shortfall is invisible
        on the underlying chart.
        """
        leg = position.option_leg
        if leg is None or exit_premium is None or position.entry_premium is None:
            return None
        expected = (
            (position.last_underlying - position.entry_underlying)
            * position.direction.sign
            * leg.delta
        )
        actual = exit_premium - position.entry_premium
        return round(actual - expected, 4)

    def _realised_pnl(self, position: ManagedPosition, exit_underlying: float,
                      exit_premium: float | None, quantity: int,
                      trade: TradeRecord) -> float:
        """Realised profit or loss in account currency, net of costs."""
        signal = position.signal
        if signal.leg_type == "OPTION" and signal.option_leg and exit_premium is not None:
            gross = (exit_premium - (position.entry_premium or 0.0)) * quantity
        elif signal.futures_leg:
            points = (exit_underlying - position.entry_underlying) * position.direction.sign
            gross = points * signal.futures_leg.contract_multiplier * quantity
        else:
            gross = 0.0
        return round(gross - trade.costs, 2)

    # -- queries -------------------------------------------------------------

    def open_markets(self) -> list[str]:
        """Markets with a live position."""
        return list(self.positions)

    def get(self, market: str) -> ManagedPosition | None:
        return self.positions.get(market)

    def snapshot(self) -> list[dict[str, Any]]:
        """Dashboard-facing summary of every open position."""
        rows = []
        for market, position in self.positions.items():
            rows.append(
                {
                    "market": market,
                    "direction": position.direction.value,
                    "setup": int(position.signal.setup_type),
                    "entry": position.entry_underlying,
                    "stop": position.current_stop,
                    "target": position.plan.target_price,
                    "last": position.last_underlying,
                    "unrealised_r": round(position.r_at(position.last_underlying), 2),
                    "mae_r": round(position.mae_r, 2),
                    "mfe_r": round(position.mfe_r, 2),
                    "bars": position.bars_held,
                    "trail": position.trail_activated,
                    "premium": position.last_premium,
                    "theta_drag": Flag.THETA_DRAG in position.flags,
                    # What this position loses if its stop fills, so the
                    # dashboard can show capital actually at risk rather than a
                    # notional allocation Beast does not have.
                    "risk_amount": round(
                        getattr(
                            self.risk.open_positions.get(market), "risk_amount", 0.0
                        ) or 0.0,
                        2,
                    ),
                    "held": _held_for(position),
                }
            )
        return rows


def _held_for(position) -> str:
    """How long the position has been open, as ``3h`` or ``42m``."""
    opened = getattr(position, "entry_time", None)
    if opened is None:
        return f"{position.bars_held} bars"
    minutes = int((datetime.now(opened.tzinfo) - opened).total_seconds() // 60)
    if minutes < 0:
        return "-"
    return f"{minutes // 60}h{minutes % 60:02d}m" if minutes >= 60 else f"{minutes}m"


def _plan_from_signal(signal: Signal):
    """Reconstruct the trade plan carried on a signal.

    The plan is fixed at entry and travels with the signal, so rebuilding it here
    is a projection, never a recomputation - nothing about the exit is decided
    after the fact (soul file 6).
    """
    from core.schemas import TradePlan

    risk_points = abs(signal.entry_price - signal.stop_price)
    return TradePlan(
        direction=signal.direction,
        entry_price=signal.entry_price,
        stop_price=signal.stop_price,
        target_price=signal.target_price,
        stop_source=signal.stop_source,
        target_r=signal.target_r,
        trail=signal.trail,
        atr=signal.atr_setup_tf,
        risk_points=risk_points,
        viable=True,
    )
