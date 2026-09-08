"""Risk management - soul file section 7, and gate G9.

Three things live here, and they are the three the immutable rules in section 13
care most about:

* **Position sizing** - linear for Gold futures, delta-based for option legs
  (7.1). Sizing an option from the premium paid systematically undersizes;
  sizing it from the raw point distance systematically oversizes. Beast sizes
  from the *premium decline when the underlying reaches the stop*.
* **Loss limits** - the percentage cap and the three-consecutive-loss trigger,
  whichever comes first, applied per market for the remainder of the session.
* **Exposure rules** - the concurrent-position cap and the Nifty/Sensex
  correlation rule.

Every number here is a ceiling, never a target. ``vol_factor`` is clamped at 1.0
precisely so volatility adjustment can only ever reduce exposure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from core.config import Config, ConfigBlockerError, get_config
from core.schemas import (
    Direction,
    FuturesLeg,
    OptionLeg,
    SizedPosition,
    TradePlan,
)


@dataclass
class OpenExposure:
    """A currently-open position, as far as the risk layer is concerned."""

    market: str
    direction: Direction
    risk_amount: float
    opened_at: datetime


@dataclass
class MarketRiskState:
    """Per-market-family loss and pause state.

    The pause is per market: an Indian-session pause does not stop XAUUSD, and
    vice versa (soul file 7).
    """

    family: str
    session_day: object | None = None
    realised_pnl: float = 0.0
    consecutive_losses: int = 0
    trades_today: int = 0
    paused: bool = False
    pause_reason: str = ""
    cooldown_until: dict[str, datetime] = field(default_factory=dict)


class RiskManager:
    """Owns sizing, loss limits and exposure caps.

    Args:
        config: Injected for tests.
        capital: Starting capital. In live mode the runner refreshes this from
            the broker; in paper mode it comes from ``risk.capital``.
    """

    def __init__(self, config: Config | None = None, capital: float | None = None) -> None:
        self.cfg = config or get_config()
        self.capital = float(capital if capital is not None else self.cfg.get("risk.capital"))
        self.state: dict[str, MarketRiskState] = {
            family: MarketRiskState(family) for family in ("indian", "gold")
        }
        self.open_positions: dict[str, OpenExposure] = {}

    # -- session lifecycle ---------------------------------------------------

    def roll_session(self, family: str, session_day: object) -> None:
        """Reset daily counters when a new session starts.

        The daily loss cap and the consecutive-loss trigger are both session
        scoped, so a fresh session clears the pause. Nothing else about the
        rules changes between sessions.
        """
        state = self.state[family]
        if state.session_day == session_day:
            return
        state.session_day = session_day
        state.realised_pnl = 0.0
        state.consecutive_losses = 0
        state.trades_today = 0
        state.paused = False
        state.pause_reason = ""
        state.cooldown_until.clear()

    # -- volatility adjustment ----------------------------------------------

    def vol_factor(self, atr_current: float, atr_median: float) -> float:
        """``clamp(ATR_median / ATR_current, floor, 1.0)`` (soul file 7).

        Capped at 1.0 so the adjustment can only shrink exposure. When the
        current ATR is unavailable the factor is 1.0 - no adjustment - rather
        than a guess in either direction.
        """
        floor = float(self.cfg.get("risk.vol_factor_floor"))
        if atr_current <= 0 or atr_median <= 0:
            return 1.0
        return float(min(1.0, max(floor, atr_median / atr_current)))

    # -- sizing --------------------------------------------------------------

    def size_futures(self, plan: TradePlan, leg: FuturesLeg, market: str,
                     vol_factor: float) -> SizedPosition:
        """Linear sizing for Gold futures (soul file 7).

        ``(entry - stop) x contract_multiplier`` is the loss per contract, and
        size rounds *down* to whole contracts. A sub-one-contract result is a
        rejection, not a rounding decision (rule 13.11).
        """
        risk_pct = self.cfg.risk_per_trade(market)
        risk_amount = self.capital * risk_pct
        stop_distance = abs(plan.entry_price - plan.stop_price)
        if stop_distance <= 0:
            return SizedPosition(False, 0, risk_amount, vol_factor, reason="zero stop distance")

        loss_per_contract = stop_distance * leg.contract_multiplier
        if loss_per_contract <= 0:
            return SizedPosition(
                False, 0, risk_amount, vol_factor, reason="non-positive loss per contract"
            )

        raw = risk_amount / loss_per_contract
        contracts = int(math.floor(raw * vol_factor))
        if contracts < 1:
            return SizedPosition(
                False,
                0,
                risk_amount,
                vol_factor,
                reason=(
                    f"sized to {raw * vol_factor:.2f} contracts; one contract risks "
                    f"{loss_per_contract:,.0f} against a {risk_amount:,.0f} budget. "
                    f"Rounding up would breach the per-trade cap."
                ),
            )
        return SizedPosition(
            True, contracts, risk_amount, vol_factor, binding_cap="risk",
            reason=f"{contracts} contract(s) at {loss_per_contract:,.0f} risk each",
        )

    def size_option(self, plan: TradePlan, leg: OptionLeg, market: str,
                    vol_factor: float) -> SizedPosition:
        """Delta-based sizing for a long option leg (soul file 7.1).

        Three caps sit on top of the risk formula and the binding one wins:

        1. **Premium outlay cap** - total premium paid must stay within
           ``options.max_premium_outlay_pct`` of capital, because premium paid is
           the theoretical maximum loss and it must stay survivable even in the
           pathological case where the position goes to zero.
        2. **Whole-lot floor** - ``lots < 1`` is rejected at G9, never rounded up.
        3. **Delta drift** - delta is read at entry and never recomputed for
           sizing. Beast never adds to a position mid-trade.
        """
        risk_pct = self.cfg.risk_per_trade(market)
        risk_amount = self.capital * risk_pct
        stop_distance = abs(plan.entry_price - plan.stop_price)

        if stop_distance <= 0 or leg.delta <= 0 or leg.lot_size <= 0:
            return SizedPosition(
                False, 0, risk_amount, vol_factor,
                reason="stop distance, delta or lot size is non-positive",
            )

        premium_loss_per_unit = stop_distance * leg.delta
        loss_per_lot = premium_loss_per_unit * leg.lot_size
        if loss_per_lot <= 0:
            return SizedPosition(
                False, 0, risk_amount, vol_factor, reason="non-positive loss per lot"
            )

        raw_lots = risk_amount / loss_per_lot
        risk_lots = int(math.floor(raw_lots * vol_factor))

        outlay_cap = self.capital * float(self.cfg.get("options.max_premium_outlay_pct"))
        cost_per_lot = leg.mid_premium * leg.lot_size
        outlay_lots = (
            int(math.floor(outlay_cap / cost_per_lot)) if cost_per_lot > 0 else 0
        )

        lots = min(risk_lots, outlay_lots)
        binding = "premium_outlay" if outlay_lots < risk_lots else "risk"

        if lots < 1:
            return SizedPosition(
                False,
                0,
                risk_amount,
                vol_factor,
                binding_cap=binding,
                reason=(
                    f"risk allows {risk_lots} lot(s), the premium outlay cap allows "
                    f"{outlay_lots}. One lot risks {loss_per_lot:,.0f} and costs "
                    f"{cost_per_lot:,.0f}. Rejected rather than rounded up (rule 13.11)."
                ),
            )

        leg.lots = lots
        leg.total_premium_outlay = round(lots * cost_per_lot, 2)
        leg.binding_cap = binding
        return SizedPosition(
            True,
            lots,
            risk_amount,
            vol_factor,
            binding_cap=binding,
            reason=(
                f"{lots} lot(s); {binding} cap binding "
                f"(risk {risk_lots}, outlay {outlay_lots})"
            ),
        )

    # -- gate G9 -------------------------------------------------------------

    def check_portfolio(self, market: str, direction: Direction,
                        risk_amount: float, now: datetime) -> tuple[bool, str]:
        """Concurrent-position cap, correlation rule and loss-limit state.

        Returns:
            ``(permitted, reason)``. The reason is logged verbatim on the
            rejection so the operator can see which cap bound.
        """
        family = self.cfg.market_family(market)
        state = self.state[family]

        if state.paused:
            return False, f"{family} paused for the session: {state.pause_reason}"

        cooldown = state.cooldown_until.get(market)
        if cooldown and now < cooldown:
            remaining = int((cooldown - now).total_seconds() // 60) + 1
            return False, f"post-loss cooldown on {market}, {remaining} min remaining"

        ok, reason = self._check_correlation(market, direction, risk_amount)
        if not ok:
            return False, reason

        cap = int(self.cfg.get("risk.max_concurrent")[family])
        if self._effective_position_count(family) >= cap:
            return False, f"{family} already at the concurrent-position cap of {cap}"

        return True, "portfolio checks clear"

    def _check_correlation(self, market: str, direction: Direction,
                           risk_amount: float) -> tuple[bool, str]:
        """Apply the Nifty/Sensex correlation rule (soul file 7).

        Simultaneous same-direction positions in both count as **one** position
        against the concurrent cap, and their combined risk may not exceed a
        single trade's risk allocation. Opposite-direction simultaneous
        positions are not permitted at all.
        """
        if not bool(self.cfg.get("risk.nifty_sensex_correlated")):
            return True, "correlation rule disabled"

        family = self.cfg.market_family(market)
        if family != "indian":
            return True, "correlation rule applies to Indian indices only"

        siblings = [
            exposure
            for key, exposure in self.open_positions.items()
            if self.cfg.market_family(exposure.market) == "indian" and key != market
        ]
        if not siblings:
            return True, "no correlated position open"

        for exposure in siblings:
            if exposure.direction is not direction:
                return False, (
                    f"{exposure.market} is open {exposure.direction.value}; opposite-direction "
                    f"simultaneous Nifty/Sensex positions are not permitted"
                )

        combined = sum(exposure.risk_amount for exposure in siblings) + risk_amount
        allowance = self.capital * self.cfg.risk_per_trade(market)
        if combined > allowance + 1e-6:
            return False, (
                f"combined Nifty/Sensex risk {combined:,.0f} exceeds a single trade's "
                f"allocation of {allowance:,.0f}"
            )
        return True, "correlated exposure within a single trade's allocation"

    def _effective_position_count(self, family: str) -> int:
        """Positions counted against the concurrent cap.

        Same-direction Nifty and Sensex positions collapse to one (soul file 7).
        """
        positions = [
            exposure
            for exposure in self.open_positions.values()
            if self.cfg.market_family(exposure.market) == family
        ]
        if family != "indian" or not bool(self.cfg.get("risk.nifty_sensex_correlated")):
            return len(positions)

        longs = [item for item in positions if item.direction is Direction.LONG]
        shorts = [item for item in positions if item.direction is Direction.SHORT]
        return (1 if longs else 0) + (1 if shorts else 0)

    # -- position bookkeeping ------------------------------------------------

    def register_open(self, market: str, direction: Direction, risk_amount: float,
                      now: datetime) -> None:
        """Record a newly opened position against the exposure caps."""
        self.open_positions[market] = OpenExposure(market, direction, risk_amount, now)

    def register_close(self, market: str, pnl: float, now: datetime) -> list[str]:
        """Record a closed trade and apply the loss limits.

        Args:
            market: The market that closed.
            pnl: Realised profit or loss in account currency.
            now: Close time, for the post-loss cooldown.

        Returns:
            Alert strings for anything that tripped - a pause, or a cooldown.
            Empty when nothing changed.
        """
        self.open_positions.pop(market, None)
        family = self.cfg.market_family(market)
        state = self.state[family]
        state.realised_pnl += pnl
        state.trades_today += 1

        alerts: list[str] = []

        if pnl < 0:
            state.consecutive_losses += 1
            minutes = int(self.cfg.get("entry.post_loss_cooldown_min"))
            state.cooldown_until[market] = now + timedelta(minutes=minutes)
            alerts.append(f"{market}: {minutes} min post-loss cooldown started")
        else:
            state.consecutive_losses = 0

        trigger = int(self.cfg.get("risk.consecutive_loss_trigger"))
        if state.consecutive_losses >= trigger and not state.paused:
            state.paused = True
            state.pause_reason = f"{trigger} consecutive losses"
            alerts.append(
                f"{family.upper()} PAUSED for the session - {state.pause_reason}. "
                f"No new entries; open positions continue to be managed."
            )

        # v3.1 caps are per-instrument (nifty 15%, sensex 10%, gold 5%) while the
        # session pause below is still family-scoped, so whichever instrument
        # trips first pauses the family. That is stricter than v3.1's per-
        # instrument reading, never looser. See DECISIONS.md D-07.
        cap_pct = self.cfg.daily_loss_cap(market)
        cap_amount = self.capital * cap_pct
        if state.realised_pnl <= -cap_amount and not state.paused:
            state.paused = True
            state.pause_reason = (
                f"daily loss cap hit ({state.realised_pnl:,.0f} vs -{cap_amount:,.0f})"
            )
            alerts.append(
                f"{family.upper()} PAUSED for the session - {state.pause_reason}. "
                f"No exceptions, no 'one more trade to win it back'."
            )
        return alerts

    def is_paused(self, market: str) -> tuple[bool, str]:
        """Whether ``market``'s family is paused for the session."""
        state = self.state[self.cfg.market_family(market)]
        return state.paused, state.pause_reason

    def headroom(self, market: str) -> dict[str, float]:
        """Remaining daily loss headroom, for the dashboard and alerts."""
        family = self.cfg.market_family(market)
        state = self.state[family]
        cap_amount = self.capital * self.cfg.daily_loss_cap(market)
        return {
            "realised_pnl": state.realised_pnl,
            "daily_cap": cap_amount,
            "remaining": max(0.0, cap_amount + state.realised_pnl),
            "consecutive_losses": float(state.consecutive_losses),
        }

    # -- convenience ---------------------------------------------------------

    def size(self, plan: TradePlan, market: str, vol_factor: float,
             option_leg: OptionLeg | None = None,
             futures_leg: FuturesLeg | None = None) -> SizedPosition:
        """Dispatch to the option or futures sizing routine."""
        try:
            if option_leg is not None:
                return self.size_option(plan, option_leg, market, vol_factor)
            if futures_leg is not None:
                return self.size_futures(plan, futures_leg, market, vol_factor)
        except ConfigBlockerError as error:
            return SizedPosition(False, 0, 0.0, vol_factor, reason=str(error))
        return SizedPosition(False, 0, 0.0, vol_factor, reason="no leg supplied to size")
