"""Exit logic - soul file section 6, including the option layer in 6.10.

    Every trade is entered with a complete exit plan - stop, target, and trail
    parameters - computed **before** the entry signal is emitted. An entry whose
    exit plan cannot be constructed is not an entry. Nothing in this section is
    decided after the fact.

Two halves live here:

* :class:`TradePlanBuilder` constructs the plan (6.1 stop, 6.2 target, 6.3 trail
  parameters) and performs the G7 viability test. It runs *before* the signal is
  emitted, and a plan it refuses to build is a rejected trade.
* :class:`ExitManager` runs the plan on a live position: trailing ratchet,
  session flatten, the premium backstop, the theta guard, and the exit-priority
  resolution in 6.8.

Everything in 6.1-6.9 is expressed in the **underlying**. For options, 6.10
defines how an underlying-level exit is executed on a premium, and adds the one
option-specific stop that has no underlying equivalent. The premium hard stop can
only ever cut risk short - it never widens it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from core.config import Config, get_config
from core.levels import LevelEngine
from core.schemas import (
    Direction,
    ExitReason,
    Flag,
    LevelTier,
    OptionLeg,
    SetupInstance,
    Signal,
    TradePlan,
    TrailPlan,
)
from core.session import SessionClock


# ---------------------------------------------------------------------------
# Plan construction (6.1 - 6.3, and the G7 viability test)
# ---------------------------------------------------------------------------


class TradePlanBuilder:
    """Builds the complete exit plan for a candidate setup."""

    def __init__(self, config: Config | None = None) -> None:
        self.cfg = config or get_config()

    def build(self, setup: SetupInstance, entry_price: float, atr_value: float,
              levels: LevelEngine, is_expiry_day: bool = False) -> TradePlan:
        """Construct stop, target and trail, and test viability.

        Args:
            setup: The detected setup, carrying its structural stop.
            entry_price: The trigger candle's close, in underlying points.
            atr_value: Setup-TF ATR - the unit for every buffer and bound.
            levels: The level engine, for the G7 feasibility test.
            is_expiry_day: Switches the trail activation to the faster
                expiry-day value (5.7.4).

        Returns:
            A :class:`TradePlan`. When ``viable`` is False, ``reject_reason``
            states which rule refused it and the caller logs a G7 rejection.
        """
        direction = setup.direction
        sign = direction.sign

        if atr_value <= 0:
            return self._reject(direction, entry_price, "ATR unavailable on the setup timeframe")

        stop_price, stop_source, problem = self._place_stop(
            setup, entry_price, atr_value, direction
        )
        if problem:
            return self._reject(direction, entry_price, problem)

        risk_points = abs(entry_price - stop_price)
        target_r = float(self.cfg.get("exit.target_r"))
        target_price = entry_price + sign * target_r * risk_points

        feasible, detail, adjusted = self._check_target(
            entry_price, target_price, risk_points, direction, levels
        )
        if not feasible:
            return self._reject(direction, entry_price, detail, stop_price, stop_source)
        if adjusted is not None:
            target_price = adjusted
            target_r = abs(target_price - entry_price) / risk_points

        trail = self._build_trail(entry_price, risk_points, direction, is_expiry_day)

        return TradePlan(
            direction=direction,
            entry_price=float(entry_price),
            stop_price=float(stop_price),
            target_price=float(target_price),
            stop_source=stop_source,
            target_r=round(target_r, 3),
            trail=trail,
            atr=float(atr_value),
            risk_points=float(risk_points),
            viable=True,
        )

    # -- 6.1 stop placement --------------------------------------------------

    def _place_stop(self, setup: SetupInstance, entry_price: float, atr_value: float,
                    direction: Direction) -> tuple[float, str, str]:
        """Structural stop plus buffer, clamped to the min/max distance bounds.

        Returns:
            ``(stop_price, source_description, problem)``. A non-empty
            ``problem`` means the trade is rejected at G7 - Beast does not size
            down to accommodate a broken structure.
        """
        buffer_atr = float(self.cfg.get("exit.stop_buffer_atr"))
        min_atr = float(self.cfg.get("exit.stop_min_atr"))
        max_atr = float(self.cfg.get("exit.stop_max_atr"))
        sign = direction.sign

        buffered = setup.structural_stop - sign * buffer_atr * atr_value
        distance = (entry_price - buffered) * sign
        source = str(setup.detail.get("stop_source", "structure"))
        source = f"{source} + {buffer_atr:.2f}ATR"

        if distance <= 0:
            return 0.0, source, (
                "structural stop sits the wrong side of the entry - stale setup"
            )

        min_distance = min_atr * atr_value
        max_distance = max_atr * atr_value

        if distance > max_distance:
            return 0.0, source, (
                f"structural stop {distance / atr_value:.2f}ATR exceeds the "
                f"{max_atr:.2f}ATR maximum; rejected rather than sized down"
            )
        if distance < min_distance:
            # Widen to the minimum. This is the one place a stop moves away from
            # entry, and it happens before the trade exists - never after.
            buffered = entry_price - sign * min_distance
            source = f"{source} (widened to the {min_atr:.2f}ATR minimum)"

        return float(buffered), source, ""

    # -- 6.2 target and the G7 feasibility test -------------------------------

    def _check_target(self, entry_price: float, target_price: float, risk_points: float,
                      direction: Direction,
                      levels: LevelEngine) -> tuple[bool, str, float | None]:
        """Reject when a Tier A opposing level blocks the target.

        The default policy is ``reject``: Beast does not shrink its R:R to make a
        marginal setup fit. The alternative policy targets the level instead with
        a 1.5R floor, and is a one-line config change.

        For Nifty/Sensex the level pool includes OI-derived Tier A levels, so a
        heavy call wall between entry and target rejects the trade exactly as a
        price-structure resistance would (4.7.1).
        """
        blocker = levels.next_opposing_level(entry_price, direction, LevelTier.A)
        if blocker is None:
            return True, "no opposing Tier A level between entry and target", None

        # The near edge is what price meets first.
        edge = blocker.low if direction is Direction.LONG else blocker.high
        blocked = (
            edge < target_price if direction is Direction.LONG else edge > target_price
        )
        if not blocked:
            return True, f"nearest opposing level {blocker.centre:g} sits beyond the target", None

        policy = str(self.cfg.get("exit.target_infeasible_policy"))
        available_r = abs(edge - entry_price) / risk_points if risk_points > 0 else 0.0

        if policy == "reject":
            return False, (
                f"opposing Tier A level ({blocker.kind.value} at {blocker.centre:g}) "
                f"sits {available_r:.2f}R away, inside the "
                f"{self.cfg.get('exit.target_r')}R target"
            ), None

        floor = float(self.cfg.get("exit.target_level_min_r"))
        if available_r < floor:
            return False, (
                f"opposing level offers only {available_r:.2f}R, below the "
                f"{floor:.1f}R floor"
            ), None
        return True, f"targeting the opposing level at {available_r:.2f}R", float(edge)

    # -- 6.3 trail parameters -------------------------------------------------

    def _build_trail(self, entry_price: float, risk_points: float, direction: Direction,
                     is_expiry_day: bool) -> TrailPlan:
        """Compute where the trail arms and how it moves."""
        activate_r = float(self.cfg.get("exit.trail_activate_r"))
        if is_expiry_day and bool(self.cfg.get("options.expiry_day")["enabled"]):
            activate_r = float(self.cfg.get("options.expiry_day")["trail_activate_r"])
        return TrailPlan(
            activate_at=float(entry_price + direction.sign * activate_r * risk_points),
            activate_r=activate_r,
            method=str(self.cfg.get("exit.trail_method")),
            mult=float(self.cfg.get("exit.trail_atr_mult")),
        )

    @staticmethod
    def _reject(direction: Direction, entry_price: float, reason: str,
                stop_price: float = 0.0, stop_source: str = "") -> TradePlan:
        """Build a non-viable plan carrying the refusal reason."""
        return TradePlan(
            direction=direction,
            entry_price=float(entry_price),
            stop_price=float(stop_price),
            target_price=0.0,
            stop_source=stop_source,
            target_r=0.0,
            trail=TrailPlan(0.0, 0.0, "none", 0.0),
            atr=0.0,
            risk_points=0.0,
            viable=False,
            reject_reason=reason,
        )


# ---------------------------------------------------------------------------
# Live position management (6.3 - 6.10)
# ---------------------------------------------------------------------------


@dataclass
class ExitDecision:
    """The verdict for one price update."""

    should_exit: bool
    reason: ExitReason | None = None
    exit_price: float = 0.0          # underlying
    exit_premium: float | None = None
    detail: str = ""


@dataclass
class ManagedPosition:
    """A live position and everything the exit rules need to track it.

    Attributes:
        signal: The emitted signal, which carries the plan.
        entry_underlying: Fill price in underlying points.
        entry_premium: Fill premium for an option leg, else ``None``.
        current_stop: The live stop, in underlying points. Only ever moves in
            the trade's favour (6.1, 6.3).
        trail_activated: Whether the trail has armed.
        extreme_since_entry: Highest high (long) or lowest low (short) since
            entry, for the chandelier trail.
        mae_r / mfe_r: Maximum adverse and favourable excursion, in R.
        theta_flagged: Whether ``THETA_DRAG`` has been recorded (6.10).
    """

    signal: Signal
    plan: TradePlan
    entry_time: datetime
    entry_underlying: float
    entry_premium: float | None = None
    current_stop: float = 0.0
    trail_activated: bool = False
    extreme_since_entry: float = 0.0
    mae_r: float = 0.0
    mfe_r: float = 0.0
    mae_premium: float | None = None
    mfe_premium: float | None = None
    bars_held: int = 0
    theta_flagged: bool = False
    last_underlying: float = 0.0
    last_premium: float | None = None
    flags: list[Flag] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.current_stop == 0.0:
            self.current_stop = self.plan.stop_price
        if self.extreme_since_entry == 0.0:
            self.extreme_since_entry = self.entry_underlying
        self.last_underlying = self.entry_underlying
        self.last_premium = self.entry_premium
        if self.entry_premium is not None:
            self.mae_premium = self.entry_premium
            self.mfe_premium = self.entry_premium

    @property
    def direction(self) -> Direction:
        return self.plan.direction

    @property
    def option_leg(self) -> OptionLeg | None:
        return self.signal.option_leg

    def r_at(self, underlying_price: float) -> float:
        """R-multiple of ``underlying_price``, in underlying terms."""
        if self.plan.risk_points <= 0:
            return 0.0
        return (underlying_price - self.entry_underlying) * self.direction.sign / self.plan.risk_points

    def premium_r_at(self, premium: float) -> float:
        """R-multiple in premium terms - the R of record for options (6.9).

        1R in premium is the premium decline expected when the underlying
        reaches the stop, which is what the position was sized against (7.1).
        """
        leg = self.option_leg
        if leg is None or self.entry_premium is None or self.plan.risk_points <= 0:
            return 0.0
        risk_premium = self.plan.risk_points * leg.delta
        if risk_premium <= 0:
            return 0.0
        return (premium - self.entry_premium) / risk_premium


class ExitManager:
    """Runs the section 6 exit rules against live prices.

    Stops, targets and trailing stops evaluate on **live price**, not candle
    close (soul file 4.2). The trailing stop is *recomputed* on each trigger-TF
    close, but it *triggers* on live price. Those are different things and the
    split is deliberate: a stop is a stop.
    """

    def __init__(self, config: Config | None = None) -> None:
        self.cfg = config or get_config()

    # -- price updates -------------------------------------------------------

    def on_price(self, position: ManagedPosition, underlying_price: float,
                 premium: float | None, now: datetime,
                 clock: SessionClock, high: float | None = None,
                 low: float | None = None,
                 bar_open: float | None = None) -> ExitDecision:
        """Evaluate every exit trigger against a live price update.

        Args:
            position: The live position.
            underlying_price: Latest underlying price.
            premium: Latest option premium, or ``None`` for futures.
            now: Current IST time.
            clock: Session clock for the flatten sequence.
            high / low: The bar's extremes when evaluating bar-by-bar (backtest).
                When supplied, the 6.8 same-candle rule applies: the **stop** is
                assumed to have filled first unless tick data proves otherwise.
            bar_open: The bar's open, used to price a gap through the stop.

        Returns:
            An :class:`ExitDecision`. Priority follows 6.8 - session hard flat,
            then stop (including the trail, tighter governs), then premium stop,
            then target.
        """
        self._update_excursions(position, underlying_price, premium, high, low)
        self._check_theta_guard(position, underlying_price, now)

        # 6.7 step 3: hard flat closes everything at market, unconditionally.
        if clock.must_flatten(now):
            return ExitDecision(
                True, ExitReason.SESSION, underlying_price, premium,
                f"hard flat at {clock.window.hard_flat.strftime('%H:%M')}",
            )

        stop = position.current_stop
        target = position.plan.target_price
        sign = position.direction.sign

        # Bar-mode: decide with the bar's extremes, never the favourable fill.
        if high is not None and low is not None:
            stop_hit = low <= stop if sign > 0 else high >= stop
            target_hit = high >= target if sign > 0 else low <= target

            # 6.8: when stop and target both sit inside one candle, assume the
            # stop filled first unless tick data proves otherwise. Never assume
            # the favourable fill - this keeps paper statistics comparable to live.
            if stop_hit and (not target_hit or bool(self.cfg.get("exit.stop_fills_first_in_bar"))):
                # A gap through the stop exits at the gap price, and the resulting
                # R-multiple is recorded as actual (worse than -1R), not clamped.
                gapped = bar_open is not None and (
                    bar_open < stop if sign > 0 else bar_open > stop
                )
                fill = float(bar_open) if gapped else float(stop)
                reason = ExitReason.TRAIL if position.trail_activated else ExitReason.SL
                detail = (
                    f"gapped through the stop, filled at {fill:.2f}"
                    if gapped
                    else ("trailing stop triggered" if position.trail_activated else "stop-loss hit")
                )
                return ExitDecision(True, reason, fill, premium, detail)
            if target_hit:
                return ExitDecision(
                    True, ExitReason.TP, float(target), premium, "fixed R:R target hit"
                )
        else:
            stop_hit = underlying_price <= stop if sign > 0 else underlying_price >= stop
            if stop_hit:
                reason = ExitReason.TRAIL if position.trail_activated else ExitReason.SL
                return ExitDecision(
                    True, reason, float(underlying_price), premium,
                    "trailing stop triggered" if position.trail_activated else "stop-loss hit",
                )
            target_hit = (
                underlying_price >= target if sign > 0 else underlying_price <= target
            )
            if target_hit:
                return ExitDecision(
                    True, ExitReason.TP, float(underlying_price), premium, "fixed R:R target hit"
                )

        # 6.10 premium hard stop - the backstop, checked after the underlying stop
        # so it can only ever cut risk short, never widen it.
        decision = self._check_premium_stop(position, underlying_price, premium)
        if decision is not None:
            return decision

        # 6.5 time stop, off by default.
        decision = self._check_time_stop(position, underlying_price, premium)
        if decision is not None:
            return decision

        return ExitDecision(False, detail="holding")

    def _check_premium_stop(self, position: ManagedPosition, underlying_price: float,
                            premium: float | None) -> ExitDecision | None:
        """The 6.10 premium hard stop: exit at 65% of entry premium.

        This exists because the delta mapping can break. IV crush, a spread
        blowout, or a slow grind that lets theta do the damage can all destroy
        premium while the underlying sits harmlessly mid-range.
        """
        leg = position.option_leg
        if leg is None or premium is None or position.entry_premium is None:
            return None
        if premium <= leg.premium_stop:
            return ExitDecision(
                True,
                ExitReason.PREMIUM_STOP,
                float(underlying_price),
                float(premium),
                (
                    f"premium {premium:.2f} at or below the {leg.premium_stop:.2f} hard stop "
                    f"({self.cfg.get('options.premium_stop_pct'):.0%} loss) while the "
                    f"underlying stop was not reached"
                ),
            )
        return None

    def _check_time_stop(self, position: ManagedPosition, underlying_price: float,
                         premium: float | None) -> ExitDecision | None:
        """The 6.5 time stop. Disabled by default and never silently active."""
        if not bool(self.cfg.get("exit.time_stop_enabled")):
            return None
        bars = int(self.cfg.get("exit.time_stop_bars"))
        min_r = float(self.cfg.get("exit.time_stop_min_r"))
        if position.bars_held >= bars and position.r_at(underlying_price) < min_r:
            return ExitDecision(
                True, ExitReason.TIME, float(underlying_price), premium,
                f"time stop: {bars} bars held without reaching +{min_r}R",
            )
        return None

    # -- trailing ------------------------------------------------------------

    def on_trigger_close(self, position: ManagedPosition, trigger_df,
                         atr_value: float, now: datetime,
                         clock: SessionClock) -> str | None:
        """Recompute the trailing stop on a trigger-TF close (6.3).

        The trail arms at ``+1.0R`` unrealised (``+0.7R`` on expiry day), moving
        the stop to breakeven plus estimated costs in one step. From that moment
        the trade cannot produce a loss. Thereafter the stop only ever tightens -
        a wider computed value is discarded (the ratchet rule).

        Returns:
            A human-readable note when the stop moved, else ``None``.
        """
        position.bars_held += 1
        if trigger_df is None or len(trigger_df) == 0:
            return None

        row = trigger_df.iloc[-1]
        high, low, close = float(row["high"]), float(row["low"]), float(row["close"])
        sign = position.direction.sign

        position.extreme_since_entry = (
            max(position.extreme_since_entry, high)
            if sign > 0
            else min(position.extreme_since_entry, low)
        )

        note: str | None = None

        # Arm the trail.
        if not position.trail_activated:
            reached = (
                close >= position.plan.trail.activate_at
                if sign > 0
                else close <= position.plan.trail.activate_at
            )
            if reached:
                position.trail_activated = True
                cost_buffer = (
                    float(self.cfg.get("exit.breakeven_cost_buffer_r"))
                    * position.plan.risk_points
                )
                breakeven = position.entry_underlying + sign * cost_buffer
                position.current_stop = self._tighten(position, breakeven)
                note = (
                    f"trail armed at +{position.plan.trail.activate_r}R; stop to "
                    f"breakeven+costs {position.current_stop:.2f}"
                )

        if position.trail_activated:
            candidate = self._trail_price(position, trigger_df, atr_value)
            if candidate is not None:
                moved = self._tighten(position, candidate)
                if moved != position.current_stop:
                    position.current_stop = moved
                    note = f"trail tightened to {moved:.2f}"

        # 6.7 step 2: inside the flatten window the trail tightens to the most
        # recent trigger-TF swing and the fixed target stops being a hold
        # condition - the trade is now managed for exit.
        if clock.in_flatten_window(now):
            recent = trigger_df.iloc[-5:]
            swing = float(recent["low"].min()) if sign > 0 else float(recent["high"].max())
            moved = self._tighten(position, swing)
            if moved != position.current_stop:
                position.current_stop = moved
                note = f"flatten window: stop tightened to the recent swing {moved:.2f}"

        return note

    def _trail_price(self, position: ManagedPosition, trigger_df,
                     atr_value: float) -> float | None:
        """Compute the raw trailing-stop candidate for the configured method."""
        method = position.plan.trail.method
        sign = position.direction.sign

        if method == "atr_chandelier":
            offset = position.plan.trail.mult * atr_value
            return position.extreme_since_entry - sign * offset

        if method == "structure_trail":
            buffer_points = float(self.cfg.get("exit.stop_buffer_atr")) * atr_value
            window = trigger_df.iloc[-10:]
            if window.empty:
                return None
            if sign > 0:
                return float(window["low"].min()) - buffer_points
            return float(window["high"].max()) + buffer_points

        return None

    @staticmethod
    def _tighten(position: ManagedPosition, candidate: float) -> float:
        """Apply the ratchet rule: the stop only ever moves toward price."""
        if position.direction is Direction.LONG:
            return max(position.current_stop, candidate)
        return min(position.current_stop, candidate)

    # -- excursions and guards -----------------------------------------------

    def _update_excursions(self, position: ManagedPosition, underlying_price: float,
                           premium: float | None, high: float | None,
                           low: float | None) -> None:
        """Track MAE and MFE in both underlying R and premium (6.9)."""
        position.last_underlying = underlying_price
        candidates = [underlying_price]
        if high is not None:
            candidates.append(high)
        if low is not None:
            candidates.append(low)

        for price in candidates:
            r_value = position.r_at(price)
            position.mfe_r = max(position.mfe_r, r_value)
            position.mae_r = min(position.mae_r, r_value)

        if premium is not None:
            position.last_premium = premium
            position.mfe_premium = max(position.mfe_premium or premium, premium)
            position.mae_premium = min(position.mae_premium or premium, premium)

    def _check_theta_guard(self, position: ManagedPosition, underlying_price: float,
                           now: datetime) -> None:
        """Flag ``THETA_DRAG`` per 6.10. Logged only - no forced action.

        This is the data that decides whether the time stop in 6.5 should be
        switched on for options specifically.
        """
        if position.option_leg is None or position.theta_flagged:
            return
        minutes = int(self.cfg.get("options.theta_guard_minutes"))
        min_r = float(self.cfg.get("options.theta_guard_min_r"))
        if now - position.entry_time < timedelta(minutes=minutes):
            return
        if position.r_at(underlying_price) < min_r:
            position.theta_flagged = True
            position.flags.append(Flag.THETA_DRAG)

    # -- reporting -----------------------------------------------------------

    @staticmethod
    def hypothetical_r_if_held(position: ManagedPosition) -> float:
        """What the trade would have made had the plan been followed to target.

        This is what makes section 8's override reporting possible: every
        override is logged with the outcome the original plan would have had.
        """
        return position.plan.target_r
