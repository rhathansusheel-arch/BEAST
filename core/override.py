"""Override and discipline enforcement - soul file section 8.

    This section exists specifically to counter the operator's known tendency to
    exit trades early even when the original analysis was correct.

The friction step is the whole mechanism. Beast will not accept a manual close on
a live position unless it has hit its stop, target or trailing stop. Any other
close request must carry an **explicit typed confirmation** acknowledging that the
override breaks the plan - and that requirement applies every time, with no
fatigue exception and no "just this once".

Every override is logged with the trade context at the time and the outcome the
original plan would have produced, so section 9 can report back weekly what the
overrides cost or saved in R.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from core.config import Config, get_config
from core.exit_manager import ExitManager, ManagedPosition
from core.schemas import ExitReason, OverrideRecord


class OverrideAction(str, Enum):
    """The three things an operator might try to do against the plan."""

    CLOSE_EARLY = "close_early"
    TIGHTEN_STOP = "tighten_stop"
    ADD_TO_LOSER = "add_to_loser"

    @property
    def confirmation_phrase(self) -> str:
        """The exact text the operator must type.

        Deliberately awkward to type. Friction is the feature - a one-key
        confirmation would be indistinguishable from no confirmation at all.
        """
        return {
            OverrideAction.CLOSE_EARLY: "CONFIRM OVERRIDE: closing against plan",
            OverrideAction.TIGHTEN_STOP: "CONFIRM OVERRIDE: tightening stop against plan",
            OverrideAction.ADD_TO_LOSER: "CONFIRM OVERRIDE: adding to a loser",
        }[self]


@dataclass
class OverrideRequest:
    """An operator's attempt to act against the plan."""

    action: OverrideAction
    market: str
    requested_at: datetime
    confirmation_text: str = ""
    new_stop: float | None = None       # for TIGHTEN_STOP
    note: str = ""


@dataclass
class OverrideVerdict:
    """Beast's answer to an override request."""

    accepted: bool
    message: str
    record: OverrideRecord | None = None
    requires_confirmation: bool = False
    expected_phrase: str = ""


class OverrideGuard:
    """Enforces the section 8 friction step and logs every attempt.

    Args:
        config: Injected for tests.

    Attributes:
        attempts: Every request seen this session, accepted or not. The count is
            what turns the psychology problem into visible data.
    """

    def __init__(self, config: Config | None = None) -> None:
        self.cfg = config or get_config()
        self.attempts: list[tuple[OverrideRequest, OverrideVerdict]] = []
        self.accepted: list[OverrideRecord] = []

    # -- the friction step ---------------------------------------------------

    def request(self, request: OverrideRequest, position: ManagedPosition,
                exit_manager: ExitManager) -> OverrideVerdict:
        """Evaluate an override request.

        Returns:
            An :class:`OverrideVerdict`. When ``requires_confirmation`` is True,
            the operator must resubmit the same request with
            ``confirmation_text`` set to ``expected_phrase`` exactly. Anything
            else - a partial match, a lowercase variant, a "y" - is refused.
        """
        verdict = self._evaluate(request, position, exit_manager)
        self.attempts.append((request, verdict))
        if verdict.accepted and verdict.record is not None:
            self.accepted.append(verdict.record)
        return verdict

    def _evaluate(self, request: OverrideRequest, position: ManagedPosition,
                  exit_manager: ExitManager) -> OverrideVerdict:
        """Apply the section 8 hard rules."""
        expected = request.action.confirmation_phrase

        # Rule 1: never widen a stop after entry (also rule 13.7). This is not an
        # override that friction can unlock - it is simply refused.
        if request.action is OverrideAction.TIGHTEN_STOP and request.new_stop is not None:
            widens = (
                request.new_stop < position.current_stop
                if position.direction.sign > 0
                else request.new_stop > position.current_stop
            )
            if widens:
                return OverrideVerdict(
                    False,
                    (
                        f"Refused. A stop is never widened after entry (rule 13.7). "
                        f"Current stop {position.current_stop:.2f}, requested "
                        f"{request.new_stop:.2f}."
                    ),
                )

        # Rule: never add to a position mid-trade (7.1, rule 13). Also not
        # unlockable - it would break the sizing the trade was entered on.
        if request.action is OverrideAction.ADD_TO_LOSER:
            return OverrideVerdict(
                False,
                (
                    "Refused. Beast never adds to a position mid-trade under any "
                    "circumstance (7.1). Position size is fixed at entry."
                ),
            )

        if request.confirmation_text.strip() != expected:
            return OverrideVerdict(
                False,
                (
                    f"Override requires explicit confirmation. Type exactly:\n"
                    f"  {expected}\n"
                    f"This breaks the plan you entered on. Current position: "
                    f"{self._describe(position)}"
                ),
                requires_confirmation=True,
                expected_phrase=expected,
            )

        record = OverrideRecord(
            timestamp=request.requested_at,
            action=request.action.value,
            confirmation_text=request.confirmation_text.strip(),
            context=self._context(position),
            hypothetical_r_if_held_to_target=exit_manager.hypothetical_r_if_held(position),
        )
        return OverrideVerdict(
            True,
            (
                f"Override accepted and logged as a rule deviation. Had the plan been "
                f"followed to target it would have returned "
                f"{record.hypothetical_r_if_held_to_target:+.2f}R."
            ),
            record=record,
        )

    # -- context and reporting ------------------------------------------------

    @staticmethod
    def _describe(position: ManagedPosition) -> str:
        """One-line position summary for the confirmation prompt."""
        return (
            f"{position.signal.market} {position.direction.value} from "
            f"{position.entry_underlying:g}, stop {position.current_stop:g}, "
            f"target {position.plan.target_price:g}, "
            f"currently {position.r_at(position.last_underlying):+.2f}R"
        )

    @staticmethod
    def _context(position: ManagedPosition) -> dict[str, Any]:
        """Full trade context at the moment of the override (section 8, point 3)."""
        return {
            "signal_id": position.signal.signal_id,
            "market": position.signal.market,
            "direction": position.direction.value,
            "setup_type": int(position.signal.setup_type),
            "entry_underlying": position.entry_underlying,
            "entry_premium": position.entry_premium,
            "current_underlying": position.last_underlying,
            "current_premium": position.last_premium,
            "current_stop": position.current_stop,
            "target": position.plan.target_price,
            "unrealised_r": round(position.r_at(position.last_underlying), 4),
            "mae_r": round(position.mae_r, 4),
            "mfe_r": round(position.mfe_r, 4),
            "bars_held": position.bars_held,
            "trail_activated": position.trail_activated,
        }

    def permitted_exit(self, reason: ExitReason) -> bool:
        """Whether ``reason`` is one of the exits that needs no override (6.6)."""
        return reason in (
            ExitReason.SL,
            ExitReason.TP,
            ExitReason.TRAIL,
            ExitReason.SESSION,
            ExitReason.TIME,
            ExitReason.PREMIUM_STOP,
        )

    def summary(self) -> dict[str, Any]:
        """Override statistics for the weekly report (section 8, point 4)."""
        accepted = self.accepted
        costs = [
            record.hypothetical_r_if_held_to_target
            for record in accepted
            if record.hypothetical_r_if_held_to_target is not None
        ]
        return {
            "attempts": len(self.attempts),
            "accepted": len(accepted),
            "refused": len(self.attempts) - len(accepted),
            "hypothetical_r_forgone": round(sum(costs), 4) if costs else 0.0,
            "by_action": {
                action.value: sum(
                    1 for request, _ in self.attempts if request.action is action
                )
                for action in OverrideAction
            },
        }

    def format_summary(self) -> str:
        """Render the override report - visible data, not a lecture."""
        data = self.summary()
        if data["attempts"] == 0:
            return "OVERRIDES: none this period. Plan followed as written."
        return (
            f"OVERRIDES: {data['attempts']} attempted, {data['accepted']} accepted, "
            f"{data['refused']} refused. Had every overridden trade been held to "
            f"target it would have returned {data['hypothetical_r_forgone']:+.2f}R."
        )
