"""Section 8 - psychology and discipline enforcement.

This section exists specifically to counter the operator's known tendency to exit trades
early even when the original analysis was correct. So the friction is deliberate:

1. Beast does not accept a manual close on a live position unless it has hit its stop,
   target or trailing stop.
2. Any override attempt requires an **explicit typed confirmation** acknowledging that the
   override breaks the plan.
3. Every override is logged with the trade context and what the plan would have produced.
4. Override frequency is reported back weekly, in R-multiples.
5. There is no "just this once" and no fatigue exception - the friction step applies every
   single time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from beast.constants import OVERRIDE_PHRASE, ExitReason
from beast.ops.precedence import resolve
from beast.schemas import OverrideRecord


class OverrideRefused(Exception):
    """An override attempt that did not carry the exact typed confirmation."""


@dataclass
class OverrideRequest:
    """An operator instruction that contradicts the plan (PRECEDENCE.md rank 5)."""

    action: str  # close_early | tighten_stop | add_to_loser
    trade_id: str
    reason: str = ""
    confirmation: Optional[str] = None
    requested_at: Optional[datetime] = None


class OverrideGuard:
    """The friction step, and the record of every time it was invoked."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.records: list[OverrideRecord] = []

    @property
    def count(self) -> int:
        return sum(1 for r in self.records if r.confirmed)

    def prompt(self, request: OverrideRequest) -> str:
        """The message the operator must answer before anything happens."""
        verdict = resolve("soul", "operator")
        return (
            f"Override requested: {request.action} on trade {request.trade_id}.\n"
            f"{verdict}\n"
            f"This breaks the plan. Type exactly: {OVERRIDE_PHRASE}"
        )

    def authorise(
        self,
        request: OverrideRequest,
        position,
        now: datetime,
        hypothetical_r: Optional[float] = None,
    ) -> OverrideRecord:
        """Accept or refuse an override, logging the attempt either way.

        A refusal is still a record: Section 8's point is to make the pattern visible, and
        an attempt that was talked out of itself is part of that pattern.
        """
        context = {
            "market": position.signal.market.value,
            "direction": position.direction.value,
            "entry_price": position.entry_price,
            "stop_price": position.stop_price,
            "target_price": position.target_price,
            "current_r": round(position.mfe_r, 2),
            "bars_held": position.bars_held,
            "trail_activated": position.trail_activated,
            "requested_action": request.action,
            "operator_reason": request.reason,
        }
        confirmed = (request.confirmation or "").strip() == OVERRIDE_PHRASE
        record = OverrideRecord(
            timestamp=now,
            trade_id=request.trade_id,
            action=request.action,
            trade_context=context,
            confirmed=confirmed,
            hypothetical_r_if_held_to_target=hypothetical_r,
        )
        self.records.append(record)
        if not confirmed:
            raise OverrideRefused(self.prompt(request))
        return record

    def exit_reason(self) -> ExitReason:
        """An authorised override closes the trade as ``OVERRIDE`` - a logged deviation."""
        return ExitReason.OVERRIDE

    def weekly_report(self) -> dict:
        """Section 8.4 - how many overrides, and what they cost or saved in R.

        "This turns the psychology problem into visible data rather than a recurring blind
        spot."
        """
        confirmed = [r for r in self.records if r.confirmed]
        cost = sum(
            r.hypothetical_r_if_held_to_target
            for r in confirmed
            if r.hypothetical_r_if_held_to_target is not None
        )
        return {
            "attempts": len(self.records),
            "confirmed": len(confirmed),
            "refused": len(self.records) - len(confirmed),
            "hypothetical_r_forgone": round(cost, 2),
            "by_action": {
                action: sum(1 for r in confirmed if r.action == action)
                for action in {r.action for r in confirmed}
            },
        }
