"""Section 13 - Immutable Core Rules, enforced as runtime guards.

These eleven rules sit above config and above the operator. Nothing in ``config/beast.yaml``
can relax them and no caller can pass a flag that skips them; a violation raises
:class:`ImmutableRuleViolation` rather than being logged and allowed through.

That is the point of the section: the rules Beast is not permitted to break are the ones it
must not be *able* to break. See PRECEDENCE.md rank 1.
"""

from __future__ import annotations

from beast.constants import Direction

RULES: dict[int, str] = {
    1: "Never risk more than the defined % per trade.",
    2: "Never exceed max concurrent positions per market.",
    3: "Never trade through a hard no-trade condition (major news, high spread).",
    4: "Never close a position early without the explicit override-confirmation step.",
    5: "Always stop trading a market for the session once the daily loss limit is hit.",
    6: "Never trade outside the defined session windows per market.",
    7: "Never widen a stop-loss after entry.",
    8: "Never enter without a complete exit plan already computed.",
    9: "Never run indicators or setups on an option premium chart.",
    10: "Never sell an option. Long CE / long PE only.",
    11: "Never round a sub-1-lot position up to 1 lot to force a trade.",
}


class ImmutableRuleViolation(Exception):
    """An attempt to break a Section 13 rule."""

    def __init__(self, rule: int, detail: str = "") -> None:
        self.rule = rule
        text = RULES[rule]
        super().__init__(f"Immutable rule {rule} violated - {text}{(' ' + detail).rstrip()}")


def assert_stop_not_widened(
    direction: Direction, current_stop: float, new_stop: float, detail: str = ""
) -> float:
    """Rule 7. A stop may move toward the trade, never away from it.

    Returns the stop that should be adopted (the tighter of the two) so callers can use
    this as a ratchet; raises only when a caller explicitly hands over a wider stop as a
    replacement rather than as a candidate.
    """
    if direction is Direction.LONG:
        if new_stop < current_stop:
            raise ImmutableRuleViolation(7, f"long stop {current_stop} -> {new_stop}. {detail}")
        return new_stop
    if new_stop > current_stop:
        raise ImmutableRuleViolation(7, f"short stop {current_stop} -> {new_stop}. {detail}")
    return new_stop


def ratchet_stop(direction: Direction, current_stop: float, candidate: float) -> float:
    """Rule 7 / 6.3 ratchet: adopt a candidate stop only when it tightens."""
    if direction is Direction.LONG:
        return max(current_stop, candidate)
    return min(current_stop, candidate)


def assert_long_option_only(action: str, option_side_cfg: str) -> None:
    """Rules 10 and Section 3.1 - long CE / long PE only, no selling, no spreads."""
    if option_side_cfg != "long_only":
        raise ImmutableRuleViolation(
            10, f"instrument config says option_side={option_side_cfg!r}; it needs its own risk section."
        )
    if action.upper() not in {"BUY", "BUY_TO_OPEN", "SELL_TO_CLOSE"}:
        raise ImmutableRuleViolation(10, f"attempted option action {action!r}.")


def assert_whole_lots(lots: int, raw_lots: float) -> int:
    """Rule 11 / 7.1 cap 2 - a sub-1-lot position is rejected, never rounded up."""
    if lots < 1:
        raise ImmutableRuleViolation(
            11, f"raw size {raw_lots:.3f} lots is below one lot; reject at G9 instead."
        )
    if lots > raw_lots + 1e-9:
        raise ImmutableRuleViolation(11, f"{raw_lots:.3f} lots rounded up to {lots}.")
    return lots


def assert_exit_plan_complete(plan) -> None:
    """Rule 8 - an entry whose exit plan cannot be constructed is not an entry (Section 6)."""
    missing = [
        name
        for name in ("entry_price", "stop_price", "target_price", "trail")
        if getattr(plan, name, None) is None
    ]
    if missing:
        raise ImmutableRuleViolation(8, f"exit plan missing {', '.join(missing)}.")


def assert_analysis_source(source: str) -> None:
    """Rule 9 / Section 3.1 - the indicator engine only ever sees the underlying.

    Every analysis entry point calls this with the frame's declared source. Passing an
    option premium series is a programming error, not a runtime condition.
    """
    if source not in {"underlying", "index_spot", "xauusd"}:
        raise ImmutableRuleViolation(
            9, f"analysis frame source is {source!r}; indicators run on the underlying only."
        )


def assert_no_add_to_position(existing_size: float) -> None:
    """7.1 cap 3 / Section 13 - Beast never adds to a position mid-trade."""
    if existing_size:
        raise ImmutableRuleViolation(
            4, "position already open; Beast never adds to or averages a live trade."
        )
