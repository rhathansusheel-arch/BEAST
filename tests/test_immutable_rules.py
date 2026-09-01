"""Section 13 - the rules Beast must not be *able* to break."""

from __future__ import annotations

import pytest

from beast.constants import Direction
from beast.ops.immutable import (
    ImmutableRuleViolation,
    assert_analysis_source,
    assert_exit_plan_complete,
    assert_long_option_only,
    assert_stop_not_widened,
    assert_whole_lots,
    ratchet_stop,
)


def test_rule_7_stop_is_never_widened():
    with pytest.raises(ImmutableRuleViolation) as exc:
        assert_stop_not_widened(Direction.LONG, 100.0, 95.0)
    assert exc.value.rule == 7
    with pytest.raises(ImmutableRuleViolation):
        assert_stop_not_widened(Direction.SHORT, 100.0, 105.0)
    assert assert_stop_not_widened(Direction.LONG, 100.0, 102.0) == 102.0


def test_rule_7_ratchet_discards_the_wider_candidate():
    assert ratchet_stop(Direction.LONG, 100.0, 98.0) == 100.0
    assert ratchet_stop(Direction.LONG, 100.0, 103.0) == 103.0
    assert ratchet_stop(Direction.SHORT, 100.0, 103.0) == 100.0
    assert ratchet_stop(Direction.SHORT, 100.0, 97.0) == 97.0


def test_rule_8_entry_requires_a_complete_exit_plan():
    class Partial:
        entry_price = 100.0
        stop_price = 95.0
        target_price = None
        trail = None

    with pytest.raises(ImmutableRuleViolation) as exc:
        assert_exit_plan_complete(Partial())
    assert exc.value.rule == 8


def test_rule_9_indicators_never_run_on_premium():
    assert_analysis_source("underlying")
    assert_analysis_source("index_spot")
    with pytest.raises(ImmutableRuleViolation) as exc:
        assert_analysis_source("option_premium")
    assert exc.value.rule == 9


def test_rule_10_options_are_only_ever_bought():
    assert_long_option_only("BUY", "long_only")
    with pytest.raises(ImmutableRuleViolation):
        assert_long_option_only("SELL", "long_only")
    with pytest.raises(ImmutableRuleViolation):
        assert_long_option_only("BUY", "sell_permitted")


def test_rule_11_sub_one_lot_is_rejected_not_rounded_up():
    with pytest.raises(ImmutableRuleViolation) as exc:
        assert_whole_lots(0, 0.87)
    assert exc.value.rule == 11
    with pytest.raises(ImmutableRuleViolation):
        assert_whole_lots(1, 0.87)  # rounding up is the violation, not just sizing small
    assert assert_whole_lots(3, 3.4) == 3
