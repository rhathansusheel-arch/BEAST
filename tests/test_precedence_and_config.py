"""The Soul File governs, and unset blockers fail closed.

These are the two structural guarantees the rest of the suite leans on: PRECEDENCE.md's
authority order, and Appendix A's config discipline.
"""

from __future__ import annotations

import pytest

from beast.config import Config, ConfigUnset
from beast.constants import Market
from beast.ops import precedence


def test_soul_file_is_present_and_hashable():
    assert precedence.SOUL_PATH.exists()
    assert precedence.soul_version().startswith("3.0")
    assert len(precedence.soul_sha256()) == 64


def test_soul_outranks_every_other_instruction_source():
    for other in ("config", "code", "operator", "legacy"):
        assert precedence.resolve("soul", other).winner == "soul"
        assert precedence.resolve(other, "soul").winner == "soul"


def test_immutable_rules_outrank_the_rest_of_the_soul_file():
    assert precedence.resolve("soul.immutable", "soul").winner == "soul.immutable"


def test_operator_loses_to_config_and_code():
    assert precedence.resolve("operator", "config").winner == "config"
    assert precedence.resolve("operator", "code").winner == "code"


def test_unset_blocker_raises_rather_than_defaulting(cfg):
    with pytest.raises(ConfigUnset):
        cfg.require("options.min_oi")
    with pytest.raises(ConfigUnset):
        cfg.capital()


def test_blockers_are_reported_per_market(cfg):
    nifty = cfg.unresolved_blockers(Market.NIFTY)
    gold = cfg.unresolved_blockers(Market.GOLD)
    assert any("options.min_oi" in item for item in nifty)
    assert not any("instruments.gold" in item for item in nifty)
    assert any("instruments.gold.contract_multiplier" in item for item in gold)


def test_populated_config_reports_no_blockers(ready_cfg):
    assert ready_cfg.unresolved_blockers(Market.NIFTY) == []
    assert ready_cfg.unresolved_blockers(Market.GOLD) == []


def test_appendix_a_carries_the_soul_file_defaults(cfg):
    assert cfg.get("entry.min_confluence") == 4
    assert cfg.get("entry.min_confluence_counter_bias") == 5
    assert cfg.get("exit.target_r") == 2.0
    assert cfg.get("exit.stop_buffer_atr") == 0.25
    assert cfg.get("exit.stop_min_atr") == 0.5
    assert cfg.get("exit.stop_max_atr") == 2.5
    assert cfg.get("options.premium_stop_pct") == 0.35
    assert cfg.get("options.max_premium_outlay_pct") == 0.10
    assert cfg.risk_per_trade(Market.NIFTY) == 0.03
    assert cfg.risk_per_trade(Market.GOLD) == 0.02
    assert cfg.daily_loss_cap(Market.NIFTY) == 0.10
    assert cfg.daily_loss_cap(Market.GOLD) == 0.05


def test_mode_is_paper(cfg):
    """Section 10 - the current phase is alert-only paper trading."""
    assert cfg.is_paper
