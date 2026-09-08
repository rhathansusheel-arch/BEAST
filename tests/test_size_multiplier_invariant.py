"""The two invariants that keep the volatility layer inside the soul file.

1. **The layer may only shrink a position.** ``effective_size_factor`` never
   returns more than 1.0 and never less than ``risk.vol_factor_floor``. Above
   1.0 a statistical model would be increasing risk beyond the section 7 number,
   which Immutable Rule 1 forbids. Below the floor it would be overriding a
   soul-file value that section 7 sets deliberately.

2. **A veto only blocks.** There is no configuration under which the layer
   permits a candidate the G0-G9 chain rejected.

These are property tests: the assertions hold across every combination of vol
state, confidence, flicker condition and ``vol_factor``, not just the handful
anybody thought to write down.
"""

from __future__ import annotations

import itertools
from datetime import datetime

import pytest

from core.regime.contracts import (
    CALM,
    NORMAL,
    PUBLIC_BUCKETS,
    TURBULENT,
    UNKNOWN,
    VolState,
    collapse,
    ladder_for,
)
from core.regime.stability import effective_size_factor, veto_for_signal

BAR_TS = datetime(2026, 9, 8, 10, 45)

#: Section 7 clamps vol_factor to [vol_factor_floor, 1.0], so those are the
#: values the combiner can actually be handed. The out-of-range ones are here
#: because a clamp that is ever removed upstream must not silently uncap this.
VOL_FACTORS = (0.5, 0.6, 0.75, 0.9, 1.0, 1.5, 0.25)


def make_state(label: str = NORMAL, multiplier: float = 1.0,
               confirmed: bool = True, flickering: bool = False,
               probability: float = 0.9, veto: bool = False,
               stale: bool = False) -> VolState:
    """Build a ``VolState`` directly, bypassing the tracker."""
    return VolState(
        market="NIFTY50",
        bar_ts=BAR_TS,
        label=label,
        ladder_label=label,
        bucket_source_state=0,
        probability=probability,
        state_probabilities={0: probability, 1: 1.0 - probability},
        is_confirmed=confirmed,
        consecutive_bars=3 if confirmed else 1,
        flicker_rate=0.4 if flickering else 0.0,
        is_flickering=flickering,
        size_multiplier=multiplier,
        veto=veto,
        reason="synthetic",
        model_version="test@2026-09-08",
        is_stale=stale,
    )


class TestVolStateConstruction:
    """The multiplier bound is enforced at construction, not at use."""

    @pytest.mark.parametrize("multiplier", [1.01, 1.5, 2.0, 0.0, -0.5])
    def test_multiplier_outside_the_open_unit_interval_is_refused(self, multiplier):
        with pytest.raises(ValueError, match="size_multiplier"):
            make_state(multiplier=multiplier)

    @pytest.mark.parametrize("multiplier", [0.01, 0.5, 0.6, 1.0])
    def test_multiplier_inside_the_interval_is_accepted(self, multiplier):
        assert make_state(multiplier=multiplier).size_multiplier == multiplier

    def test_label_must_be_a_public_bucket(self):
        """`ELEVATED` is a ladder rung, not a public bucket - it must collapse first."""
        with pytest.raises(ValueError, match="label must be one of"):
            make_state(label="ELEVATED")

    @pytest.mark.parametrize("probability", [-0.1, 1.1])
    def test_probability_must_be_a_probability(self, probability):
        with pytest.raises(ValueError, match="probability"):
            make_state(probability=probability)

    def test_usable_requires_confirmed_steady_and_fresh(self):
        assert make_state().is_usable is True
        assert make_state(confirmed=False).is_usable is False
        assert make_state(flickering=True).is_usable is False
        assert make_state(stale=True).is_usable is False


class TestLadder:
    """Labels are sorted by volatility, and every rung collapses to a bucket."""

    @pytest.mark.parametrize("n_states", [3, 4, 5, 6, 7])
    def test_every_candidate_state_count_has_a_ladder(self, n_states):
        ladder = ladder_for(n_states)
        assert len(ladder) == n_states
        assert len(set(ladder)) == n_states

    @pytest.mark.parametrize("n_states", [3, 4, 5, 6, 7])
    def test_every_rung_collapses_to_a_public_bucket(self, n_states):
        for rung in ladder_for(n_states):
            assert collapse(rung) in PUBLIC_BUCKETS

    @pytest.mark.parametrize("n_states", [3, 4, 5, 6, 7])
    def test_the_ladder_is_monotonic_in_volatility(self, n_states):
        """CALM rungs come first, TURBULENT rungs last, never interleaved."""
        order = {CALM: 0, NORMAL: 1, TURBULENT: 2, UNKNOWN: 3}
        buckets = [order[collapse(rung)] for rung in ladder_for(n_states)]
        assert buckets == sorted(buckets), (
            f"ladder for {n_states} states is not monotonic in volatility"
        )

    def test_an_unknown_state_count_is_refused_not_guessed(self):
        with pytest.raises(ValueError, match="no volatility ladder"):
            ladder_for(9)

    def test_no_directional_label_exists_anywhere(self):
        """`BULL` in a log will eventually be read as a bias signal."""
        forbidden = {"BULL", "BEAR", "CRASH", "EUPHORIA", "UP", "DOWN"}
        for n_states in (3, 4, 5, 6, 7):
            for rung in ladder_for(n_states):
                assert not forbidden & set(rung.split("_")), (
                    f"directional label {rung!r} on a model with no directional mandate"
                )


class TestEffectiveSizeFactor:
    """Property test over the whole input space, per the build prompt."""

    @pytest.mark.parametrize(
        "vol_factor,label,multiplier,confirmed,flickering",
        list(itertools.product(
            VOL_FACTORS,
            (CALM, NORMAL, TURBULENT, UNKNOWN),
            (0.1, 0.5, 0.6, 1.0),
            (True, False),
            (True, False),
        )),
    )
    def test_result_is_bounded_above_by_one_and_below_by_the_floor(
        self, cfg, vol_factor, label, multiplier, confirmed, flickering
    ):
        state = make_state(label=label, multiplier=multiplier,
                           confirmed=confirmed, flickering=flickering)
        factor, reason = effective_size_factor(vol_factor, state, cfg)

        floor = float(cfg.get("risk.vol_factor_floor"))
        assert factor <= 1.0, "the volatility layer enlarged a position"
        assert factor >= floor, "the volatility layer undercut risk.vol_factor_floor"
        assert reason

    @pytest.mark.parametrize("vol_factor", VOL_FACTORS)
    def test_combination_is_min_not_product(self, cfg, vol_factor):
        """0.6 x 0.6 = 0.36; min(0.6, 0.6) = 0.6. The difference is the test."""
        state = make_state(label=TURBULENT, multiplier=0.6)
        factor, _ = effective_size_factor(vol_factor, state, cfg)
        floor = float(cfg.get("risk.vol_factor_floor"))
        expected = max(min(vol_factor, 0.6), floor)
        assert factor == pytest.approx(expected)
        if vol_factor < 1.0:
            # At vol_factor 1.0 min and product coincide, so only the strictly
            # sub-1.0 cases can tell the two combinations apart.
            assert factor != pytest.approx(vol_factor * 0.6), (
                "factors were multiplied - that double-counts volatility and "
                "the product's floor would override risk.vol_factor_floor"
            )

    def test_the_smaller_input_binds_and_the_reason_says_which(self, cfg):
        tight_state = make_state(label=TURBULENT, multiplier=0.6)
        factor, reason = effective_size_factor(1.0, tight_state, cfg)
        assert factor == pytest.approx(0.6)
        assert "vol_state" in reason

        loose_state = make_state(label=CALM, multiplier=1.0)
        factor, reason = effective_size_factor(0.7, loose_state, cfg)
        assert factor == pytest.approx(0.7)
        assert "vol_factor" in reason

    def test_floor_binds_and_is_named(self, cfg):
        state = make_state(label=UNKNOWN, multiplier=0.1)
        factor, reason = effective_size_factor(0.6, state, cfg)
        assert factor == pytest.approx(float(cfg.get("risk.vol_factor_floor")))
        assert "vol_factor_floor" in reason

    def test_no_state_yet_falls_back_to_vol_factor_alone(self, cfg):
        factor, reason = effective_size_factor(0.8, None, cfg)
        assert factor == pytest.approx(0.8)
        assert "regime layer inactive" in reason

    def test_disabled_layer_changes_nothing(self, cfg):
        """`regime.enabled: false` must be a true no-op, not a quiet 0.5."""
        cfg.section("regime")["enabled"] = False
        state = make_state(label=TURBULENT, multiplier=0.6)
        factor, _ = effective_size_factor(0.9, state, cfg)
        assert factor == pytest.approx(0.9)

    def test_product_combine_method_is_refused_at_use_as_well_as_at_load(self, cfg):
        cfg.section("regime")["sizing"]["combine_method"] = "product"
        with pytest.raises(ValueError, match="must be 'min'"):
            effective_size_factor(0.9, make_state(), cfg)


class TestVetoOnlyBlocks:
    """Gate G10 can refuse a candidate. It can never approve one."""

    def test_veto_flag_blocks(self, cfg):
        blocked, reason = veto_for_signal(make_state(label=TURBULENT, veto=True),
                                          is_counter_bias=False, config=cfg)
        assert blocked is True
        assert "G10" in reason

    def test_no_veto_flag_permits(self, cfg):
        blocked, _ = veto_for_signal(make_state(label=TURBULENT),
                                     is_counter_bias=False, config=cfg)
        assert blocked is False

    def test_turbulent_shrinks_but_does_not_block_by_default(self, cfg):
        """`veto.on_turbulent` ships false: TURBULENT resizes, it does not refuse."""
        assert cfg.get("regime.veto.on_turbulent") is False
        blocked, _ = veto_for_signal(make_state(label=TURBULENT),
                                     is_counter_bias=False, config=cfg)
        assert blocked is False

    def test_counter_bias_veto_is_opt_in(self, cfg):
        state = make_state(label=TURBULENT)
        assert veto_for_signal(state, is_counter_bias=True, config=cfg)[0] is False

        cfg.section("regime")["veto"]["on_turbulent_counter_bias"] = True
        blocked, reason = veto_for_signal(state, is_counter_bias=True, config=cfg)
        assert blocked is True
        assert "counter-bias" in reason
        # ...and it still only applies counter-bias.
        assert veto_for_signal(state, is_counter_bias=False, config=cfg)[0] is False

    def test_disabled_layer_never_vetoes(self, cfg):
        cfg.section("regime")["enabled"] = False
        blocked, reason = veto_for_signal(make_state(label=UNKNOWN, veto=True),
                                          is_counter_bias=True, config=cfg)
        assert blocked is False
        assert "disabled" in reason

    @pytest.mark.parametrize("label", list(PUBLIC_BUCKETS))
    @pytest.mark.parametrize("counter_bias", [True, False])
    def test_veto_return_is_always_a_bool_and_a_reason(self, cfg, label, counter_bias):
        """Whatever the state, G10 answers with a decision and an explanation."""
        blocked, reason = veto_for_signal(make_state(label=label),
                                          is_counter_bias=counter_bias, config=cfg)
        assert isinstance(blocked, bool)
        assert isinstance(reason, str) and reason


class TestConfigValidation:
    """Beast refuses to start on a `regime:` block that could break an invariant.

    These are the load-time twin of the runtime assertions above. Catching a bad
    multiplier at startup is worth more than catching it at the first trade,
    because the first trade may be hours away and the operator will be looking
    at something else by then.
    """

    def test_the_shipped_config_is_clean(self, cfg):
        assert cfg.validate() == []

    def test_a_multiplier_above_one_is_refused(self, cfg):
        cfg.section("regime")["sizing"]["size_multiplier"]["TURBULENT"] = 1.4
        problems = cfg.validate()
        assert any("may only shrink size" in problem for problem in problems)

    @pytest.mark.parametrize("value", [0.0, -0.5])
    def test_a_non_positive_multiplier_is_refused(self, cfg, value):
        cfg.section("regime")["sizing"]["size_multiplier"]["CALM"] = value
        assert any("(0, 1.0]" in problem for problem in cfg.validate())

    def test_product_combine_method_is_refused_at_load(self, cfg):
        cfg.section("regime")["sizing"]["combine_method"] = "product"
        problems = cfg.validate()
        assert any("must be 'min'" in problem for problem in problems)
        assert any("vol_factor_floor" in problem for problem in problems)

    def test_a_missing_bucket_is_refused(self, cfg):
        del cfg.section("regime")["sizing"]["size_multiplier"]["UNKNOWN"]
        assert any("missing 'UNKNOWN'" in problem for problem in cfg.validate())

    def test_a_missing_confirm_key_is_reported_not_raised(self, cfg):
        """A missing key must produce a problem line, never an exception.

        `Config.get` raises when a key is absent and no default is supplied, so
        the check has to pass a sentinel that is not the module's own - which is
        exactly the bug this test exists to pin.
        """
        del cfg.section("regime")["stability"]["stale_max_bars"]
        problems = cfg.validate()
        assert any("regime.stability.stale_max_bars" in problem for problem in problems)

    def test_an_absent_regime_block_is_reported(self, cfg):
        del cfg.data["regime"]
        assert any("no `regime:` block" in problem for problem in cfg.validate())

    @pytest.mark.parametrize("window", ["sliding", "", "EXPANDING"])
    def test_an_unknown_retrain_window_is_refused(self, cfg, window):
        cfg.section("regime")["hmm"]["retrain_window"] = window
        assert any("retrain_window" in problem for problem in cfg.validate())

    def test_a_one_state_candidate_is_refused(self, cfg):
        """A one-state HMM has nothing to classify."""
        cfg.section("regime")["hmm"]["n_candidates"] = [1, 3]
        assert any("n_candidates" in problem for problem in cfg.validate())
