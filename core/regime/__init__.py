"""The volatility-regime layer.

The HMM here is a **volatility classifier**. It answers one question - is this
market calm, normal or turbulent - and it does not predict direction, generate
signals or pick setups. Beast already has a regime classifier: soul file 4.4,
rule-based, ADX/DI/BB on the bias timeframe, outputting
``TREND_UP | TREND_DOWN | RANGE``, which governs which setups are permitted.
This layer does not replace it, modify it, or vote on it.

Two concepts, two names, everywhere::

    regime     -> 4.4's TREND_UP | TREND_DOWN | RANGE
    vol_state  -> CALM | NORMAL | TURBULENT | UNKNOWN

The layer's entire surface area to the rest of Beast is
:class:`~core.regime.contracts.VolState`, plus the two functions in
:mod:`core.regime.stability` that gate G10 and section 7 sizing will call::

    vol_state = tracker.observe(bar_ts, posterior, engine, session_id)
    blocked, why = veto_for_signal(vol_state, is_counter_bias)
    factor, bound_by = effective_size_factor(vol_factor, vol_state)

Three invariants hold everywhere in this package, and each is asserted in code
rather than left to a docstring:

1. ``size_multiplier`` never exceeds 1.0 and combines with section 7's
   ``vol_factor`` by ``min()``, floored at ``risk.vol_factor_floor``. The layer
   may only ever shrink a position.
2. ``veto`` only blocks. Nothing here can permit a trade the G0-G9 chain
   rejected.
3. Inference is **filtered** - the forward algorithm alone, ``P(state_t |
   observations_1..t)``. No ``predict()``, no ``predict_proba()``, no smoothing.

The layer is not yet wired into the entry pipeline. That is Phase 2: gate G10
after G9, and ``effective_size_factor`` inside G9's sizing. Until then it
computes and logs, and changes nothing.
"""

from core.regime.contracts import (
    CALM,
    NORMAL,
    PUBLIC_BUCKETS,
    TURBULENT,
    UNKNOWN,
    ModelMetadata,
    VolState,
)
from core.regime.hmm_engine import ModelUnusable, VolatilityHMM
from core.regime.stability import (
    StabilityTracker,
    effective_size_factor,
    veto_for_signal,
)
from core.regime.vol_features import (
    CausalZScoreScaler,
    compute_features,
    feature_columns,
    feature_hash,
)

__all__ = [
    "CALM",
    "NORMAL",
    "TURBULENT",
    "UNKNOWN",
    "PUBLIC_BUCKETS",
    "VolState",
    "ModelMetadata",
    "VolatilityHMM",
    "ModelUnusable",
    "StabilityTracker",
    "effective_size_factor",
    "veto_for_signal",
    "CausalZScoreScaler",
    "compute_features",
    "feature_columns",
    "feature_hash",
]
