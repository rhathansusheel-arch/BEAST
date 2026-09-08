"""The volatility layer's entire surface area to the rest of Beast.

Two dataclasses and a label ladder. Nothing else in ``core/regime/`` is imported
by anything outside it, and nothing here imports from the entry pipeline, the
exit manager or the risk manager. That narrowness is the point: the layer has to
be provably correct in isolation before it is allowed to influence a trade.

Naming, and why it is enforced rather than documented
-----------------------------------------------------
* ``regime``    - reserved for soul file 4.4: ``TREND_UP | TREND_DOWN | RANGE``,
  rule-based, ADX/DI/BB on the bias timeframe. It governs which setups are
  permitted. The HMM does not replace it, modify it, or vote on it.
* ``vol_state`` - this layer's output: ``CALM | NORMAL | TURBULENT | UNKNOWN``.

Two concepts, two names, everywhere - variables, log fields, the Appendix B
signal object, the Appendix C trade log, the dashboard. Collapsing them would
be the single easiest way to end up with a volatility model quietly steering
direction.

Labels are volatility-sorted, never return-sorted
-------------------------------------------------
States are ordered by mean realized volatility ascending. They are deliberately
*not* labelled ``CRASH / BEAR / BULL / EUPHORIA`` by mean return, for two
reasons. Those are directional labels on a model with no directional mandate,
and once the string ``"BULL"`` exists in a log somebody will eventually read it
as bias. Return-sorted labels are also unstable across retrains: this week's
``BULL`` is not next week's, which makes any longitudinal analysis of them
meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# --- the public three-bucket API -------------------------------------------

CALM = "CALM"
NORMAL = "NORMAL"
TURBULENT = "TURBULENT"
UNKNOWN = "UNKNOWN"

#: The buckets anything downstream is allowed to branch on. The ladder below may
#: grow when BIC picks a larger model; this tuple does not.
PUBLIC_BUCKETS: tuple[str, ...] = (CALM, NORMAL, TURBULENT, UNKNOWN)

#: Volatility ladders by state count, ascending in mean realized volatility.
#: BIC picks the state count; nothing downstream should have to know which it
#: picked, so :data:`LADDER_TO_BUCKET` collapses whichever ladder was used back
#: to the three public buckets.
LADDERS: dict[int, tuple[str, ...]] = {
    3: (CALM, NORMAL, TURBULENT),
    4: (CALM, NORMAL, "ELEVATED", TURBULENT),
    5: ("VERY_CALM", CALM, NORMAL, "ELEVATED", TURBULENT),
    6: ("VERY_CALM", CALM, NORMAL, "ELEVATED", TURBULENT, "EXTREME"),
    7: ("VERY_CALM", CALM, NORMAL, "ELEVATED", TURBULENT, "EXTREME", "EXTREME_2"),
}

#: How each ladder rung collapses to a public bucket. Kept as data rather than
#: as a string prefix rule so that adding a rung is a one-line, reviewable
#: change instead of a silent reclassification of everything above it.
LADDER_TO_BUCKET: dict[str, str] = {
    "VERY_CALM": CALM,
    CALM: CALM,
    NORMAL: NORMAL,
    "ELEVATED": TURBULENT,
    TURBULENT: TURBULENT,
    "EXTREME": TURBULENT,
    "EXTREME_2": TURBULENT,
}


def ladder_for(n_states: int) -> tuple[str, ...]:
    """Return the volatility ladder for an ``n_states`` model.

    Args:
        n_states: Number of HMM states, as selected by BIC.

    Returns:
        Labels ordered by ascending mean realized volatility.

    Raises:
        ValueError: No ladder is defined for ``n_states``. Extending the ladder
            is a deliberate act - it must stay monotonic in volatility - so an
            unknown state count fails rather than falling back to integers.
    """
    if n_states not in LADDERS:
        raise ValueError(
            f"no volatility ladder defined for {n_states} states; "
            f"defined ladders: {sorted(LADDERS)}"
        )
    return LADDERS[n_states]


def collapse(label: str) -> str:
    """Collapse a ladder label to one of the three public buckets."""
    return LADDER_TO_BUCKET.get(label, UNKNOWN)


@dataclass(frozen=True)
class ModelMetadata:
    """Everything needed to decide whether a persisted model may still be used.

    Attributes:
        market: ``NIFTY`` | ``SENSEX`` | ``GOLD``.
        n_states: State count chosen by BIC.
        bic: The winning BIC value.
        bic_margin: Gap to the runner-up. A margin of about 2 is noise, not a
            decision, and is logged as such rather than presented as a verdict.
        log_likelihood: Of the selected model on its training data.
        converged: Whether EM reported convergence.
        n_iter: Iterations EM actually ran.
        train_start / train_end: Bias-TF bar timestamps bounding the fit.
        n_samples: Training rows.
        feature_list: Ordered feature column names.
        feature_hash: Digest of ``feature_list`` plus the feature-pipeline
            version. Checked at load; a mismatch refuses the model rather than
            scoring new features against old means. Silent feature drift is how
            these systems rot.
        label_map: ``{state id: ladder label}``.
        model_version: ``<feature_hash prefix>@<train_end date>``, the string
            that goes on every signal.
        trained_at: When the fit ran.
        scaler_mean / scaler_scale: The causal z-score scaler fitted on the
            training window only, persisted with the model because applying a
            different scaler to the same model is the same bug as changing the
            features.
    """

    market: str
    n_states: int
    bic: float
    bic_margin: float
    log_likelihood: float
    converged: bool
    n_iter: int
    train_start: datetime | None
    train_end: datetime | None
    n_samples: int
    feature_list: tuple[str, ...]
    feature_hash: str
    label_map: dict[int, str]
    model_version: str
    trained_at: datetime
    scaler_mean: tuple[float, ...] = ()
    scaler_scale: tuple[float, ...] = ()

    @property
    def bic_margin_is_noise(self) -> bool:
        """True when the top two candidates are within ~2 BIC of each other.

        At that distance the state count was not really chosen by the data. The
        model is still usable; the log should just not claim a verdict it does
        not have.
        """
        return abs(self.bic_margin) < 2.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "market": self.market,
            "n_states": self.n_states,
            "bic": round(self.bic, 3),
            "bic_margin": round(self.bic_margin, 3),
            "bic_margin_is_noise": self.bic_margin_is_noise,
            "log_likelihood": round(self.log_likelihood, 3),
            "converged": self.converged,
            "n_iter": self.n_iter,
            "train_start": self.train_start.isoformat() if self.train_start else None,
            "train_end": self.train_end.isoformat() if self.train_end else None,
            "n_samples": self.n_samples,
            "feature_list": list(self.feature_list),
            "feature_hash": self.feature_hash,
            "label_map": {int(k): v for k, v in self.label_map.items()},
            "model_version": self.model_version,
            "trained_at": self.trained_at.isoformat(),
        }


@dataclass(frozen=True)
class VolState:
    """One market's volatility read at one bias-TF bar close.

    This is the whole contract. Anything downstream that wants to know about
    volatility reads these fields and nothing else.

    Attributes:
        market: ``NIFTY`` | ``SENSEX`` | ``GOLD``.
        bar_ts: IST close of the bias-TF bar this was computed on.
        label: One of :data:`PUBLIC_BUCKETS`.
        ladder_label: The finer rung actually selected, e.g. ``ELEVATED``.
            Logged for audit; nothing branches on it.
        bucket_source_state: Raw HMM state id, for audit.
        probability: ``P(state | observations 1..t)`` - filtered, not smoothed.
        state_probabilities: The full filtered posterior.
        is_confirmed: Survived the N-bar persistence check.
        consecutive_bars: How long the current state has held, counting across
            the session boundary when ``carry_across_sessions`` is true.
        flicker_rate: State changes per bar over the flicker window.
        is_flickering: Change count over the window exceeded the threshold.
        size_multiplier: In ``(0, 1.0]``. Combined with section 7's
            ``vol_factor`` by ``min()``, never by product.
        veto: True only ever *blocks*. It can never permit a trade that the
            G0-G9 chain rejected.
        reason: Human-readable, goes into the trade log.
        vol_state_source: Always ``"underlying"``. Recorded explicitly rather
            than assumed - see the class note below.
        model_version: From :class:`ModelMetadata`.
        data_delay_minutes: 15 for Sensex, whose feed runs behind (4.6, 12).
            Its bias bars are therefore effectively one bar stale.
        is_stale: True while holding a previous state through an inference
            failure rather than computing a fresh one.

    On ``vol_state_source``
    -----------------------
    The HMM trains and infers on the **underlying** series for all three markets
    - Nifty 50 spot, Sensex spot, XAUUSD spot. An option premium series carries
    theta decay and IV shifts, so it falls while the index is flat; a volatility
    model fitted to that learns decay, not market state.

    Under the operator's path-B resolution of Conflict 1, analysis is on the
    underlying everywhere, so this matches the instrument being risked and the
    field is a statement of fact. It is still recorded on every signal, because
    if that resolution is ever revisited the field silently becomes a modelling
    assumption - vol state computed on a different instrument from the one at
    risk - and an assumption that was already being logged is one somebody can
    find.
    """

    market: str
    bar_ts: datetime
    label: str
    bucket_source_state: int
    probability: float
    state_probabilities: dict[int, float]
    is_confirmed: bool
    consecutive_bars: int
    flicker_rate: float
    is_flickering: bool
    size_multiplier: float
    veto: bool
    reason: str
    model_version: str
    ladder_label: str = UNKNOWN
    vol_state_source: str = "underlying"
    data_delay_minutes: int = 0
    is_stale: bool = False
    state_probabilities_raw: dict[int, float] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Assert the invariants in code, not just in the docstring.

        Raises:
            ValueError: ``size_multiplier`` outside ``(0, 1.0]``, or a label
                outside the public buckets. A multiplier above 1.0 would let a
                statistical model increase risk beyond the section 7 number,
                which is Immutable Rule 1 territory; it must be impossible to
                construct, not merely discouraged.
        """
        if not 0.0 < self.size_multiplier <= 1.0:
            raise ValueError(
                f"VolState.size_multiplier must be in (0, 1.0], got "
                f"{self.size_multiplier!r}. The volatility layer may only "
                f"shrink a position, never enlarge one."
            )
        if self.label not in PUBLIC_BUCKETS:
            raise ValueError(
                f"VolState.label must be one of {PUBLIC_BUCKETS}, got {self.label!r}"
            )
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(
                f"VolState.probability must be in [0, 1], got {self.probability!r}"
            )

    @property
    def is_usable(self) -> bool:
        """True when this read is confirmed, not flickering and not stale."""
        return self.is_confirmed and not self.is_flickering and not self.is_stale

    def to_dict(self) -> dict[str, Any]:
        """Flat record for the signal object, the trade log and the dashboard."""
        return {
            "market": self.market,
            "bar_ts": self.bar_ts.isoformat() if self.bar_ts else None,
            "vol_state": self.label,
            "vol_state_ladder": self.ladder_label,
            "vol_state_raw": self.bucket_source_state,
            "vol_state_probability": round(self.probability, 4),
            "vol_state_confirmed": self.is_confirmed,
            "vol_state_consecutive_bars": self.consecutive_bars,
            "vol_state_flicker_rate": round(self.flicker_rate, 4),
            "vol_state_flickering": self.is_flickering,
            "vol_state_stale": self.is_stale,
            "size_multiplier": round(self.size_multiplier, 4),
            "veto": self.veto,
            "reason": self.reason,
            "vol_state_source": self.vol_state_source,
            "model_version": self.model_version,
            "data_delay_minutes": self.data_delay_minutes,
        }


def unknown_state(market: str, bar_ts: datetime, reason: str,
                  size_multiplier: float, model_version: str = "none",
                  data_delay_minutes: int = 0, veto: bool = False) -> VolState:
    """Build the fail-safe ``UNKNOWN`` state.

    Used on model-load failure, NaN features, inference exception, or once the
    stale-hold budget is spent. A model bug must not silently halt Beast, and
    must not silently let it trade at full size either - hence a reduced
    multiplier and ``veto=False`` by default.
    """
    return VolState(
        market=market,
        bar_ts=bar_ts,
        label=UNKNOWN,
        ladder_label=UNKNOWN,
        bucket_source_state=-1,
        probability=0.0,
        state_probabilities={},
        is_confirmed=False,
        consecutive_bars=0,
        flicker_rate=0.0,
        is_flickering=False,
        size_multiplier=size_multiplier,
        veto=veto,
        reason=reason,
        model_version=model_version,
        data_delay_minutes=data_delay_minutes,
        is_stale=False,
    )
