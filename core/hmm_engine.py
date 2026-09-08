"""Regime detection.

This module holds two regime classifiers, and the distinction between them is
deliberate and load-bearing:

* :class:`RuleRegimeClassifier` implements soul file 4.4 exactly - ADX, DI and
  the Bollinger basis on the bias timeframe, producing ``TREND_UP``,
  ``TREND_DOWN`` or ``RANGE`` and the set of setups each permits. **This is the
  authoritative classifier.** Gate G3 consults this and nothing else by default.

* :class:`HMMRegimeEngine` fits a Gaussian hidden Markov model over volatility
  and momentum features and reports a latent volatility state. It is a *context*
  layer: recorded on every signal, available to section 9's learning loop, and
  displayed on the dashboard - but it does not gate a trade unless the operator
  explicitly sets ``hmm.as_gate: true``.

The reason for that separation is soul file 13 rule 9 and section 9's bound on
learning: "Beast does not invent new setup types on its own. Learning is
confined to *how strictly* it applies the setups already defined in Section 5."
A statistical model silently vetoing trades would be exactly the unconstrained
learning the document rules out. Turning ``hmm.as_gate`` on is therefore a
conscious operator decision with its own line in the config, not a default.
"""

from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from core.config import Config, get_config
from core.schemas import Direction, Regime, SetupType

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:  # pragma: no cover - the rule classifier works without it
    GaussianHMM = None  # type: ignore[assignment]
    HMM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Soul file 4.4 - the authoritative classifier
# ---------------------------------------------------------------------------


@dataclass
class RegimeState:
    """The bias-timeframe verdict for one evaluation cycle.

    Attributes:
        regime: The classified regime.
        adx: ADX on the bias timeframe, recorded for the signal.
        plus_di / minus_di: The DI pair.
        close_vs_basis: Signed distance from the Bollinger basis, in points.
        detail: Human-readable explanation, used in the reason line.
    """

    regime: Regime
    adx: float
    plus_di: float
    minus_di: float
    close_vs_basis: float
    detail: str = ""


class RuleRegimeClassifier:
    """Soul file 4.4 regime classifier, evaluated on each bias-TF close."""

    def __init__(self, config: Config | None = None) -> None:
        self.cfg = config or get_config()

    def classify(self, bias_df: pd.DataFrame) -> RegimeState:
        """Classify the market into exactly one regime.

        Args:
            bias_df: Bias-timeframe frame with indicators attached, trimmed to
                closed candles.

        Returns:
            A :class:`RegimeState`. When the bias frame is too short or an
            indicator is still warming up, the answer is ``RANGE`` - the most
            restrictive regime - rather than a guess.
        """
        threshold = float(self.cfg.get("indicators.adx_trend_threshold"))

        if bias_df.empty:
            return RegimeState(Regime.RANGE, 0.0, 0.0, 0.0, 0.0, "no bias-TF data")

        row = bias_df.iloc[-1]
        required = ("adx", "plus_di", "minus_di", "bb_mid", "close")
        if any(pd.isna(row.get(col)) for col in required):
            return RegimeState(Regime.RANGE, 0.0, 0.0, 0.0, 0.0, "bias-TF indicators warming up")

        adx_value = float(row["adx"])
        plus_di = float(row["plus_di"])
        minus_di = float(row["minus_di"])
        close_vs_basis = float(row["close"]) - float(row["bb_mid"])

        if adx_value < threshold:
            return RegimeState(
                Regime.RANGE, adx_value, plus_di, minus_di, close_vs_basis,
                f"ADX {adx_value:.1f} < {threshold:.0f}",
            )

        if plus_di > minus_di and close_vs_basis > 0:
            return RegimeState(
                Regime.TREND_UP, adx_value, plus_di, minus_di, close_vs_basis,
                f"ADX {adx_value:.1f}, +DI > -DI, close above BB basis",
            )
        if minus_di > plus_di and close_vs_basis < 0:
            return RegimeState(
                Regime.TREND_DOWN, adx_value, plus_di, minus_di, close_vs_basis,
                f"ADX {adx_value:.1f}, -DI > +DI, close below BB basis",
            )

        # ADX is elevated but the DI and BB conditions disagree - 4.4 sends this
        # to RANGE rather than picking the stronger of the two signals.
        return RegimeState(
            Regime.RANGE, adx_value, plus_di, minus_di, close_vs_basis,
            "DI/BB conditions disagree",
        )

    def permitted_setups(self, regime: Regime) -> dict[Direction, set[int]]:
        """Return which setup types each direction may use under ``regime``.

        This is the 4.4 permission table, read literally:

        * ``TREND_UP``   - Long: 1, 3, 4. Short: 2 only (counter-trend reversal).
        * ``TREND_DOWN`` - Short: 1, 3, 4. Long: 2 only.
        * ``RANGE``      - setups 2 and 3 only, both directions; 1 and 4 are
          suppressed (configurable via ``entry.range_regime_allowed_setups``).
        """
        if regime is Regime.TREND_UP:
            return {Direction.LONG: {1, 3, 4}, Direction.SHORT: {2}}
        if regime is Regime.TREND_DOWN:
            return {Direction.SHORT: {1, 3, 4}, Direction.LONG: {2}}
        allowed = set(int(item) for item in self.cfg.get("entry.range_regime_allowed_setups"))
        return {Direction.LONG: set(allowed), Direction.SHORT: set(allowed)}

    def is_permitted(self, regime: Regime, setup_type: SetupType,
                     direction: Direction) -> bool:
        """True when ``setup_type`` may be taken in ``direction`` under ``regime``."""
        return int(setup_type) in self.permitted_setups(regime).get(direction, set())

    @staticmethod
    def is_counter_bias(regime: Regime, direction: Direction) -> bool:
        """True when ``direction`` opposes the prevailing bias-TF regime.

        Counter-bias reversals require 5-of-6 instead of 4-of-6, and the level
        being reversed at must be Tier A (soul file 4.4). ``RANGE`` has no
        prevailing direction, so nothing taken in it is counter-bias.
        """
        if regime is Regime.TREND_UP:
            return direction is Direction.SHORT
        if regime is Regime.TREND_DOWN:
            return direction is Direction.LONG
        return False


# ---------------------------------------------------------------------------
# HMM overlay - context only
# ---------------------------------------------------------------------------


@dataclass
class HMMState:
    """The HMM overlay's current read.

    Attributes:
        state: Index of the most likely latent state, or ``None`` when unusable.
        label: Interpretable label, e.g. ``"low_vol_up"``.
        confidence: Posterior probability of ``state`` on the latest bar.
        n_states: How many states the selected model has.
        stable: False while a new state has not yet persisted ``stability_bars``.
        flickering: True when the state changed more than ``flicker_threshold``
            times inside ``flicker_window`` bars.
        usable: True only when fitted, confident, stable and not flickering.
        detail: Why it is or is not usable.
    """

    state: int | None = None
    label: str = "UNKNOWN"
    confidence: float = 0.0
    n_states: int = 0
    stable: bool = False
    flickering: bool = False
    usable: bool = False
    detail: str = "not fitted"

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "n_states": self.n_states,
            "stable": self.stable,
            "flickering": self.flickering,
            "usable": self.usable,
            "detail": self.detail,
        }


@dataclass
class _StateHistory:
    """Rolling record of accepted states, for stability and flicker tests."""

    window: int
    values: deque = field(default_factory=deque)

    def push(self, state: int) -> None:
        self.values.append(state)
        while len(self.values) > self.window:
            self.values.popleft()

    def changes(self) -> int:
        return sum(
            1 for prev, curr in zip(self.values, list(self.values)[1:]) if prev != curr
        )

    def run_length(self) -> int:
        """How many consecutive bars the latest state has held."""
        if not self.values:
            return 0
        latest = self.values[-1]
        count = 0
        for value in reversed(self.values):
            if value != latest:
                break
            count += 1
        return count


class HMMRegimeEngine:
    """Gaussian HMM over volatility/momentum features - context only.

    Model selection is by BIC across ``hmm.n_candidates`` state counts, each fit
    ``hmm.n_init`` times from different seeds. BIC rather than log-likelihood
    because likelihood always improves with more states, and a seven-state model
    of a five-state market produces exactly the flickering the stability and
    flicker guards below exist to catch.

    Args:
        config: Injected for tests.

    Raises:
        RuntimeError: Only from :meth:`fit`, and only when ``hmm.enabled`` is
            true while ``hmmlearn`` is not installed.
    """

    def __init__(self, config: Config | None = None) -> None:
        self.cfg = config or get_config()
        section = self.cfg.section("hmm")
        self.enabled = bool(section.get("enabled", True))
        self.as_gate = bool(section.get("as_gate", False))
        self.n_candidates: Sequence[int] = list(section.get("n_candidates", [3, 4, 5]))
        self.n_init = int(section.get("n_init", 10))
        self.covariance_type = str(section.get("covariance_type", "full"))
        self.min_train_bars = int(section.get("min_train_bars", 252))
        self.stability_bars = int(section.get("stability_bars", 3))
        self.min_confidence = float(section.get("min_confidence", 0.55))
        self.refit_every_bars = int(section.get("refit_every_bars", 126))
        self.random_state = int(section.get("random_state", 42))

        self.model: Any = None
        self.feature_columns: list[str] = []
        self.state_labels: dict[int, str] = {}
        self._history = _StateHistory(int(section.get("flicker_window", 20)))
        self._flicker_threshold = int(section.get("flicker_threshold", 4))
        self._bars_since_fit = 0
        self._scaler_mean: np.ndarray | None = None
        self._scaler_std: np.ndarray | None = None

    # -- fitting -------------------------------------------------------------

    def fit(self, features: pd.DataFrame) -> bool:
        """Fit the model, selecting the state count by BIC.

        Args:
            features: Output of ``data.feature_engineering.hmm_features``.

        Returns:
            True when a model was fitted. False when there is not enough
            history - which is not an error, just an instruction to keep using
            the rule classifier alone until enough bars accumulate.

        Raises:
            RuntimeError: ``hmmlearn`` is missing while the overlay is enabled.
        """
        if not self.enabled:
            return False
        if not HMM_AVAILABLE:
            raise RuntimeError(
                "hmm.enabled is true but hmmlearn is not installed. "
                "Install it, or set hmm.enabled: false in config/beast_config.yaml."
            )
        if len(features) < self.min_train_bars:
            return False

        self.feature_columns = list(features.columns)
        raw = features.to_numpy(dtype=float)

        # Standardise: the features differ by orders of magnitude (log returns
        # near 1e-4, ADX near 0.3) and a full covariance fit on raw scales is
        # numerically miserable.
        self._scaler_mean = raw.mean(axis=0)
        self._scaler_std = raw.std(axis=0)
        self._scaler_std[self._scaler_std == 0] = 1.0
        observations = (raw - self._scaler_mean) / self._scaler_std

        best_model, best_bic = None, np.inf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for n_states in self.n_candidates:
                if n_states >= len(observations):
                    continue
                for seed in range(self.n_init):
                    try:
                        candidate = GaussianHMM(
                            n_components=int(n_states),
                            covariance_type=self.covariance_type,
                            n_iter=200,
                            random_state=self.random_state + seed,
                        )
                        candidate.fit(observations)
                        log_likelihood = candidate.score(observations)
                    except Exception:
                        # A degenerate restart is expected; the other seeds cover it.
                        continue
                    if not np.isfinite(log_likelihood):
                        continue
                    bic = self._bic(candidate, log_likelihood, observations.shape)
                    if bic < best_bic:
                        best_bic, best_model = bic, candidate

        if best_model is None:
            return False

        self.model = best_model
        self._bars_since_fit = 0
        self._label_states(observations)
        return True

    def _bic(self, model: Any, log_likelihood: float,
             shape: tuple[int, int]) -> float:
        """Bayesian information criterion for a fitted Gaussian HMM."""
        n_samples, n_features = shape
        n_states = model.n_components
        transitions = n_states * (n_states - 1)
        starts = n_states - 1
        means = n_states * n_features
        if self.covariance_type == "full":
            covariances = n_states * n_features * (n_features + 1) / 2
        elif self.covariance_type == "diag":
            covariances = n_states * n_features
        elif self.covariance_type == "tied":
            covariances = n_features * (n_features + 1) / 2
        else:  # spherical
            covariances = n_states
        n_params = transitions + starts + means + covariances
        return -2.0 * log_likelihood + n_params * np.log(n_samples)

    def _label_states(self, observations: np.ndarray) -> None:
        """Attach interpretable labels by ranking states on volatility and drift.

        Without this the states are arbitrary integers that change identity on
        every refit, which makes the section 9 breakdown by regime meaningless.
        Ranking by realised volatility gives a stable ordering across refits.
        """
        if self.model is None:
            return
        try:
            ret_idx = self.feature_columns.index("ret")
            vol_idx = self.feature_columns.index("atr_pct")
        except ValueError:
            self.state_labels = {i: f"state_{i}" for i in range(self.model.n_components)}
            return

        means = np.asarray(self.model.means_)
        order = np.argsort(means[:, vol_idx])
        tiers = ["low_vol", "mid_vol", "high_vol"]
        for rank, state in enumerate(order):
            tier = tiers[min(rank * len(tiers) // max(1, len(order)), len(tiers) - 1)]
            drift = means[state, ret_idx]
            direction = "up" if drift > 0.05 else ("down" if drift < -0.05 else "flat")
            self.state_labels[int(state)] = f"{tier}_{direction}"

    # -- inference -----------------------------------------------------------

    def update(self, features: pd.DataFrame) -> HMMState:
        """Score the latest bar and apply the stability and flicker guards.

        Args:
            features: Feature frame ending at the latest closed bias-TF bar.

        Returns:
            An :class:`HMMState`. ``usable`` is True only when the model is
            fitted, the posterior clears ``min_confidence``, the state has held
            for ``stability_bars``, and the series is not flickering.
        """
        if not self.enabled:
            return HMMState(detail="hmm disabled in config")
        if self.model is None or features.empty:
            return HMMState(detail="model not fitted")

        self._bars_since_fit += 1
        observations = features[self.feature_columns].to_numpy(dtype=float)
        if self._scaler_mean is not None and self._scaler_std is not None:
            observations = (observations - self._scaler_mean) / self._scaler_std

        try:
            posteriors = self.model.predict_proba(observations)
        except Exception as error:  # a shape change or a degenerate model
            return HMMState(detail=f"inference failed: {error}")

        latest = posteriors[-1]
        state = int(np.argmax(latest))
        confidence = float(latest[state])
        self._history.push(state)

        changes = self._history.changes()
        flickering = changes > self._flicker_threshold
        stable = self._history.run_length() >= self.stability_bars
        confident = confidence >= self.min_confidence

        reasons = []
        if not confident:
            reasons.append(f"confidence {confidence:.2f} < {self.min_confidence:.2f}")
        if not stable:
            reasons.append(f"held {self._history.run_length()}/{self.stability_bars} bars")
        if flickering:
            reasons.append(f"{changes} changes in {self._history.window} bars")

        return HMMState(
            state=state,
            label=self.state_labels.get(state, f"state_{state}"),
            confidence=confidence,
            n_states=int(self.model.n_components),
            stable=stable,
            flickering=flickering,
            usable=confident and stable and not flickering,
            detail="; ".join(reasons) if reasons else "usable",
        )

    def needs_refit(self) -> bool:
        """True when the periodic refit cadence has elapsed."""
        return self.model is None or self._bars_since_fit >= self.refit_every_bars

    def blocks_entry(self, state: HMMState) -> tuple[bool, str]:
        """Whether the HMM should veto an entry.

        Returns ``(False, ...)`` unless ``hmm.as_gate`` is explicitly enabled.
        Even then the veto only applies when the overlay is *confidently*
        unusable, so a cold-start model never silently stops Beast trading.
        """
        if not self.as_gate or not self.enabled:
            return False, "hmm is context-only"
        if self.model is None:
            return False, "hmm not fitted; not blocking"
        if not state.usable:
            return True, f"hmm gate: state not usable ({state.detail})"
        return False, f"hmm state {state.label} usable"
