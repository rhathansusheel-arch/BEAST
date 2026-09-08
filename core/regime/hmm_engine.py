"""Training, model selection and **filtered** inference for the volatility layer.

The one thing to understand about this module
---------------------------------------------
It never calls ``model.predict()`` or ``model.predict_proba()``.

Both run the forward-*backward* algorithm (Viterbi, in ``predict``'s case) over
the entire sequence, which revises the estimate for bar *t* using bars after
*t*. On historical data that is look-ahead bias in its purest form: it will make
a backtest look excellent and live trading look nothing like it, because live
there are no future bars to revise with.

What is implemented instead is the **forward algorithm alone**::

    alpha_0 = startprob * emission(obs_0)
    alpha_t = (alpha_{t-1} @ transmat) * emission(obs_t)

normalised at every step, giving ``P(state_t | observations_1..t)`` - the
filtered posterior, which uses only past and present. All of it is done in log
space with ``logsumexp``; the naive product underflows to zero within a few
hundred bars and then silently reports a uniform posterior.

:meth:`VolatilityHMM.step` caches ``alpha_{t-1}`` so the live loop costs O(1)
per bar rather than re-running the whole sequence each time. :meth:`filtered`
is the batch equivalent used for backtests and for the look-ahead tests, and the
two are required to agree.

Model selection
---------------
Each candidate state count is fitted ``n_init`` times from different seeds and
scored by ``BIC = -2 * log_likelihood + n_params * log(n_samples)``; lowest
wins. BIC rather than raw likelihood, because likelihood always improves with
more states, and an over-stated model of an under-stated market produces exactly
the state flickering that :mod:`core.regime.stability` then has to suppress.

Every candidate's BIC, log-likelihood, convergence flag and iteration count is
logged, along with the winning margin. A margin of about 2 BIC between the top
two is noise, not a decision, and the log says so rather than implying the data
chose.
"""

from __future__ import annotations

import json
import logging
import pickle
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from core.config import Config, get_config
from core.regime.contracts import ModelMetadata, collapse, ladder_for
from core.regime.vol_features import (
    CausalZScoreScaler,
    assert_volume_policy,
    feature_columns,
    feature_hash,
)

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:  # pragma: no cover - the layer degrades to UNKNOWN
    GaussianHMM = None  # type: ignore[assignment]
    HMM_AVAILABLE = False

LOGGER = logging.getLogger("beast.regime.hmm")

#: The feature whose per-state mean orders the volatility ladder. Ascending
#: mean realized volatility, never mean return.
LADDER_SORT_FEATURE = "realized_vol_20"

#: Floor for a single state's log emission density.
#:
#: A state whose fitted covariance is near-singular can return ``-inf`` for an
#: observation far off its low-variance direction. That is a legitimate "this
#: state did not produce this bar", but as ``-inf`` it hard-zeroes the state for
#: the rest of the sequence: no amount of later evidence can bring back a
#: posterior that is exactly zero, because the transition term is a product.
#: Flooring keeps the arithmetic finite and the state recoverable. At -1e4 the
#: state's posterior is still indistinguishable from zero on any bar where a
#: rival state has a sane density, so nothing about the model's behaviour
#: changes - only its numerical failure mode.
EMISSION_LOG_FLOOR = -1e4


class ModelUnusable(RuntimeError):
    """A persisted model may not be used.

    Raised for a feature-hash mismatch, an expired model, or a corrupt file.
    Callers treat it as a fail-safe condition and fall back to ``UNKNOWN``
    rather than as a crash: a model problem must not halt Beast, and must not
    let it trade at full size either.
    """


@dataclass
class CandidateReport:
    """One fitted candidate, kept so the selection is auditable after the fact."""

    n_states: int
    seed: int
    bic: float
    log_likelihood: float
    converged: bool
    n_iter: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_states": self.n_states,
            "seed": self.seed,
            "bic": round(self.bic, 3),
            "log_likelihood": round(self.log_likelihood, 3),
            "converged": self.converged,
            "n_iter": self.n_iter,
        }


class VolatilityHMM:
    """One market's volatility model. Never pooled across markets.

    Nifty, Sensex and Gold each get their own. They have different sessions and
    different volatility distributions, and Sensex carries a 15-minute feed
    delay (4.6, 12) that the other two do not - pooling them would fit one set
    of states to three different processes.

    Args:
        market: ``NIFTY50`` | ``SENSEX`` | ``XAUUSD`` and friends.
        config: Injected for tests.

    Attributes:
        model: The fitted ``GaussianHMM``, or ``None``.
        metadata: :class:`ModelMetadata` for the fitted model.
        candidates: Every candidate considered in the last training run.
    """

    def __init__(self, market: str, config: Config | None = None) -> None:
        self.cfg = config or get_config()
        self.market = market.upper()
        self.model: Any = None
        self.metadata: ModelMetadata | None = None
        self.candidates: list[CandidateReport] = []
        self.scaler = CausalZScoreScaler(int(self.cfg.get("regime.features.zscore_lookback")))
        self._alpha: np.ndarray | None = None      # cached log-alpha for O(1) stepping
        self._log_transmat: np.ndarray | None = None
        self._log_startprob: np.ndarray | None = None

    # -- training ------------------------------------------------------------

    def train(self, raw_features: pd.DataFrame,
              min_train_bars: int | None = None) -> ModelMetadata:
        """Fit the model, selecting the state count by BIC.

        Args:
            raw_features: **Unstandardised** bias-TF feature matrix from
                :func:`core.regime.vol_features.compute_features`. Standardising
                happens here, causally, so that the scaler is fitted on exactly
                the training window and nothing else.
            min_train_bars: Overrides ``regime.hmm.min_train_bars`` for tests.

        Returns:
            The fitted model's :class:`ModelMetadata`.

        Raises:
            RuntimeError: ``hmmlearn`` is not installed.
            ValueError: Not enough training bars, the volume policy does not
                match, or no candidate converged.
        """
        if not HMM_AVAILABLE:
            raise RuntimeError(
                "hmmlearn is not installed. Install it, or set regime.enabled: "
                "false in config/beast_config.yaml - in which case sizing falls "
                "back to section 7's vol_factor alone."
            )
        assert_volume_policy(raw_features, self.market, self.cfg)

        required = int(
            min_train_bars if min_train_bars is not None
            else self.cfg.regime_min_train_bars(self.market)
        )
        if len(raw_features) < required:
            raise ValueError(
                f"{self.market}: {len(raw_features)} bias-TF bars is below the "
                f"{required}-bar minimum. Two years of bias-TF history is the "
                f"target: Indian 15M is ~25 bars/session, Gold 30M ~32."
            )

        scaled = self.scaler.fit_transform(raw_features)
        observations = scaled.to_numpy(dtype=float)
        columns = tuple(raw_features.columns)

        best_model, best_bic, best_report = None, np.inf, None
        runner_up_bic = np.inf
        self.candidates = []

        section = self.cfg.section("regime")["hmm"]
        base_seed = int(section.get("random_state", 42))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for n_states in section["n_candidates"]:
                n_states = int(n_states)
                if n_states >= len(observations):
                    continue
                for offset in range(int(section["n_init"])):
                    candidate = GaussianHMM(
                        n_components=n_states,
                        covariance_type=str(section["covariance_type"]),
                        n_iter=200,
                        random_state=base_seed + offset,
                    )
                    try:
                        candidate.fit(observations)
                        log_likelihood = float(candidate.score(observations))
                    except Exception as error:      # a degenerate restart
                        LOGGER.debug("%s: n=%d seed=%d failed: %s",
                                     self.market, n_states, base_seed + offset, error)
                        continue
                    if not np.isfinite(log_likelihood):
                        continue

                    bic = self._bic(candidate, log_likelihood, observations.shape)
                    report = CandidateReport(
                        n_states=n_states,
                        seed=base_seed + offset,
                        bic=bic,
                        log_likelihood=log_likelihood,
                        converged=bool(candidate.monitor_.converged),
                        n_iter=int(candidate.monitor_.iter),
                    )
                    self.candidates.append(report)

                    if bic < best_bic:
                        runner_up_bic, best_bic = best_bic, bic
                        best_model, best_report = candidate, report
                    elif bic < runner_up_bic:
                        runner_up_bic = bic

        if best_model is None or best_report is None:
            raise ValueError(
                f"{self.market}: no HMM candidate converged over "
                f"{len(observations)} bars"
            )

        self.model = best_model
        self._cache_log_parameters()
        label_map = self._label_states(columns)

        train_start = raw_features.index[0]
        train_end = raw_features.index[-1]
        digest = feature_hash(columns)
        margin = float(runner_up_bic - best_bic) if np.isfinite(runner_up_bic) else float("inf")

        self.metadata = ModelMetadata(
            market=self.market,
            n_states=int(best_model.n_components),
            bic=float(best_bic),
            bic_margin=margin,
            log_likelihood=best_report.log_likelihood,
            converged=best_report.converged,
            n_iter=best_report.n_iter,
            train_start=train_start.to_pydatetime(),
            train_end=train_end.to_pydatetime(),
            n_samples=len(observations),
            feature_list=columns,
            feature_hash=digest,
            label_map=label_map,
            model_version=f"{digest}@{train_end.date().isoformat()}",
            trained_at=datetime.now(timezone.utc),
        )
        self._log_selection()
        return self.metadata

    def _bic(self, model: Any, log_likelihood: float,
             shape: tuple[int, int]) -> float:
        """``-2 * log_likelihood + n_params * log(n_samples)``.

        Free parameters counted explicitly rather than taken from the library,
        so that changing ``covariance_type`` changes the penalty correctly
        instead of quietly comparing models on different scales.
        """
        n_samples, n_features = shape
        n_states = int(model.n_components)
        transitions = n_states * (n_states - 1)
        starts = n_states - 1
        means = n_states * n_features
        covariance_type = str(model.covariance_type)
        if covariance_type == "full":
            covariances = n_states * n_features * (n_features + 1) / 2
        elif covariance_type == "diag":
            covariances = n_states * n_features
        elif covariance_type == "tied":
            covariances = n_features * (n_features + 1) / 2
        else:                                   # spherical
            covariances = n_states
        n_params = transitions + starts + means + covariances
        return -2.0 * log_likelihood + n_params * float(np.log(n_samples))

    def _label_states(self, columns: tuple[str, ...]) -> dict[int, str]:
        """Order states by ascending mean realized volatility and label them.

        Not by mean return. A directional label on a model with no directional
        mandate will eventually be read as a bias signal, and return-sorted
        labels are unstable across retrains in a way volatility-sorted ones are
        not.
        """
        index = columns.index(LADDER_SORT_FEATURE)
        means = np.asarray(self.model.means_)
        order = np.argsort(means[:, index])
        ladder = ladder_for(int(self.model.n_components))
        return {int(state): ladder[rank] for rank, state in enumerate(order)}

    def _log_selection(self) -> None:
        """Record the whole selection, not just the winner."""
        assert self.metadata is not None
        for report in sorted(self.candidates, key=lambda item: item.bic):
            LOGGER.info("%s candidate %s", self.market, report.to_dict())
        verdict = (
            "margin is noise (< 2 BIC) - the state count was not really chosen "
            "by the data"
            if self.metadata.bic_margin_is_noise
            else f"won by {self.metadata.bic_margin:.1f} BIC"
        )
        LOGGER.info(
            "%s selected n_states=%d, %s, model_version=%s",
            self.market, self.metadata.n_states, verdict, self.metadata.model_version,
        )

    # -- inference -----------------------------------------------------------

    def _cache_log_parameters(self) -> None:
        """Pre-compute log transition and start probabilities."""
        with np.errstate(divide="ignore"):
            log_transmat = np.log(np.asarray(self.model.transmat_))
            log_startprob = np.log(np.asarray(self.model.startprob_))
        # EM drives unused transitions to exactly zero, and log(0) is -inf. Same
        # floor, same reason as the emissions: a -inf makes a state permanently
        # unreachable rather than merely very unlikely.
        self._log_transmat = np.maximum(log_transmat, EMISSION_LOG_FLOOR)
        self._log_startprob = np.maximum(log_startprob, EMISSION_LOG_FLOOR)
        self._alpha = None

    def _log_emissions(self, observations: np.ndarray) -> np.ndarray:
        """``log P(obs_t | state)`` for every bar and state.

        Computed from ``means_`` and ``covars_`` with scipy rather than through
        hmmlearn's private ``_compute_log_likelihood``. ``covars_`` expands to
        full matrices for every ``covariance_type``, so one code path covers
        them all, and no private API can change underneath this.
        """
        means = np.asarray(self.model.means_)
        covars = np.asarray(self.model.covars_)
        densities = np.column_stack([
            multivariate_normal.logpdf(
                observations, mean=means[state], cov=covars[state],
                allow_singular=True,
            )
            for state in range(int(self.model.n_components))
        ])
        densities = np.nan_to_num(
            densities, nan=EMISSION_LOG_FLOOR,
            neginf=EMISSION_LOG_FLOOR, posinf=0.0,
        )
        return np.maximum(densities, EMISSION_LOG_FLOOR)

    def predict_vol_state_filtered(self, features_up_to_now: np.ndarray) -> np.ndarray:
        """``P(state_t | observations_1..t)`` for every ``t``.

        Args:
            features_up_to_now: **Standardised** observation matrix, shape
                ``(T, n_features)``. Rows must be in time order and must contain
                only bars that had closed by the bar being scored.

        Returns:
            Array of shape ``(T, n_states)``. Row ``t`` uses rows ``0..t`` and
            nothing later - which is what makes it a filtered posterior rather
            than a smoothed one, and what the look-ahead tests verify.

        Uses only past and present data::

            alpha_0 = startprob * emission(obs_0)
            alpha_t = (alpha_{t-1} @ transmat) * emission(obs_t)

        in log space, normalised at each step with ``logsumexp``.
        """
        if self.model is None:
            raise ModelUnusable(f"{self.market}: no model fitted")
        observations = np.atleast_2d(np.asarray(features_up_to_now, dtype=float))
        log_emissions = self._log_emissions(observations)
        n_bars, n_states = log_emissions.shape

        posteriors = np.empty((n_bars, n_states), dtype=float)
        log_alpha = self._log_startprob + log_emissions[0]
        log_alpha -= logsumexp(log_alpha)
        posteriors[0] = np.exp(log_alpha)

        for bar in range(1, n_bars):
            # logsumexp over the previous state axis, i.e. alpha @ transmat.
            log_alpha = (
                logsumexp(log_alpha[:, None] + self._log_transmat, axis=0)
                + log_emissions[bar]
            )
            log_alpha -= logsumexp(log_alpha)
            posteriors[bar] = np.exp(log_alpha)

        return posteriors

    def reset_stream(self) -> None:
        """Forget the cached ``alpha``, so the next :meth:`step` restarts the chain.

        Called when a session's model is loaded and frozen, and whenever the
        live stream is known to have a hole in it - stepping across a gap with a
        stale alpha would carry a belief formed before the gap into bars after
        it as though nothing had happened.
        """
        self._alpha = None

    def step(self, observation: np.ndarray) -> np.ndarray:
        """Advance the filter by exactly one bar. O(1) in sequence length.

        Args:
            observation: One standardised feature row.

        Returns:
            The filtered posterior over states after this bar.

        Equivalent to the last row of :meth:`predict_vol_state_filtered` over
        the whole stream, and :mod:`tests.test_look_ahead` asserts that
        equivalence - a live loop and a backtest that disagree about the same
        bar would make every backtest number unreadable.
        """
        if self.model is None:
            raise ModelUnusable(f"{self.market}: no model fitted")
        row = np.asarray(observation, dtype=float).reshape(1, -1)
        log_emission = self._log_emissions(row)[0]

        if self._alpha is None:
            log_alpha = self._log_startprob + log_emission
        else:
            log_alpha = (
                logsumexp(self._alpha[:, None] + self._log_transmat, axis=0)
                + log_emission
            )
        log_alpha -= logsumexp(log_alpha)
        self._alpha = log_alpha
        return np.exp(log_alpha)

    def label_for(self, state: int) -> tuple[str, str]:
        """Return ``(ladder_label, public_bucket)`` for a raw state id."""
        if self.metadata is None:
            return ("UNKNOWN", "UNKNOWN")
        ladder_label = self.metadata.label_map.get(int(state), "UNKNOWN")
        return ladder_label, collapse(ladder_label)

    # -- persistence ---------------------------------------------------------

    def save(self, directory: str | Path | None = None) -> Path:
        """Persist the model, its scaler and its metadata.

        Returns:
            Path of the written ``.pkl``. A sidecar ``.json`` carries the
            metadata in readable form, because the question "what is running?"
            should not require unpickling anything.
        """
        if self.model is None or self.metadata is None:
            raise ModelUnusable(f"{self.market}: nothing to save")
        target = Path(directory or self.cfg.get("regime.paths.model_dir", "./models"))
        target.mkdir(parents=True, exist_ok=True)
        path = target / f"{self.market.lower()}_vol_hmm.pkl"
        with path.open("wb") as handle:
            pickle.dump(
                {"model": self.model, "metadata": self.metadata,
                 "scaler_lookback": self.scaler.lookback,
                 "scaler_min_periods": self.scaler.min_periods},
                handle,
            )
        path.with_suffix(".json").write_text(
            json.dumps(self.metadata.to_dict(), indent=2), encoding="utf-8"
        )
        return path

    @classmethod
    def load(cls, market: str, config: Config | None = None,
             directory: str | Path | None = None,
             now: datetime | None = None) -> "VolatilityHMM":
        """Load a persisted model and verify it may still be used.

        Raises:
            ModelUnusable: The file is missing or corrupt, the feature hash does
                not match the current pipeline, or the model is older than
                ``regime.hmm.max_model_age_days``.

        The feature-hash check is the important one. If the feature set changed
        and the model did not, scoring proceeds against means and covariances
        fitted to different quantities, and produces numbers that look entirely
        reasonable and mean nothing.
        """
        cfg = config or get_config()
        engine = cls(market, cfg)
        target = Path(directory or cfg.get("regime.paths.model_dir", "./models"))
        path = target / f"{engine.market.lower()}_vol_hmm.pkl"
        if not path.exists():
            raise ModelUnusable(f"{engine.market}: no persisted model at {path}")

        try:
            with path.open("rb") as handle:
                payload = pickle.load(handle)
        except Exception as error:
            raise ModelUnusable(f"{engine.market}: model file unreadable: {error}") from error

        metadata: ModelMetadata = payload["metadata"]
        expected = feature_hash(tuple(feature_columns(cfg.regime_use_volume(market))))
        if metadata.feature_hash != expected:
            raise ModelUnusable(
                f"{engine.market}: feature hash mismatch. Model was fitted on "
                f"{metadata.feature_hash}, the current pipeline produces {expected}. "
                f"Retrain before trading - scoring new features against old "
                f"means is silent, not loud."
            )

        max_age = int(cfg.get("regime.hmm.max_model_age_days"))
        reference = now or datetime.now(timezone.utc)
        age_days = (reference - metadata.trained_at).days
        if age_days > max_age:
            raise ModelUnusable(
                f"{engine.market}: model is {age_days} days old, limit is {max_age}"
            )

        engine.model = payload["model"]
        engine.metadata = metadata
        engine.scaler = CausalZScoreScaler(
            int(payload.get("scaler_lookback", cfg.get("regime.features.zscore_lookback"))),
            payload.get("scaler_min_periods"),
        )
        engine._cache_log_parameters()
        return engine
