"""HMM regime detection engine.

DESIGN PHILOSOPHY: this is a VOLATILITY CLASSIFIER, not a price-direction
predictor. It fits a Gaussian HMM to standardized market features and infers
which of a small number of latent volatility/return regimes the market is
currently in (calm, moderate, turbulent). The strategy layer uses that
classification to set portfolio allocation — fully invested when conditions
are calm, reduced when turbulent — never to bet on direction.

*** NO LOOK-AHEAD BIAS ***
`GaussianHMM.predict()` runs the Viterbi algorithm, which processes an
entire sequence and can revise past states using future observations. Using
it for backtesting or live inference is look-ahead bias and will make
results look better than they can ever be live. Every regime call in this
module is instead computed with the forward algorithm (filtered inference):
P(state_t | obs_1:t), which depends only on data up to and including t. See
`predict_regime_filtered` / `predict_regime_proba` / `_forward_log_alpha`.
"""

from __future__ import annotations

import pickle
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, ClassVar, Optional

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from monitoring.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RegimeInfo:
    """Static metadata describing one labeled regime, consumed by the
    strategy/risk layers to decide how aggressively to trade it."""

    regime_id: int
    regime_name: str
    expected_return: float
    expected_volatility: float
    recommended_strategy_type: str
    max_leverage_allowed: float
    max_position_size_pct: float
    min_confidence_to_act: float


@dataclass
class RegimeState:
    """A single point-in-time regime read, as produced by `HMMEngine.observe`."""

    label: str
    state_id: int
    probability: float
    state_probabilities: dict[str, float]
    timestamp: Any
    is_confirmed: bool
    consecutive_bars: int


# Regime-label sets, ordered from lowest to highest mean return. Hidden
# states are ranked by mean return (ascending) after training and assigned
# labels positionally from these lists.
_REGIME_LABEL_SETS: dict[int, list[str]] = {
    3: ["BEAR", "NEUTRAL", "BULL"],
    4: ["CRASH", "BEAR", "BULL", "EUPHORIA"],
    5: ["CRASH", "BEAR", "NEUTRAL", "BULL", "EUPHORIA"],
    6: ["CRASH", "STRONG_BEAR", "WEAK_BEAR", "WEAK_BULL", "STRONG_BULL", "EUPHORIA"],
    7: ["CRASH", "STRONG_BEAR", "WEAK_BEAR", "NEUTRAL", "WEAK_BULL", "STRONG_BULL", "EUPHORIA"],
}

# How defensively the strategy layer should treat each label. Per the design
# philosophy this is really about volatility, not direction — EUPHORIA is
# flagged alongside the bear labels because blow-off tops carry elevated
# realized vol and tail risk just like sell-offs do.
_VOLATILITY_TIER: dict[str, str] = {
    "CRASH": "extreme",
    "STRONG_BEAR": "high",
    "BEAR": "high",
    "WEAK_BEAR": "moderate",
    "NEUTRAL": "low",
    "WEAK_BULL": "moderate",
    "BULL": "low",
    "STRONG_BULL": "moderate",
    "EUPHORIA": "high",
}

# tier -> (max_leverage_allowed, max_position_size_pct, min_confidence_to_act, recommended_strategy_type)
_TIER_DEFAULTS: dict[str, tuple[float, float, float, str]] = {
    "low": (1.25, 0.15, 0.55, "trend_following"),
    "moderate": (1.0, 0.12, 0.60, "balanced"),
    "high": (0.5, 0.08, 0.70, "defensive"),
    "extreme": (0.0, 0.05, 0.80, "capital_preservation"),
}


class HMMEngine:
    """Selects, fits, and runs filtered inference for a Gaussian HMM regime classifier."""

    REGIME_LABEL_SETS: ClassVar[dict[int, list[str]]] = _REGIME_LABEL_SETS

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Args:
            config: The `hmm` section of settings.yaml — n_candidates, n_init,
                covariance_type, min_train_bars, stability_bars,
                flicker_window, flicker_threshold, min_confidence — plus the
                optional `return_feature`/`volatility_feature` column-name
                overrides used for regime labeling (default "return_1" /
                "realized_vol").
        """
        self.config = config
        self.model: Optional[GaussianHMM] = None
        self.n_states: Optional[int] = None
        self.feature_names: list[str] = []
        self.regime_labels: dict[int, str] = {}
        self.regime_info: dict[int, RegimeInfo] = {}
        self.training_metadata: dict[str, Any] = {}
        self._label_to_state: dict[str, int] = {}
        self._search_results: list[dict[str, Any]] = []
        self.reset_state()

    # ------------------------------------------------------------------
    # Training / model selection
    # ------------------------------------------------------------------

    def select_n_states(self, features: pd.DataFrame) -> int:
        """Train every candidate in `n_candidates`, score each by BIC
        (BIC = -2*log_likelihood + n_params*log(n_samples), lower is better),
        log all candidate scores, and return the winning n_components.

        Fitted models are cached on `self._search_results` so `fit()` doesn't
        need to retrain the winner.
        """
        X = features.to_numpy()
        n_init = self.config.get("n_init", 10)
        covariance_type = self.config.get("covariance_type", "full")
        candidates = self.config.get("n_candidates", [3, 4, 5, 6, 7])

        results: list[dict[str, Any]] = []
        for k in candidates:
            model, log_likelihood, bic = self._fit_best_of_n_init(X, k, n_init, covariance_type)
            results.append(
                {"n_components": k, "model": model, "log_likelihood": log_likelihood, "bic": bic}
            )
            logger.info(
                "HMM candidate n_components=%d log_likelihood=%.2f bic=%.2f converged=%s iters=%d",
                k, log_likelihood, bic, model.monitor_.converged, model.monitor_.iter,
            )

        best = min(results, key=lambda r: r["bic"])
        logger.info(
            "Selected n_components=%d (BIC=%.2f) from candidates %s",
            best["n_components"], best["bic"], list(candidates),
        )
        self._search_results = results
        return best["n_components"]

    def fit(self, features: pd.DataFrame) -> "HMMEngine":
        """Search candidate state counts by BIC, fit the winner, and label its
        regimes by mean return (ascending). Requires at least `min_train_bars` rows.

        The model is designed for 2+ years of daily data (>=504 bars); the
        enforced floor is `min_train_bars` from settings.yaml (default 252).
        """
        min_bars = self.config.get("min_train_bars", 252)
        if len(features) < min_bars:
            raise ValueError(f"Need at least {min_bars} bars to train, got {len(features)}")

        self.feature_names = list(features.columns)
        best_k = self.select_n_states(features)
        best = next(r for r in self._search_results if r["n_components"] == best_k)

        self.model = best["model"]
        self.n_states = best_k
        self.regime_labels = self._label_regimes(self.model, best_k)
        self._label_to_state = {v: k for k, v in self.regime_labels.items()}
        self.regime_info = self._build_regime_info_table()

        self.training_metadata = {
            "n_regimes": best_k,
            "bic": best["bic"],
            "log_likelihood": best["log_likelihood"],
            "training_date": datetime.now(timezone.utc),
            "labels": dict(self.regime_labels),
            "candidate_bics": {r["n_components"]: r["bic"] for r in self._search_results},
        }

        self.reset_state()
        return self

    def _fit_best_of_n_init(
        self, X: np.ndarray, n_components: int, n_init: int, covariance_type: str
    ) -> tuple[GaussianHMM, float, float]:
        """Fit `n_init` random restarts of GaussianHMM(n_components) and keep the
        highest-log-likelihood fit. Returns (best_model, log_likelihood, bic)."""
        best_model: Optional[GaussianHMM] = None
        best_ll = -np.inf
        for seed in range(n_init):
            model = GaussianHMM(
                n_components=n_components,
                covariance_type=covariance_type,
                n_iter=self.config.get("n_iter", 1000),
                random_state=seed,
            )
            model.fit(X)
            log_likelihood = model.score(X)
            if log_likelihood > best_ll:
                best_ll = log_likelihood
                best_model = model
        assert best_model is not None
        n_params = self._n_free_params(n_components, X.shape[1])
        bic = -2.0 * best_ll + n_params * np.log(X.shape[0])
        return best_model, best_ll, bic

    @staticmethod
    def _n_free_params(n_components: int, n_features: int) -> int:
        """Free parameter count for a full-covariance GaussianHMM, used in the BIC penalty."""
        startprob_params = n_components - 1
        transmat_params = n_components * (n_components - 1)
        means_params = n_components * n_features
        cov_params = n_components * n_features * (n_features + 1) // 2
        return startprob_params + transmat_params + means_params + cov_params

    def _return_feature_index(self) -> int:
        name = self.config.get("return_feature", "return_1")
        if name not in self.feature_names:
            raise ValueError(
                f"Return feature '{name}' not found among trained features {self.feature_names}"
            )
        return self.feature_names.index(name)

    def _label_regimes(self, model: GaussianHMM, n_components: int) -> dict[int, str]:
        """Rank hidden states by mean return (ascending: lowest -> CRASH/BEAR,
        highest -> BULL/EUPHORIA) and assign the label set for `n_components`."""
        return_idx = self._return_feature_index()
        mean_returns = model.means_[:, return_idx]
        order = np.argsort(mean_returns)
        label_set = self.REGIME_LABEL_SETS[n_components]
        return {int(state_id): label_set[rank] for rank, state_id in enumerate(order)}

    def _build_regime_info_table(self) -> dict[int, RegimeInfo]:
        vol_name = self.config.get("volatility_feature", "realized_vol")
        vol_idx = self.feature_names.index(vol_name) if vol_name in self.feature_names else None
        return_idx = self._return_feature_index()

        info: dict[int, RegimeInfo] = {}
        for state_id in range(self.n_states):
            label = self.regime_labels[state_id]
            mean_return = float(self.model.means_[state_id, return_idx])
            mean_vol = float(self.model.means_[state_id, vol_idx]) if vol_idx is not None else float("nan")
            info[state_id] = self._build_regime_info(state_id, label, mean_return, mean_vol)
        return info

    @staticmethod
    def _build_regime_info(state_id: int, label: str, mean_return: float, mean_volatility: float) -> RegimeInfo:
        tier = _VOLATILITY_TIER.get(label, "moderate")
        max_leverage, max_position, min_confidence, strategy_type = _TIER_DEFAULTS[tier]
        return RegimeInfo(
            regime_id=state_id,
            regime_name=label,
            expected_return=mean_return,
            expected_volatility=mean_volatility,
            recommended_strategy_type=strategy_type,
            max_leverage_allowed=max_leverage,
            max_position_size_pct=max_position,
            min_confidence_to_act=min_confidence,
        )

    # ------------------------------------------------------------------
    # Filtered inference (forward algorithm only — no look-ahead)
    # ------------------------------------------------------------------

    def _emission_log_prob(self, X: np.ndarray) -> np.ndarray:
        """Log emission probability log P(obs_t | state=k) for every t, k."""
        log_prob = np.empty((X.shape[0], self.n_states))
        for k in range(self.n_states):
            dist = multivariate_normal(
                mean=self.model.means_[k], cov=self.model.covars_[k], allow_singular=True
            )
            log_prob[:, k] = dist.logpdf(X)
        return log_prob

    def _forward_log_alpha(self, log_B: np.ndarray) -> np.ndarray:
        """Log-space forward algorithm (filtering only): log P(state_t=k | obs_1:t).

        Uses only startprob_/transmat_/log_B up to and including t at every
        step — the piece that makes this safe from look-ahead bias, unlike
        Viterbi (`model.predict`), which is revised by future observations.
        """
        T = log_B.shape[0]
        log_startprob = np.log(self.model.startprob_ + 1e-300)
        log_transmat = np.log(self.model.transmat_ + 1e-300)

        log_alpha = np.empty_like(log_B)
        log_alpha[0] = log_startprob + log_B[0]
        log_alpha[0] -= logsumexp(log_alpha[0])

        for t in range(1, T):
            log_predict = logsumexp(log_alpha[t - 1][:, None] + log_transmat, axis=0)
            log_alpha[t] = log_predict + log_B[t]
            log_alpha[t] -= logsumexp(log_alpha[t])

        return log_alpha

    def predict_regime_proba(self, features_up_to_now: pd.DataFrame) -> pd.DataFrame:
        """Filtered regime probability distribution P(state_t | obs_1:t) for
        every t in `features_up_to_now`, via the forward algorithm only.

        Pure function of the input prefix: calling this with a longer window
        never changes the probabilities already computed for earlier rows,
        which is what `tests/test_look_ahead.py` verifies.
        """
        self._require_fitted()
        X = features_up_to_now[self.feature_names].to_numpy()
        log_B = self._emission_log_prob(X)
        log_alpha = self._forward_log_alpha(log_B)
        proba = np.exp(log_alpha)
        columns = [self.regime_labels[k] for k in range(self.n_states)]
        return pd.DataFrame(proba, index=features_up_to_now.index, columns=columns)

    def predict_regime_filtered(self, features_up_to_now: pd.DataFrame) -> pd.Series:
        """Most-likely filtered regime label at every t in `features_up_to_now`.

        DO NOT replace this with `self.model.predict(...)` — see the module
        docstring. This is the safe, no-look-ahead entry point for backtesting.
        """
        proba = self.predict_regime_proba(features_up_to_now)
        return proba.idxmax(axis=1)

    # ------------------------------------------------------------------
    # Incremental (stateful) inference for live trading / step backtests
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """Clear all incremental/live state: the cached forward-algorithm
        alpha, raw regime history, and the stability/flicker trackers. Call
        before starting a new backtest run or live session (also called
        automatically at the end of `fit()`)."""
        self._log_alpha_cache: Optional[np.ndarray] = None
        self._raw_history: deque[str] = deque(maxlen=self.config.get("flicker_window", 20))
        self._confirmed_label: Optional[str] = None
        self._confirmed_streak: int = 0
        self._pending_label: Optional[str] = None
        self._pending_streak: int = 0
        self._last_change_confirmed: bool = False

    def _forward_step(self, obs: np.ndarray) -> np.ndarray:
        """One incremental forward-algorithm update using the cached alpha
        from the previous call — O(n_states^2) per bar instead of recomputing
        the full history, for use in the live/backtest loop via `observe`."""
        log_B_t = self._emission_log_prob(obs.reshape(1, -1))[0]
        if self._log_alpha_cache is None:
            log_alpha_t = np.log(self.model.startprob_ + 1e-300) + log_B_t
        else:
            log_transmat = np.log(self.model.transmat_ + 1e-300)
            log_predict = logsumexp(self._log_alpha_cache[:, None] + log_transmat, axis=0)
            log_alpha_t = log_predict + log_B_t
        log_alpha_t -= logsumexp(log_alpha_t)
        self._log_alpha_cache = log_alpha_t
        return np.exp(log_alpha_t)

    def observe(self, feature_row: pd.Series, timestamp: Any = None) -> RegimeState:
        """Process one new bar: update the cached filtered distribution,
        apply the regime-stability/flicker filter, and return the resulting
        RegimeState. This is the stateful entry point for live trading and
        step-by-step backtesting; `predict_regime_filtered`/`predict_regime_proba`
        remain pure batch functions for offline analysis and the look-ahead test.
        """
        self._require_fitted()
        obs = feature_row[self.feature_names].to_numpy(dtype=float)
        proba = self._forward_step(obs)
        raw_label = self.regime_labels[int(np.argmax(proba))]

        self._raw_history.append(raw_label)
        confirmed_this_bar = self._apply_stability_filter(raw_label, timestamp)
        self._last_change_confirmed = confirmed_this_bar

        state_probabilities = {self.regime_labels[k]: float(proba[k]) for k in range(self.n_states)}

        return RegimeState(
            label=self._confirmed_label,
            state_id=self._label_to_state[self._confirmed_label],
            probability=state_probabilities[self._confirmed_label],
            state_probabilities=state_probabilities,
            timestamp=timestamp,
            is_confirmed=self._pending_label is None,
            consecutive_bars=self._confirmed_streak,
        )

    def _apply_stability_filter(self, raw_label: str, timestamp: Any) -> bool:
        """Update pending/confirmed regime state given the newest raw label.

        A regime change is logged (WARNING) the moment the raw filtered call
        first diverges from the currently confirmed regime. It only becomes
        official — logged at INFO as a confirmation — once it has persisted
        for `stability_bars` consecutive observations; until then the
        previously confirmed regime remains active, and `RegimeState.is_confirmed`
        is False so the strategy layer can reduce sizes during the transition.

        Returns True only on the bar where a change is newly confirmed.
        """
        stability_bars = self.config.get("stability_bars", 3)

        if self._confirmed_label is None:
            self._confirmed_label = raw_label
            self._confirmed_streak = 1
            self._pending_label = None
            self._pending_streak = 0
            return False

        if raw_label == self._confirmed_label:
            self._confirmed_streak += 1
            self._pending_label = None
            self._pending_streak = 0
            return False

        if raw_label != self._pending_label:
            logger.warning(
                "Regime change detected: %s -> %s at %s (pending confirmation)",
                self._confirmed_label, raw_label, timestamp,
            )
            self._pending_label = raw_label
            self._pending_streak = 1
        else:
            self._pending_streak += 1

        if self._pending_streak >= stability_bars:
            previous_label = self._confirmed_label
            self._confirmed_label = raw_label
            self._confirmed_streak = 1
            self._pending_label = None
            self._pending_streak = 0
            logger.info(
                "Regime change confirmed: %s -> %s at %s", previous_label, self._confirmed_label, timestamp
            )
            return True

        return False

    # ------------------------------------------------------------------
    # Stability / flicker / transition-matrix accessors
    # ------------------------------------------------------------------

    def get_regime_stability(self) -> int:
        """Consecutive bars the currently confirmed regime has held."""
        return self._confirmed_streak

    def get_transition_matrix(self) -> pd.DataFrame:
        """Learned regime transition probabilities, labeled by regime name."""
        self._require_fitted()
        labels = [self.regime_labels[k] for k in range(self.n_states)]
        return pd.DataFrame(self.model.transmat_, index=labels, columns=labels)

    def detect_regime_change(self) -> bool:
        """True only if the most recent `observe()` call confirmed a regime
        change (i.e. passed the stability filter, not just a single noisy bar)."""
        return self._last_change_confirmed

    def get_regime_flicker_rate(self) -> int:
        """Number of raw regime-label changes within the trailing `flicker_window` bars."""
        history = list(self._raw_history)
        return sum(1 for prev, curr in zip(history, history[1:]) if prev != curr)

    def is_flickering(self) -> bool:
        """True if the flicker rate exceeds `flicker_threshold` — signals
        unstable/noisy regime calls that should push the strategy into
        uncertainty mode (reduced sizing) regardless of the nominal regime."""
        return self.get_regime_flicker_rate() > self.config.get("flicker_threshold", 4)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist the trained model + metadata to `path` via pickle."""
        self._require_fitted()
        payload = {
            "model": self.model,
            "n_states": self.n_states,
            "feature_names": self.feature_names,
            "regime_labels": self.regime_labels,
            "regime_info": self.regime_info,
            "training_metadata": self.training_metadata,
            "config": self.config,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        logger.info("Saved HMM model (n_regimes=%d) to %s", self.n_states, path)

    @classmethod
    def load(cls, path: str) -> "HMMEngine":
        """Load a previously trained HMMEngine from a pickle file written by `save`."""
        with open(path, "rb") as f:
            payload = pickle.load(f)
        engine = cls(config=payload["config"])
        engine.model = payload["model"]
        engine.n_states = payload["n_states"]
        engine.feature_names = payload["feature_names"]
        engine.regime_labels = payload["regime_labels"]
        engine._label_to_state = {v: k for k, v in engine.regime_labels.items()}
        engine.regime_info = payload["regime_info"]
        engine.training_metadata = payload["training_metadata"]
        engine.reset_state()
        return engine

    def _require_fitted(self) -> None:
        if self.model is None:
            raise RuntimeError("HMMEngine.fit() must be called before inference")
