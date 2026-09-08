"""Configuration loader - the single source of truth for every threshold.

Soul file, "Config discipline":

    Every value marked ``[DEFAULT - pending confirmation]`` must be surfaced in
    the single config block in Appendix A, not hardcoded in logic. Appendix A is
    the only place a threshold may be literal; everything else reads from it.

This module enforces that contract:

* :func:`get_config` returns a process-wide singleton read from
  ``config/beast_config.yaml``.
* :meth:`Config.get` does dotted lookups (``cfg.get("exit.target_r")``).
* :meth:`Config.require` raises :class:`ConfigBlockerError` when a value is
  ``None``. Several config keys are deliberately ``null`` in the shipped file
  (option lot sizes, liquidity floors, gold contract specs). The soul file is
  explicit that an unset filter is treated as *failing*, not passing - so any
  code path that needs one of those values must call ``require`` and let the
  trade be rejected rather than silently substituting a guess.

Secrets never live in ``beast_config.yaml``. They come from the environment (loaded
from ``.env``) or from ``config/credentials.yaml``, in that order of precedence.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

try:  # optional at import time so tests do not need the dependency installed
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - trivial shim
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "beast_config.yaml"
DEFAULT_CREDENTIALS_PATH = PROJECT_ROOT / "config" / "credentials.yaml"

_SENTINEL = object()


class ConfigError(Exception):
    """Raised when the configuration file itself is malformed or missing."""


class ConfigBlockerError(ConfigError):
    """Raised when a required config value is unset.

    This is not a bug - it is the soul file's designed refusal. ``min_oi``,
    ``lot_size``, ``contract_multiplier`` and friends ship as ``null`` and Beast
    must refuse to trade the affected instrument until the operator supplies a
    real number (open items 15, 17, 18, 19).
    """

    def __init__(self, key: str, hint: str = "") -> None:
        message = (
            f"Required config value '{key}' is unset. Beast treats an unset "
            f"threshold as FAILING, not passing."
        )
        if hint:
            message += f" {hint}"
        super().__init__(message)
        self.key = key


@dataclass
class Config:
    """Immutable-ish view over the parsed settings file.

    Attributes:
        data: The raw parsed YAML tree.
        path: Where it was read from, for error messages and reload.
        credentials: Secrets merged from ``credentials.yaml`` and the process
            environment. Kept separate from ``data`` so that logging the config
            can never leak a key.
    """

    data: dict[str, Any]
    path: Path
    credentials: dict[str, Any] = field(default_factory=dict, repr=False)

    # -- lookups --------------------------------------------------------------

    def get(self, dotted_key: str, default: Any = _SENTINEL) -> Any:
        """Return the value at ``dotted_key``.

        Args:
            dotted_key: e.g. ``"exit.target_r"`` or ``"options.delta_band"``.
            default: Returned when the key is absent. When omitted, a missing
                key raises :class:`ConfigError` - a typo in a config key should
                fail loudly, not silently behave like an unset threshold.

        Raises:
            ConfigError: The key is absent and no ``default`` was supplied.
        """
        node: Any = self.data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is _SENTINEL:
                    raise ConfigError(
                        f"Missing config key '{dotted_key}' in {self.path}"
                    )
                return default
            node = node[part]
        return node

    def require(self, dotted_key: str, hint: str = "") -> Any:
        """Return the value at ``dotted_key``, refusing ``None``.

        Use this for every BLOCKER-flagged value. Callers should let the
        exception propagate to the gate that owns the decision (G8 for option
        legs, G9 for sizing) so the rejection is logged with the right gate ID.

        Raises:
            ConfigBlockerError: The value is ``None``.
        """
        value = self.get(dotted_key, None)
        if value is None:
            raise ConfigBlockerError(dotted_key, hint)
        return value

    def section(self, name: str) -> dict[str, Any]:
        """Return a whole config section as a dict."""
        value = self.get(name)
        if not isinstance(value, dict):
            raise ConfigError(f"Config section '{name}' is not a mapping")
        return value

    # -- convenience ----------------------------------------------------------

    @property
    def mode(self) -> str:
        """``"paper"`` or ``"live"`` (soul file section 10)."""
        return str(self.get("mode", "paper")).lower()

    @property
    def is_paper(self) -> bool:
        """True when no real order may be placed.

        Two independent switches must both be off before Beast can trade live:
        the top-level ``mode`` and ``broker.paper_trading``. Either one being
        set to paper wins.
        """
        return self.mode != "live" or bool(self.get("broker.paper_trading", True))

    def market_family(self, market: str) -> str:
        """Map a market to its session/timeframe family.

        Args:
            market: ``"NIFTY"``, ``"NIFTY50"``, ``"SENSEX"``, ``"GOLD"`` or
                ``"XAUUSD"`` (case-insensitive).

        Returns:
            ``"indian"`` or ``"gold"`` - the key used in the ``sessions``,
            ``timeframes``, ``risk.daily_loss_cap`` and ``risk.max_concurrent``
            sections.
        """
        token = market.upper()
        if token.startswith("NIFTY") or token.startswith("SENSEX"):
            return "indian"
        if token.startswith("GOLD") or token.startswith("XAU"):
            return "gold"
        raise ConfigError(f"Unknown market '{market}'")

    def instrument_key(self, market: str) -> str:
        """Map a market to its key under the ``instruments`` section."""
        token = market.upper()
        if token.startswith("NIFTY"):
            return "nifty"
        if token.startswith("SENSEX"):
            return "sensex"
        if token.startswith("GOLD") or token.startswith("XAU"):
            return "gold"
        raise ConfigError(f"Unknown market '{market}'")

    def timeframes(self, market: str) -> dict[str, str]:
        """Return the bias/setup/trigger cascade for ``market`` (soul file 4.3)."""
        return dict(self.section("timeframes")[self.market_family(market)])

    def session(self, market: str) -> dict[str, Any]:
        """Return the session window for ``market`` (soul file 3)."""
        return dict(self.section("sessions")[self.market_family(market)])

    def risk_per_trade(self, market: str) -> float:
        """Return the per-trade risk fraction for ``market`` (soul file 7)."""
        return float(self.get("risk.risk_per_trade")[self.instrument_key(market)])

    def daily_loss_cap(self, market: str) -> float:
        """Return the daily loss cap fraction for ``market`` (soul file 7).

        v3.1 moved this from one combined Indian-session number to a per-
        instrument cap (``{nifty: 0.15, sensex: 0.10, gold: 0.05}``), so the
        lookup key is the instrument, not the session family. A config still
        written the v3.0 way - keyed by ``indian``/``gold`` - is read through
        the family key instead, so an operator's older file keeps working.
        """
        caps = self.get("risk.daily_loss_cap")
        key = self.instrument_key(market)
        if key in caps:
            return float(caps[key])
        family = self.market_family(market)
        if family in caps:
            return float(caps[family])
        raise ConfigError(
            f"risk.daily_loss_cap has no entry for '{key}' or '{family}'"
        )

    def credential(self, dotted_key: str, env_var: str | None = None) -> str | None:
        """Return a secret, environment first, then ``credentials.yaml``.

        Args:
            dotted_key: e.g. ``"zerodha.api_key"``.
            env_var: Environment variable checked first, e.g. ``ZERODHA_API_KEY``.
        """
        if env_var:
            from_env = os.environ.get(env_var)
            if from_env:
                return from_env
        node: Any = self.credentials
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        return node or None

    # -- validation -----------------------------------------------------------

    def unset_blockers(self) -> list[str]:
        """Return the dotted keys that are still ``null``.

        Used by ``main.py`` at startup and by the dashboard header so the
        operator can see exactly which instruments Beast will refuse to trade
        and why, instead of discovering it as a silent absence of signals.
        """
        watched = [
            "instruments.nifty.lot_size",
            "instruments.nifty.strike_interval",
            "instruments.sensex.lot_size",
            "instruments.sensex.strike_interval",
            "instruments.gold.venue",
            "instruments.gold.contract_multiplier",
            "instruments.gold.tick_size",
            "instruments.gold.tick_value",
            "options.min_oi",
            "options.min_volume",
            "options.min_premium",
            "data.gold_spread_max",
        ]
        return [key for key in watched if self.get(key, None) is None]

    def validate(self) -> list[str]:
        """Check internal consistency. Returns a list of human-readable problems.

        This catches contradictions the YAML parser cannot - a minimum stop
        distance above the maximum, a confluence requirement above six, an
        option side other than ``long_only`` (soul file 13 rule 10).
        """
        problems: list[str] = []

        if self.get("exit.stop_min_atr") >= self.get("exit.stop_max_atr"):
            problems.append("exit.stop_min_atr must be below exit.stop_max_atr")

        for key in ("entry.min_confluence", "entry.min_confluence_counter_bias"):
            value = int(self.get(key))
            if not 1 <= value <= 6:
                problems.append(f"{key} must be between 1 and 6 (got {value})")

        if self.get("entry.min_confluence_counter_bias") < self.get("entry.min_confluence"):
            problems.append(
                "entry.min_confluence_counter_bias must be >= entry.min_confluence"
            )

        for name in ("nifty", "sensex"):
            side = self.get(f"instruments.{name}.option_side", "long_only")
            if side != "long_only":
                problems.append(
                    f"instruments.{name}.option_side must be 'long_only'. Soul file "
                    f"rule 13.10 forbids selling options without a dedicated risk "
                    f"section; this is not a config toggle."
                )

        floor = float(self.get("risk.vol_factor_floor"))
        if not 0 < floor <= 1.0:
            problems.append("risk.vol_factor_floor must be in (0, 1.0]")

        low, high = self.get("options.delta_band")
        if not 0 < low < high < 1:
            problems.append("options.delta_band must satisfy 0 < low < high < 1")

        target_delta = float(self.get("options.target_delta"))
        if not low <= target_delta <= high:
            problems.append(
                "options.target_delta must sit inside options.delta_band"
            )

        if not self.get("ai.advisory_only", True):
            problems.append(
                "ai.advisory_only must be true. The AI layer narrates decisions; "
                "it never makes them (soul file 13)."
            )

        problems.extend(self._validate_regime())
        return problems

    # -- regime layer ---------------------------------------------------------

    #: Keys the ``regime:`` block must define. Each is marked ``# CONFIRM`` in
    #: the config and carries a default that has never been reviewed against
    #: live data, so an absent key is a missing decision, not a missing value.
    REGIME_REQUIRED_KEYS = (
        "regime.enabled",
        "regime.hmm.n_candidates",
        "regime.hmm.n_init",
        "regime.hmm.covariance_type",
        "regime.hmm.min_train_bars",
        "regime.hmm.retrain_interval_days",
        "regime.hmm.retrain_window",
        "regime.hmm.max_model_age_days",
        "regime.features.zscore_lookback",
        "regime.features.use_volume",
        "regime.stability.confirm_bars",
        "regime.stability.flicker_window",
        "regime.stability.flicker_threshold",
        "regime.stability.min_confidence",
        "regime.stability.stale_max_bars",
        "regime.stability.carry_across_sessions",
        "regime.sizing.combine_method",
        "regime.sizing.size_multiplier",
        "regime.sizing.uncertainty_size_mult",
        "regime.veto.on_turbulent",
        "regime.veto.on_turbulent_counter_bias",
        "regime.veto.on_unknown",
    )

    def _validate_regime(self) -> list[str]:
        """Validate the ``regime:`` block. Beast refuses to start on any problem.

        Three invariants matter more than the rest:

        * **Every multiplier sits in ``(0, 1.0]``.** The volatility layer may
          only ever shrink a position. A multiplier above 1.0 would let a
          statistical model increase risk beyond the section 7 number, which is
          Immutable Rule 1 territory.
        * **``combine_method`` is ``min``.** Multiplying the HMM's multiplier by
          section 7's ``vol_factor`` double-counts volatility - they measure
          largely the same thing - and the product's floor would be
          ``0.5 x 0.5 = 0.25``, silently overriding ``risk.vol_factor_floor``.
        * **Every key is present.** A missing key is an unmade decision.
        """
        problems: list[str] = []
        if self.get("regime", None) is None:
            return ["config has no `regime:` block (see soul file v3.2 diff)"]

        # A local sentinel, not the module-level `_SENTINEL`: passing that one
        # as the default makes `get` treat it as "no default supplied" and
        # raise, which would turn a missing key into a crash instead of a line
        # in the problem list.
        missing = object()
        for key in self.REGIME_REQUIRED_KEYS:
            if self.get(key, missing) is missing:
                problems.append(f"missing required config key '{key}'")

        multipliers = self.get("regime.sizing.size_multiplier", {}) or {}
        for label, value in multipliers.items():
            if not 0 < float(value) <= 1.0:
                problems.append(
                    f"regime.sizing.size_multiplier.{label} must be in (0, 1.0] "
                    f"(got {value}). The volatility layer may only shrink size."
                )
        for label in ("CALM", "NORMAL", "TURBULENT", "UNKNOWN"):
            if label not in multipliers:
                problems.append(
                    f"regime.sizing.size_multiplier is missing '{label}'"
                )

        uncertainty = self.get("regime.sizing.uncertainty_size_mult", None)
        if uncertainty is not None and not 0 < float(uncertainty) <= 1.0:
            problems.append(
                "regime.sizing.uncertainty_size_mult must be in (0, 1.0] "
                f"(got {uncertainty})"
            )

        combine = self.get("regime.sizing.combine_method", "min")
        if combine != "min":
            problems.append(
                f"regime.sizing.combine_method must be 'min', got '{combine}'. "
                f"Multiplying the vol_state multiplier by section 7's vol_factor "
                f"double-counts volatility and breaks risk.vol_factor_floor."
            )

        confidence = self.get("regime.stability.min_confidence", None)
        if confidence is not None and not 0 < float(confidence) < 1:
            problems.append("regime.stability.min_confidence must be in (0, 1)")

        for name in ("confirm_bars", "flicker_window", "flicker_threshold",
                     "stale_max_bars"):
            value = self.get(f"regime.stability.{name}", None)
            if value is not None and int(value) < 0:
                problems.append(f"regime.stability.{name} must not be negative")

        candidates = self.get("regime.hmm.n_candidates", []) or []
        if any(int(n) < 2 for n in candidates):
            problems.append("regime.hmm.n_candidates entries must all be >= 2")

        window = self.get("regime.hmm.retrain_window", "expanding")
        if window not in ("expanding", "rolling"):
            problems.append(
                f"regime.hmm.retrain_window must be 'expanding' or 'rolling', "
                f"got '{window}'"
            )

        return problems

    def regime_enabled(self) -> bool:
        """True when the volatility layer may influence sizing.

        When false, sizing falls back to section 7's ``vol_factor`` alone with
        no other change in behaviour - the layer still computes and logs a
        ``VolState`` so the operator can see what it *would* have done.
        """
        return bool(self.get("regime.enabled", False))

    def regime_min_train_bars(self, market: str) -> int:
        """Minimum bias-TF bars before a model may be fitted for ``market``."""
        return int(self.get("regime.hmm.min_train_bars")[self.instrument_key(market)])

    def regime_use_volume(self, market: str) -> bool:
        """Whether volume features are admissible for ``market``.

        False for Gold: XAUUSD spot is OTC, and a feed's "volume" is its own
        tick count - a measure of the provider's update rate, not traded size.
        """
        return bool(self.get("regime.features.use_volume")[self.instrument_key(market)])


_lock = threading.Lock()
_instance: Config | None = None


def load_config(
    path: str | Path | None = None,
    credentials_path: str | Path | None = None,
) -> Config:
    """Read and return a fresh :class:`Config` (does not touch the singleton).

    Args:
        path: Settings file. Defaults to ``config/beast_config.yaml``.
        credentials_path: Optional secrets file. Missing is fine - the
            environment is the primary source.

    Raises:
        ConfigError: The settings file is missing or is not a mapping.
    """
    settings_path = Path(path) if path else DEFAULT_SETTINGS_PATH
    if not settings_path.exists():
        raise ConfigError(f"Settings file not found: {settings_path}")

    with settings_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ConfigError(f"Settings file {settings_path} did not parse to a mapping")

    load_dotenv(PROJECT_ROOT / ".env", override=False)

    creds: dict[str, Any] = {}
    creds_path = Path(credentials_path) if credentials_path else DEFAULT_CREDENTIALS_PATH
    if creds_path.exists():
        with creds_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        if isinstance(loaded, dict):
            creds = loaded

    return Config(data=data, path=settings_path, credentials=creds)


def get_config(reload: bool = False) -> Config:
    """Return the process-wide config singleton.

    Args:
        reload: Re-read from disk instead of returning the cached instance.
    """
    global _instance
    with _lock:
        if _instance is None or reload:
            _instance = load_config()
        return _instance


def set_config(config: Config) -> None:
    """Install a config instance - used by tests to inject fixtures."""
    global _instance
    with _lock:
        _instance = config


def describe_unset(keys: Iterable[str]) -> str:
    """Format unset blocker keys for an operator-facing message."""
    keys = list(keys)
    if not keys:
        return "All blocker config values are populated."
    lines = ["Unset config values (Beast will refuse the affected trades):"]
    lines.extend(f"  - {key}" for key in keys)
    return "\n".join(lines)
