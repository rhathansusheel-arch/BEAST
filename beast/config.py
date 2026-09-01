"""Appendix A config access.

The Soul File's config discipline: *every* value marked
``[DEFAULT - pending confirmation]`` lives in ``config/beast.yaml`` and nowhere else.
Logic reads thresholds from here; it never carries its own literals.

Two rules this module enforces on behalf of the Soul File:

* **Unset means fail.** A ``null`` blocker (``options.min_oi``, Gold contract specs,
  ``capital``, ``data.gold_spread_max``) is treated as *failing*, not as "no limit"
  (5.7.3, 3.1). :meth:`Config.require` raises rather than substituting a guess.
* **One home per threshold.** :meth:`Config.get` uses dotted paths so callers reference
  Appendix A directly instead of copying values into their own defaults.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "beast.yaml"

_MISSING = object()


class ConfigError(Exception):
    """A config path is absent, or a blocker value is unset."""


class ConfigUnset(ConfigError):
    """A required Appendix A value is ``null``.

    Raised - never defaulted around - because the Soul File treats an unset blocker as a
    hard failure. Section 3.1: "Beast will refuse to size a Gold trade until they are
    populated." Section 5.7.3: "an unset liquidity filter is treated as failing."
    """


class Config:
    """Read-only accessor over the Appendix A block."""

    def __init__(self, data: dict[str, Any], source: Path | None = None) -> None:
        self._data = data
        self.source = source

    # -- construction ----------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        p = Path(path) if path is not None else DEFAULT_CONFIG_PATH
        with open(p, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            raise ConfigError(f"{p} did not parse to a mapping")
        return cls(data, source=p)

    def with_overrides(self, **dotted: Any) -> "Config":
        """Return a copy with dotted paths replaced.

        Used by tests and by the learning loop (Section 9), which may raise a setup's
        required confluence 4 -> 5 and nothing else.
        """
        data = copy.deepcopy(self._data)
        for path, value in dotted.items():
            node = data
            parts = path.split("__") if "__" in path else path.split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
        return Config(data, source=self.source)

    # -- access ----------------------------------------------------------------

    def get(self, path: str, default: Any = _MISSING) -> Any:
        """Fetch a dotted path. Missing paths raise unless a default is given."""
        node: Any = self._data
        for part in path.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            elif default is not _MISSING:
                return default
            else:
                raise ConfigError(f"config path not found: {path}")
        return node

    def require(self, path: str, why: str = "") -> Any:
        """Fetch a dotted path that must be populated.

        Raises :class:`ConfigUnset` when the value is ``None``. This is the mechanism
        behind "Beast refuses to trade rather than guess".
        """
        value = self.get(path)
        if value is None:
            detail = f" - {why}" if why else ""
            raise ConfigUnset(f"{path} is unset in Appendix A{detail}")
        return value

    def is_set(self, path: str) -> bool:
        return self.get(path, None) is not None

    def section(self, path: str) -> dict[str, Any]:
        value = self.get(path)
        if not isinstance(value, dict):
            raise ConfigError(f"config path is not a section: {path}")
        return value

    # -- Soul-File-shaped helpers ---------------------------------------------

    @property
    def mode(self) -> str:
        """Section 10 - ``paper`` (alert-only) or ``live``."""
        return str(self.get("mode", "paper"))

    @property
    def is_paper(self) -> bool:
        return self.mode != "live"

    def timeframes(self, market) -> dict[str, str]:
        """Section 4.3 cascade for a market: bias / setup / trigger."""
        return dict(self.get(f"timeframes.{market.session_key}"))

    def session(self, market) -> dict[str, Any]:
        """Section 3 trading window for a market."""
        return dict(self.get(f"sessions.{market.session_key}"))

    def risk_per_trade(self, market) -> float:
        return float(self.get(f"risk.risk_per_trade.{market.instrument_key}"))

    def daily_loss_cap(self, market) -> float:
        return float(self.get(f"risk.daily_loss_cap.{market.session_key}"))

    def max_concurrent(self, market) -> int:
        return int(self.get(f"risk.max_concurrent.{market.session_key}"))

    def instrument(self, market) -> dict[str, Any]:
        return dict(self.get(f"instruments.{market.instrument_key}"))

    def capital(self) -> float:
        """Account capital. Unset is a hard failure - every Section 7 cap is a % of it."""
        return float(self.require("capital", "Section 7 sizes every trade as a % of capital"))

    # -- readiness -------------------------------------------------------------

    def unresolved_blockers(self, market=None) -> list[str]:
        """Appendix A values that are ``null`` and therefore block trading.

        Mirrors "Open Items 17, 18, 19 - the blockers; Beast cannot trade without them".
        Pass a market to scope the report to that market's blockers.
        """
        checks: list[tuple[str, str]] = [
            ("capital", "Section 7 - risk per trade is a % of capital"),
        ]
        option_checks = [
            ("options.min_oi", "5.7.3 - unset liquidity filter fails, not passes"),
            ("options.min_volume", "5.7.3 - unset liquidity filter fails, not passes"),
            ("options.min_premium", "5.7.3 - unset liquidity filter fails, not passes"),
        ]
        gold_checks = [
            ("instruments.gold.venue", "3.1 / Open Item 17 - Gold contract specs"),
            ("instruments.gold.contract_multiplier", "3.1 / Open Item 17"),
            ("instruments.gold.tick_size", "3.1 / Open Item 17"),
            ("instruments.gold.tick_value", "3.1 / Open Item 17"),
            ("data.gold_spread_max", "5.6 / Open Item 15 - high-spread rule unenforceable"),
        ]

        if market is None:
            checks += option_checks + gold_checks
            for name in ("nifty", "sensex"):
                checks += [
                    (f"instruments.{name}.lot_size", "Open Item 18 - lot size"),
                    (f"instruments.{name}.strike_interval", "Open Item 18 - strike ladder"),
                ]
        elif market.is_option_market:
            key = market.instrument_key
            checks += option_checks
            checks += [
                (f"instruments.{key}.lot_size", "Open Item 18 - lot size"),
                (f"instruments.{key}.strike_interval", "Open Item 18 - strike ladder"),
            ]
        else:
            checks += gold_checks

        return [f"{path} ({why})" for path, why in checks if not self.is_set(path)]
