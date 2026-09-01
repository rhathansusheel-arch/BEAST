"""Enumerations and codes fixed by the Soul File.

Nothing here is configurable: these are the vocabulary of the Soul File itself
(regimes, setup types, gate IDs, exit reasons, flags). Thresholds live in
``config/beast.yaml`` — see PRECEDENCE.md.
"""

from __future__ import annotations

from enum import Enum


class Market(str, Enum):
    """Section 3 — the three markets Beast trades."""

    NIFTY = "NIFTY"
    SENSEX = "SENSEX"
    GOLD = "GOLD"

    @property
    def session_key(self) -> str:
        """Config key under ``sessions`` / ``timeframes`` / ``daily_loss_cap``."""
        return "gold" if self is Market.GOLD else "indian"

    @property
    def instrument_key(self) -> str:
        """Config key under ``instruments`` / ``risk.risk_per_trade``."""
        return self.name.lower()

    @property
    def underlying(self) -> str:
        return {"NIFTY": "NIFTY50", "SENSEX": "SENSEX", "GOLD": "XAUUSD"}[self.value]

    @property
    def is_option_market(self) -> bool:
        """Section 3.1 — Nifty/Sensex are traded as long options, Gold as futures."""
        return self is not Market.GOLD


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"

    @property
    def sign(self) -> int:
        return 1 if self is Direction.LONG else -1

    @property
    def opposite(self) -> "Direction":
        return Direction.SHORT if self is Direction.LONG else Direction.LONG


class Regime(str, Enum):
    """Section 4.4."""

    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE = "RANGE"


class ConfluenceMode(str, Enum):
    """Section 5.3 — the two alignment columns."""

    TREND_CONTINUATION = "TREND_CONTINUATION"
    REVERSAL = "REVERSAL"


class Read(str, Enum):
    """A single indicator's read for the confluence tally (5.3)."""

    BULL = "bull"
    BEAR = "bear"
    NEUTRAL = "neutral"


class Tier(str, Enum):
    """Section 4.5 — level tiers."""

    A = "A"
    B = "B"


class ZoneKind(str, Enum):
    SUPPORT = "SUPPORT"
    RESISTANCE = "RESISTANCE"


class Gate(str, Enum):
    """Section 5.1 — the ordered entry pipeline."""

    G0 = "G0"  # Session
    G1 = "G1"  # Data integrity
    G2 = "G2"  # No-trade conditions
    G3 = "G3"  # Regime / bias
    G4 = "G4"  # Setup detection
    G5 = "G5"  # Confluence
    G6 = "G6"  # Trigger
    G7 = "G7"  # Trade viability
    G8 = "G8"  # Instrument selection
    G9 = "G9"  # Risk & portfolio


GATE_NAMES: dict[Gate, str] = {
    Gate.G0: "Session",
    Gate.G1: "Data integrity",
    Gate.G2: "No-trade conditions",
    Gate.G3: "Regime / bias",
    Gate.G4: "Setup detection",
    Gate.G5: "Confluence",
    Gate.G6: "Trigger",
    Gate.G7: "Trade viability",
    Gate.G8: "Instrument selection",
    Gate.G9: "Risk & portfolio",
}


class ExitReason(str, Enum):
    """Section 6.6 — the only permitted ways a position closes."""

    SL = "SL"
    TP = "TP"
    TRAIL = "TRAIL"
    SESSION = "SESSION"
    TIME = "TIME"
    PREMIUM_STOP = "PREMIUM_STOP"
    OVERRIDE = "OVERRIDE"


class Flag(str, Enum):
    """Appendix B ``flags``."""

    SENSEX_DELAY = "SENSEX_DELAY"
    HIGH_SPREAD = "HIGH_SPREAD"
    NEWS_NEAR = "NEWS_NEAR"
    EXPIRY_DAY = "EXPIRY_DAY"
    IV_ELEVATED = "IV_ELEVATED"
    IV_CRUSH_RISK = "IV_CRUSH_RISK"
    PCR_EXTREME = "PCR_EXTREME"
    THETA_DRAG = "THETA_DRAG"
    STALE_FEED = "STALE_FEED"
    STALE_CHAIN = "STALE_CHAIN"
    GAP_OPEN = "GAP_OPEN"


class OITag(str, Enum):
    """Section 4.7.2 — price/OI change interpretation."""

    LONG_BUILDUP = "LONG_BUILDUP"
    SHORT_BUILDUP = "SHORT_BUILDUP"
    SHORT_COVERING = "SHORT_COVERING"
    LONG_UNWINDING = "LONG_UNWINDING"


# Section 5.2 — the four setup types, and which confluence column each uses.
SETUP_NAMES: dict[int, str] = {
    1: "Trendline Breakout/Breakdown",
    2: "Reversal at Support/Resistance",
    3: "Order Block Retest",
    4: "Indicator Confluence Trend Continuation",
}

#: Short labels used in the Section 11 one-liners.
SETUP_SHORT: dict[int, str] = {
    1: "trendline break",
    2: "reversal",
    3: "order block retest",
    4: "trend continuation",
}

SETUP_MODES: dict[int, ConfluenceMode] = {
    1: ConfluenceMode.TREND_CONTINUATION,
    2: ConfluenceMode.REVERSAL,
    3: ConfluenceMode.TREND_CONTINUATION,
    4: ConfluenceMode.TREND_CONTINUATION,
}

# Section 4.4 — permitted setups per regime. Setup 2 is the only counter-bias setup.
REGIME_PERMITS: dict[Regime, dict[Direction, tuple[int, ...]]] = {
    Regime.TREND_UP: {Direction.LONG: (1, 3, 4), Direction.SHORT: (2,)},
    Regime.TREND_DOWN: {Direction.SHORT: (1, 3, 4), Direction.LONG: (2,)},
    # RANGE is config-driven (`entry.range_regime_allowed_setups`) — see regime.py.
    Regime.RANGE: {},
}

IST = "Asia/Kolkata"

#: Section 8 — the exact typed acknowledgement required to override the plan.
OVERRIDE_PHRASE = "CONFIRM OVERRIDE: closing against plan."

#: The six indicators counted in the 4-of-6 confluence engine (5.3).
#: ATR is deliberately absent — 4.1 states it is a utility, never counted.
CONFLUENCE_INDICATORS: tuple[str, ...] = ("adx", "stoch", "macd", "rsi", "bb", "vwap")
