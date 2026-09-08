"""Data contracts - Appendix B (Signal) and Appendix C (Trade Log).

The soul file is explicit that these schemas are load-bearing:

    Appendix B defines the Signal object and Appendix C the Trade Log record.
    Emit and persist exactly these fields - Section 9's learning loop and
    Section 11's reporting both read from them.

Everything here is a plain dataclass with a ``to_dict`` that emits the exact
Appendix field names, so the JSON on disk matches the specification verbatim and
the learning loop can be written against the document rather than against the
code.

A note on units, because it is the single most important structural rule in the
document (soul file 3.1): every price field named ``*_underlying``, ``entry_price``,
``stop_price`` or ``target_price`` is expressed in **underlying points** - index
points for Nifty/Sensex, dollars for XAUUSD. Premium values live only inside
:class:`OptionLeg` and the premium-suffixed trade-log fields. Nothing in the
analysis layer ever sees a premium.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Direction(str, Enum):
    """Trade direction, expressed on the underlying."""

    LONG = "LONG"
    SHORT = "SHORT"

    @property
    def sign(self) -> int:
        """+1 for long, -1 for short. Lets one formula serve both sides."""
        return 1 if self is Direction.LONG else -1

    @property
    def opposite(self) -> "Direction":
        return Direction.SHORT if self is Direction.LONG else Direction.LONG


class Regime(str, Enum):
    """Bias-timeframe regime (soul file 4.4)."""

    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE = "RANGE"


class SetupType(int, Enum):
    """The four setup types (soul file 5.2)."""

    TRENDLINE_BREAK = 1
    SR_REVERSAL = 2
    ORDER_BLOCK_RETEST = 3
    TREND_CONTINUATION = 4

    @property
    def label(self) -> str:
        return {
            1: "Trendline breakout/breakdown",
            2: "Reversal at support/resistance",
            3: "Order block retest",
            4: "Indicator confluence trend continuation",
        }[int(self)]


class ConfluenceMode(str, Enum):
    """Which column of the 5.3 table applies.

    Setup 2 reads the Reversal column; setups 1, 3 and 4 read the
    Trend-Continuation column. Beast never mixes the two within one count.
    """

    TREND_CONTINUATION = "TREND_CONTINUATION"
    REVERSAL = "REVERSAL"


class IndicatorRead(str, Enum):
    """One indicator's verdict. ``NEUTRAL`` counts toward neither side."""

    BULL = "bull"
    BEAR = "bear"
    NEUTRAL = "neutral"


class LevelTier(str, Enum):
    """Level importance (soul file 4.5)."""

    A = "A"  # bias-TF, session-structural, or option-chain OI levels
    B = "B"  # setup-TF only


class LevelKind(str, Enum):
    """Where a level came from - recorded so 4.7's convergence bonus is auditable."""

    SWING_CLUSTER = "swing_cluster"
    PRIOR_DAY_HIGH = "prior_day_high"
    PRIOR_DAY_LOW = "prior_day_low"
    PRIOR_DAY_CLOSE = "prior_day_close"
    SESSION_HIGH = "session_high"
    SESSION_LOW = "session_low"
    OVERNIGHT_HIGH = "overnight_high"
    OVERNIGHT_LOW = "overnight_low"
    MAX_CALL_OI = "max_call_oi"
    MAX_PUT_OI = "max_put_oi"
    SECOND_MAX_CALL_OI = "second_max_call_oi"
    SECOND_MAX_PUT_OI = "second_max_put_oi"
    MAX_OI_CHANGE_CALL = "max_oi_change_call"
    MAX_OI_CHANGE_PUT = "max_oi_change_put"
    MAX_PAIN = "max_pain"


class ExitReason(str, Enum):
    """The only ways a live position may close (soul file 6.6)."""

    SL = "SL"
    TP = "TP"
    TRAIL = "TRAIL"
    SESSION = "SESSION"
    TIME = "TIME"
    PREMIUM_STOP = "PREMIUM_STOP"
    OVERRIDE = "OVERRIDE"


class Gate(str, Enum):
    """Entry pipeline gate IDs (soul file 5.1).

    Every rejection is logged against the gate that killed it so section 9 can
    see whether Beast is missing trades at G5 (confluence too strict) or G7
    (structure too wide).
    """

    G0_SESSION = "G0"
    G1_DATA = "G1"
    G2_NO_TRADE = "G2"
    G3_REGIME = "G3"
    G4_SETUP = "G4"
    G5_CONFLUENCE = "G5"
    G6_TRIGGER = "G6"
    G7_VIABILITY = "G7"
    G8_INSTRUMENT = "G8"
    G9_RISK = "G9"


class OITag(str, Enum):
    """Price/OI change interpretation (soul file 4.7.2)."""

    LONG_BUILDUP = "LONG_BUILDUP"
    SHORT_BUILDUP = "SHORT_BUILDUP"
    SHORT_COVERING = "SHORT_COVERING"
    LONG_UNWINDING = "LONG_UNWINDING"
    UNKNOWN = "UNKNOWN"


class Flag(str, Enum):
    """Context flags carried on a signal (Appendix B ``flags``)."""

    SENSEX_DELAY = "SENSEX_DELAY"
    HIGH_SPREAD = "HIGH_SPREAD"
    NEWS_NEAR = "NEWS_NEAR"
    EXPIRY_DAY = "EXPIRY_DAY"
    IV_ELEVATED = "IV_ELEVATED"
    IV_CRUSH_RISK = "IV_CRUSH_RISK"
    PCR_EXTREME = "PCR_EXTREME"
    THETA_DRAG = "THETA_DRAG"
    STALE_CHAIN = "STALE_CHAIN"
    GAP_SESSION = "GAP_SESSION"
    HMM_UNSTABLE = "HMM_UNSTABLE"


# ---------------------------------------------------------------------------
# Level engine outputs (soul file 4.5, 4.7.1)
# ---------------------------------------------------------------------------


@dataclass
class Zone:
    """A support/resistance zone with a deterministic lifetime.

    Attributes:
        zone_id: Stable identifier, used for the re-entry cap and blacklist.
        kind: Provenance of the level.
        tier: A (major) or B (minor).
        low: Lower price bound, in underlying points.
        high: Upper price bound.
        centre: Zone centre - the mean of the clustered swings, or the strike.
        strength: Touches + one per prior rejection, plus the 4.7 convergence
            bonus where an OI level and a price level agree. Reported for
            transparency; it does not gate the trade.
        is_support: True when the zone is currently read as support.
        touches: Number of confirmed touches.
        created_at: When the zone was first formed.
        alive: Set False once price closes beyond it by > 0.5 x ATR.
        flipped_from: Set when a dead resistance becomes candidate support.
        converged_with: Ids of levels merged into this one (4.7 bonus).
    """

    zone_id: str
    kind: LevelKind
    tier: LevelTier
    low: float
    high: float
    centre: float
    strength: float = 1.0
    is_support: bool = True
    touches: int = 1
    created_at: datetime | None = None
    alive: bool = True
    flipped_from: str | None = None
    converged_with: list[str] = field(default_factory=list)

    def contains(self, price: float) -> bool:
        """True when ``price`` sits inside the zone bounds."""
        return self.low <= price <= self.high

    def distance_to(self, price: float) -> float:
        """Absolute distance from ``price`` to the nearest zone edge (0 if inside)."""
        if self.contains(price):
            return 0.0
        return min(abs(price - self.low), abs(price - self.high))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        payload["tier"] = self.tier.value
        payload["created_at"] = self.created_at.isoformat() if self.created_at else None
        return payload


@dataclass
class Trendline:
    """A least-squares trendline with >= 3 confirmed anchors (soul file 4.5)."""

    line_id: str
    is_support: bool           # True: rising line under lows; False: falling over highs
    slope: float               # price units per bar
    intercept: float           # price at anchor_start_index
    anchor_indices: list[int] = field(default_factory=list)
    anchor_prices: list[float] = field(default_factory=list)
    start_index: int = 0
    max_deviation: float = 0.0
    alive: bool = True
    broken_at_index: int | None = None

    def value_at(self, index: int) -> float:
        """Price of the fitted line at bar ``index``."""
        return self.intercept + self.slope * (index - self.start_index)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OrderBlock:
    """The last opposing candle before an impulse that broke structure."""

    ob_id: str
    direction: Direction       # direction of the impulse the OB precedes
    low: float
    high: float
    formed_index: int
    formed_at: datetime | None = None
    retests: int = 0
    fresh: bool = True
    alive: bool = True
    expires_at: datetime | None = None

    @property
    def far_edge(self) -> float:
        """The edge a close beyond which kills the block permanently."""
        return self.low if self.direction is Direction.LONG else self.high

    def contains(self, price: float) -> bool:
        return self.low <= price <= self.high

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["direction"] = self.direction.value
        payload["formed_at"] = self.formed_at.isoformat() if self.formed_at else None
        payload["expires_at"] = self.expires_at.isoformat() if self.expires_at else None
        return payload


# ---------------------------------------------------------------------------
# Setup / plan / sizing
# ---------------------------------------------------------------------------


@dataclass
class SetupInstance:
    """One detected setup awaiting a trigger (soul file 5.2, 5.5).

    A setup instance is the unit the "one instance, one signal" rule applies to.
    Re-arming requires a fresh detection, so ``consumed`` is sticky.
    """

    setup_id: str
    setup_type: SetupType
    direction: Direction
    ref_id: str                       # id of the trendline / zone / OB / pullback leg
    detected_index: int
    detected_at: datetime
    structural_stop: float            # raw structural price, before the ATR buffer
    counter_bias: bool = False
    expires_after_index: int = 0      # setup-TF index by which the trigger must fire
    consumed: bool = False
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def confluence_mode(self) -> ConfluenceMode:
        """Setup 2 is the only reversal; everything else is trend-continuation."""
        return (
            ConfluenceMode.REVERSAL
            if self.setup_type is SetupType.SR_REVERSAL
            else ConfluenceMode.TREND_CONTINUATION
        )


@dataclass
class ConfluenceResult:
    """Outcome of the 4-of-6 engine (soul file 5.3).

    All six reads are retained including the neutrals - section 9 needs the
    misses to compute per-indicator hit rates.
    """

    mode: ConfluenceMode
    direction: Direction
    reads: dict[str, IndicatorRead]
    aligned: int
    opposing: int
    neutral: int
    required: int
    passed: bool
    rejected_by_conflict: bool = False
    notes: dict[str, str] = field(default_factory=dict)

    def reads_as_str(self) -> dict[str, str]:
        return {name: read.value for name, read in self.reads.items()}


@dataclass
class TrailPlan:
    """Trailing-stop parameters, fixed at entry (soul file 6.3)."""

    activate_at: float          # underlying price at which the trail arms
    activate_r: float
    method: str                 # atr_chandelier | structure_trail
    mult: float


@dataclass
class TradePlan:
    """The complete exit plan, computed before the signal is emitted.

    Soul file 6: "An entry whose exit plan cannot be constructed is not an
    entry." Every field here is in underlying points.
    """

    direction: Direction
    entry_price: float
    stop_price: float
    target_price: float
    stop_source: str
    target_r: float
    trail: TrailPlan
    atr: float
    risk_points: float          # 1R, fixed at entry
    viable: bool = True
    reject_reason: str | None = None

    @property
    def r_multiple_at(self) -> Any:
        """Return a callable mapping a price to its R-multiple."""

        def _r(price: float) -> float:
            if self.risk_points <= 0:
                return 0.0
            return (price - self.entry_price) * self.direction.sign / self.risk_points

        return _r


@dataclass
class OptionLeg:
    """The tradable option leg produced by G8 (soul file 5.7.5)."""

    expiry: date
    dte: int
    strike: float
    option_type: str            # "CE" or "PE"
    delta: float
    iv: float
    mid_premium: float
    bid: float
    ask: float
    oi: int
    volume: int
    premium_stop: float
    lot_size: int
    tradingsymbol: str = ""
    lots: int = 0
    total_premium_outlay: float = 0.0
    binding_cap: str = ""       # "risk" | "premium_outlay"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["expiry"] = self.expiry.isoformat()
        return payload


@dataclass
class FuturesLeg:
    """The tradable futures contract produced by G8 (soul file 5.7.6)."""

    contract: str
    contract_multiplier: float
    tick_size: float
    tick_value: float
    expiry: date | None = None
    contracts: int = 0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["expiry"] = self.expiry.isoformat() if self.expiry else None
        return payload


@dataclass
class SizedPosition:
    """Output of G9 (soul file 7, 7.1)."""

    permitted: bool
    quantity: int               # lots for options, contracts for futures
    risk_amount: float
    vol_factor: float
    binding_cap: str = ""
    reason: str = ""


@dataclass
class ChainContext:
    """Option-chain context recorded on every Nifty/Sensex signal (4.7)."""

    max_call_oi_strike: float | None = None
    max_put_oi_strike: float | None = None
    second_max_call_oi_strike: float | None = None
    second_max_put_oi_strike: float | None = None
    max_oi_change_strike: float | None = None
    max_pain: float | None = None
    pcr: float | None = None
    atm_iv: float | None = None
    iv_percentile: float | None = None
    oi_tag: OITag = OITag.UNKNOWN
    level_convergence: bool = False
    snapshot_age_sec: float | None = None
    stale: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["oi_tag"] = self.oi_tag.value
        return payload


# ---------------------------------------------------------------------------
# Appendix B - Signal
# ---------------------------------------------------------------------------


@dataclass
class Signal:
    """A qualifying trade signal. Field names match Appendix B exactly."""

    market: str                       # NIFTY | SENSEX | GOLD
    underlying: str                   # NIFTY50 | SENSEX | XAUUSD
    direction: Direction
    setup_type: SetupType
    setup_ref: str
    regime: Regime
    counter_bias: bool
    confluence_mode: ConfluenceMode
    confluence_count: dict[str, int]
    indicator_reads: dict[str, str]
    timeframes: dict[str, str]
    entry_price: float
    stop_price: float
    stop_source: str
    target_price: float
    target_r: float
    trail: TrailPlan
    risk_pct: float
    vol_factor: float
    atr_setup_tf: float
    timestamp_ist: datetime
    leg_type: str                     # OPTION | FUTURES
    option_leg: OptionLeg | None = None
    futures_leg: FuturesLeg | None = None
    chain_context: ChainContext | None = None
    hmm_context: dict[str, Any] = field(default_factory=dict)
    flags: list[Flag] = field(default_factory=list)
    mode: str = "paper"
    reason_line: str = ""
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the Appendix B JSON shape."""
        leg: dict[str, Any] = {"type": self.leg_type}
        if self.option_leg is not None:
            leg["_option_only"] = self.option_leg.to_dict()
        if self.futures_leg is not None:
            leg["_futures_only"] = self.futures_leg.to_dict()

        return {
            "signal_id": self.signal_id,
            "timestamp_ist": self.timestamp_ist.isoformat(),
            "market": self.market,
            "underlying": self.underlying,
            "direction": self.direction.value,
            "setup_type": int(self.setup_type),
            "setup_ref": self.setup_ref,
            "regime": self.regime.value,
            "counter_bias": self.counter_bias,
            "confluence_mode": self.confluence_mode.value,
            "confluence_count": dict(self.confluence_count),
            "indicator_reads": dict(self.indicator_reads),
            "timeframes": dict(self.timeframes),
            "entry_price": self.entry_price,
            "stop_price": self.stop_price,
            "stop_source": self.stop_source,
            "target_price": self.target_price,
            "target_r": self.target_r,
            "trail": asdict(self.trail),
            "risk_pct": self.risk_pct,
            "vol_factor": self.vol_factor,
            "atr_setup_tf": self.atr_setup_tf,
            "leg": leg,
            "chain_context": (
                self.chain_context.to_dict() if self.chain_context else None
            ),
            "hmm_context": dict(self.hmm_context),
            "flags": [flag.value for flag in self.flags],
            "mode": self.mode,
            "reason_line": self.reason_line,
        }


# ---------------------------------------------------------------------------
# Appendix C - Trade log and rejection log
# ---------------------------------------------------------------------------


@dataclass
class OverrideRecord:
    """A logged Section 8 override."""

    timestamp: datetime
    action: str                 # close_early | tighten_stop | add_to_loser
    confirmation_text: str
    context: dict[str, Any] = field(default_factory=dict)
    hypothetical_r_if_held_to_target: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        return payload


@dataclass
class TradeRecord:
    """Appendix C: the signal object plus everything the exit produced."""

    signal: Signal
    entry_time: datetime
    entry_fill_price: float                  # underlying
    exit_time: datetime | None = None
    exit_price: float | None = None          # underlying
    exit_reason: ExitReason | None = None
    r_multiple: float | None = None          # premium-based for options (6.9)
    underlying_r_multiple: float | None = None
    mae_r: float | None = None
    mfe_r: float | None = None
    bars_held: int = 0
    trail_activated: bool = False
    hypothetical_r_if_held_to_target: float | None = None
    override: OverrideRecord | None = None
    slippage: float = 0.0
    costs: float = 0.0

    # options-only, both sides of the trade (Appendix C)
    entry_premium: float | None = None
    exit_premium: float | None = None
    entry_underlying: float | None = None
    exit_underlying: float | None = None
    premium_r_multiple: float | None = None
    mae_premium: float | None = None
    mfe_premium: float | None = None
    delta_at_entry: float | None = None
    iv_at_entry: float | None = None
    iv_at_exit: float | None = None
    dte: int | None = None
    theta_cost_estimate: float | None = None
    slippage_premium: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = self.signal.to_dict()
        payload.update(
            {
                "entry_fill_price": self.entry_fill_price,
                "entry_time": self.entry_time.isoformat(),
                "exit_time": self.exit_time.isoformat() if self.exit_time else None,
                "exit_price": self.exit_price,
                "exit_reason": self.exit_reason.value if self.exit_reason else None,
                "r_multiple": self.r_multiple,
                "underlying_r_multiple": self.underlying_r_multiple,
                "mae_r": self.mae_r,
                "mfe_r": self.mfe_r,
                "bars_held": self.bars_held,
                "trail_activated": self.trail_activated,
                "hypothetical_r_if_held_to_target": self.hypothetical_r_if_held_to_target,
                "override": self.override.to_dict() if self.override else None,
                "slippage": self.slippage,
                "costs": self.costs,
                "entry_premium": self.entry_premium,
                "exit_premium": self.exit_premium,
                "entry_underlying": self.entry_underlying,
                "exit_underlying": self.exit_underlying,
                "premium_r_multiple": self.premium_r_multiple,
                "mae_premium": self.mae_premium,
                "mfe_premium": self.mfe_premium,
                "delta_at_entry": self.delta_at_entry,
                "iv_at_entry": self.iv_at_entry,
                "iv_at_exit": self.iv_at_exit,
                "dte": self.dte,
                "theta_cost_estimate": self.theta_cost_estimate,
                "slippage_premium": self.slippage_premium,
            }
        )
        return payload


@dataclass
class Rejection:
    """Rejection-log row (Appendix C, separate table).

    Section 9 reads this to answer "where do signals die" - too strict at G5, or
    too wide at G7.
    """

    timestamp: datetime
    instrument: str
    failed_gate: Gate
    gate_detail: str
    setup_type: SetupType | None = None
    direction: Direction | None = None
    confluence_count: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "instrument": self.instrument,
            "setup_type": int(self.setup_type) if self.setup_type else None,
            "direction": self.direction.value if self.direction else None,
            "failed_gate": self.failed_gate.value,
            "gate_detail": self.gate_detail,
            "confluence_count": self.confluence_count,
        }
