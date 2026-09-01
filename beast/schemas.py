"""Appendix B (Signal) and Appendix C (Trade Log / Rejection Log) schemas.

Section 9's learning loop and Section 11's reporting both read from these records, so the
field names here are the Soul File's field names verbatim. ``to_dict`` emits exactly the
Appendix B shape - nothing added, nothing renamed.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from beast.constants import (
    ConfluenceMode,
    Direction,
    ExitReason,
    Gate,
    Market,
    Regime,
)


def new_id() -> str:
    return str(uuid.uuid4())


def _iso(ts: datetime | None) -> str | None:
    return ts.isoformat() if ts is not None else None


@dataclass
class OptionLeg:
    """Appendix B ``leg._option_only`` - emitted by G8 (5.7.5)."""

    expiry: str
    dte: int
    strike: float
    option_type: str  # CE | PE
    delta: float
    iv: float
    mid_premium: float
    bid: float
    ask: float
    oi: int
    volume: int
    premium_stop: float
    lots: int = 0
    lot_size: int = 0
    total_premium_outlay: float = 0.0
    binding_cap: str = ""  # risk | premium_outlay

    def to_dict(self) -> dict[str, Any]:
        return {"type": "OPTION", "_option_only": self.__dict__.copy()}


@dataclass
class FuturesLeg:
    """Appendix B ``leg._futures_only`` - emitted by G8 (5.7.6)."""

    contract: str
    contract_multiplier: float
    contracts: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"type": "FUTURES", "_futures_only": self.__dict__.copy()}


@dataclass
class ChainContext:
    """Appendix B ``chain_context`` - recorded on every Nifty/Sensex signal (4.7)."""

    max_call_oi_strike: Optional[float] = None
    max_put_oi_strike: Optional[float] = None
    max_oi_change_strike: Optional[float] = None
    pcr: Optional[float] = None
    iv_percentile: Optional[float] = None
    oi_tag: Optional[str] = None
    level_convergence: bool = False

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class TrailSpec:
    """Appendix B ``trail`` - computed at entry, never after the fact (Section 6)."""

    activate_at: float
    method: str
    mult: float

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class Signal:
    """Appendix B - the Signal object. Emitted only after every gate in 5.1 passes."""

    market: Market
    direction: Direction
    setup_type: int
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
    trail: TrailSpec
    risk_pct: float
    vol_factor: float
    atr_setup_tf: float
    leg: dict[str, Any]
    chain_context: Optional[ChainContext] = None
    flags: list[str] = field(default_factory=list)
    mode: str = "paper"
    reason_line: str = ""
    timestamp_ist: Optional[datetime] = None
    signal_id: str = field(default_factory=new_id)
    soul_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "timestamp_ist": _iso(self.timestamp_ist),
            "market": self.market.value,
            "underlying": self.market.underlying,
            "direction": self.direction.value,
            "setup_type": self.setup_type,
            "setup_ref": self.setup_ref,
            "regime": self.regime.value,
            "counter_bias": self.counter_bias,
            "confluence_mode": self.confluence_mode.value,
            "confluence_count": self.confluence_count,
            "indicator_reads": self.indicator_reads,
            "timeframes": self.timeframes,
            "entry_price": self.entry_price,
            "stop_price": self.stop_price,
            "stop_source": self.stop_source,
            "target_price": self.target_price,
            "target_r": self.target_r,
            "trail": self.trail.to_dict(),
            "risk_pct": self.risk_pct,
            "vol_factor": self.vol_factor,
            "atr_setup_tf": self.atr_setup_tf,
            "leg": self.leg,
            "chain_context": self.chain_context.to_dict() if self.chain_context else None,
            "flags": list(self.flags),
            "mode": self.mode,
            "reason_line": self.reason_line,
            "soul_sha256": self.soul_sha256,
        }


@dataclass
class Rejection:
    """Appendix C rejection log - one row per candidate killed at a gate (5.1, Section 9)."""

    timestamp: datetime
    instrument: str
    setup_type: Optional[int]
    direction: Optional[str]
    failed_gate: Gate
    gate_detail: str
    confluence_count: Optional[dict[str, int]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": _iso(self.timestamp),
            "instrument": self.instrument,
            "setup_type": self.setup_type,
            "direction": self.direction,
            "failed_gate": self.failed_gate.value,
            "gate_detail": self.gate_detail,
            "confluence_count": self.confluence_count,
        }


@dataclass
class OverrideRecord:
    """Section 8 - every override attempt, honoured or refused."""

    timestamp: datetime
    trade_id: str
    action: str
    trade_context: dict[str, Any]
    confirmed: bool
    hypothetical_r_if_held_to_target: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["timestamp"] = _iso(self.timestamp)
        return d


@dataclass
class TradeRecord:
    """Appendix C - the Signal object plus everything the exit records (6.9).

    For options both sides are stored. The R-multiple of record is the **premium-based**
    one (6.9) - that is the actual money - with ``underlying_r_multiple`` alongside it, so
    Section 9 can see whether strike selection is leaking edge.
    """

    signal: Signal
    entry_time: Optional[datetime] = None
    entry_fill_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[ExitReason] = None
    r_multiple: Optional[float] = None
    mae_r: Optional[float] = None
    mfe_r: Optional[float] = None
    bars_held: int = 0
    trail_activated: bool = False
    hypothetical_r_if_held_to_target: Optional[float] = None
    override: Optional[OverrideRecord] = None
    slippage: float = 0.0
    costs: float = 0.0

    # Options-only (6.9, Appendix C)
    entry_premium: Optional[float] = None
    exit_premium: Optional[float] = None
    entry_underlying: Optional[float] = None
    exit_underlying: Optional[float] = None
    premium_r_multiple: Optional[float] = None
    underlying_r_multiple: Optional[float] = None
    mae_premium: Optional[float] = None
    mfe_premium: Optional[float] = None
    delta_at_entry: Optional[float] = None
    iv_at_entry: Optional[float] = None
    iv_at_exit: Optional[float] = None
    dte: Optional[int] = None
    theta_cost_estimate: Optional[float] = None
    slippage_premium: float = 0.0

    trade_id: str = field(default_factory=new_id)

    @property
    def market(self) -> Market:
        return self.signal.market

    @property
    def setup_type(self) -> int:
        return self.signal.setup_type

    @property
    def is_closed(self) -> bool:
        return self.exit_reason is not None

    def to_dict(self) -> dict[str, Any]:
        d = self.signal.to_dict()
        d.update(
            {
                "trade_id": self.trade_id,
                "entry_time": _iso(self.entry_time),
                "entry_fill_price": self.entry_fill_price,
                "exit_time": _iso(self.exit_time),
                "exit_price": self.exit_price,
                "exit_reason": self.exit_reason.value if self.exit_reason else None,
                "r_multiple": self.r_multiple,
                "mae_r": self.mae_r,
                "mfe_r": self.mfe_r,
                "bars_held": self.bars_held,
                "trail_activated": self.trail_activated,
                "hypothetical_r_if_held_to_target": self.hypothetical_r_if_held_to_target,
                "override": self.override.to_dict() if self.override else None,
                "slippage": self.slippage,
                "costs": self.costs,
            }
        )
        if self.signal.market.is_option_market:
            d.update(
                {
                    "entry_premium": self.entry_premium,
                    "exit_premium": self.exit_premium,
                    "entry_underlying": self.entry_underlying,
                    "exit_underlying": self.exit_underlying,
                    "premium_r_multiple": self.premium_r_multiple,
                    "underlying_r_multiple": self.underlying_r_multiple,
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
        return d
