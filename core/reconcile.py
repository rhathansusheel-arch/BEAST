"""Startup reconcile: make every broker position explainable and protected.

Runs at startup step 6, before feeds start and before any entry is possible.
The question it answers is not "what does Beast remember" but "what does the
venue hold, and is each of those positions covered by a stop Beast can vouch
for". The venue is asked directly - never the in-memory tracker, never the
snapshot - because both of those are what Beast *believed*, and after a hard
kill belief is exactly what cannot be trusted.

Order of operations, and why that order
---------------------------------------
1. Enumerate the venue's positions and pending orders under Beast's magic.
   A venue that cannot be asked is **unknown**, not flat: entries for its
   markets are refused and the operator is paged. Assuming flat is how a
   position ends up running on nothing.
2. Match each position to a plan in the journal - the ``signals`` row with no
   ``trades`` row, which is what an open trade looks like there.
3. Verify the stop. Absent -> placed at the plan's level immediately, before
   anything else at startup completes. Wider than the plan -> alert and SAFE
   mode, **unchanged**: section 8 refuses a widening outright and the reverse
   correction here would be a loosening in disguise if the resting stop is
   the true one. Tighter than the plan -> left alone: section 6.3's ratchet
   only tightens, so a tighter stop is legitimate trail state.
4. Restore the target when absent. Rehydrate trail state under the ratchet,
   asserting the restored stop is never looser than the venue's.
5. Cancel orphan pending orders - Beast's magic, no matching position.
6. SAFE mode for anything Beast cannot explain: a protective stop, entries
   paused for that market, operator paged. Beast does not manage a position it
   cannot explain; it protects it and says so.

Idempotency
-----------
Every stop write goes through ``set_position_stop``, which re-reads before
sending, sends only a change, and re-reads after. ``TRADE_ACTION_SLTP`` is a
no-op when the value is already resting, so running this twice is safe by
construction - which is what a watchdog restart into a half-finished reconcile
requires.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from broker import BrokerClient, BrokerOrder, BrokerPosition, OrderSide
from broker.order_executor import parse_client_tag
from core.config import Config
from core.schemas import Direction

logger = logging.getLogger("beast.reconcile")


@dataclass
class Plan:
    """What the journal knows about an open trade."""

    signal_id: str
    market: str
    direction: Direction
    entry_price: float
    stop_price: float
    target_price: float
    opened_at: datetime | None
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def plan_id(self) -> str:
        return self.signal_id.replace("-", "")[:8]


@dataclass
class Match:
    position: BrokerPosition
    plan: Plan | None
    confidence: float
    how: str


@dataclass
class Action:
    kind: str            # stop_placed | stop_repaired | target_restored | order_cancelled | safe_stop_placed
    market: str
    ticket: int
    detail: str
    before: float | None = None
    after: float | None = None
    confirmed: bool = False


@dataclass
class ReconcileReport:
    unreachable: dict[str, str] = field(default_factory=dict)      # market -> why
    positions: list[BrokerPosition] = field(default_factory=list)
    matches: list[Match] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)
    disagreements: list[str] = field(default_factory=list)
    safe_mode: dict[str, str] = field(default_factory=dict)        # market -> reason
    adopted: dict[str, dict[str, Any]] = field(default_factory=dict)  # market -> rehydrated state

    @property
    def repaired(self) -> bool:
        return any(a.kind != "noop" for a in self.actions)

    @property
    def entries_to_pause(self) -> dict[str, str]:
        paused = dict(self.safe_mode)
        for market, why in self.unreachable.items():
            paused.setdefault(market, f"broker unreachable at startup: {why}")
        return paused

    def lines(self) -> list[str]:
        out = []
        for market, why in self.unreachable.items():
            out.append(f"UNREACHABLE {market}: {why} - entries refused, nothing assumed")
        for m in self.matches:
            p = m.position
            out.append(
                f"POSITION {p.market} #{p.ticket} {p.side.value} {p.volume:g} @ {p.entry_price} "
                f"sl={p.sl or 'NONE'} tp={p.tp or 'NONE'} -> "
                f"{'plan ' + m.plan.plan_id if m.plan else 'NO PLAN'} ({m.how}, {m.confidence:.2f})"
            )
        for a in self.actions:
            out.append(f"ACTION {a.kind} {a.market} #{a.ticket}: {a.detail}"
                       + (f" [{a.before} -> {a.after}]" if a.before is not None else "")
                       + ("" if a.confirmed else "  NOT CONFIRMED"))
        for d in self.disagreements:
            out.append(f"DISAGREEMENT {d}")
        for market, why in self.safe_mode.items():
            out.append(f"SAFE MODE {market}: {why}")
        return out or ["no broker positions, no pending orders, nothing to reconcile"]


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def reconcile(brokers: dict[str, BrokerClient], positions, journal, risk, alerts,
              cfg: Config, now: datetime, markets: list[str] | None = None,
              history=None) -> ReconcileReport:
    """Reconcile every market's venue state against the journal.

    Args:
        brokers: Adapter by name, as ``main.py`` holds them.
        positions: The in-process ``PositionTracker`` - written to, never read
            as a source of truth.
        journal: ``monitoring.journal.Journal``.
        risk: ``RiskManager`` - only its capital is read here.
        alerts: ``AlertManager``.
        history: Optional ``(market, timeframe, bars) -> DataFrame`` for the
            SAFE-mode ATR. Defaults to the routed broker's ``history``.
    """
    report = ReconcileReport()
    markets = [m.upper() for m in (markets or cfg.get("broker.symbols"))]
    routing = {str(k).upper(): str(v) for k, v in cfg.get("broker.routing").items()}
    threshold = float(cfg.get("ops.reconcile_match_threshold", 0.7))

    plans = _open_plans(journal)

    for market in markets:
        broker = brokers.get(routing.get(market, ""))
        if broker is None:
            report.unreachable[market] = "no broker routed"
            continue
        if not broker.is_connected():
            report.unreachable[market] = f"{broker.name} not connected"
            continue

        held = broker.open_positions()
        pending = broker.pending_orders()
        if held is None or pending is None:
            report.unreachable[market] = f"{broker.name} could not list positions/orders"
            continue

        held = [p for p in held if p.market.upper() == market]
        pending = [o for o in pending if o.market.upper() == market]
        report.positions.extend(held)

        # -- orphan pending orders (step 5) --------------------------------
        open_tickets = {p.ticket for p in held}
        for order in pending:
            if order.position_ticket and order.position_ticket in open_tickets:
                continue
            ok = broker.cancel_order(str(order.ticket))
            report.actions.append(Action(
                "order_cancelled", market, order.ticket,
                f"orphan {order.side.value} {order.volume:g} @ {order.price} ({order.comment!r})",
                confirmed=bool(ok),
            ))
            if not ok:
                report.disagreements.append(
                    f"{market}: orphan order #{order.ticket} could not be cancelled")

        # -- each position (steps 2-4, 6) -----------------------------------
        for position in held:
            match = _match(position, plans.get(market, []), now)
            report.matches.append(match)
            if match.plan is None or match.confidence < threshold:
                _safe_mode(report, broker, position, match, cfg, history, market)
                continue
            if not _verify_stop(report, broker, position, match.plan, cfg):
                continue        # SAFE mode: flagged and left exactly as found
            _restore_target(report, broker, position, match.plan)
            _adopt(report, position, match.plan, now)

    for market, why in report.unreachable.items():
        alerts.api_lost(market, f"reconcile: {why}")
    for market, why in report.safe_mode.items():
        alerts.circuit_breaker(market, f"SAFE MODE: {why}", tripped=True)
    if report.repaired:
        alerts.circuit_breaker("SYSTEM", "startup reconcile repaired broker state - "
                               + "; ".join(a.detail for a in report.actions), tripped=True)
    return report


# ---------------------------------------------------------------------------
# steps
# ---------------------------------------------------------------------------


def _open_plans(journal) -> dict[str, list[Plan]]:
    """Open trades as the journal sees them: a signal with no trade row."""
    by_market: dict[str, list[Plan]] = {}
    try:
        rows = journal.open_signals_without_trades()
    except Exception as error:
        logger.error("journal unavailable for reconcile: %s", error)
        return by_market
    for row in rows:
        try:
            payload = json.loads(row.get("payload") or "{}")
        except (TypeError, ValueError):
            payload = {}
        opened = _parse_ts(row.get("timestamp_ist"))
        plan = Plan(
            signal_id=str(row["signal_id"]),
            market=str(row["market"]).upper(),
            direction=Direction(str(row["direction"])),
            entry_price=float(row.get("entry_price") or 0.0),
            stop_price=float(row.get("stop_price") or 0.0),
            target_price=float(row.get("target_price") or 0.0),
            opened_at=opened,
            payload=payload,
        )
        by_market.setdefault(plan.market, []).append(plan)
    return by_market


def _match(position: BrokerPosition, plans: list[Plan], now: datetime) -> Match:
    """Best plan for a position, with how confident that pairing is.

    Exact comment tag beats everything. Failing that, same direction and an
    entry within the match window; failing that, same direction alone at a
    confidence below the threshold, so it lands in SAFE mode.
    """
    want_side = OrderSide.BUY if position.side is OrderSide.BUY else OrderSide.SELL
    same_side = [p for p in plans
                 if (p.direction is Direction.LONG) == (want_side is OrderSide.BUY)]

    tag = parse_client_tag(position.comment)
    if tag is not None:
        for plan in same_side:
            if plan.plan_id == tag[2]:
                return Match(position, plan, 1.0, "comment tag")

    if position.opened_at is not None:
        window = timedelta(minutes=10)
        near = [p for p in same_side if p.opened_at is not None
                and abs(_naive(p.opened_at) - _naive(position.opened_at)) <= window]
        if len(near) == 1:
            return Match(position, near[0], 0.85, "direction + entry time")
        if len(near) > 1:
            near.sort(key=lambda p: abs(p.entry_price - position.entry_price))
            return Match(position, near[0], 0.6, "direction + time, ambiguous")

    if len(same_side) == 1:
        return Match(position, same_side[0], 0.5, "direction only")
    return Match(position, None, 0.0, "no plan")


def _verify_stop(report: ReconcileReport, broker: BrokerClient, position: BrokerPosition,
                 plan: Plan, cfg: Config) -> bool:
    """Step 3 and 4: stop present, full volume, never widened.

    Returns False when the position has been put into SAFE mode and must be
    left exactly as found - no target restore, no adoption.
    """
    market = position.market
    is_long = position.side is OrderSide.BUY
    plan_stop = plan.stop_price
    was_present = position.stop_present
    was_covering = position.stop_covers_volume
    before = position.sl or None

    if not was_present or not was_covering:
        why = "absent" if not was_present else \
              f"covers {position.stop_volume:g} of {position.volume:g}"
        result = broker.set_position_stop(position.ticket, plan_stop, plan.target_price or None)
        report.actions.append(Action(
            "stop_placed" if not was_present else "stop_repaired",
            market, position.ticket, f"stop was {why}; set to plan {plan_stop}: {result.message}",
            before=before, after=plan_stop, confirmed=result.accepted,
        ))
        if not result.accepted:
            report.safe_mode[market] = f"could not place the stop on #{position.ticket}: {result.message}"
            return False
        position.sl = plan_stop
        return True

    resting = position.sl
    wider = resting < plan_stop if is_long else resting > plan_stop
    tighter = resting > plan_stop if is_long else resting < plan_stop
    if wider:
        report.disagreements.append(
            f"{market} #{position.ticket}: resting stop {resting} is WIDER than the plan's "
            f"{plan_stop}. Left unchanged - section 8 refuses a widening and this code "
            f"will not decide which one is right. SAFE mode."
        )
        report.safe_mode[market] = f"resting stop {resting} wider than plan {plan_stop}"
        return False
    if tighter:
        report.actions.append(Action(
            "noop", market, position.ticket,
            f"resting stop {resting} is tighter than plan {plan_stop}: trail state, kept",
            confirmed=True,
        ))
    return True


def _restore_target(report: ReconcileReport, broker: BrokerClient,
                    position: BrokerPosition, plan: Plan) -> None:
    if position.tp > 0 or plan.target_price <= 0:
        if position.tp > 0 and plan.target_price > 0 and abs(position.tp - plan.target_price) > 1e-9:
            report.disagreements.append(
                f"{position.market} #{position.ticket}: resting target {position.tp} differs "
                f"from plan {plan.target_price}; left as is")
        return
    result = broker.set_position_stop(position.ticket, position.sl, plan.target_price)
    report.actions.append(Action(
        "target_restored", position.market, position.ticket,
        f"target was absent; set to plan {plan.target_price}: {result.message}",
        before=None, after=plan.target_price, confirmed=result.accepted,
    ))


def _adopt(report: ReconcileReport, position: BrokerPosition, plan: Plan, now: datetime) -> None:
    """Rehydrate trail state under the 6.3 ratchet.

    The journal does not persist trail state, so it is reconstructed from the
    one fact the venue keeps: the resting stop. A stop tighter than the plan's
    can only have got there through the trail, so the trail is armed and the
    ratchet level is that stop. The restored stop is then, by construction,
    never looser than what is resting - asserted rather than assumed.
    """
    is_long = position.side is OrderSide.BUY
    resting = position.sl
    restored = max(resting, plan.stop_price) if is_long else min(resting, plan.stop_price)
    if resting > 0:
        looser = restored < resting if is_long else restored > resting
        assert not looser, "restored stop must never be looser than the venue's"
    trail_armed = resting > 0 and (resting > plan.stop_price if is_long else resting < plan.stop_price)
    report.adopted[position.market] = {
        "signal_id": plan.signal_id,
        "ticket": position.ticket,
        "direction": plan.direction.value,
        "entry_underlying": position.entry_price,
        "volume": position.volume,
        "current_stop": restored,
        "target": plan.target_price,
        "trail_activated": trail_armed,
        "extreme_since_entry": position.entry_price,   # unknown; the loop updates it
        "entry_time": (plan.opened_at or now).isoformat(),
        "payload": plan.payload,
    }


def _safe_mode(report: ReconcileReport, broker: BrokerClient, position: BrokerPosition,
               match: Match, cfg: Config, history, market: str) -> None:
    """Step 6: protect what cannot be explained, then pause and page."""
    why = ("no plan in the journal" if match.plan is None
           else f"plan match confidence {match.confidence:.2f} below threshold ({match.how})")
    report.safe_mode[market] = why

    if position.stop_present and position.stop_covers_volume:
        report.actions.append(Action("noop", market, position.ticket,
                                     f"SAFE: resting stop {position.sl} kept", confirmed=True))
        return

    stop = _safe_stop_level(broker, position, cfg, history, market)
    if stop is None:
        report.disagreements.append(
            f"{market} #{position.ticket}: NO PROTECTIVE STOP could be computed - no ATR and "
            f"instruments.{cfg.instrument_key(market)}.safe_mode_fallback_points is unset. "
            f"The position is running on nothing. Set the stop by hand NOW.")
        return

    result = broker.set_position_stop(position.ticket, stop, None)
    report.actions.append(Action(
        "safe_stop_placed", market, position.ticket,
        f"SAFE-mode stop at {stop}: {result.message}",
        before=position.sl or None, after=stop, confirmed=result.accepted,
    ))
    if not result.accepted:
        report.disagreements.append(
            f"{market} #{position.ticket}: SAFE-mode stop could not be placed: {result.message}")


def _safe_stop_level(broker: BrokerClient, position: BrokerPosition, cfg: Config,
                     history, market: str) -> float | None:
    """``entry +/- multiple x ATR``, clamped, or the configured fallback, or None."""
    instrument = cfg.instrument_key(market)
    multiple = float(cfg.get("ops.safe_mode_stop_atr_multiple", 2.0))
    is_long = position.side is OrderSide.BUY
    point = float(cfg.get(f"instruments.{instrument}.point", 0) or 0)
    stops_level = int(cfg.get(f"instruments.{instrument}.stops_level_points", 0) or 0)
    minimum = stops_level * point

    distance: float | None = None
    atr = _atr(history or broker.history, market, cfg)
    if atr is not None and atr > 0:
        distance = multiple * atr
    else:
        fallback = cfg.get(f"instruments.{instrument}.safe_mode_fallback_points", None)
        if fallback is not None and point > 0:
            distance = float(fallback) * point
    if distance is None:
        return None

    distance = max(distance, minimum)
    stop = position.entry_price - distance if is_long else position.entry_price + distance

    # never further from price than a stop that already exists
    if position.stop_present:
        stop = max(stop, position.sl) if is_long else min(stop, position.sl)
    digits = len(f"{point:.10f}".rstrip("0").split(".")[1]) if point else 2
    return round(stop, digits)


def _atr(history, market: str, cfg: Config, period: int = 14) -> float | None:
    """Wilder ATR on the setup timeframe from the routed feed, or None."""
    try:
        timeframe = cfg.timeframes(market)["setup"]
        frame = history(market, timeframe, period * 4)
    except Exception as error:
        logger.warning("SAFE-mode ATR unavailable for %s: %s", market, error)
        return None
    if frame is None or len(frame) < period + 1:
        return None
    high, low, close = frame["high"], frame["low"], frame["close"]
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    value = tr.ewm(alpha=1.0 / period, adjust=False).mean().iloc[-1]
    return float(value) if pd.notna(value) else None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _parse_ts(text) -> datetime | None:
    try:
        return datetime.fromisoformat(str(text))
    except (TypeError, ValueError):
        return None


def _naive(value: datetime) -> datetime:
    return value.replace(tzinfo=None) if value.tzinfo else value
