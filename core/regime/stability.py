"""Persistence, flicker, uncertainty and the size multiplier.

A raw filtered posterior is not a tradeable signal. It changes every bar, and a
volatility model that reclassifies the market every fifteen minutes would resize
positions for reasons that are in the model, not in the market. This module is
the buffer between the two: it decides when a state has held long enough to be
believed, when the series is changing too fast to believe any of it, and what
multiplier falls out the other side.

The four guards
---------------
* **Confirmation.** A new state must persist ``confirm_bars`` bars before
  ``is_confirmed`` becomes true. Until then the state is reported - it is
  useful to see - but it is sized as uncertain.
* **Confidence.** A posterior below ``min_confidence`` is treated as
  unconfirmed no matter how long it has held. A state the model is not sure
  about is not a state.
* **Flicker.** More than ``flicker_threshold`` changes inside
  ``flicker_window`` bars forces uncertainty mode for as long as it lasts.
* **Staleness.** On inference failure the last confirmed state is held for up to
  ``stale_max_bars``, then the layer emits ``UNKNOWN``. A model bug should not
  silently halt Beast, and should not let it run at full size either.

What this module must never do
------------------------------
Touch direction, setup type, entry price, stop, target, trail, strike selection
or exit logic. Once a position is open, section 6 governs it alone: a flip to
``TURBULENT`` mid-trade does **not** close the position - the stop does. The
only outputs are a multiplier that can shrink the next entry and a veto that can
block it.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from core.config import Config, get_config
from core.regime.contracts import (
    TURBULENT,
    UNKNOWN,
    VolState,
    unknown_state,
)
from core.regime.hmm_engine import VolatilityHMM

LOGGER = logging.getLogger("beast.regime.stability")

#: Sensex carries a 15-minute feed delay (4.6, 12) and is excluded from trigger
#: logic tighter than 5M, so its 15M bias bars are effectively one bar stale.
#: Acceptable for a volatility layer, but every Sensex VolState says so.
DATA_DELAY_MINUTES = {"SENSEX": 15}


@dataclass
class _StateHistory:
    """Rolling record of accepted states, for the flicker and run-length tests."""

    window: int
    values: deque[int] = field(default_factory=deque)

    def push(self, state: int) -> None:
        self.values.append(int(state))
        while len(self.values) > self.window:
            self.values.popleft()

    def changes(self) -> int:
        """How many times the state changed inside the window."""
        items = list(self.values)
        return sum(1 for previous, current in zip(items, items[1:]) if previous != current)

    def flicker_rate(self) -> float:
        """Changes per bar over the window."""
        if len(self.values) < 2:
            return 0.0
        return self.changes() / (len(self.values) - 1)


class StabilityTracker:
    """Turns a stream of filtered posteriors into a stream of ``VolState``.

    One tracker per market, matching one model per market.

    Args:
        market: ``NIFTY50`` | ``SENSEX`` | ``XAUUSD`` and friends.
        config: Injected for tests.

    Attributes:
        last_confirmed: The most recent confirmed state, carried across the
            session boundary when ``carry_across_sessions`` is true.
    """

    def __init__(self, market: str, config: Config | None = None) -> None:
        self.cfg = config or get_config()
        self.market = market.upper()
        section = self.cfg.section("regime")["stability"]
        self.confirm_bars = int(section["confirm_bars"])
        self.flicker_threshold = int(section["flicker_threshold"])
        self.min_confidence = float(section["min_confidence"])
        self.stale_max_bars = int(section["stale_max_bars"])
        self.carry_across_sessions = bool(section["carry_across_sessions"])

        self._history = _StateHistory(int(section["flicker_window"]))
        self._current_state: int | None = None
        self._consecutive = 0
        self._stale_bars = 0
        self._session_id: Any = None
        self.last_confirmed: VolState | None = None

    # -- the main entry point ------------------------------------------------

    def observe(self, bar_ts: datetime, posterior: np.ndarray,
                engine: VolatilityHMM, session_id: Any = None) -> VolState:
        """Fold one bar's filtered posterior into a ``VolState``.

        Args:
            bar_ts: IST close of the bias-TF bar.
            posterior: ``P(state | observations 1..t)`` for this bar.
            engine: The market's model, for the state-to-label map and the
                model version.
            session_id: Session tag for this bar. Used only to detect a session
                boundary; see :meth:`_handle_session_boundary`.

        Returns:
            The state for this bar, with the multiplier and veto already
            resolved.
        """
        self._handle_session_boundary(session_id)
        self._stale_bars = 0

        probabilities = np.asarray(posterior, dtype=float)
        state = int(np.argmax(probabilities))
        confidence = float(probabilities[state])

        if self._current_state is None or state != self._current_state:
            self._current_state = state
            self._consecutive = 1
        else:
            self._consecutive += 1
        self._history.push(state)

        flicker_rate = self._history.flicker_rate()
        is_flickering = self._history.changes() > self.flicker_threshold
        confident = confidence >= self.min_confidence
        persisted = self._consecutive >= self.confirm_bars
        is_confirmed = confident and persisted and not is_flickering

        ladder_label, bucket = engine.label_for(state)
        reasons: list[str] = []
        if not confident:
            reasons.append(f"confidence {confidence:.2f} < {self.min_confidence:.2f}")
        if not persisted:
            reasons.append(f"held {self._consecutive}/{self.confirm_bars} bars")
        if is_flickering:
            reasons.append(
                f"{self._history.changes()} changes in {self._history.window} bars"
            )

        multiplier = self._multiplier_for(bucket, is_confirmed, is_flickering)
        veto = self._veto_for(bucket)

        vol_state = VolState(
            market=self.market,
            bar_ts=bar_ts,
            label=bucket,
            ladder_label=ladder_label,
            bucket_source_state=state,
            probability=confidence,
            state_probabilities={
                index: float(value) for index, value in enumerate(probabilities)
            },
            is_confirmed=is_confirmed,
            consecutive_bars=self._consecutive,
            flicker_rate=flicker_rate,
            is_flickering=is_flickering,
            size_multiplier=multiplier,
            veto=veto,
            reason="; ".join(reasons) if reasons else f"{bucket} confirmed",
            model_version=engine.metadata.model_version if engine.metadata else "none",
            data_delay_minutes=DATA_DELAY_MINUTES.get(self._delay_key(), 0),
            is_stale=False,
        )
        if is_confirmed:
            self.last_confirmed = vol_state
        return vol_state

    def on_failure(self, bar_ts: datetime, reason: str) -> VolState:
        """Fail-safe path: hold the last confirmed state, then give up.

        Args:
            bar_ts: The bar that could not be scored.
            reason: What went wrong - a model-load failure, NaN features, or an
                inference exception.

        Returns:
            The last confirmed state, re-stamped and marked ``is_stale``, for up
            to ``stale_max_bars`` bars. After that, ``UNKNOWN`` at
            ``uncertainty_size_mult`` with ``veto=False``.

        The bounded hold exists because the alternative failure modes are both
        bad: dropping straight to ``UNKNOWN`` on one transient exception throws
        away a perfectly good state, and holding forever means a model that died
        at 09:30 is still sizing trades at 15:00.
        """
        self._stale_bars += 1
        sizing = self.cfg.section("regime")["sizing"]

        if self.last_confirmed is not None and self._stale_bars <= self.stale_max_bars:
            LOGGER.warning(
                "%s: holding %s (stale %d/%d): %s",
                self.market, self.last_confirmed.label,
                self._stale_bars, self.stale_max_bars, reason,
            )
            held = self.last_confirmed
            return VolState(
                market=held.market,
                bar_ts=bar_ts,
                label=held.label,
                ladder_label=held.ladder_label,
                bucket_source_state=held.bucket_source_state,
                probability=held.probability,
                state_probabilities=dict(held.state_probabilities),
                is_confirmed=held.is_confirmed,
                consecutive_bars=held.consecutive_bars,
                flicker_rate=held.flicker_rate,
                is_flickering=held.is_flickering,
                size_multiplier=float(sizing["uncertainty_size_mult"]),
                veto=held.veto,
                reason=f"stale hold {self._stale_bars}/{self.stale_max_bars}: {reason}",
                model_version=held.model_version,
                data_delay_minutes=held.data_delay_minutes,
                is_stale=True,
            )

        LOGGER.warning("%s: vol_state UNKNOWN: %s", self.market, reason)
        return unknown_state(
            market=self.market,
            bar_ts=bar_ts,
            reason=reason,
            size_multiplier=float(sizing["uncertainty_size_mult"]),
            model_version=(
                self.last_confirmed.model_version if self.last_confirmed else "none"
            ),
            data_delay_minutes=DATA_DELAY_MINUTES.get(self._delay_key(), 0),
            veto=bool(self.cfg.get("regime.veto.on_unknown", False)),
        )

    # -- internals -----------------------------------------------------------

    def _delay_key(self) -> str:
        """Normalise the market name for the feed-delay table."""
        if self.market.startswith("SENSEX"):
            return "SENSEX"
        if self.market.startswith("NIFTY"):
            return "NIFTY"
        return "GOLD"

    def _handle_session_boundary(self, session_id: Any) -> None:
        """Carry state across a session boundary, or reset at it.

        With ``carry_across_sessions: true`` the consecutive-bar count continues
        over the boundary and the last confirmed state survives. Resetting to
        ``UNKNOWN`` every morning instead would put the first ``confirm_bars``
        of every session in uncertainty mode - on a 25-bar Indian session that
        is the whole first hour, every day, for no reason the market gave.
        """
        if session_id is None or session_id == self._session_id:
            self._session_id = session_id
            return
        first_session = self._session_id is None
        self._session_id = session_id
        if first_session or self.carry_across_sessions:
            return
        self._current_state = None
        self._consecutive = 0
        self._history = _StateHistory(self._history.window)
        self.last_confirmed = None

    def _multiplier_for(self, bucket: str, is_confirmed: bool,
                        is_flickering: bool) -> float:
        """Resolve the size multiplier for one bar.

        Uncertainty - unconfirmed or flickering - takes
        ``uncertainty_size_mult``. Otherwise the per-bucket multiplier applies.
        Whichever is smaller wins, so an uncertain read of a turbulent market
        never sizes larger than a confirmed one.
        """
        sizing = self.cfg.section("regime")["sizing"]
        table = sizing["size_multiplier"]
        bucket_multiplier = float(table.get(bucket, table[UNKNOWN]))
        if not is_confirmed or is_flickering:
            return min(bucket_multiplier, float(sizing["uncertainty_size_mult"]))
        return bucket_multiplier

    def _veto_for(self, bucket: str) -> bool:
        """The direction-independent half of the veto decision.

        ``on_turbulent_counter_bias`` needs the proposed direction and the 4.4
        regime, neither of which belongs in this module. Gate G10 combines them
        with :func:`veto_for_signal`.
        """
        veto = self.cfg.section("regime")["veto"]
        if bucket == TURBULENT and bool(veto.get("on_turbulent", False)):
            return True
        if bucket == UNKNOWN and bool(veto.get("on_unknown", False)):
            return True
        return False


# ---------------------------------------------------------------------------
# The two functions gate G10 and sizing will call
# ---------------------------------------------------------------------------


def veto_for_signal(vol_state: VolState, is_counter_bias: bool,
                    config: Config | None = None) -> tuple[bool, str]:
    """Whether gate ``G10_VOL_STATE`` blocks this candidate.

    Args:
        vol_state: This bar's state.
        is_counter_bias: Whether the proposed direction opposes the 4.4 regime.

    Returns:
        ``(blocked, reason)``.

    G10 is evaluated **after** G9 and immediately before Emit. Placing it last
    is deliberate: a candidate that already failed G3 must be logged as a G3
    rejection, or section 9's "where signals die" analytics stop meaning
    anything. The volatility layer is the last word, not the first.

    A veto can only ever block. There is no configuration under which this
    function permits a trade the G0-G9 chain rejected.
    """
    cfg = config or get_config()
    if not cfg.regime_enabled():
        return False, "regime layer disabled"
    if vol_state.veto:
        return True, f"G10: vol_state {vol_state.label} vetoes ({vol_state.reason})"
    if (
        is_counter_bias
        and vol_state.label == TURBULENT
        and bool(cfg.get("regime.veto.on_turbulent_counter_bias", False))
    ):
        return True, "G10: counter-bias entry refused in a TURBULENT vol_state"
    return False, f"G10: vol_state {vol_state.label} permits"


def effective_size_factor(vol_factor: float, vol_state: VolState | None,
                          config: Config | None = None) -> tuple[float, str]:
    """Combine section 7's ``vol_factor`` with the volatility layer's multiplier.

    Args:
        vol_factor: ``clamp(ATR_median_20d / ATR_current, floor, 1.0)`` from
            section 7.
        vol_state: This bar's state, or ``None`` when the layer has produced
            nothing yet.

    Returns:
        ``(effective_factor, binding_reason)`` - the reason names which input
        bound, so section 9 can measure whether the layer earned its place.

    Combined by ``min()``, never by product
    ---------------------------------------
    Section 7's ``vol_factor`` and the HMM's ``size_multiplier`` measure largely
    the same quantity. Multiplying them double-counts volatility, and the
    product's floor would be ``0.5 x 0.5 = 0.25`` - silently overriding
    ``risk.vol_factor_floor``, which is a soul-file value. The soul file wins,
    so the combination is ``min()`` and the floor is the one already in
    Appendix A. There is no second floor in the ``regime:`` block.
    """
    cfg = config or get_config()
    floor = float(cfg.get("risk.vol_factor_floor"))

    if vol_state is None or not cfg.regime_enabled():
        effective = max(min(float(vol_factor), 1.0), floor)
        return effective, "vol_factor only (regime layer inactive)"

    if str(cfg.get("regime.sizing.combine_method", "min")) != "min":
        raise ValueError(
            "regime.sizing.combine_method must be 'min'. A product combination "
            "double-counts volatility and breaks risk.vol_factor_floor."
        )

    combined = min(float(vol_factor), float(vol_state.size_multiplier))
    effective = max(combined, floor)

    if effective <= floor and combined < floor:
        reason = f"floored at risk.vol_factor_floor {floor:.2f}"
    elif float(vol_state.size_multiplier) < float(vol_factor):
        reason = f"vol_state {vol_state.label} multiplier {vol_state.size_multiplier:.2f}"
    else:
        reason = f"section 7 vol_factor {float(vol_factor):.2f}"

    assert effective <= 1.0, "effective size factor must never exceed 1.0"
    return effective, reason
