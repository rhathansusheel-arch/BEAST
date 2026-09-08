"""Alerting for the conditions the soul file says the operator must hear about.

Section 11: alerts are delivered the same way as signals - short, factual, no
padding. The alert-worthy events are enumerated by the document rather than
invented here:

* ``HIGH SPREAD`` (4.6, 5.6) - XAUUSD spread above the ceiling.
* ``DATA STALE`` (4.6, 6.8) - the feed went stale, with the last known price,
  position and stop. Beast does not guess.
* ``SENSEX DELAY`` (12) - appended to every Sensex signal.
* ``NEWS BLACKOUT`` (5.6) - entries suppressed around a scheduled release.
* ``LOSS LIMIT PAUSE`` (7, 10) - the market is paused for the session.
* ``OVERRIDE`` (8) - a rule deviation was accepted.
* ``CONFIG BLOCKER`` (3.1, 5.7.3) - an unset threshold is refusing trades.

Beyond those, seven operational conditions are alerted because an operator who
does not know about them cannot act on the ones above:

* ``VOL_STATE_CHANGE`` - a **confirmed** volatility-state transition.
* ``REGIME_CHANGE`` - a section 4.4 ``TREND_UP/TREND_DOWN/RANGE`` transition.
* ``CIRCUIT_BREAKER`` - any condition now stopping new entries.
* ``LARGE_PNL`` - the day's realised P&L crossed a fraction of the daily cap.
* ``FEED_DOWN`` - the bar feed stopped answering.
* ``API_LOST`` - a broker session dropped.
* ``MODEL_RETRAINED`` - a volatility model was refitted.
* ``FLICKER_EXCEEDED`` - the volatility state is changing too fast to believe.

Rate limiting exists so a persistent condition - a wide spread that lasts an hour -
produces one alert rather than seven hundred. The limit is one per (kind, market)
per ``monitoring.alert_rate_limit_minutes`` (15), so a spread alert never
suppresses a loss-limit pause, and a Nifty alert never suppresses the same kind
on Gold.

Critical kinds bypass the limit entirely. A circuit breaker, a lost broker
session and a dead feed are each things where the *second* occurrence is as
important as the first, because it means the condition did not clear.
"""

from __future__ import annotations

import json
import logging
import smtplib
import ssl
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.message import EmailMessage
from enum import Enum
from typing import Any

from core.config import Config, get_config
from monitoring.logger import log_alert

logger = logging.getLogger("beast.alerts")


class AlertKind(str, Enum):
    """Alert categories, matching the soul file's named conditions."""

    HIGH_SPREAD = "HIGH_SPREAD"
    DATA_STALE = "DATA_STALE"
    SENSEX_DELAY = "SENSEX_DELAY"
    NEWS_BLACKOUT = "NEWS_BLACKOUT"
    LOSS_LIMIT_PAUSE = "LOSS_LIMIT_PAUSE"
    OVERRIDE = "OVERRIDE"
    CONFIG_BLOCKER = "CONFIG_BLOCKER"
    SIGNAL = "SIGNAL"
    TRADE_CLOSED = "TRADE_CLOSED"
    ERROR = "ERROR"

    # -- operational conditions ---------------------------------------------
    # Not named by the soul file, but each is a state the operator has to know
    # about to act on the ones that are.
    VOL_STATE_CHANGE = "VOL_STATE_CHANGE"
    REGIME_CHANGE = "REGIME_CHANGE"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"
    LARGE_PNL = "LARGE_PNL"
    FEED_DOWN = "FEED_DOWN"
    API_LOST = "API_LOST"
    MODEL_RETRAINED = "MODEL_RETRAINED"
    FLICKER_EXCEEDED = "FLICKER_EXCEEDED"

    @property
    def critical(self) -> bool:
        """Critical alerts bypass rate limiting.

        A loss-limit pause, an accepted override and a stale feed while in
        position are all things the operator must see the moment they happen.
        """
        return self in (
            AlertKind.LOSS_LIMIT_PAUSE,
            AlertKind.OVERRIDE,
            AlertKind.DATA_STALE,
            AlertKind.ERROR,
            AlertKind.CIRCUIT_BREAKER,
            AlertKind.FEED_DOWN,
            AlertKind.API_LOST,
        )


@dataclass
class Alert:
    """One alert."""

    kind: AlertKind
    market: str
    message: str
    at: datetime
    payload: dict[str, Any] | None = None

    def format(self) -> str:
        """Render in the section 11 tone: short, factual, no padding."""
        return f"[{self.kind.value}] {self.market}: {self.message}"


class AlertManager:
    """Dispatches alerts to the console, email and a webhook.

    Args:
        config: Injected for tests.
        transport: Optional callable ``(Alert) -> None`` replacing every
            outbound channel - used by tests and by the dashboard, which renders
            alerts itself rather than printing them.
    """

    def __init__(self, config: Config | None = None, transport=None) -> None:
        self.cfg = config or get_config()
        self.transport = transport
        self.history: list[Alert] = []
        self._last_sent: dict[tuple[str, str], datetime] = {}
        self.rate_limit = timedelta(
            minutes=int(self.cfg.get("monitoring.alert_rate_limit_minutes"))
        )

    # -- dispatch ------------------------------------------------------------

    def send(self, kind: AlertKind, market: str, message: str,
             at: datetime | None = None, payload: dict[str, Any] | None = None) -> bool:
        """Emit an alert, subject to rate limiting.

        Returns:
            True when the alert was dispatched, False when it was suppressed as
            a duplicate inside the rate-limit window.
        """
        at = at or datetime.now()
        alert = Alert(kind, market, message, at, payload)
        self.history.append(alert)

        key = (kind.value, market)
        last = self._last_sent.get(key)
        if not kind.critical and last is not None and at - last < self.rate_limit:
            return False
        self._last_sent[key] = at

        if self.transport is not None:
            self.transport(alert)
            return True

        # Routed through log_alert so the record carries event="alert" and lands
        # in alerts.log as well as main.log, with the runtime context attached.
        log_alert(
            alert.format(),
            {"kind": kind.value, "market": market, **(payload or {})},
        )
        self._email(alert)
        self._webhook(alert)
        return True

    # -- convenience wrappers -------------------------------------------------

    def high_spread(self, market: str, spread: float, ceiling: float) -> None:
        """4.6 / 5.6: no entry, alert instead, wait for normalisation."""
        self.send(
            AlertKind.HIGH_SPREAD, market,
            f"HIGH SPREAD ALERT - {spread:.2f} above the {ceiling:.2f} ceiling. "
            f"No new entries. Open positions unaffected; exits still honour their triggers.",
        )

    def data_stale(self, market: str, last_price: float, position: str | None,
                   stop: float | None) -> None:
        """4.6 / 6.8: alert with last known price, position and stop; do not guess."""
        detail = f"last known price {last_price:.2f}"
        if position:
            detail += f"; position {position}, stop {stop:.2f}" if stop else f"; position {position}"
        else:
            detail += "; no open position"
        self.send(
            AlertKind.DATA_STALE, market,
            f"FEED STALE - new entries suppressed. {detail}. Beast does not guess a price.",
        )

    def loss_limit_pause(self, family: str, reason: str) -> None:
        """7 / 10: auto-pause for the remainder of the session."""
        self.send(
            AlertKind.LOSS_LIMIT_PAUSE, family.upper(),
            f"PAUSED for the session - {reason}. No new entries in this market. "
            f"Open positions continue to be managed to their exits.",
        )

    def config_blocker(self, keys: list[str]) -> None:
        """3.1 / 5.7.3: an unset threshold is refusing trades."""
        if not keys:
            return
        self.send(
            AlertKind.CONFIG_BLOCKER, "CONFIG",
            "Unset config values are blocking trades: " + ", ".join(keys)
            + ". An unset threshold is treated as failing, not passing.",
        )

    # -- operational conditions ----------------------------------------------

    def vol_state_change(self, market: str, previous: str, current: str,
                         probability: float, consecutive_bars: int) -> None:
        """A **confirmed** volatility-state transition.

        Only confirmed transitions are alerted. An unconfirmed state changes
        with the posterior and would page the operator several times an hour
        for a market that never actually moved regime - which is precisely the
        flicker the stability layer exists to absorb, and re-emitting it as an
        alert would undo that work.
        """
        self.send(
            AlertKind.VOL_STATE_CHANGE, market,
            f"vol_state {previous} -> {current} "
            f"(p={probability:.2f}, held {consecutive_bars} bars). "
            f"Position sizing for new entries changes; open positions are "
            f"unaffected - section 6 governs those.",
            payload={"from": previous, "to": current, "probability": probability},
        )

    def regime_change(self, market: str, previous: str, current: str,
                      adx: float) -> None:
        """A section 4.4 regime transition - this one changes permitted setups."""
        self.send(
            AlertKind.REGIME_CHANGE, market,
            f"regime {previous} -> {current} (ADX {adx:.1f}). "
            f"Permitted setups change per 4.4.",
            payload={"from": previous, "to": current, "adx": adx},
        )

    def circuit_breaker(self, market: str, reason: str,
                        tripped: bool = True) -> None:
        """Entries stopped, or resumed. Open positions are never affected.

        Sent on both edges. An operator told entries stopped and never told they
        resumed will assume Beast is still halted, and will either intervene
        unnecessarily or stop trusting the alert.
        """
        if tripped:
            message = (
                f"CIRCUIT BREAKER - no new entries: {reason}. "
                f"Open positions continue to be managed; their stops are live."
            )
        else:
            message = f"circuit breaker cleared: {reason}. Entries resume."
        self.send(AlertKind.CIRCUIT_BREAKER, market, message,
                  payload={"reason": reason, "tripped": tripped})

    def large_pnl(self, family: str, realised: float, cap_amount: float,
                  fraction: float) -> None:
        """The day's realised P&L crossed a fraction of the daily loss cap.

        Expressed against the cap rather than as an absolute number, because
        "down 45,000" means nothing without knowing that the cap is 75,000. The
        useful question is how much room is left before section 7 stops the
        session, and that is what this says.
        """
        remaining = max(0.0, cap_amount + realised)
        self.send(
            AlertKind.LARGE_PNL, family.upper(),
            f"day P&L {realised:+,.0f} - {abs(fraction):.0%} of the "
            f"{cap_amount:,.0f} daily cap. {remaining:,.0f} of room left.",
            payload={"realised": realised, "cap": cap_amount, "fraction": fraction},
        )

    def feed_down(self, market: str, cycles: int, has_position: bool) -> None:
        """The bar feed stopped answering.

        Severity depends entirely on whether a position is open. With no
        position this is an inconvenience: entries pause, nothing is at risk.
        With one, the in-process stop evaluation has lost its input and the
        resting stop at the broker is the only protection left - which the
        operator has to be told explicitly rather than left to infer.
        """
        tail = (
            " AN OPEN POSITION IS AFFECTED: in-process stop evaluation has no "
            "input. The resting stop at the broker is the only protection until "
            "the feed returns."
            if has_position else
            " No position open; entries are paused until it returns."
        )
        self.send(
            AlertKind.FEED_DOWN, market,
            f"data feed down for {cycles} cycles.{tail}",
            payload={"cycles": cycles, "has_position": has_position},
        )

    def api_lost(self, broker: str, detail: str = "") -> None:
        """A broker session dropped."""
        self.send(
            AlertKind.API_LOST, broker.upper(),
            f"broker session lost{': ' + detail if detail else ''}. "
            f"Trades routed to it are refused until it reconnects.",
            payload={"broker": broker, "detail": detail},
        )

    def api_restored(self, broker: str, latency_ms: float | None = None) -> None:
        """A broker session came back."""
        tail = f" ({latency_ms:.0f}ms)" if latency_ms is not None else ""
        self.send(
            AlertKind.API_LOST, broker.upper(),
            f"broker session restored{tail}.",
            payload={"broker": broker, "restored": True},
        )

    def model_retrained(self, market: str, model_version: str, n_states: int,
                        n_samples: int, margin_is_noise: bool) -> None:
        """A volatility model was refitted.

        Worth an alert because it is the one thing that changes ``vol_state``
        without the market changing. A state that shifts the morning after a
        retrain shifted because the model did.
        """
        caveat = (
            " BIC margin was noise - the state count was not really chosen by "
            "the data."
            if margin_is_noise else ""
        )
        self.send(
            AlertKind.MODEL_RETRAINED, market,
            f"volatility model retrained: {model_version}, {n_states} states "
            f"over {n_samples:,} bias-TF bars.{caveat}",
            payload={
                "model_version": model_version, "n_states": n_states,
                "n_samples": n_samples, "bic_margin_is_noise": margin_is_noise,
            },
        )

    def flicker_exceeded(self, market: str, changes: int, window: int,
                         threshold: int) -> None:
        """The volatility state is changing too fast to be believed.

        Not an error. The layer has already fallen back to the uncertainty
        multiplier by the time this is sent - the alert exists so the operator
        knows *why* size dropped, rather than finding smaller positions and no
        explanation.
        """
        self.send(
            AlertKind.FLICKER_EXCEEDED, market,
            f"vol_state flickering - {changes} changes in {window} bars "
            f"(threshold {threshold}). Sizing is in uncertainty mode until it "
            f"settles.",
            payload={"changes": changes, "window": window, "threshold": threshold},
        )

    def override(self, market: str, action: str, hypothetical_r: float | None) -> None:
        """8: every accepted override is logged as a rule deviation."""
        tail = (
            f" Plan-to-target outcome would have been {hypothetical_r:+.2f}R."
            if hypothetical_r is not None
            else ""
        )
        self.send(
            AlertKind.OVERRIDE, market,
            f"OVERRIDE ACCEPTED - {action}. Logged as a rule deviation.{tail}",
        )

    # -- channels ------------------------------------------------------------

    def _email(self, alert: Alert) -> None:
        """Send by SMTP when configured. Failures are logged, never raised."""
        settings = self.cfg.get("monitoring.alerts")
        if not settings.get("email_enabled"):
            return
        recipient = settings.get("email_to")
        host = settings.get("smtp_host")
        if not recipient or not host:
            logger.error("Email alerts enabled but email_to or smtp_host is unset")
            return

        message = EmailMessage()
        message["Subject"] = f"Beast {alert.kind.value} - {alert.market}"
        message["From"] = settings.get("email_from") or recipient
        message["To"] = recipient
        message.set_content(f"{alert.at:%Y-%m-%d %H:%M:%S IST}\n\n{alert.format()}")

        try:
            with smtplib.SMTP(host, int(settings.get("smtp_port", 587)), timeout=10) as server:
                server.starttls(context=ssl.create_default_context())
                user = self.cfg.credential("alerts.smtp_user", "BEAST_SMTP_USER")
                password = self.cfg.credential("alerts.smtp_password", "BEAST_SMTP_PASSWORD")
                if user and password:
                    server.login(user, password)
                server.send_message(message)
        except Exception as error:
            logger.error("Email alert failed: %s", error)

    def _webhook(self, alert: Alert) -> None:
        """POST to a webhook when configured. Failures are logged, never raised."""
        settings = self.cfg.get("monitoring.alerts")
        if not settings.get("webhook_enabled"):
            return
        url = self.cfg.credential("alerts.webhook_url", "BEAST_ALERT_WEBHOOK_URL") or settings.get(
            "webhook_url"
        )
        if not url:
            logger.error("Webhook alerts enabled but no URL is configured")
            return
        try:
            import requests

            requests.post(
                url,
                data=json.dumps(
                    {
                        "kind": alert.kind.value,
                        "market": alert.market,
                        "message": alert.message,
                        "at": alert.at.isoformat(),
                        "payload": alert.payload,
                    },
                    default=str,
                ),
                headers={"Content-Type": "application/json"},
                timeout=5,
            )
        except Exception as error:
            logger.error("Webhook alert failed: %s", error)

    # -- queries -------------------------------------------------------------

    def recent(self, limit: int = 20) -> list[Alert]:
        """The most recent alerts, newest last - used by the dashboard."""
        return self.history[-limit:]
