"""The Claude reasoning/narration layer.

**This layer never makes a trading decision.** Every entry, exit, size and
rejection is produced by the deterministic pipeline in ``core/signal_generator.py``
and ``core/exit_manager.py``. Claude's job here is to explain what the rules
decided, in the strict-risk-manager tone section 11 specifies, and to write the
weekly review section 9 asks for.

That separation is not stylistic. The soul file's value comes from *not*
deviating - "I trade the plan, not the feeling" - and a model that could veto or
create trades would be a source of exactly the discretionary drift sections 8 and
13 exist to prevent. So:

* The analyst is called **after** a signal is fully formed, with the signal as
  input. It cannot change any field.
* Every call is wrapped so that an API failure, a refusal, or a timeout degrades
  to the deterministic reason line. Beast must keep trading when Claude is down.
* ``ai.advisory_only`` is validated at startup and must be ``true``.

Model: ``claude-opus-5`` with adaptive thinking. Effort is configurable and
defaults to ``low`` - narration is a formatting task, not a reasoning-heavy one,
and the weekly review raises it deliberately.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from core.config import Config, get_config
from core.schemas import Signal

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:  # pragma: no cover - Beast runs fine without the narration layer
    anthropic = None  # type: ignore[assignment]
    ANTHROPIC_AVAILABLE = False

logger = logging.getLogger("beast.ai")

SYSTEM_PROMPT = """You are the narration layer for Beast, a rules-bound intraday \
trading agent that trades Nifty 50 and Sensex as long index options and XAUUSD as \
futures.

Your role is strictly explanatory. The trading decision has already been made by a \
deterministic rule engine before you are called. You must never suggest a different \
entry, stop, target, size or direction, never say a trade should or should not have \
been taken, and never add analysis that could be read as a recommendation. If the \
data looks wrong to you, say so as an observation and stop there.

Tone: strict risk manager. Calm, factual, unemotional. No hype on wins, no \
self-flagellation on losses. No padding, no preamble, no filler openers. Prefer \
plain sentences over lists unless the content is genuinely a list.

Vocabulary you must use correctly:
- All analysis is on the UNDERLYING (index points / gold price). Option premium is \
never used for analysis - only for execution and the premium hard stop.
- 1R is the entry-to-stop distance, fixed at entry.
- The confluence engine reads six indicators (ADX+DI, Stochastic, MACD, RSI, \
Bollinger Bands, session VWAP) in one of two modes: Trend-Continuation for setups \
1, 3 and 4, Reversal for setup 2.
- Option-chain data (OI walls, PCR, IV) is context and contributes levels; it never \
adds to the confluence count."""


@dataclass
class AIResponse:
    """Result of one analyst call.

    Attributes:
        text: The generated text, or the deterministic fallback.
        used_model: False when the fallback was used - the caller can log which.
        detail: Why the fallback was used, when it was.
    """

    text: str
    used_model: bool = True
    detail: str = ""


class AIAnalyst:
    """Thin, failure-tolerant wrapper over the Claude Messages API.

    Args:
        config: Injected for tests.
        client: Injected Anthropic client, for tests and for callers that want to
            share one client across the process.
    """

    def __init__(self, config: Config | None = None, client: Any = None) -> None:
        self.cfg = config or get_config()
        self.enabled = bool(self.cfg.get("ai.enabled", True))
        self.model = str(self.cfg.get("ai.model"))
        self.max_tokens = int(self.cfg.get("ai.max_tokens"))
        self.effort = str(self.cfg.get("ai.effort", "low"))
        self._client = client
        self._unavailable_reason = ""

        if not bool(self.cfg.get("ai.advisory_only", True)):
            raise RuntimeError(
                "ai.advisory_only is false. The AI layer must never be in the "
                "decision path (soul file section 13)."
            )

    # -- client --------------------------------------------------------------

    @property
    def client(self) -> Any:
        """Lazily construct the Anthropic client.

        Credentials resolve from the environment (``ANTHROPIC_API_KEY``) or an
        ``ant auth login`` profile; nothing is hardcoded.
        """
        if self._client is not None:
            return self._client
        if not ANTHROPIC_AVAILABLE:
            self._unavailable_reason = "the anthropic package is not installed"
            return None
        try:
            self._client = anthropic.Anthropic()
        except Exception as error:
            self._unavailable_reason = f"client construction failed: {error}"
            return None
        return self._client

    def available(self) -> bool:
        """True when narration can be attempted."""
        return self.enabled and self.client is not None

    # -- public API ----------------------------------------------------------

    def narrate_signal(self, signal: Signal, context_note: str = "") -> AIResponse:
        """Expand a signal's deterministic reason line into two or three sentences.

        The deterministic ``signal.reason_line`` is always the fallback, so a
        failure here costs formatting, never information.
        """
        if not self.enabled or not bool(self.cfg.get("ai.narrate_signals", True)):
            return AIResponse(signal.reason_line, False, "narration disabled")

        payload = signal.to_dict()
        prompt = (
            "Here is a signal the rule engine has already emitted. Write two or three "
            "sentences for the operator explaining, in order: what the underlying plan "
            "is and why it qualified, then what was bought to express it and what the "
            "binding constraint on size was. State the underlying plan first and the "
            "instrument second. Do not evaluate whether the trade is a good idea.\n\n"
            f"Deterministic one-liner (authoritative, do not contradict):\n{signal.reason_line}\n\n"
            f"Full signal record:\n{json.dumps(payload, indent=2, default=str)}"
        )
        if context_note:
            prompt += f"\n\nAdditional context: {context_note}"

        return self._call(prompt, fallback=signal.reason_line, effort=self.effort)

    def weekly_review(self, summary: dict[str, Any], override_summary: str,
                      fallback_text: str) -> AIResponse:
        """Write the section 9 weekly review from the computed statistics.

        The numbers are computed by ``core/learning.py``; Claude turns them into
        prose and highlights which of them the operator should act on. It must
        not invent numbers, and the prompt says so.
        """
        if not self.enabled or not bool(self.cfg.get("ai.weekly_review", True)):
            return AIResponse(fallback_text, False, "weekly review disabled")

        prompt = (
            "Write the weekly performance review for the operator. Use only the numbers "
            "given; do not estimate, extrapolate or invent any figure. Cover: overall win "
            "rate and average R, the best and worst setup type, where signals are dying "
            "by gate (G5 means confluence is rejecting them, G7 means structure is too "
            "wide, G8 means no tradable strike), the override count and what it cost, and "
            "any setup currently raised to 5-of-6 on negative expectancy. If the option "
            "diagnostics show underlying R consistently ahead of premium R, say plainly "
            "that the leak is in strike selection rather than in the analysis.\n\n"
            "Six to twelve sentences. No headings unless they genuinely help.\n\n"
            f"Statistics:\n{json.dumps(summary, indent=2, default=str)}\n\n"
            f"Override summary:\n{override_summary}"
        )
        # The review reasons over a week of statistics; it earns more effort than
        # per-signal narration does.
        return self._call(prompt, fallback=fallback_text, effort="medium", stream=True)

    def explain_rejections(self, histogram: dict[str, int], fallback_text: str) -> AIResponse:
        """Summarise where signals are dying, for the dashboard footer."""
        if not self.enabled:
            return AIResponse(fallback_text, False, "ai disabled")
        prompt = (
            "These are today's entry-pipeline rejections by gate ID. In at most three "
            "sentences, say where signals are dying and what that implies about whether "
            "the filters are too strict or the market simply is not offering setups. Do "
            "not recommend changing any threshold.\n\n"
            "G0 session, G1 data integrity, G2 no-trade conditions, G3 regime, G4 setup, "
            "G5 confluence, G6 trigger, G7 viability, G8 instrument, G9 risk.\n\n"
            f"{json.dumps(histogram, indent=2)}"
        )
        return self._call(prompt, fallback=fallback_text, effort="low")

    # -- transport -----------------------------------------------------------

    def _call(self, prompt: str, fallback: str, effort: str,
              stream: bool = False) -> AIResponse:
        """Make one Messages API call, degrading to ``fallback`` on any failure.

        Uses adaptive thinking and server-side refusal fallbacks. If the beta
        parameters are rejected by the endpoint, the call is retried once without
        them - a narration layer must not be the thing that breaks on an API
        change while the market is open.
        """
        if not self.available():
            return AIResponse(fallback, False, self._unavailable_reason or "ai unavailable")

        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": effort},
        }

        for attempt, extra in enumerate(
            (
                {
                    "betas": ["server-side-fallback-2026-07-01"],
                    "fallbacks": "default",
                },
                {},
            )
        ):
            try:
                payload = {**request, **extra}
                messages = self.client.beta.messages if extra else self.client.messages
                if stream:
                    with messages.stream(**payload) as handle:
                        message = handle.get_final_message()
                else:
                    message = messages.create(**payload)
                return self._read(message, fallback)
            except Exception as error:
                if attempt == 0:
                    logger.debug("AI call retrying without beta params: %s", error)
                    continue
                logger.warning("AI narration unavailable, using deterministic text: %s", error)
                return AIResponse(fallback, False, str(error))

        return AIResponse(fallback, False, "exhausted AI call attempts")

    @staticmethod
    def _read(message: Any, fallback: str) -> AIResponse:
        """Extract text from a response, handling the refusal stop reason.

        ``stop_reason == "refusal"`` arrives as an HTTP 200, so it must be
        checked before reading content.
        """
        stop_reason = getattr(message, "stop_reason", None)
        if stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            return AIResponse(fallback, False, f"model declined ({category})")

        parts = [
            block.text
            for block in getattr(message, "content", [])
            if getattr(block, "type", None) == "text"
        ]
        text = "\n".join(part for part in parts if part).strip()
        if not text:
            return AIResponse(fallback, False, "empty response")
        return AIResponse(text, True)
