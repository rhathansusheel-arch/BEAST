"""Retry with exponential backoff for broker calls.

A broker API is the one dependency Beast cannot work around: a transient 502 on
a quote lookup is noise, but the same failure on an exit order is a position with
no stop. Retrying is therefore not optional, and neither is bounding it - a call
that retries forever is indistinguishable from a hung process, and the watchdog
would be right to kill it.

Three retries with exponential backoff, jittered. Jitter matters because Beast
evaluates three markets on the same bar boundary: without it, three simultaneous
failures produce three simultaneous retries, then three more, hammering an API
that is already unwell.

What is deliberately *not* retried
----------------------------------
Anything that changes state and might already have succeeded. A submit that
times out may well have reached the exchange, and a blind resend is how one
position becomes two. The soul file's answer to that is idempotency - a
deterministic client order id, so a resend is rejected as a duplicate rather
than filled twice - and that belongs to the ops layer, which is not built yet.
Until it is, :func:`with_retry` is for **reads**: quotes, history, positions,
account. Order submission uses it only where the adapter itself guarantees
idempotency, and :class:`RetryPolicy.idempotent` has to be set explicitly to say
so.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

LOGGER = logging.getLogger("beast.broker.retry")

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """How hard to try, and how long to wait between attempts.

    Attributes:
        attempts: Total attempts including the first. 3 means one call and two
            retries.
        base_delay: Seconds before the first retry.
        multiplier: Backoff factor - 1s, 2s, 4s at the default.
        max_delay: Ceiling, so a long backoff cannot outlive a trigger bar.
        jitter: Fraction of the delay randomised, to desynchronise markets
            that failed on the same bar boundary.
        idempotent: True only when re-issuing the call cannot double an effect.
            Read calls are idempotent. A bare order submission is not.
    """

    attempts: int = 3
    base_delay: float = 1.0
    multiplier: float = 2.0
    max_delay: float = 8.0
    jitter: float = 0.25
    idempotent: bool = True


DEFAULT_POLICY = RetryPolicy()


class BrokerCallFailed(RuntimeError):
    """Every attempt failed.

    Carries the last underlying exception as ``__cause__`` so the traceback the
    operator sees is the broker's, not this wrapper's.
    """

    def __init__(self, description: str, attempts: int) -> None:
        super().__init__(f"{description} failed after {attempts} attempt(s)")
        self.description = description
        self.attempts = attempts


def with_retry(call: Callable[[], T], description: str,
               policy: RetryPolicy = DEFAULT_POLICY,
               sleep: Callable[[float], None] = time.sleep) -> T:
    """Call ``call``, retrying transient failures with exponential backoff.

    Args:
        call: A zero-argument callable. Bind arguments with a lambda or
            ``functools.partial``.
        description: What is being attempted, for the log and the exception.
        policy: Attempts and backoff shape.
        sleep: Injected so tests do not actually wait.

    Returns:
        Whatever ``call`` returned.

    Raises:
        BrokerCallFailed: Every attempt raised. The last exception is attached
            as ``__cause__``.
        ValueError: ``policy.idempotent`` is false and more than one attempt was
            requested. Retrying a non-idempotent call is a bug, not a
            configuration choice, so it fails at the call site rather than
            quietly resending an order.
    """
    if not policy.idempotent and policy.attempts > 1:
        raise ValueError(
            f"refusing to retry non-idempotent call {description!r}: a resend "
            f"after an ambiguous failure can double a position. Give the call a "
            f"deterministic client order id first."
        )

    last: BaseException | None = None
    delay = policy.base_delay

    for attempt in range(1, policy.attempts + 1):
        try:
            return call()
        except Exception as error:
            last = error
            if attempt >= policy.attempts:
                break
            wait = min(delay, policy.max_delay)
            wait *= 1.0 + random.uniform(-policy.jitter, policy.jitter)
            LOGGER.warning(
                "%s failed (attempt %d/%d): %s - retrying in %.1fs",
                description, attempt, policy.attempts, error, wait,
            )
            sleep(max(0.0, wait))
            delay *= policy.multiplier

    LOGGER.error("%s failed after %d attempts: %s", description, policy.attempts, last)
    raise BrokerCallFailed(description, policy.attempts) from last


def try_call(call: Callable[[], T], description: str, default: T,
             policy: RetryPolicy = DEFAULT_POLICY,
             sleep: Callable[[float], None] = time.sleep) -> T:
    """:func:`with_retry`, but returning ``default`` instead of raising.

    For calls whose failure Beast can carry on without - a quote used for a
    dashboard line, an optional account refresh. Never for a call whose result
    gates a trading decision: there, the failure has to reach the caller so the
    decision can be refused rather than made on a default.
    """
    try:
        return with_retry(call, description, policy, sleep)
    except BrokerCallFailed:
        return default
