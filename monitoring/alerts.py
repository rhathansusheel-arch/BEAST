"""Email/webhook alerts for critical events."""

from __future__ import annotations

from typing import Any


class AlertManager:
    """Sends rate-limited alerts via email and/or webhook."""

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Args:
            config: The `monitoring` section of settings.yaml, including
                `alert_rate_limit_minutes`.
        """
        self.config = config
        self._last_sent: dict[str, float] = {}

    def send(self, message: str, severity: str = "info") -> None:
        """Send an alert, respecting the configured rate limit per message key."""
        raise NotImplementedError

    def _is_rate_limited(self, key: str) -> bool:
        """Check whether an alert with this key was sent within the rate-limit window."""
        raise NotImplementedError
