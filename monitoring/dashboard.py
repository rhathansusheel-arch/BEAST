"""Terminal-based live dashboard (built on rich)."""

from __future__ import annotations

from typing import Any

from broker.position_tracker import PositionTracker


class Dashboard:
    """Renders a live-refreshing terminal dashboard of positions, P&L, and regime state."""

    def __init__(self, position_tracker: PositionTracker, refresh_seconds: int = 5) -> None:
        self.position_tracker = position_tracker
        self.refresh_seconds = refresh_seconds

    def render(self) -> Any:
        """Build the current dashboard layout (rich.Table/Layout)."""
        raise NotImplementedError

    def run(self) -> None:
        """Start the live-refreshing dashboard loop."""
        raise NotImplementedError
