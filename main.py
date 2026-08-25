"""Entry point for regime-trader."""

from __future__ import annotations

import argparse
from typing import Any


def load_config(path: str = "config/settings.yaml") -> dict[str, Any]:
    """Load and return the settings.yaml configuration."""
    raise NotImplementedError


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments (e.g. --backtest, --live, --config)."""
    raise NotImplementedError


def main() -> None:
    """Wire up config, broker, data, core, and monitoring components and start trading."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
