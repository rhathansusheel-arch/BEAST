"""Structured logging setup."""

from __future__ import annotations

import logging


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a configured logger with a consistent format, creating it once per name."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
    return logger
