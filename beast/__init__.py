"""Beast - a rules-bound intraday trading agent.

The single source of truth for every decision in this package is the Soul File at
``beast/soul/BEAST_SOUL_v3.md``. Where any instruction conflicts with it, the Soul File
wins - see PRECEDENCE.md.
"""

from beast.config import Config
from beast.constants import Direction, Market

__all__ = ["Config", "Market", "Direction"]
