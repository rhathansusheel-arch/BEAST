"""The terminal dashboard.

Six panels, refreshed every ``monitoring.dashboard_refresh_seconds`` (5)::

    +- VOLATILITY & REGIME --------------------------------------+
    | NIFTY50  CALM (72%) | confirmed 14 bars | flicker 1/20      |
    |          regime TREND_UP (ADX 27.4) | size x1.00            |
    +- PORTFOLIO ------------------------------------------------+
    | Equity Rs 5,05,230 | Day +Rs 340 (+0.07%)                   |
    | Open risk 3.0% of capital | Positions 1/3 indian, 0/1 gold  |
    +- POSITIONS ------------------------------------------------+
    | NIFTY50 | LONG | 24,180 | +1.2% | stop 24,130 | 3h          |
    +- RECENT SIGNALS -------------------------------------------+
    | 14:30 | NIFTY50 | Setup 1 LONG | 5/6 confluence            |
    +- RISK STATUS ----------------------------------------------+
    | Day loss  0.3% / 15%  [####------] | Losses 0/3            |
    +- SYSTEM ---------------------------------------------------+
    | Data OK | zerodha OK 23ms | model 2d | PAPER                |
    +------------------------------------------------------------+

The dashboard is a pure view. It reads state handed to :meth:`update` and never
queries the engine, so a slow terminal cannot delay an exit.

On the numbers it shows, and the ones it does not
-------------------------------------------------
* **Volatility state, not direction.** ``CALM | NORMAL | TURBULENT | UNKNOWN``,
  shown beside - never merged with - section 4.4's ``TREND_UP | TREND_DOWN |
  RANGE``. Two concepts, two names, on two lines. A single "regime" field
  showing a directional label on a volatility model is how a volatility read
  ends up being traded as a bias.
* **Open risk, not allocation or leverage.** Beast sizes each trade at a fixed
  % of capital and holds long options and futures outright. It has no target
  allocation to display and nothing to rebalance towards, and it cannot be
  levered - Immutable Rule 10 permits long CE/PE only. What matters, and what
  is shown, is how much capital is at risk right now against section 7's caps.
* **Daily loss against the section 7 cap, not drawdown from peak.** Section 7
  defines exactly two session-stopping conditions: the per-instrument daily loss
  cap and three consecutive losses. A peak-drawdown limit is not in the document,
  and displaying one would imply a rule that does not exist and that nothing in
  the engine enforces.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from core.config import Config, get_config

try:
    from rich.console import Console, Group
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.live import Live
    RICH_AVAILABLE = True
except ImportError:  # pragma: no cover - the plain renderer covers this
    RICH_AVAILABLE = False

#: Thresholds for the risk bars, as a fraction of the limit consumed.
CAUTION_AT = 0.50
DANGER_AT = 0.80

#: Colours per volatility bucket. UNKNOWN is yellow rather than red: it means
#: the model could not answer, which reduces size but is not itself a hazard.
VOL_COLOURS = {
    "CALM": "green",
    "NORMAL": "cyan",
    "TURBULENT": "red",
    "UNKNOWN": "yellow",
}

REGIME_COLOURS = {
    "TREND_UP": "green",
    "TREND_DOWN": "red",
    "RANGE": "yellow",
}


def risk_bar(used: float, limit: float, width: int = 10) -> tuple[str, str]:
    """Render a consumption bar and pick its colour.

    Args:
        used: How much of the limit is consumed, in the limit's units.
        limit: The limit. Zero or negative yields an empty bar rather than a
            division error - an unset limit is not a full one.

    Returns:
        ``(bar, colour)``. Colour steps green -> yellow -> red at
        :data:`CAUTION_AT` and :data:`DANGER_AT`.
    """
    if limit <= 0:
        return "-" * width, "dim"
    fraction = max(0.0, min(1.0, abs(used) / limit))
    filled = int(round(fraction * width))
    bar = "#" * filled + "-" * (width - filled)
    if fraction >= DANGER_AT:
        return bar, "red"
    if fraction >= CAUTION_AT:
        return bar, "yellow"
    return bar, "green"


def _money(value: float) -> str:
    """Format in rupees. Beast trades Indian index options and MCX/COMEX gold."""
    return f"Rs {value:,.0f}"


class Dashboard:
    """Renders live state to the terminal.

    Args:
        config: Injected for tests.
    """

    def __init__(self, config: Config | None = None) -> None:
        self.cfg = config or get_config()
        self.refresh_seconds = int(self.cfg.get("monitoring.dashboard_refresh_seconds"))
        self.console = Console() if RICH_AVAILABLE else None
        self._live: Any = None
        self.state: dict[str, Any] = {
            "now": datetime.now(),
            "capital": float(self.cfg.get("risk.capital")),
            "day_pnl": 0.0,
            "markets": {},
            "positions": [],
            "gates": {},
            "signals": [],
            "alerts": [],
            "blockers": [],
            "risk": {},
            "system": {},
        }

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Take over the terminal."""
        if not RICH_AVAILABLE:
            return
        self._live = Live(
            self.render(), console=self.console,
            refresh_per_second=max(1, 1 // max(1, self.refresh_seconds)) or 1,
            screen=False,
        )
        self._live.start()

    def stop(self) -> None:
        """Release the terminal."""
        if self._live is not None:
            self._live.stop()
            self._live = None

    def update(self, **state: Any) -> None:
        """Merge new state and redraw."""
        self.state.update(state)
        self.state["now"] = state.get("now", datetime.now())
        if self._live is not None:
            self._live.update(self.render())

    # -- rendering -----------------------------------------------------------

    def render(self) -> Any:
        """Build the full layout."""
        if not RICH_AVAILABLE:
            return self.render_plain()
        return Group(
            self._volatility_panel(),
            self._portfolio_panel(),
            self._positions_panel(),
            self._signals_panel(),
            self._risk_panel(),
            self._system_panel(),
        )

    def _volatility_panel(self) -> Any:
        """Volatility state and section 4.4 regime, side by side, never merged."""
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold", width=10)
        table.add_column()

        markets = self.state.get("markets", {})
        if not markets:
            table.add_row("-", Text("no market evaluated yet", style="dim"))

        for market, info in markets.items():
            label = str(info.get("vol_label", "UNKNOWN"))
            colour = VOL_COLOURS.get(label, "white")
            confirmed = info.get("vol_confirmed")
            probability = float(info.get("vol_probability") or 0.0)
            bars = int(info.get("vol_bars") or 0)
            changes = int(info.get("vol_flicker_changes") or 0)
            window = int(info.get("vol_flicker_window") or 0)
            multiplier = float(info.get("size_multiplier") or 1.0)

            line = Text()
            line.append(f"{label}", style=f"bold {colour}")
            line.append(f" ({probability:.0%})", style=colour)
            line.append(" | ")
            line.append(
                f"{'confirmed' if confirmed else 'unconfirmed'} {bars} bars",
                style="white" if confirmed else "yellow",
            )
            line.append(" | ")
            flickering = bool(info.get("vol_flickering"))
            line.append(
                f"flicker {changes}/{window}",
                style="red" if flickering else "dim",
            )
            line.append(f" | size x{multiplier:.2f}",
                        style="yellow" if multiplier < 1.0 else "dim")
            if info.get("data_delay_minutes"):
                line.append(f" | feed {info['data_delay_minutes']}m behind",
                            style="yellow")
            table.add_row(market, line)

            regime = str(info.get("regime", "-"))
            regime_line = Text()
            regime_line.append("regime ", style="dim")
            regime_line.append(regime, style=REGIME_COLOURS.get(regime, "dim"))
            if info.get("adx") is not None:
                regime_line.append(f" (ADX {float(info['adx']):.1f})", style="dim")
            entries = str(info.get("entries", "allowed"))
            regime_line.append(
                f" | entries {entries}",
                style="red" if entries != "allowed" else "dim",
            )
            table.add_row("", regime_line)

        return Panel(table, title="VOLATILITY & REGIME", title_align="left",
                     border_style="blue")

    def _portfolio_panel(self) -> Any:
        """Equity, the day's P&L, and how much capital is actually at risk."""
        capital = float(self.state.get("capital", 0.0))
        risk_rows = self.state.get("risk", {}) or {}
        day_pnl = sum(
            float(row.get("realised_pnl", 0.0))
            for row in _unique_by_family(risk_rows).values()
        )
        pct = (day_pnl / capital) if capital else 0.0

        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold", width=10)
        table.add_column()

        pnl_style = "green" if day_pnl >= 0 else "red"
        equity_line = Text()
        equity_line.append(f"{_money(capital)}", style="bold")
        equity_line.append("  |  day ", style="dim")
        equity_line.append(f"{day_pnl:+,.0f} ({pct:+.2%})", style=pnl_style)
        table.add_row("Equity", equity_line)

        # Open risk, not allocation: the sum of what the open positions can
        # lose if every stop fills, as a fraction of capital.
        open_risk = sum(
            float(row.get("risk_amount", 0.0) or 0.0)
            for row in self.state.get("positions", [])
        )
        open_pct = (open_risk / capital) if capital else 0.0
        counts = self.state.get("system", {}).get("position_counts", {})
        exposure = Text()
        exposure.append(f"{_money(open_risk)} ({open_pct:.2%} of capital)")
        if counts:
            exposure.append("  |  ", style="dim")
            exposure.append(
                " ".join(
                    f"{family} {used}/{cap}" for family, (used, cap) in counts.items()
                ),
                style="dim",
            )
        table.add_row("At risk", exposure)

        return Panel(table, title="PORTFOLIO", title_align="left",
                     border_style="blue")

    def _positions_panel(self) -> Any:
        rows = self.state.get("positions", [])
        if not rows:
            return Panel(Text("flat", style="dim"), title="POSITIONS",
                         title_align="left", border_style="blue")

        table = Table.grid(padding=(0, 2))
        for _ in range(6):
            table.add_column()
        for row in rows:
            r_multiple = float(row.get("unrealised_r", 0.0) or 0.0)
            table.add_row(
                Text(str(row.get("market", "?")), style="bold"),
                Text(str(row.get("direction", "?")),
                     style="green" if str(row.get("direction")) == "LONG" else "red"),
                f"{float(row.get('entry', 0.0)):,.2f}",
                Text(f"{r_multiple:+.2f}R",
                     style="green" if r_multiple >= 0 else "red"),
                Text(f"stop {float(row.get('stop', 0.0)):,.2f}", style="dim"),
                Text(str(row.get("held", "")), style="dim"),
            )
        return Panel(table, title="POSITIONS", title_align="left",
                     border_style="blue")

    def _signals_panel(self) -> Any:
        lines = self.state.get("signals", [])[-4:]
        body = (
            Group(*[Text(line) for line in lines])
            if lines else Text("no signals yet today", style="dim")
        )
        return Panel(body, title="RECENT SIGNALS", title_align="left",
                     border_style="blue")

    def _risk_panel(self) -> Any:
        """The two section 7 session-stopping conditions, and nothing else."""
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold", width=10)
        table.add_column()

        risk_rows = self.state.get("risk", {}) or {}
        if not risk_rows:
            table.add_row("-", Text("no risk state yet", style="dim"))

        trigger = int(self.cfg.get("risk.consecutive_loss_trigger"))
        for market, row in risk_rows.items():
            realised = float(row.get("realised_pnl", 0.0))
            cap = float(row.get("daily_cap", 0.0))
            losses = int(row.get("consecutive_losses", 0))

            bar, colour = risk_bar(min(0.0, realised) * -1.0, cap)
            used_pct = (abs(min(0.0, realised)) / cap) if cap else 0.0
            cap_pct = self.cfg.daily_loss_cap(market)

            line = Text()
            line.append(f"day loss {used_pct * cap_pct:.2%} / {cap_pct:.0%}  ")
            line.append(f"[{bar}]", style=colour)
            line.append(f"  {_money(max(0.0, cap + realised))} left", style="dim")
            line.append("  |  ")
            loss_bar, loss_colour = risk_bar(losses, trigger, width=3)
            line.append(f"losses {losses}/{trigger} ", style="dim")
            line.append(f"[{loss_bar}]", style=loss_colour)
            if row.get("paused"):
                line.append("  PAUSED", style="bold red")
            table.add_row(market, line)

        return Panel(table, title="RISK STATUS", title_align="left",
                     border_style="blue")

    def _system_panel(self) -> Any:
        system = self.state.get("system", {})
        line = Text()

        feeds_ok = system.get("feeds_ok", True)
        line.append("data ", style="dim")
        line.append("OK" if feeds_ok else "DOWN",
                    style="green" if feeds_ok else "bold red")

        for name, info in (system.get("brokers") or {}).items():
            line.append("  |  ", style="dim")
            line.append(f"{name} ", style="dim")
            connected = info.get("connected")
            line.append("OK" if connected else "LOST",
                        style="green" if connected else "bold red")
            if connected and info.get("latency_ms") is not None:
                line.append(f" {info['latency_ms']:.0f}ms", style="dim")

        models = system.get("models") or {}
        if models:
            line.append("  |  models ", style="dim")
            line.append(
                " ".join(f"{market} {age}" for market, age in models.items()),
                style="dim",
            )

        line.append("  |  ", style="dim")
        line.append(
            "PAPER" if self.cfg.is_paper else "LIVE",
            style="yellow" if self.cfg.is_paper else "bold red",
        )

        body: Any = line
        blockers = self.state.get("blockers", [])
        if blockers:
            body = Group(
                line,
                Text(f"refusing trades - unset: {', '.join(blockers)}", style="red"),
            )
        return Panel(body, title="SYSTEM", title_align="left", border_style="blue")

    # -- plain rendering -----------------------------------------------------

    def render_plain(self) -> str:
        """Text rendering for terminals without ``rich``."""
        capital = float(self.state.get("capital", 0.0))
        risk_rows = self.state.get("risk", {}) or {}
        day_pnl = sum(
            float(row.get("realised_pnl", 0.0))
            for row in _unique_by_family(risk_rows).values()
        )
        parts = [
            f"BEAST {self.cfg.mode.upper()} "
            f"({'paper' if self.cfg.is_paper else 'LIVE'}) "
            f"equity {capital:,.0f} day {day_pnl:+,.0f} "
            f"{self.state['now']:%H:%M:%S}",
        ]
        for market, info in self.state.get("markets", {}).items():
            parts.append(
                f"  {market}: vol_state={info.get('vol_label', '-')} "
                f"({float(info.get('vol_probability') or 0):.0%}"
                f"{'' if info.get('vol_confirmed') else ', unconfirmed'}) "
                f"regime={info.get('regime', '-')} "
                f"entries={info.get('entries', 'allowed')}"
            )
        for row in self.state.get("positions", []):
            parts.append(
                f"  POS {row.get('market')} {row.get('direction')} "
                f"entry {float(row.get('entry', 0)):.2f} "
                f"stop {float(row.get('stop', 0)):.2f} "
                f"R {float(row.get('unrealised_r', 0)):+.2f}"
            )
        for market, row in risk_rows.items():
            cap = float(row.get("daily_cap", 0.0))
            realised = float(row.get("realised_pnl", 0.0))
            bar, _ = risk_bar(min(0.0, realised) * -1.0, cap)
            parts.append(
                f"  RISK {market} [{bar}] {realised:+,.0f} of {cap:,.0f} cap"
                f"{' PAUSED' if row.get('paused') else ''}"
            )
        for text in self.state.get("signals", [])[-3:]:
            parts.append(f"  SIGNAL {text}")
        for text in self.state.get("alerts", [])[-3:]:
            parts.append(f"  ALERT  {text}")
        if self.state.get("blockers"):
            parts.append(f"  BLOCKED: {', '.join(self.state['blockers'])}")
        return "\n".join(parts)

    def print_once(self) -> None:
        """Print a single snapshot - used by ``--once`` runs and by CI."""
        if RICH_AVAILABLE and self.console is not None:
            self.console.print(self.render())
        else:
            print(self.render_plain())


def _unique_by_family(risk_rows: dict[str, dict]) -> dict[str, dict]:
    """Collapse per-market risk rows to one row per session family.

    Nifty and Sensex share one ``MarketRiskState``, so summing the rows as they
    arrive would double-count the Indian session's P&L. Keyed on the realised
    figure and cap together, since two markets in one family report identical
    values by construction.
    """
    seen: dict[tuple, dict] = {}
    for row in risk_rows.values():
        key = (round(float(row.get("realised_pnl", 0.0)), 6),
               round(float(row.get("daily_cap", 0.0)), 6),
               int(row.get("consecutive_losses", 0)))
        seen.setdefault(key, row)
    return {str(index): row for index, row in enumerate(seen.values())}
