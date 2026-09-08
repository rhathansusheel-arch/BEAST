"""Order placement, modification and cancellation.

The executor is the only place in the system that can send an order. Everything
above it produces intent; this turns intent into a broker call - or, in the
current operational mode, into a simulated fill.

Soul file section 10, paper trading phase:

    Beast operates in **alert-only** mode. It identifies and logs every
    qualifying trade as if live, but places no real orders. All performance
    tracking runs identically to live mode so the data is comparable later. Fills
    are simulated at the trigger candle's close, with the conservative
    assumptions in 6.8.

So the paper path is not a stub - it produces the same fill records, the same
slippage and cost accounting, and the same journal rows the live path will. The
only difference is that nothing leaves the process.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from broker import (
    BrokerClient,
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderType,
)
from core.config import Config, get_config
from core.schemas import Direction, ExitReason, Signal

logger = logging.getLogger("beast.executor")


@dataclass
class Fill:
    """A completed entry or exit.

    Attributes:
        price: Fill price in the traded instrument's units - premium for an
            option leg, underlying price for futures.
        underlying_price: The underlying at fill time, always recorded, because
            every level in section 6 is expressed in the underlying.
        slippage: Modelled slippage in instrument units.
        costs: Brokerage and charges estimate.
    """

    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    underlying_price: float
    at: datetime
    paper: bool = True
    slippage: float = 0.0
    costs: float = 0.0
    message: str = ""


class OrderExecutor:
    """Places entries and exits through a broker adapter.

    Args:
        brokers: Adapters keyed by the name used in ``broker.routing``.
        config: Injected for tests.
    """

    def __init__(self, brokers: dict[str, BrokerClient], config: Config | None = None) -> None:
        self.cfg = config or get_config()
        self.brokers = brokers

    # -- routing -------------------------------------------------------------

    def broker_for(self, market: str) -> BrokerClient | None:
        """Return the adapter configured to serve ``market``."""
        routing = self.cfg.get("broker.routing")
        name = routing.get(market.upper())
        if name is None:
            logger.error("No broker routed for %s", market)
            return None
        return self.brokers.get(name)

    # -- entries -------------------------------------------------------------

    def enter(self, signal: Signal, underlying_price: float, now: datetime) -> Fill | None:
        """Open the position described by ``signal``.

        Long CE for a bullish underlying signal, long PE for a bearish one, or a
        long/short futures contract for Gold. Options are only ever bought
        (rule 13.10), which is why the option branch has no ``SELL`` path.

        Returns:
            A :class:`Fill`, or ``None`` when the broker refused.
        """
        broker = self.broker_for(signal.market)
        if broker is None:
            return None

        if signal.leg_type == "OPTION":
            leg = signal.option_leg
            if leg is None or leg.lots < 1:
                logger.error("Option signal without a sized leg: %s", signal.signal_id)
                return None
            symbol = leg.tradingsymbol or f"{signal.market}{leg.strike:g}{leg.option_type}"
            quantity = leg.lots * leg.lot_size
            side = OrderSide.BUY           # always long: CE for up, PE for down
            reference_price = leg.mid_premium
        else:
            leg = signal.futures_leg
            if leg is None or leg.contracts < 1:
                logger.error("Futures signal without a sized leg: %s", signal.signal_id)
                return None
            symbol = leg.contract
            quantity = leg.contracts
            side = OrderSide.BUY if signal.direction is Direction.LONG else OrderSide.SELL
            reference_price = underlying_price

        request = OrderRequest(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType[str(self.cfg.get("broker.order_type")).upper()],
            limit_price=reference_price,
            tag=signal.signal_id[:20],
            metadata={"signal_id": signal.signal_id, "leg": signal.leg_type,
                      "lots": leg.lots if signal.leg_type == "OPTION" else leg.contracts},
        )
        result = broker.place_order(request)
        return self._to_fill(result, request, reference_price, underlying_price, now, entering=True)

    # -- exits ---------------------------------------------------------------

    def exit(self, signal: Signal, quantity: int, reason: ExitReason,
             underlying_price: float, premium: float | None, now: datetime) -> Fill | None:
        """Square off a position.

        Exits always execute. High spread blocks *entries* only - if the option's
        spread is wide at exit time, Beast exits anyway and logs the slippage
        (6.10). Holding a position because the exit is expensive is exactly the
        reasoning section 8 exists to prevent.
        """
        broker = self.broker_for(signal.market)
        if broker is None:
            return None

        if signal.leg_type == "OPTION":
            leg = signal.option_leg
            symbol = leg.tradingsymbol if leg else signal.market
            side = OrderSide.SELL          # closing a long option
            reference_price = premium if premium is not None else 0.0
            lots = leg.lots if leg else 1
        else:
            leg = signal.futures_leg
            symbol = leg.contract if leg else signal.market
            side = OrderSide.SELL if signal.direction is Direction.LONG else OrderSide.BUY
            reference_price = underlying_price
            lots = leg.contracts if leg else quantity

        request = OrderRequest(
            symbol=symbol,
            side=side,
            quantity=quantity,
            # A flatten is never a limit order - it must complete.
            order_type=OrderType.MARKET,
            limit_price=reference_price,
            tag=f"{reason.value}-{signal.signal_id[:12]}",
            metadata={"signal_id": signal.signal_id, "exit_reason": reason.value,
                      "lots": lots},
        )
        result = broker.place_order(request)
        return self._to_fill(result, request, reference_price, underlying_price, now, entering=False)

    # -- cancellation --------------------------------------------------------

    def cancel(self, market: str, order_id: str) -> bool:
        """Cancel a resting order."""
        broker = self.broker_for(market)
        return bool(broker and broker.cancel_order(order_id))

    # -- fill modelling ------------------------------------------------------

    def _to_fill(self, result: OrderResult, request: OrderRequest, reference_price: float,
                 underlying_price: float, now: datetime, entering: bool) -> Fill | None:
        """Convert a broker result into a :class:`Fill`, modelling costs.

        Slippage is applied *against* the position on both entry and exit -
        never assume the favourable fill (6.8). This keeps paper-mode statistics
        honest and comparable to live.
        """
        if not result.accepted:
            logger.error("Order refused: %s", result.message)
            return None

        slippage_pct = float(self.cfg.get("backtest.slippage_pct"))
        direction = 1 if request.side is OrderSide.BUY else -1
        base_price = result.average_price or reference_price
        slippage = abs(base_price) * slippage_pct
        fill_price = base_price + direction * slippage

        # Charges are per lot per side, not per unit of quantity.
        lots = int(request.metadata.get("lots", 1) or 1)
        costs = float(self.cfg.get("backtest.commission_per_lot")) * max(1, lots)

        return Fill(
            order_id=result.order_id,
            symbol=request.symbol,
            side=request.side,
            quantity=result.filled_quantity or request.quantity,
            price=round(fill_price, 4),
            underlying_price=round(underlying_price, 4),
            at=now,
            paper=result.paper,
            slippage=round(slippage, 4),
            costs=round(costs, 2),
            message=result.message,
        )
