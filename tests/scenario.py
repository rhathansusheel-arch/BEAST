"""Hand-built underlying scenarios.

Random walks rarely produce a textbook setup, and a test that waits for one proves nothing.
These build the shapes 5.2 describes so the whole G0-G9 chain can be exercised
deterministically.

Both scenarios prepend **prior sessions**. That is not padding: the bias timeframe is 15M
and Bollinger Bands need 20 periods, so a single 6-hour session cannot even define the
regime classifier's inputs - Beast would sit in ``RANGE`` on NaN and never permit Setup 1
or 4. Multi-day history is a real precondition of the framework, not a test artefact.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

SESSION_OPEN = (9, 15)
SESSION_MINUTES = 375  # 09:15 - 15:30


def _bar(ts, o, h, l, c, v=2000.0):
    return {"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": v}


def _oscillating_session(day: datetime, price: float, amplitude: float = 5.0) -> tuple[list, float]:
    """A quiet prior session, used only to warm the bias-TF indicators."""
    rows = []
    ts = day.replace(hour=SESSION_OPEN[0], minute=SESSION_OPEN[1], second=0, microsecond=0)
    for i in range(SESSION_MINUTES):
        step = amplitude if (i // 20) % 2 == 0 else -amplitude
        o = price
        c = price + step
        rows.append(_bar(ts, o, max(o, c) + 1.0, min(o, c) - 1.0, c))
        price = c
        ts += timedelta(minutes=1)
    return rows, price


def _warmup(sessions: int, first_day: datetime, price: float) -> tuple[list, float]:
    rows: list = []
    for offset in range(sessions, 0, -1):
        day_rows, price = _oscillating_session(first_day - timedelta(days=offset), price)
        rows += day_rows
    return rows, price


def trend_continuation_pullback(
    day: datetime | None = None, base: float = 24000.0, warmup_sessions: int = 3
) -> pd.DataFrame:
    """Setup 4 shaped: trend, shallow pullback to a reference, resumption.

    Trend-continuation is the setup type that can realistically clear 4-of-6 on the
    Trend-Continuation column - ADX rising with DI aligned, RSI above 50 and rising, MACD
    above signal with an expanding histogram, price walking the upper band and holding
    VWAP. A reversal has ADX and VWAP structurally opposing it, which is exactly why 5.3
    makes reversals harder to qualify.
    """
    day = day or datetime(2026, 9, 1)
    rows, price = _warmup(warmup_sessions, day, base)
    ts = day.replace(hour=SESSION_OPEN[0], minute=SESSION_OPEN[1])

    # A trend with breathing room. A perfectly monotone ramp has no fractal swings at all
    # (4.5), so it can never produce the pullback leg Setup 4 is defined against.
    for i in range(180):
        step = -2.2 if i % 11 in (7, 8) else 2.4
        o, c = price, price + step
        rows.append(_bar(ts, o, max(o, c) + 0.6, min(o, c) - 0.6, c))
        price, ts = c, ts + timedelta(minutes=1)

    for _ in range(15):  # shallow pullback, well inside the 61.8% disqualifier
        o, c = price, price - 3.0
        rows.append(_bar(ts, o, o + 0.5, c - 0.8, c))
        price, ts = c, ts + timedelta(minutes=1)

    for i in range(10):  # basing at the low - the retest Setup 3 triggers on
        o, c = price, price + (3.0 if i % 2 else -3.0)
        rows.append(_bar(ts, o, max(o, c) + 0.8, min(o, c) - 0.8, c))
        price, ts = c, ts + timedelta(minutes=1)

    for _ in range(40):  # resumption, reclaiming the pullback reference
        o, c = price, price + 4.0
        rows.append(_bar(ts, o, c + 0.8, o - 0.5, c))
        price, ts = c, ts + timedelta(minutes=1)

    return pd.DataFrame(rows).set_index("timestamp")


def reversal_at_session_low(
    day: datetime | None = None, base: float = 24400.0, warmup_sessions: int = 3
) -> pd.DataFrame:
    """Setup 2 shaped: decline into the session low, deep-wick rejection, recovery."""
    day = day or datetime(2026, 9, 1)
    rows, price = _warmup(warmup_sessions, day, base)
    ts = day.replace(hour=SESSION_OPEN[0], minute=SESSION_OPEN[1])

    for _ in range(120):  # orderly decline
        o, c = price, price - 1.6
        rows.append(_bar(ts, o, max(o, c) + 0.8, min(o, c) - 0.8, c))
        price, ts = c, ts + timedelta(minutes=1)

    for _ in range(40):  # acceleration into the low
        o, c = price, price - 3.0
        rows.append(_bar(ts, o, o + 0.5, c - 1.5, c))
        price, ts = c, ts + timedelta(minutes=1)

    low = price - 20.0
    for i in range(5):  # the rejection: deep wick, close back through the zone
        o = price
        c = price + 6.0 if i >= 2 else price - 2.0
        rows.append(_bar(ts, o, max(o, c) + 1.0, low if i < 3 else min(o, c) - 1.0, c))
        price, ts = c, ts + timedelta(minutes=1)

    for _ in range(30):  # recovery through the zone edge - trigger territory
        o, c = price, price + 4.0
        rows.append(_bar(ts, o, c + 1.0, o - 0.8, c))
        price, ts = c, ts + timedelta(minutes=1)

    return pd.DataFrame(rows).set_index("timestamp")
