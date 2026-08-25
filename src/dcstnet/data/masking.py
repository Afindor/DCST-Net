from __future__ import annotations

import datetime as dt
from collections.abc import Iterable


def is_available(departure_date: dt.date, booking_interval: int, cutoff: dt.date) -> bool:
    """Return whether a surface cell was observable by the request cutoff."""
    return departure_date - dt.timedelta(days=booking_interval) <= cutoff


def valid_horizon(booking_interval: int, requested_horizon: int) -> int:
    return max(0, min(requested_horizon, booking_interval - 1))


def causal_forward_fill(values: Iterable[float | None]) -> list[float | None]:
    """Fill only from earlier elements in the supplied event-time order."""
    result: list[float | None] = []
    last: float | None = None
    for value in values:
        if value is not None:
            last = float(value)
        result.append(last)
    return result
