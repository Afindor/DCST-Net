from __future__ import annotations

import dataclasses
import datetime as dt
from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from ..data.schema import ItineraryRecord, PriceRecord


@dataclasses.dataclass(frozen=True)
class Aggregate:
    mean: float
    count: int


class HierarchicalRouteFallback:
    def __init__(
        self,
        metadata: dict[str, ItineraryRecord],
        flight: dict[str, Aggregate],
        od: dict[tuple[str, str], Aggregate],
        airport: dict[str, Aggregate],
        global_aggregate: Aggregate,
        minimum_count: int,
    ) -> None:
        self.metadata = metadata
        self.flight = flight
        self.od = od
        self.airport = airport
        self.global_aggregate = global_aggregate
        self.minimum_count = minimum_count

    @classmethod
    def fit(
        cls,
        itineraries: list[ItineraryRecord],
        prices: list[PriceRecord],
        cutoff: dt.date,
        minimum_count: int = 8,
    ) -> HierarchicalRouteFallback:
        metadata = {record.itinerary_id: record for record in itineraries}
        flight_values: dict[str, list[float]] = defaultdict(list)
        od_values: dict[tuple[str, str], list[float]] = defaultdict(list)
        airport_values: dict[str, list[float]] = defaultdict(list)
        global_values = []
        for price in prices:
            if price.booking_date > cutoff or price.itinerary_id not in metadata:
                continue
            route = metadata[price.itinerary_id]
            flight_values[route.flight_id].append(price.price)
            od_values[(route.origin, route.destination)].append(price.price)
            airport_values[route.origin].append(price.price)
            airport_values[route.destination].append(price.price)
            global_values.append(price.price)
        if not global_values:
            raise ValueError("Fallback fitting requires at least one as-of price")

        def aggregate(values: dict[object, list[float]]) -> dict[object, Aggregate]:
            return {
                key: Aggregate(float(np.mean(group)), len(group)) for key, group in values.items()
            }

        return cls(
            metadata=metadata,
            flight=aggregate(flight_values),
            od=aggregate(od_values),
            airport=aggregate(airport_values),
            global_aggregate=Aggregate(float(np.mean(global_values)), len(global_values)),
            minimum_count=minimum_count,
        )

    def estimate(self, itinerary_id: str) -> float:
        route = self.metadata.get(itinerary_id)
        if route is None:
            return self.global_aggregate.mean
        candidates = [
            self.flight.get(route.flight_id),
            self.od.get((route.origin, route.destination)),
        ]
        airport_candidates = [
            value
            for value in (self.airport.get(route.origin), self.airport.get(route.destination))
            if value is not None
        ]
        if airport_candidates:
            candidates.append(
                Aggregate(
                    mean=float(np.mean([value.mean for value in airport_candidates])),
                    count=sum(value.count for value in airport_candidates),
                )
            )
        for candidate in candidates:
            if candidate is not None and candidate.count >= self.minimum_count:
                return candidate.mean
        return self.global_aggregate.mean


class LightweightTrajectoryFallback:
    def forecast(self, observed_prices: Sequence[float], horizon: int) -> list[float]:
        if not observed_prices:
            raise ValueError("Fallback requires at least one observed price")
        if horizon < 1:
            raise ValueError("horizon must be positive")
        values = np.asarray(observed_prices, dtype=np.float64)
        recent = values[-7:]
        slope = float(np.median(np.diff(recent))) if recent.size > 1 else 0.0
        slope = float(np.clip(slope, -0.1 * recent[-1], 0.1 * recent[-1]))
        return [max(0.01, float(recent[-1] + slope * step)) for step in range(1, horizon + 1)]
