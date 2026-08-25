from __future__ import annotations

import dataclasses
import datetime as dt
import math
from collections import defaultdict

import numpy as np
import torch

from .schema import CoBookingRecord, ItineraryRecord, PriceRecord


@dataclasses.dataclass
class RouteGraph:
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    node_ids: tuple[str, ...]
    itinerary_nodes: dict[str, tuple[int, int, int]]
    snapshot_date: dt.date

    def to(self, device: torch.device | str) -> RouteGraph:
        return RouteGraph(
            node_features=self.node_features.to(device),
            edge_index=self.edge_index.to(device),
            edge_type=self.edge_type.to(device),
            node_ids=self.node_ids,
            itinerary_nodes=self.itinerary_nodes,
            snapshot_date=self.snapshot_date,
        )


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator == 0.0 else float(left @ right / denominator)


def build_route_graph(
    itineraries: list[ItineraryRecord],
    co_bookings: list[CoBookingRecord],
    snapshot_date: dt.date,
    top_k: int,
    co_booking_percentile: float,
    jaccard_threshold: float,
    frequency_by_itinerary: dict[str, float] | None = None,
) -> RouteGraph:
    """Build an as-of heterogeneous route graph and apply directed top-k pruning."""
    latest_co_bookings: dict[tuple[str, str], CoBookingRecord] = {}
    for record in co_bookings:
        if record.snapshot_date > snapshot_date:
            continue
        pair = tuple(sorted((record.airport_a, record.airport_b)))
        previous = latest_co_bookings.get(pair)
        if previous is None or record.snapshot_date > previous.snapshot_date:
            latest_co_bookings[pair] = record
    eligible = list(latest_co_bookings.values())
    airports = sorted(
        {record.origin for record in itineraries} | {record.destination for record in itineraries}
    )
    flights = sorted({record.flight_id for record in itineraries})
    node_ids = tuple(
        [f"airport:{value}" for value in airports] + [f"flight:{value}" for value in flights]
    )
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    airport_index = {value: node_index[f"airport:{value}"] for value in airports}
    flight_index = {value: node_index[f"flight:{value}"] for value in flights}

    frequency_by_flight: dict[str, float] = defaultdict(float)
    airport_frequency: dict[str, float] = defaultdict(float)
    signatures: dict[str, set[str]] = defaultdict(set)
    for record in itineraries:
        frequency = (
            frequency_by_itinerary.get(record.itinerary_id, 0.0)
            if frequency_by_itinerary is not None
            else record.frequency
        )
        frequency_by_flight[record.flight_id] += frequency
        airport_frequency[record.origin] += frequency
        airport_frequency[record.destination] += frequency
        signatures[record.flight_id].update((f"O:{record.origin}", f"D:{record.destination}"))

    airport_position = {airport: index for index, airport in enumerate(airports)}
    vectors: dict[str, np.ndarray] = {}
    for flight in flights:
        vector = np.zeros(len(airports), dtype=np.float64)
        related = [record for record in itineraries if record.flight_id == flight]
        for record in related:
            frequency = (
                frequency_by_itinerary.get(record.itinerary_id, 0.0)
                if frequency_by_itinerary is not None
                else record.frequency
            )
            for airport in (record.origin, record.destination):
                denominator = max(airport_frequency[airport], 1.0)
                vector[airport_position[airport]] += frequency / denominator
        vectors[flight] = vector

    candidates: dict[int, list[tuple[float, str, int, int]]] = defaultdict(list)

    counts = np.asarray([record.count for record in eligible], dtype=np.float64)
    cutoff = float(np.percentile(counts, co_booking_percentile)) if counts.size else math.inf
    for record in eligible:
        if record.count < cutoff:
            continue
        if record.airport_a not in airport_index or record.airport_b not in airport_index:
            continue
        left = airport_index[record.airport_a]
        right = airport_index[record.airport_b]
        candidates[left].append((record.count, node_ids[right], right, 1))
        candidates[right].append((record.count, node_ids[left], left, 1))

    for position, left_flight in enumerate(flights):
        left_signature = signatures[left_flight]
        for right_flight in flights[position + 1 :]:
            right_signature = signatures[right_flight]
            union = left_signature | right_signature
            similarity = len(left_signature & right_signature) / max(len(union), 1)
            if similarity < jaccard_threshold:
                continue
            score = _cosine(vectors[left_flight], vectors[right_flight])
            left = flight_index[left_flight]
            right = flight_index[right_flight]
            candidates[left].append((score, node_ids[right], right, 2))
            candidates[right].append((score, node_ids[left], left, 2))

    for record in itineraries:
        flight = flight_index[record.flight_id]
        for airport_name in (record.origin, record.destination):
            airport = airport_index[airport_name]
            candidates[flight].append((2.0, node_ids[airport], airport, 3))
            candidates[airport].append((2.0, node_ids[flight], flight, 3))

    edges: dict[tuple[int, int], int] = {}
    for destination in range(len(node_ids)):
        ranked = sorted(candidates[destination], key=lambda item: (-item[0], item[1]))
        for _, _, source, relation in ranked[:top_k]:
            edges[(source, destination)] = relation
        edges[(destination, destination)] = 0
    ordered_edges = sorted(edges)
    edge_index = torch.tensor(ordered_edges, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor([edges[edge] for edge in ordered_edges], dtype=torch.long)

    maximum_frequency = max([*frequency_by_flight.values(), *airport_frequency.values(), 1.0])
    degree = np.zeros(len(node_ids), dtype=np.float32)
    for source, destination in edges:
        degree[source] += 1.0
        degree[destination] += 1.0
    features = np.zeros((len(node_ids), 8), dtype=np.float32)
    for airport, index in airport_index.items():
        features[index, 0] = 1.0
        features[index, 2] = math.log1p(airport_frequency[airport]) / math.log1p(maximum_frequency)
        features[index, 3] = degree[index] / max(float(degree.max()), 1.0)
    for flight, index in flight_index.items():
        features[index, 1] = 1.0
        features[index, 2] = math.log1p(frequency_by_flight[flight]) / math.log1p(maximum_frequency)
        features[index, 3] = degree[index] / max(float(degree.max()), 1.0)
        vector = vectors[flight]
        features[index, 4] = float(vector.mean()) if vector.size else 0.0
        features[index, 5] = float(vector.max()) if vector.size else 0.0
        features[index, 6] = float(np.count_nonzero(vector)) / max(len(airports), 1)
    features[:, 7] = 1.0

    itinerary_nodes = {
        record.itinerary_id: (
            airport_index[record.origin],
            airport_index[record.destination],
            flight_index[record.flight_id],
        )
        for record in itineraries
    }
    return RouteGraph(
        node_features=torch.from_numpy(features),
        edge_index=edge_index,
        edge_type=edge_type,
        node_ids=node_ids,
        itinerary_nodes=itinerary_nodes,
        snapshot_date=snapshot_date,
    )


class AsOfGraphStore:
    """Cache weekly graph snapshots constructed only from data available by cutoff."""

    def __init__(
        self,
        itineraries: list[ItineraryRecord],
        co_bookings: list[CoBookingRecord],
        prices: list[PriceRecord],
        top_k: int,
        co_booking_percentile: float,
        jaccard_threshold: float,
    ) -> None:
        self.itineraries = itineraries
        self.co_bookings = co_bookings
        self.prices = prices
        self.top_k = top_k
        self.co_booking_percentile = co_booking_percentile
        self.jaccard_threshold = jaccard_threshold
        self.refresh_dates = sorted({record.snapshot_date for record in co_bookings})
        if not self.refresh_dates:
            raise ValueError("At least one graph snapshot is required")
        self._cache: dict[dt.date, RouteGraph] = {}

    def cutoff_for(self, request_date: dt.date) -> dt.date:
        eligible = [value for value in self.refresh_dates if value <= request_date]
        if not eligible:
            raise ValueError(f"No completed graph snapshot is available by {request_date}")
        return eligible[-1]

    def for_date(self, request_date: dt.date) -> RouteGraph:
        cutoff = self.cutoff_for(request_date)
        if cutoff not in self._cache:
            frequencies: dict[str, float] = defaultdict(float)
            for record in self.prices:
                if record.booking_date <= cutoff:
                    frequencies[record.itinerary_id] += 1.0
            self._cache[cutoff] = build_route_graph(
                itineraries=self.itineraries,
                co_bookings=self.co_bookings,
                snapshot_date=cutoff,
                top_k=self.top_k,
                co_booking_percentile=self.co_booking_percentile,
                jaccard_threshold=self.jaccard_threshold,
                frequency_by_itinerary=dict(frequencies),
            )
        return self._cache[cutoff]


def initial_route_snapshot(graph: RouteGraph, embedding_dim: int, seed: int) -> torch.Tensor:
    """Create a deterministic first-refresh node snapshot from as-of graph features."""
    generator = torch.Generator().manual_seed(seed)
    projection = torch.randn(graph.node_features.shape[1], embedding_dim, generator=generator).to(
        graph.node_features.device
    ) / math.sqrt(graph.node_features.shape[1])
    source, destination = graph.edge_index
    aggregated = torch.zeros_like(graph.node_features)
    aggregated.index_add_(0, destination, graph.node_features[source])
    counts = torch.zeros(graph.node_features.shape[0], dtype=graph.node_features.dtype)
    counts.index_add_(0, destination, torch.ones_like(destination, dtype=graph.node_features.dtype))
    states = 0.5 * graph.node_features + 0.5 * aggregated / counts.clamp_min(1.0).unsqueeze(1)
    return torch.nn.functional.normalize(states @ projection, dim=-1)
