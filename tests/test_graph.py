from __future__ import annotations

import datetime as dt
import unittest
from pathlib import Path

from dcstnet.data.graph import AsOfGraphStore
from dcstnet.data.schema import read_co_bookings, read_itineraries, read_prices

ROOT = Path(__file__).resolve().parents[1]


class GraphTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.itineraries = read_itineraries(ROOT / "data/example/itineraries.csv")
        cls.prices = read_prices(ROOT / "data/example/prices.csv")
        cls.co_bookings = read_co_bookings(ROOT / "data/example/co_bookings.csv")
        cls.store = AsOfGraphStore(
            cls.itineraries,
            cls.co_bookings,
            cls.prices,
            top_k=4,
            co_booking_percentile=85.0,
            jaccard_threshold=0.3,
        )

    def test_snapshot_never_exceeds_request_date(self) -> None:
        request_date = dt.date(2024, 1, 10)
        graph = self.store.for_date(request_date)
        self.assertEqual(graph.snapshot_date, dt.date(2024, 1, 7))
        self.assertLessEqual(graph.snapshot_date, request_date)

    def test_pruned_graph_has_bounded_in_degree(self) -> None:
        graph = self.store.for_date(dt.date(2024, 2, 1))
        destination = graph.edge_index[1]
        for node in range(len(graph.node_ids)):
            self.assertLessEqual(int((destination == node).sum()), 5)
        self.assertTrue({0, 1, 2, 3}.issubset(set(graph.edge_type.tolist())))


if __name__ == "__main__":
    unittest.main()
