from __future__ import annotations

import unittest

import numpy as np

from dcstnet.evaluation.bootstrap import user_level_bootstrap
from dcstnet.evaluation.metrics import evaluate_predictions


class MetricsTest(unittest.TestCase):
    def test_alert_denominators_and_timing(self) -> None:
        rows = [
            {
                "itinerary_id": "A",
                "departure_date": "2024-02-10",
                "booking_date": "2024-02-01",
                "actual": [90.0, 70.0],
                "predicted": [88.0, 72.0],
                "mask": [True, True],
                "current_price": 110.0,
                "threshold": 30.0,
            },
            {
                "itinerary_id": "B",
                "departure_date": "2024-02-11",
                "booking_date": "2024-02-01",
                "actual": [100.0, 99.0],
                "predicted": [70.0, 68.0],
                "mask": [True, True],
                "current_price": 110.0,
                "threshold": 30.0,
            },
        ]
        metrics = evaluate_predictions(rows)
        self.assertEqual(metrics["true_positive"], 1)
        self.assertEqual(metrics["false_positive"], 1)
        self.assertEqual(metrics["alert_precision"], 0.5)

    def test_bootstrap_uses_user_aggregates(self) -> None:
        result = user_level_bootstrap(
            np.array([1.0, 1.2, 0.9, 1.1]),
            np.array([1.1, 1.3, 1.0, 1.2]),
            resamples=200,
            seed=7,
        )
        self.assertGreater(result["relative_effect"], 0.0)
        self.assertLessEqual(result["ci_lower"], result["ci_upper"])


if __name__ == "__main__":
    unittest.main()
