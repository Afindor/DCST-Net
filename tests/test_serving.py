from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dcstnet.serving.cache import TrajectoryCache
from dcstnet.serving.fallback import LightweightTrajectoryFallback
from dcstnet.serving.policy import decide_alert


class ServingTest(unittest.TestCase):
    def test_cache_version_and_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = TrajectoryCache(Path(directory) / "cache.sqlite")
            cache.put("A", "2024-02-20", 10, "2024-02-10", [105.0, 80.0, 75.0])
            trajectory = cache.get("A", "2024-02-20", 10, "2024-02-10")
            self.assertEqual(trajectory, [105.0, 80.0, 75.0])
            decision = decide_alert(120.0, trajectory, absolute_threshold=40.0)
            self.assertTrue(decision.alert)
            self.assertEqual(decision.predicted_best_price, 75.0)
            self.assertIsNone(cache.get("A", "2024-02-20", 10, "2024-02-17"))
            cache.publish_refresh("2024-02-17", [("A", "2024-02-20", 3, [92.0, 87.0])])
            self.assertEqual(cache.get_active("A", "2024-02-20", 3), [92.0, 87.0])

    def test_lightweight_fallback_is_deterministic(self) -> None:
        model = LightweightTrajectoryFallback()
        first = model.forecast([100.0, 98.0, 97.0], 3)
        second = model.forecast([100.0, 98.0, 97.0], 3)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)


if __name__ == "__main__":
    unittest.main()
