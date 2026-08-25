from __future__ import annotations

import unittest
from pathlib import Path

from dcstnet.pipeline import prepare_context

ROOT = Path(__file__).resolve().parents[1]


class DataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.context = prepare_context(ROOT / "configs/demo.toml")

    def test_splits_are_nonempty_and_ordered(self) -> None:
        self.assertGreater(len(self.context.data.train), 0)
        self.assertGreater(len(self.context.data.validation), 0)
        self.assertGreater(len(self.context.data.test), 0)
        train_dates = {sample["booking_date"] for sample in self.context.data.train.samples}
        test_dates = {sample["booking_date"] for sample in self.context.data.test.samples}
        self.assertLess(max(train_dates), min(test_dates))

    def test_targets_do_not_enter_booking_history(self) -> None:
        sample = self.context.data.train[0]
        interval_positions = int(sample["booking_mask"].sum())
        self.assertGreater(interval_positions, 0)
        self.assertEqual(sample["target"].shape[0], self.context.config.data.horizon)
        normalized_current = (
            float(sample["current_interval"]) / self.context.config.data.max_interval
        )
        valid_targets = sample["target_intervals"][sample["target_mask"].bool()]
        self.assertTrue(bool((valid_targets < normalized_current).all()))

    def test_normalization_is_fitted_for_each_itinerary(self) -> None:
        self.assertEqual(
            set(self.context.data.normalization), {"ODF001", "ODF002", "ODF003", "ODF004"}
        )


if __name__ == "__main__":
    unittest.main()
