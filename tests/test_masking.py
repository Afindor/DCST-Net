from __future__ import annotations

import datetime as dt
import unittest

from dcstnet.data.masking import causal_forward_fill, is_available, valid_horizon


class MaskingTest(unittest.TestCase):
    def test_event_time_availability(self) -> None:
        cutoff = dt.date(2024, 1, 10)
        self.assertTrue(is_available(dt.date(2024, 1, 15), 5, cutoff))
        self.assertFalse(is_available(dt.date(2024, 1, 15), 4, cutoff))

    def test_valid_horizon_stops_before_departure(self) -> None:
        self.assertEqual(valid_horizon(2, 15), 1)
        self.assertEqual(valid_horizon(20, 15), 15)
        self.assertEqual(valid_horizon(1, 15), 0)

    def test_forward_fill_is_causal(self) -> None:
        self.assertEqual(
            causal_forward_fill([None, 10.0, None, 12.0, None]),
            [None, 10.0, 10.0, 12.0, 12.0],
        )


if __name__ == "__main__":
    unittest.main()
