from __future__ import annotations

import unittest

import torch

from dcstnet.losses import decision_aware_loss, normalized_soft_min


class LossTest(unittest.TestCase):
    def test_normalized_soft_min_has_no_horizon_shift_for_constant_values(self) -> None:
        predictions = torch.tensor([[3.0, 3.0, 3.0], [7.0, 7.0, 0.0]])
        mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
        result = normalized_soft_min(predictions, mask, rho=2.0)
        torch.testing.assert_close(result, torch.tensor([3.0, 7.0]))

    def test_decision_loss_backpropagates(self) -> None:
        predictions = torch.tensor([[8.0, 6.0]], requires_grad=True)
        total, parts = decision_aware_loss(
            predictions=predictions,
            targets=torch.tensor([[9.0, 5.0]]),
            mask=torch.ones(1, 2),
            current_price=torch.tensor([10.0]),
            threshold=torch.tensor([3.0]),
            opportunity=torch.tensor([1.0]),
            rho=2.0,
            alert_lambda=0.5,
            false_alert_weight=1.2,
            missed_opportunity_weight=0.8,
        )
        total.backward()
        self.assertIsNotNone(predictions.grad)
        self.assertGreaterEqual(float(parts["price_loss"]), 0.0)


if __name__ == "__main__":
    unittest.main()
