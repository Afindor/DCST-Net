from __future__ import annotations

import unittest

import torch

from dcstnet.config import load_config
from dcstnet.models.dcst import DCSTNet


class PhaseBoundaryTest(unittest.TestCase):
    def test_detached_trend_receives_no_gradient(self) -> None:
        config = load_config("configs/demo.toml")
        model = DCSTNet(
            node_feature_dim=8,
            graph_hidden_dim=config.graph.hidden_dim,
            graph_embedding_dim=config.graph.embedding_dim,
            graph_heads=config.graph.heads,
            graph_layers=config.graph.layers,
            receptive_fields=config.calendar.receptive_fields,
            calendar_hidden_dim=config.calendar.hidden_dim,
        )
        node_features = torch.randn(7, 8)
        edge_index = torch.stack((torch.arange(7), torch.arange(7)))
        edge_type = torch.zeros(7, dtype=torch.long)
        node_indices = torch.tensor([[0, 1, 4], [1, 2, 5]])
        calendar_values = torch.randn(2, 3, 8)
        calendar_mask = torch.ones_like(calendar_values)
        target_intervals = torch.rand(2, 3)
        phase_a_trend = torch.randn(2, 3, requires_grad=True)
        predictions = model(
            node_features,
            edge_index,
            edge_type,
            node_indices,
            calendar_values,
            calendar_mask,
            target_intervals,
            phase_a_trend.detach(),
        )
        predictions.square().mean().backward()
        self.assertIsNone(phase_a_trend.grad)


if __name__ == "__main__":
    unittest.main()
