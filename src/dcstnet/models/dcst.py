from __future__ import annotations

import torch
from torch import nn

from .calendar import MaskedCalendarChannel
from .graph import RouteGraphEncoder


class DCSTNet(nn.Module):
    def __init__(
        self,
        node_feature_dim: int,
        graph_hidden_dim: int,
        graph_embedding_dim: int,
        graph_heads: int,
        graph_layers: int,
        receptive_fields: tuple[int, ...],
        calendar_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.route_encoder = RouteGraphEncoder(
            input_dim=node_feature_dim,
            hidden_dim=graph_hidden_dim,
            embedding_dim=graph_embedding_dim,
            heads=graph_heads,
            layers=graph_layers,
        )
        route_dim = graph_embedding_dim * 3
        self.calendar = MaskedCalendarChannel(
            receptive_fields=receptive_fields,
            route_dim=route_dim,
            hidden_dim=calendar_hidden_dim,
        )
        self.fusion = nn.Sequential(
            nn.Linear(route_dim + 2, calendar_hidden_dim),
            nn.ReLU(),
            nn.Linear(calendar_hidden_dim, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        node_indices: torch.Tensor,
        calendar_values: torch.Tensor,
        calendar_mask: torch.Tensor,
        target_intervals: torch.Tensor,
        detached_trend: torch.Tensor,
    ) -> torch.Tensor:
        node_states = self.route_encoder(node_features, edge_index, edge_type)
        route_embedding = self.route_encoder.itinerary_embeddings(node_states, node_indices)
        periodic = self.calendar(calendar_values, calendar_mask, route_embedding, target_intervals)
        batch_size, horizon = periodic.shape
        route = route_embedding.unsqueeze(1).expand(-1, horizon, -1)
        fusion_input = torch.cat(
            (route, periodic.unsqueeze(-1), detached_trend.unsqueeze(-1)), dim=-1
        )
        correction = self.fusion(fusion_input).squeeze(-1)
        return correction + periodic + detached_trend
