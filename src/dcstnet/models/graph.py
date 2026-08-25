from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def segment_softmax(scores: torch.Tensor, index: torch.Tensor, segments: int) -> torch.Tensor:
    maximum = torch.full((segments,), -torch.inf, dtype=scores.dtype, device=scores.device)
    maximum.scatter_reduce_(0, index, scores, reduce="amax", include_self=True)
    shifted = scores - maximum[index]
    numerator = shifted.exp()
    denominator = torch.zeros(segments, dtype=scores.dtype, device=scores.device)
    denominator.index_add_(0, index, numerator)
    return numerator / denominator[index].clamp_min(1e-12)


class MultiHeadGraphAttention(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, heads: int, relations: int = 4) -> None:
        super().__init__()
        if output_dim % heads:
            raise ValueError("output_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = output_dim // heads
        self.projection = nn.Linear(input_dim, output_dim, bias=False)
        self.source_attention = nn.Parameter(torch.empty(heads, self.head_dim))
        self.destination_attention = nn.Parameter(torch.empty(heads, self.head_dim))
        self.relation_bias = nn.Parameter(torch.zeros(relations, heads))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.xavier_uniform_(self.source_attention)
        nn.init.xavier_uniform_(self.destination_attention)
        nn.init.zeros_(self.relation_bias)
        nn.init.zeros_(self.bias)

    def forward(
        self, features: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor
    ) -> torch.Tensor:
        source, destination = edge_index
        projected = self.projection(features).view(features.shape[0], self.heads, self.head_dim)
        source_states = projected[source]
        destination_states = projected[destination]
        scores = F.leaky_relu(
            (source_states * self.source_attention).sum(-1)
            + (destination_states * self.destination_attention).sum(-1),
            negative_slope=0.2,
        )
        scores = scores + self.relation_bias[edge_type]
        messages = []
        for head in range(self.heads):
            weights = segment_softmax(scores[:, head], destination, features.shape[0])
            aggregated = torch.zeros(
                features.shape[0], self.head_dim, dtype=features.dtype, device=features.device
            )
            aggregated.index_add_(0, destination, weights.unsqueeze(1) * source_states[:, head])
            messages.append(aggregated)
        return torch.cat(messages, dim=-1) + self.bias


class RouteGraphEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        heads: int,
        layers: int,
    ) -> None:
        super().__init__()
        dimensions = [input_dim] + [hidden_dim] * max(layers - 1, 0) + [embedding_dim]
        self.layers = nn.ModuleList(
            MultiHeadGraphAttention(dimensions[index], dimensions[index + 1], heads)
            for index in range(layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(value) for value in dimensions[1:])

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        states = node_features
        for layer, norm in zip(self.layers, self.norms, strict=True):
            states = norm(layer(states, edge_index, edge_type))
            states = F.elu(states)
        return states

    @staticmethod
    def itinerary_embeddings(node_states: torch.Tensor, node_indices: torch.Tensor) -> torch.Tensor:
        selected = node_states[node_indices]
        return selected.flatten(start_dim=1)
