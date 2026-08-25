from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MaskedCalendarChannel(nn.Module):
    def __init__(
        self,
        receptive_fields: tuple[int, ...],
        route_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.receptive_fields = receptive_fields
        self.convolutions = nn.ModuleList(
            nn.Conv1d(1, hidden_dim, kernel_size=field) for field in receptive_fields
        )
        branch_dim = hidden_dim + route_dim + 2
        self.branch_heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(branch_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            for _ in receptive_fields
        )
        self.gate = nn.Sequential(
            nn.Linear(len(receptive_fields) * hidden_dim + route_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(receptive_fields)),
        )

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        route_embedding: torch.Tensor,
        target_intervals: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, horizon, lookback = values.shape
        flat_values = (values * mask).view(batch_size * horizon, 1, lookback)
        flat_mask = mask.view(batch_size * horizon, 1, lookback)
        route = (
            route_embedding.unsqueeze(1).expand(-1, horizon, -1).reshape(batch_size * horizon, -1)
        )
        intervals = target_intervals.reshape(batch_size * horizon, 1)
        coverage = flat_mask.mean(dim=-1)

        branch_states = []
        branch_predictions = []
        for field, convolution, head in zip(
            self.receptive_fields, self.convolutions, self.branch_heads, strict=True
        ):
            if lookback < field:
                padded_values = F.pad(flat_values, (field - lookback, 0))
                padded_mask = F.pad(flat_mask, (field - lookback, 0))
            else:
                padded_values = flat_values
                padded_mask = flat_mask
            activations = F.relu(convolution(padded_values))
            valid_counts = F.conv1d(
                padded_mask,
                torch.ones(1, 1, field, device=values.device, dtype=values.dtype),
            )
            valid_windows = (valid_counts > 0).to(values.dtype)
            pooled = (activations * valid_windows).sum(dim=-1) / valid_windows.sum(
                dim=-1
            ).clamp_min(1.0)
            branch_states.append(pooled)
            branch_predictions.append(head(torch.cat((pooled, route, intervals, coverage), dim=-1)))
        gate_input = torch.cat((*branch_states, route, coverage), dim=-1)
        weights = torch.softmax(self.gate(gate_input), dim=-1)
        predictions = torch.cat(branch_predictions, dim=-1)
        return (weights * predictions).sum(dim=-1).view(batch_size, horizon)
