from __future__ import annotations

import math

import torch
from torch import nn


def cosine_alpha_schedule(steps: int, offset: float = 0.008) -> torch.Tensor:
    timeline = torch.linspace(0, steps, steps + 1, dtype=torch.float64) / steps
    alpha_bar = torch.cos((timeline + offset) / (1.0 + offset) * math.pi / 2.0).square()
    alpha_bar = alpha_bar / alpha_bar[0]
    return alpha_bar[1:].clamp(1e-5, 0.999999).float()


def timestep_embedding(timesteps: torch.Tensor, dimension: int) -> torch.Tensor:
    half = dimension // 2
    scale = math.log(10_000.0) / max(half - 1, 1)
    frequencies = torch.exp(-scale * torch.arange(half, device=timesteps.device))
    angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
    if dimension % 2:
        embedding = torch.nn.functional.pad(embedding, (0, 1))
    return embedding


class ResidualDenoiser(nn.Module):
    def __init__(self, context_dim: int, hidden_dim: int, time_dim: int = 32) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.network = nn.Sequential(
            nn.Linear(1 + context_dim + time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, noisy_residual: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        time_features = timestep_embedding(timesteps, self.time_dim)
        inputs = torch.cat((noisy_residual.unsqueeze(1), context, time_features), dim=1)
        return self.network(inputs).squeeze(1)


class GaussianResidualDiffusion(nn.Module):
    def __init__(self, denoiser: ResidualDenoiser, diffusion_steps: int) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.diffusion_steps = diffusion_steps
        self.register_buffer("alpha_bar", cosine_alpha_schedule(diffusion_steps))

    def loss(self, residual: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        timesteps = torch.randint(
            0, self.diffusion_steps, (residual.shape[0],), device=residual.device
        )
        noise = torch.randn_like(residual)
        alpha = self.alpha_bar[timesteps]
        noisy = alpha.sqrt() * residual + (1.0 - alpha).sqrt() * noise
        prediction = self.denoiser(noisy, timesteps, context)
        return torch.nn.functional.mse_loss(prediction, noise)

    @torch.no_grad()
    def ddim_sample(self, context: torch.Tensor, steps: int) -> torch.Tensor:
        if steps < 1 or steps > self.diffusion_steps:
            raise ValueError("Invalid number of DDIM steps")
        schedule = (
            torch.linspace(self.diffusion_steps - 1, 0, steps, device=context.device).round().long()
        )
        sample = torch.zeros(context.shape[0], device=context.device)
        for position, timestep in enumerate(schedule):
            current = torch.full(
                (context.shape[0],), int(timestep.item()), device=context.device, dtype=torch.long
            )
            predicted_noise = self.denoiser(sample, current, context)
            alpha_current = self.alpha_bar[timestep]
            predicted_clean = (
                sample - (1.0 - alpha_current).sqrt() * predicted_noise
            ) / alpha_current.sqrt()
            if position == len(schedule) - 1:
                sample = predicted_clean
            else:
                alpha_next = self.alpha_bar[schedule[position + 1]]
                sample = (
                    alpha_next.sqrt() * predicted_clean
                    + (1.0 - alpha_next).sqrt() * predicted_noise
                )
        return sample


def booking_summary(
    values: torch.Tensor, mask: torch.Tensor, window: int
) -> tuple[torch.Tensor, torch.Tensor]:
    masked = values * mask
    count = mask.sum(dim=1).clamp_min(1.0)
    mean = masked.sum(dim=1) / count
    variance = (((values - mean.unsqueeze(1)) * mask).square().sum(dim=1) / count).clamp_min(0.0)
    positions = torch.arange(values.shape[1], device=values.device, dtype=values.dtype)
    position_mean = (positions.unsqueeze(0) * mask).sum(dim=1) / count
    covariance = (
        (positions.unsqueeze(0) - position_mean.unsqueeze(1)) * (values - mean.unsqueeze(1)) * mask
    ).sum(dim=1)
    position_variance = (
        ((positions.unsqueeze(0) - position_mean.unsqueeze(1)).square() * mask)
        .sum(dim=1)
        .clamp_min(1.0)
    )
    slope = covariance / position_variance
    tail_values = values[:, -window:]
    tail_mask = mask[:, -window:]
    tail_mean = (tail_values * tail_mask).sum(dim=1) / tail_mask.sum(dim=1).clamp_min(1.0)
    summary = torch.stack((mean, variance.sqrt(), slope, count / values.shape[1]), dim=1)
    return summary, tail_mean


def residual_context(
    booking_values: torch.Tensor,
    booking_mask: torch.Tensor,
    route_embedding: torch.Tensor,
    target_intervals: torch.Tensor,
    moving_average_window: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    summary, moving_average = booking_summary(booking_values, booking_mask, moving_average_window)
    batch_size, horizon = target_intervals.shape
    expanded_summary = summary.unsqueeze(1).expand(-1, horizon, -1)
    expanded_route = route_embedding.unsqueeze(1).expand(-1, horizon, -1)
    context = torch.cat((expanded_summary, expanded_route, target_intervals.unsqueeze(-1)), dim=-1)
    trend = moving_average.unsqueeze(1).expand(-1, horizon)
    return context, trend
