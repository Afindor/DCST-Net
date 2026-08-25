from __future__ import annotations

import torch


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def normalized_soft_min(predictions: torch.Tensor, mask: torch.Tensor, rho: float) -> torch.Tensor:
    if rho <= 0:
        raise ValueError("rho must be positive")
    large = torch.finfo(predictions.dtype).max / 100.0
    scaled = torch.where(mask.bool(), -predictions / rho, -large)
    count = mask.sum(dim=1).clamp_min(1.0)
    return -rho * (torch.logsumexp(scaled, dim=1) - count.log())


def decision_aware_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    current_price: torch.Tensor,
    threshold: torch.Tensor,
    opportunity: torch.Tensor,
    rho: float,
    alert_lambda: float,
    false_alert_weight: float,
    missed_opportunity_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    price_loss = masked_mean((predictions - targets).square(), mask)
    future_price = normalized_soft_min(predictions, mask, rho)
    predicted_drop = current_price - future_price
    false_alert = (1.0 - opportunity) * torch.relu(predicted_drop - threshold).square()
    missed_opportunity = opportunity * torch.relu(threshold - predicted_drop).square()
    alert_loss = (
        false_alert_weight * false_alert + missed_opportunity_weight * missed_opportunity
    ).mean()
    total = price_loss + alert_lambda * alert_loss
    return total, {
        "loss": total.detach(),
        "price_loss": price_loss.detach(),
        "alert_loss": alert_loss.detach(),
    }
