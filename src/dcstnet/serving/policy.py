from __future__ import annotations

import dataclasses
from collections.abc import Sequence


@dataclasses.dataclass(frozen=True)
class AlertDecision:
    alert: bool
    current_price: float
    predicted_best_price: float
    predicted_drop: float
    threshold: float


def decide_alert(
    current_price: float,
    trajectory: Sequence[float],
    absolute_threshold: float,
    relative_threshold: float | None = None,
) -> AlertDecision:
    if not trajectory:
        raise ValueError("A trajectory must contain at least one price")
    if absolute_threshold < 0:
        raise ValueError("absolute_threshold must be non-negative")
    best = min(float(value) for value in trajectory)
    drop = float(current_price) - best
    threshold = float(absolute_threshold)
    if relative_threshold is not None:
        if relative_threshold < 0:
            raise ValueError("relative_threshold must be non-negative")
        threshold = max(threshold, float(current_price) * relative_threshold)
    return AlertDecision(
        alert=drop >= threshold,
        current_price=float(current_price),
        predicted_best_price=best,
        predicted_drop=drop,
        threshold=threshold,
    )
