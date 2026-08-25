from __future__ import annotations

from statistics import NormalDist

import numpy as np


def user_level_bootstrap(
    control: np.ndarray,
    treatment: np.ndarray,
    resamples: int = 1000,
    seed: int = 2027,
) -> dict[str, float]:
    """Estimate a relative treatment effect from already aggregated user values."""
    if control.ndim != 1 or treatment.ndim != 1:
        raise ValueError("Inputs must contain one aggregate per user")
    if control.size == 0 or treatment.size == 0:
        raise ValueError("Both groups must be non-empty")
    generator = np.random.default_rng(seed)
    effects = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        control_sample = generator.choice(control, size=control.size, replace=True)
        treatment_sample = generator.choice(treatment, size=treatment.size, replace=True)
        effects[index] = treatment_sample.mean() / control_sample.mean() - 1.0
    point = treatment.mean() / control.mean() - 1.0
    normal = NormalDist()
    fraction_below = float(np.mean(effects < point))
    fraction_below = min(
        max(fraction_below, 1.0 / (2.0 * resamples)), 1.0 - 1.0 / (2.0 * resamples)
    )
    bias = normal.inv_cdf(fraction_below)
    lower_probability = normal.cdf(2.0 * bias + normal.inv_cdf(0.025))
    upper_probability = normal.cdf(2.0 * bias + normal.inv_cdf(0.975))
    lower, upper = np.quantile(effects, (lower_probability, upper_probability))
    p_value = 2.0 * min(float(np.mean(effects <= 0.0)), float(np.mean(effects >= 0.0)))
    return {
        "relative_effect": float(point),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "two_sided_p": min(p_value, 1.0),
        "resamples": float(resamples),
    }
