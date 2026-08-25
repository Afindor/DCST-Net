from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("nan")


def evaluate_predictions(rows: list[dict[str, object]]) -> dict[str, float | int]:
    squared_errors: list[float] = []
    absolute_percentage_errors: list[float] = []
    true_positive = false_positive = true_negative = false_negative = 0
    timing_errors: list[float] = []
    decisions: dict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)

    for row in rows:
        actual = np.asarray(row["actual"], dtype=np.float64)
        predicted = np.asarray(row["predicted"], dtype=np.float64)
        mask = np.asarray(row["mask"], dtype=bool)
        actual_valid = actual[mask]
        predicted_valid = predicted[mask]
        errors = predicted_valid - actual_valid
        squared_errors.extend(np.square(errors).tolist())
        absolute_percentage_errors.extend(
            (np.abs(errors) / np.maximum(np.abs(actual_valid), 1e-6)).tolist()
        )
        current = float(row["current_price"])
        threshold = float(row["threshold"])
        predicted_crossings = np.flatnonzero(current - predicted_valid >= threshold)
        actual_crossings = np.flatnonzero(current - actual_valid >= threshold)
        alert = int(predicted_crossings.size > 0)
        opportunity = int(actual_crossings.size > 0)
        if alert and opportunity:
            true_positive += 1
            timing_errors.append(float(abs(predicted_crossings[0] - actual_crossings[0])))
        elif alert:
            false_positive += 1
        elif opportunity:
            false_negative += 1
        else:
            true_negative += 1
        decisions[(str(row["itinerary_id"]), str(row["departure_date"]))].append(
            (str(row["booking_date"]), alert)
        )

    churn_values = []
    for values in decisions.values():
        ordered = sorted(values)
        for left, right in zip(ordered, ordered[1:], strict=False):
            churn_values.append(float(left[1] != right[1]))

    return {
        "samples": len(rows),
        "trajectory_points": len(squared_errors),
        "rmse": math.sqrt(float(np.mean(squared_errors))) if squared_errors else float("nan"),
        "mape_percent": 100.0 * float(np.mean(absolute_percentage_errors))
        if absolute_percentage_errors
        else float("nan"),
        "alert_precision": _safe_divide(true_positive, true_positive + false_positive),
        "false_alert_rate": _safe_divide(false_positive, false_positive + true_negative),
        "missed_opportunity_rate": _safe_divide(false_negative, true_positive + false_negative),
        "timing_error_days": float(np.mean(timing_errors)) if timing_errors else float("nan"),
        "churn": float(np.mean(churn_values)) if churn_values else float("nan"),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
    }
