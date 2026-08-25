from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dcstnet.evaluation.bootstrap import user_level_bootstrap  # noqa: E402


def _read(path: Path, group: str, metric: str) -> np.ndarray:
    values = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"user_id", "group", metric}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
        seen = set()
        for row in reader:
            if row["group"] != group:
                continue
            if row["user_id"] in seen:
                raise ValueError("Input must contain one aggregate row per user")
            seen.add(row["user_id"])
            values.append(float(row[metric]))
    return np.asarray(values, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap an online metric at user level")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--control-label", default="control")
    parser.add_argument("--treatment-label", default="treatment")
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2027)
    arguments = parser.parse_args()
    control = _read(arguments.input, arguments.control_label, arguments.metric)
    treatment = _read(arguments.input, arguments.treatment_label, arguments.metric)
    result = user_level_bootstrap(control, treatment, arguments.resamples, arguments.seed)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
