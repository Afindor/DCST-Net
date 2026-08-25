from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dcstnet.pipeline import prepare_context  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export matched booking-date arrays for official baseline implementations"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    context = prepare_context(arguments.config)
    dataset = getattr(context.data, arguments.split)
    arrays = {
        "sample_ids": np.asarray(
            [sample["sample_id"] for sample in dataset.samples], dtype=np.str_
        ),
        "booking_values": np.stack(
            [sample["booking_values"].numpy() for sample in dataset.samples]
        ),
        "booking_mask": np.stack([sample["booking_mask"].numpy() for sample in dataset.samples]),
        "targets": np.stack([sample["target"].numpy() for sample in dataset.samples]),
        "target_mask": np.stack([sample["target_mask"].numpy() for sample in dataset.samples]),
        "means": np.asarray([sample["mean"].item() for sample in dataset.samples]),
        "stds": np.asarray([sample["std"].item() for sample in dataset.samples]),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.output, **arrays)
    print(json.dumps({"samples": len(dataset), "output": str(arguments.output.resolve())}))


if __name__ == "__main__":
    main()
