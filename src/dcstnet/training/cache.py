from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class TrendCache:
    def __init__(self, values: dict[str, np.ndarray]) -> None:
        self.values = values

    def get_batch(self, sample_ids: list[str], device: torch.device) -> torch.Tensor:
        missing = [sample_id for sample_id in sample_ids if sample_id not in self.values]
        if missing:
            raise KeyError(f"Trend cache is missing {len(missing)} samples")
        matrix = np.stack([self.values[sample_id] for sample_id in sample_ids])
        return torch.from_numpy(matrix).to(device=device, dtype=torch.float32)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        sample_ids = np.asarray(sorted(self.values), dtype=np.str_)
        matrix = np.stack([self.values[sample_id] for sample_id in sample_ids])
        np.savez_compressed(destination, sample_ids=sample_ids, trends=matrix)

    @classmethod
    def load(cls, path: str | Path) -> TrendCache:
        with np.load(path, allow_pickle=False) as archive:
            sample_ids = archive["sample_ids"]
            matrix = archive["trends"]
        return cls({str(sample_id): matrix[index] for index, sample_id in enumerate(sample_ids)})
