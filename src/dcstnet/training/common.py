from __future__ import annotations

import datetime as dt
import json
import os
import random
import tempfile
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Sampler

from ..config import ExperimentConfig
from ..data.dataset import AirfareDataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


class DateBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: AirfareDataset,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        groups: dict[str, list[int]] = defaultdict(list)
        for index, sample in enumerate(dataset.samples):
            groups[str(sample["booking_date"])].append(index)
        self.groups = groups

    def __iter__(self) -> Iterator[list[int]]:
        generator = random.Random(self.seed + self.epoch)
        dates = sorted(self.groups)
        if self.shuffle:
            generator.shuffle(dates)
        for date in dates:
            indices = self.groups[date].copy()
            if self.shuffle:
                generator.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                yield indices[start : start + self.batch_size]
        self.epoch += 1

    def __len__(self) -> int:
        return sum(
            (len(indices) + self.batch_size - 1) // self.batch_size
            for indices in self.groups.values()
        )


def move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def batch_date(batch: dict[str, object]) -> dt.date:
    values = batch["booking_date"]
    if not isinstance(values, list) or not values:
        raise ValueError("Expected collated booking_date values")
    unique = set(values)
    if len(unique) != 1:
        raise ValueError("Each batch must contain one request date")
    return dt.date.fromisoformat(values[0])


def atomic_torch_save(payload: object, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=destination.name, dir=destination.parent)
    os.close(descriptor)
    try:
        torch.save(payload, temporary_name)
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def write_manifest(
    config: ExperimentConfig,
    path: str | Path,
    artifact_paths: dict[str, str | Path],
) -> None:
    import hashlib

    manifest = {
        "config_sha256": config.digest(),
        "seed": config.seed,
        "artifacts": {},
    }
    for name, raw_path in sorted(artifact_paths.items()):
        artifact_path = Path(raw_path)
        digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        manifest["artifacts"][name] = {
            "path": str(artifact_path.resolve()),
            "sha256": digest,
            "bytes": artifact_path.stat().st_size,
        }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
