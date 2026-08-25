from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..config import ExperimentConfig
from ..data.dataset import AirfareDataset
from ..data.graph import AsOfGraphStore
from ..training.cache import TrendCache
from ..training.common import DateBatchSampler, batch_date, move_batch, resolve_device
from ..training.phase_b import create_model
from .metrics import evaluate_predictions


@torch.no_grad()
def evaluate_checkpoint(
    dataset: AirfareDataset,
    graph_store: AsOfGraphStore,
    trend_cache: TrendCache,
    config: ExperimentConfig,
    checkpoint_path: str | Path,
    output_dir: str | Path,
) -> dict[str, float | int]:
    device = resolve_device(config.device)
    sampler = DateBatchSampler(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        seed=config.seed,
    )
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=config.training.num_workers)
    first_batch = next(iter(loader))
    first_graph = graph_store.for_date(batch_date(first_batch))
    model = create_model(config, first_graph.node_features.shape[1]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint["config_sha256"] != config.digest():
        raise ValueError("Phase-B checkpoint and configuration do not match")
    model.load_state_dict(checkpoint["model"])
    model.eval()

    rows: list[dict[str, object]] = []
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        graph = graph_store.for_date(batch_date(batch)).to(device)
        trend = trend_cache.get_batch(batch["sample_id"], device).detach()
        predictions = model(
            graph.node_features,
            graph.edge_index,
            graph.edge_type,
            batch["node_indices"],
            batch["calendar_values"],
            batch["calendar_mask"],
            batch["target_intervals"],
            trend,
        )
        actual_raw = batch["target"] * batch["std"].unsqueeze(1) + batch["mean"].unsqueeze(1)
        predicted_raw = predictions * batch["std"].unsqueeze(1) + batch["mean"].unsqueeze(1)
        current_raw = batch["current_price"] * batch["std"] + batch["mean"]
        threshold_raw = batch["threshold"] * batch["std"]
        for index, sample_id in enumerate(batch["sample_id"]):
            rows.append(
                {
                    "sample_id": str(sample_id),
                    "itinerary_id": str(batch["itinerary_id"][index]),
                    "booking_date": str(batch["booking_date"][index]),
                    "departure_date": str(batch["departure_date"][index]),
                    "actual": actual_raw[index].cpu().tolist(),
                    "predicted": predicted_raw[index].cpu().tolist(),
                    "mask": batch["target_mask"][index].bool().cpu().tolist(),
                    "current_price": float(current_raw[index].cpu()),
                    "threshold": float(threshold_raw[index].cpu()),
                }
            )
    metrics = evaluate_predictions(rows)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    horizon = config.data.horizon
    fields = (
        [
            "sample_id",
            "itinerary_id",
            "booking_date",
            "departure_date",
            "current_price",
            "threshold",
        ]
        + [f"actual_t{step}" for step in range(1, horizon + 1)]
        + [f"predicted_t{step}" for step in range(1, horizon + 1)]
        + [f"valid_t{step}" for step in range(1, horizon + 1)]
    )
    with (destination / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {key: row[key] for key in fields[:6]}
            for step in range(horizon):
                flat[f"actual_t{step + 1}"] = row["actual"][step]
                flat[f"predicted_t{step + 1}"] = row["predicted"][step]
                flat[f"valid_t{step + 1}"] = int(row["mask"][step])
            writer.writerow(flat)
    return metrics
