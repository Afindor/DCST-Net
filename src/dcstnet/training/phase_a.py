from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import ExperimentConfig
from ..data.dataset import AirfareDataset
from ..data.graph import AsOfGraphStore, initial_route_snapshot
from ..models.diffusion import GaussianResidualDiffusion, ResidualDenoiser, residual_context
from .cache import TrendCache
from .common import (
    DateBatchSampler,
    atomic_torch_save,
    batch_date,
    move_batch,
    resolve_device,
    seed_everything,
)


def _loader(
    dataset: AirfareDataset, config: ExperimentConfig, shuffle: bool
) -> DataLoader[dict[str, object]]:
    sampler = DateBatchSampler(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=shuffle,
        seed=config.seed,
    )
    return DataLoader(dataset, batch_sampler=sampler, num_workers=config.training.num_workers)


def _route_embedding(
    batch: dict[str, object],
    graph_store: AsOfGraphStore,
    embedding_dim: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    graph = graph_store.for_date(batch_date(batch)).to(device)
    snapshot = initial_route_snapshot(graph, embedding_dim, seed).detach()
    node_indices = batch["node_indices"]
    if not isinstance(node_indices, torch.Tensor):
        raise TypeError("node_indices must be a tensor")
    return snapshot[node_indices].flatten(start_dim=1)


def _model(config: ExperimentConfig, route_dim: int) -> GaussianResidualDiffusion:
    context_dim = 4 + route_dim + 1
    denoiser = ResidualDenoiser(context_dim, config.residual.hidden_dim)
    return GaussianResidualDiffusion(denoiser, config.residual.diffusion_steps)


def _epoch_loss(
    model: GaussianResidualDiffusion,
    loader: DataLoader[dict[str, object]],
    graph_store: AsOfGraphStore,
    config: ExperimentConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    model.train(optimizer is not None)
    total = 0.0
    batches = 0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        route = _route_embedding(
            batch, graph_store, config.graph.embedding_dim, config.seed, device
        )
        context, moving_average = residual_context(
            batch["booking_values"],
            batch["booking_mask"],
            route,
            batch["target_intervals"],
            config.residual.moving_average_window,
        )
        residual = batch["target"] - moving_average
        valid = batch["target_mask"].bool()
        loss = model.loss(residual[valid], context[valid])
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        total += float(loss.detach().cpu())
        batches += 1
    return total / max(batches, 1)


def train_phase_a(
    train: AirfareDataset,
    validation: AirfareDataset,
    graph_store: AsOfGraphStore,
    config: ExperimentConfig,
    checkpoint_path: str | Path,
) -> dict[str, float]:
    seed_everything(config.seed)
    device = resolve_device(config.device)
    route_dim = config.graph.embedding_dim * 3
    model = _model(config, route_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    train_loader = _loader(train, config, shuffle=True)
    validation_loader = _loader(validation, config, shuffle=False)
    best = float("inf")
    stale_epochs = 0
    history: dict[str, float] = {}
    for epoch in range(config.training.max_epochs):
        train_loss = _epoch_loss(model, train_loader, graph_store, config, device, optimizer)
        with torch.no_grad():
            validation_loss = _epoch_loss(
                model, validation_loader, graph_store, config, device, None
            )
        history = {
            "phase_a_epoch": float(epoch + 1),
            "phase_a_train_loss": train_loss,
            "phase_a_validation_loss": validation_loss,
        }
        if validation_loss < best:
            best = validation_loss
            stale_epochs = 0
            atomic_torch_save(
                {
                    "model": model.state_dict(),
                    "config_sha256": config.digest(),
                    "validation_loss": best,
                    "epoch": epoch + 1,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= config.training.patience:
                break
    return history | {"phase_a_best_validation_loss": best}


@torch.no_grad()
def materialize_trends(
    datasets: list[AirfareDataset],
    graph_store: AsOfGraphStore,
    config: ExperimentConfig,
    checkpoint_path: str | Path,
    cache_path: str | Path,
) -> TrendCache:
    device = resolve_device(config.device)
    route_dim = config.graph.embedding_dim * 3
    model = _model(config, route_dim).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint["config_sha256"] != config.digest():
        raise ValueError("Phase-A checkpoint and configuration do not match")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    values: dict[str, np.ndarray] = {}
    for dataset in datasets:
        for raw_batch in _loader(dataset, config, shuffle=False):
            batch = move_batch(raw_batch, device)
            route = _route_embedding(
                batch, graph_store, config.graph.embedding_dim, config.seed, device
            )
            context, moving_average = residual_context(
                batch["booking_values"],
                batch["booking_mask"],
                route,
                batch["target_intervals"],
                config.residual.moving_average_window,
            )
            residual = model.ddim_sample(
                context.reshape(-1, context.shape[-1]), config.residual.ddim_steps
            ).view_as(moving_average)
            trend = moving_average + residual
            for index, sample_id in enumerate(batch["sample_id"]):
                values[str(sample_id)] = trend[index].cpu().numpy().astype(np.float32)
    cache = TrendCache(values)
    cache.save(cache_path)
    return cache
