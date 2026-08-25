from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..config import ExperimentConfig
from ..data.dataset import AirfareDataset
from ..data.graph import AsOfGraphStore
from ..losses import decision_aware_loss
from ..models.dcst import DCSTNet
from .cache import TrendCache
from .common import (
    DateBatchSampler,
    atomic_torch_save,
    batch_date,
    move_batch,
    resolve_device,
    seed_everything,
)


def create_model(config: ExperimentConfig, node_feature_dim: int) -> DCSTNet:
    return DCSTNet(
        node_feature_dim=node_feature_dim,
        graph_hidden_dim=config.graph.hidden_dim,
        graph_embedding_dim=config.graph.embedding_dim,
        graph_heads=config.graph.heads,
        graph_layers=config.graph.layers,
        receptive_fields=config.calendar.receptive_fields,
        calendar_hidden_dim=config.calendar.hidden_dim,
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


def _forward(
    model: DCSTNet,
    batch: dict[str, object],
    graph_store: AsOfGraphStore,
    trend_cache: TrendCache,
    device: torch.device,
) -> torch.Tensor:
    graph = graph_store.for_date(batch_date(batch)).to(device)
    detached_trend = trend_cache.get_batch(batch["sample_id"], device).detach()
    return model(
        graph.node_features,
        graph.edge_index,
        graph.edge_type,
        batch["node_indices"],
        batch["calendar_values"],
        batch["calendar_mask"],
        batch["target_intervals"],
        detached_trend,
    )


def _run_epoch(
    model: DCSTNet,
    loader: DataLoader[dict[str, object]],
    graph_store: AsOfGraphStore,
    trend_cache: TrendCache,
    config: ExperimentConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    model.train(optimizer is not None)
    total_loss = 0.0
    batches = 0
    true_positive = 0
    predicted_positive = 0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        predictions = _forward(model, batch, graph_store, trend_cache, device)
        loss, _ = decision_aware_loss(
            predictions=predictions,
            targets=batch["target"],
            mask=batch["target_mask"],
            current_price=batch["current_price"],
            threshold=batch["threshold"],
            opportunity=batch["opportunity"],
            rho=config.training.softmin_rho,
            alert_lambda=config.training.alert_lambda,
            false_alert_weight=config.training.false_alert_weight,
            missed_opportunity_weight=config.training.missed_opportunity_weight,
        )
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        valid_predictions = predictions.masked_fill(~batch["target_mask"].bool(), torch.inf)
        alerts = batch["current_price"] - valid_predictions.min(dim=1).values >= batch["threshold"]
        opportunities = batch["opportunity"].bool()
        true_positive += int((alerts & opportunities).sum().cpu())
        predicted_positive += int(alerts.sum().cpu())
        total_loss += float(loss.detach().cpu())
        batches += 1
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    return total_loss / max(batches, 1), precision


def train_phase_b(
    train: AirfareDataset,
    validation: AirfareDataset,
    graph_store: AsOfGraphStore,
    trend_cache: TrendCache,
    config: ExperimentConfig,
    checkpoint_path: str | Path,
) -> dict[str, float]:
    seed_everything(config.seed)
    device = resolve_device(config.device)
    first_graph = graph_store.for_date(batch_date(next(iter(_loader(train, config, False)))))
    model = create_model(config, first_graph.node_features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    train_loader = _loader(train, config, shuffle=True)
    validation_loader = _loader(validation, config, shuffle=False)
    best_precision = -1.0
    stale_epochs = 0
    history: dict[str, float] = {}
    for epoch in range(config.training.max_epochs):
        train_loss, train_precision = _run_epoch(
            model, train_loader, graph_store, trend_cache, config, device, optimizer
        )
        with torch.no_grad():
            validation_loss, validation_precision = _run_epoch(
                model,
                validation_loader,
                graph_store,
                trend_cache,
                config,
                device,
                None,
            )
        history = {
            "phase_b_epoch": float(epoch + 1),
            "phase_b_train_loss": train_loss,
            "phase_b_train_precision": train_precision,
            "phase_b_validation_loss": validation_loss,
            "phase_b_validation_precision": validation_precision,
        }
        if validation_precision > best_precision:
            best_precision = validation_precision
            stale_epochs = 0
            atomic_torch_save(
                {
                    "model": model.state_dict(),
                    "config_sha256": config.digest(),
                    "validation_precision": best_precision,
                    "epoch": epoch + 1,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= config.training.patience:
                break
    return history | {"phase_b_best_validation_precision": best_precision}
