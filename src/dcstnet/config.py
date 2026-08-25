from __future__ import annotations

import dataclasses
import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any


@dataclasses.dataclass(frozen=True)
class DataConfig:
    prices_path: str
    itineraries_path: str
    co_bookings_path: str
    train_end: str
    validation_end: str
    test_end: str
    max_interval: int = 90
    lookback: int = 30
    missing_threshold: float = 0.5
    horizon: int = 15
    alert_threshold: float = 50.0


@dataclasses.dataclass(frozen=True)
class GraphConfig:
    layers: int = 2
    heads: int = 4
    top_k: int = 12
    hidden_dim: int = 64
    embedding_dim: int = 32
    co_booking_percentile: float = 85.0
    jaccard_threshold: float = 0.3


@dataclasses.dataclass(frozen=True)
class CalendarConfig:
    receptive_fields: tuple[int, ...] = (3, 7, 14, 30)
    hidden_dim: int = 64


@dataclasses.dataclass(frozen=True)
class ResidualConfig:
    moving_average_window: int = 7
    diffusion_steps: int = 1000
    ddim_steps: int = 20
    hidden_dim: int = 128


@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 512
    max_epochs: int = 50
    patience: int = 7
    softmin_rho: float = 2.0
    alert_lambda: float = 0.5
    false_alert_weight: float = 1.2
    missed_opportunity_weight: float = 0.8
    num_workers: int = 0


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    output_dir: str
    device: str
    data: DataConfig
    graph: GraphConfig
    calendar: CalendarConfig
    residual: ResidualConfig
    training: TrainingConfig

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def digest(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing [{key}] section")
    return value


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).resolve()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    experiment = _section(raw, "experiment")
    base = config_path.parent.parent

    data_values = _section(raw, "data").copy()
    for key in ("prices_path", "itineraries_path", "co_bookings_path"):
        candidate = Path(data_values[key])
        if not candidate.is_absolute():
            data_values[key] = str((base / candidate).resolve())

    calendar_values = _section(raw, "calendar").copy()
    calendar_values["receptive_fields"] = tuple(calendar_values["receptive_fields"])

    output_dir = Path(experiment["output_dir"])
    if not output_dir.is_absolute():
        output_dir = (base / output_dir).resolve()

    config = ExperimentConfig(
        seed=int(experiment["seed"]),
        output_dir=str(output_dir),
        device=str(experiment.get("device", "auto")),
        data=DataConfig(**data_values),
        graph=GraphConfig(**_section(raw, "graph")),
        calendar=CalendarConfig(**calendar_values),
        residual=ResidualConfig(**_section(raw, "residual")),
        training=TrainingConfig(**_section(raw, "training")),
    )
    validate_config(config)
    return config


def validate_config(config: ExperimentConfig) -> None:
    if config.data.horizon < 1 or config.data.horizon >= config.data.max_interval:
        raise ValueError("horizon must be in [1, max_interval)")
    if not 0.0 <= config.data.missing_threshold < 1.0:
        raise ValueError("missing_threshold must be in [0, 1)")
    if config.graph.hidden_dim % config.graph.heads:
        raise ValueError("graph hidden_dim must be divisible by heads")
    if config.graph.embedding_dim % config.graph.heads:
        raise ValueError("graph embedding_dim must be divisible by heads")
    if config.residual.ddim_steps > config.residual.diffusion_steps:
        raise ValueError("ddim_steps cannot exceed diffusion_steps")
    if any(field < 1 for field in config.calendar.receptive_fields):
        raise ValueError("calendar receptive fields must be positive")
