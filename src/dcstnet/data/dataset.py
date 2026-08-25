from __future__ import annotations

import dataclasses
import datetime as dt
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import DataConfig
from .masking import is_available, valid_horizon
from .schema import PriceRecord


@dataclasses.dataclass(frozen=True)
class Normalization:
    mean: float
    std: float


@dataclasses.dataclass
class PreparedData:
    train: AirfareDataset
    validation: AirfareDataset
    test: AirfareDataset
    normalization: dict[str, Normalization]


class AirfareDataset(Dataset[dict[str, torch.Tensor | str]]):
    def __init__(self, samples: list[dict[str, torch.Tensor | str]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        return self.samples[index]


def _compute_normalization(
    records: Sequence[PriceRecord], train_end: dt.date
) -> dict[str, Normalization]:
    values: dict[str, list[float]] = defaultdict(list)
    global_values: list[float] = []
    for record in records:
        if record.booking_date <= train_end:
            values[record.itinerary_id].append(record.price)
            global_values.append(record.price)
    if not global_values:
        raise ValueError("The training split contains no prices")
    global_mean = float(np.mean(global_values))
    global_std = max(float(np.std(global_values)), 1.0)
    result: dict[str, Normalization] = {}
    for itinerary_id, prices in values.items():
        if len(prices) < 4:
            result[itinerary_id] = Normalization(global_mean, global_std)
        else:
            result[itinerary_id] = Normalization(
                float(np.mean(prices)), max(float(np.std(prices)), 1.0)
            )
    return result


def _calendar_view(
    itinerary_id: str,
    departure_date: dt.date,
    target_interval: int,
    cutoff: dt.date,
    lookback: int,
    lookup: dict[tuple[str, dt.date, int], float],
    normalization: Normalization,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros(lookback, dtype=np.float32)
    mask = np.zeros(lookback, dtype=np.float32)
    last_value: float | None = None
    for offset in range(lookback, 0, -1):
        candidate_date = departure_date - dt.timedelta(days=offset)
        position = lookback - offset
        key = (itinerary_id, candidate_date, target_interval)
        if not is_available(candidate_date, target_interval, cutoff):
            continue
        if key in lookup:
            last_value = lookup[key]
        if last_value is not None:
            values[position] = (last_value - normalization.mean) / normalization.std
            mask[position] = 1.0
    return values, mask


def _booking_view(
    itinerary_id: str,
    departure_date: dt.date,
    current_interval: int,
    max_interval: int,
    lookup: dict[tuple[str, dt.date, int], float],
    normalization: Normalization,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros(max_interval, dtype=np.float32)
    mask = np.zeros(max_interval, dtype=np.float32)
    intervals = list(range(max_interval, current_interval - 1, -1))
    start = max_interval - len(intervals)
    last_value: float | None = None
    for position, interval in enumerate(intervals, start=start):
        key = (itinerary_id, departure_date, interval)
        if key in lookup:
            last_value = lookup[key]
        if last_value is not None:
            values[position] = (last_value - normalization.mean) / normalization.std
            mask[position] = 1.0
    return values, mask


def _sample_id(record: PriceRecord) -> str:
    return "|".join(
        (
            record.itinerary_id,
            record.departure_date.isoformat(),
            record.booking_date.isoformat(),
        )
    )


def prepare_data(
    records: Sequence[PriceRecord],
    itinerary_nodes: dict[str, tuple[int, int, int]],
    config: DataConfig,
) -> PreparedData:
    train_end = dt.date.fromisoformat(config.train_end)
    validation_end = dt.date.fromisoformat(config.validation_end)
    test_end = dt.date.fromisoformat(config.test_end)
    if not train_end < validation_end < test_end:
        raise ValueError("Expected train_end < validation_end < test_end")

    filtered = [
        record
        for record in records
        if record.itinerary_id in itinerary_nodes and record.booking_interval <= config.max_interval
    ]
    normalization = _compute_normalization(filtered, train_end)
    lookup: dict[tuple[str, dt.date, int], float] = {}
    for record in filtered:
        key = (record.itinerary_id, record.departure_date, record.booking_interval)
        if key in lookup:
            raise ValueError(f"Duplicate price cell: {key}")
        lookup[key] = record.price

    by_itinerary: dict[str, list[PriceRecord]] = defaultdict(list)
    for record in filtered:
        by_itinerary[record.itinerary_id].append(record)
    eligible_itineraries = set()
    for itinerary_id, itinerary_records in by_itinerary.items():
        possible = len({(item.departure_date, item.booking_interval) for item in itinerary_records})
        observed_span = (
            len({item.departure_date for item in itinerary_records}) * config.max_interval
        )
        missing_rate = 1.0 - possible / max(observed_span, 1)
        if missing_rate <= config.missing_threshold:
            eligible_itineraries.add(itinerary_id)
    if not eligible_itineraries:
        raise ValueError("All itineraries exceed the configured missing-rate threshold")

    splits: dict[str, list[dict[str, torch.Tensor | str]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for record in sorted(
        filtered, key=lambda item: (item.booking_date, item.itinerary_id, item.departure_date)
    ):
        if record.itinerary_id not in eligible_itineraries:
            continue
        horizon = valid_horizon(record.booking_interval, config.horizon)
        if horizon < 1:
            continue
        target_prices: list[float] = []
        target_mask = np.zeros(config.horizon, dtype=np.float32)
        target_intervals = np.zeros(config.horizon, dtype=np.float32)
        calendar_values = np.zeros((config.horizon, config.lookback), dtype=np.float32)
        calendar_mask = np.zeros_like(calendar_values)
        norm = normalization[record.itinerary_id]
        for tau in range(1, horizon + 1):
            target_interval = record.booking_interval - tau
            target = lookup.get((record.itinerary_id, record.departure_date, target_interval))
            if target is None:
                target_prices.append(0.0)
                continue
            target_prices.append((target - norm.mean) / norm.std)
            target_mask[tau - 1] = 1.0
            target_intervals[tau - 1] = target_interval / config.max_interval
            view, view_mask = _calendar_view(
                record.itinerary_id,
                record.departure_date,
                target_interval,
                record.booking_date,
                config.lookback,
                lookup,
                norm,
            )
            calendar_values[tau - 1] = view
            calendar_mask[tau - 1] = view_mask
        while len(target_prices) < config.horizon:
            target_prices.append(0.0)
        if target_mask.sum() == 0:
            continue
        booking_values, booking_mask = _booking_view(
            record.itinerary_id,
            record.departure_date,
            record.booking_interval,
            config.max_interval,
            lookup,
            norm,
        )
        opportunity = float(
            record.price
            - min(
                value * norm.std + norm.mean
                for value, valid in zip(target_prices, target_mask, strict=True)
                if valid
            )
            >= config.alert_threshold
        )
        origin_node, destination_node, flight_node = itinerary_nodes[record.itinerary_id]
        sample: dict[str, torch.Tensor | str] = {
            "sample_id": _sample_id(record),
            "itinerary_id": record.itinerary_id,
            "booking_date": record.booking_date.isoformat(),
            "departure_date": record.departure_date.isoformat(),
            "calendar_values": torch.from_numpy(calendar_values),
            "calendar_mask": torch.from_numpy(calendar_mask),
            "booking_values": torch.from_numpy(booking_values),
            "booking_mask": torch.from_numpy(booking_mask),
            "target": torch.tensor(target_prices, dtype=torch.float32),
            "target_mask": torch.from_numpy(target_mask),
            "target_intervals": torch.from_numpy(target_intervals),
            "current_interval": torch.tensor(record.booking_interval, dtype=torch.long),
            "current_price": torch.tensor((record.price - norm.mean) / norm.std),
            "threshold": torch.tensor(config.alert_threshold / norm.std),
            "opportunity": torch.tensor(opportunity),
            "mean": torch.tensor(norm.mean),
            "std": torch.tensor(norm.std),
            "node_indices": torch.tensor(
                [origin_node, destination_node, flight_node], dtype=torch.long
            ),
            "is_holiday": torch.tensor(float(record.is_holiday)),
        }
        if record.booking_date <= train_end:
            splits["train"].append(sample)
        elif record.booking_date <= validation_end:
            splits["validation"].append(sample)
        elif record.booking_date <= test_end:
            splits["test"].append(sample)

    if any(not splits[name] for name in splits):
        sizes = ", ".join(f"{name}={len(value)}" for name, value in splits.items())
        raise ValueError(f"Every split must contain at least one valid target ({sizes})")
    return PreparedData(
        train=AirfareDataset(splits["train"]),
        validation=AirfareDataset(splits["validation"]),
        test=AirfareDataset(splits["test"]),
        normalization=normalization,
    )
