from __future__ import annotations

import dataclasses
import datetime as dt
import json
from pathlib import Path

from .config import ExperimentConfig, load_config
from .data.dataset import PreparedData, prepare_data
from .data.graph import AsOfGraphStore
from .data.schema import read_co_bookings, read_itineraries, read_prices
from .evaluation.replay import evaluate_checkpoint
from .training.cache import TrendCache
from .training.common import write_manifest
from .training.phase_a import materialize_trends, train_phase_a
from .training.phase_b import train_phase_b


@dataclasses.dataclass
class PipelineContext:
    config: ExperimentConfig
    data: PreparedData
    graph_store: AsOfGraphStore
    output_dir: Path

    @property
    def phase_a_checkpoint(self) -> Path:
        return self.output_dir / "checkpoints" / "phase_a.pt"

    @property
    def trend_cache_path(self) -> Path:
        return self.output_dir / "cache" / "phase_a_trends.npz"

    @property
    def phase_b_checkpoint(self) -> Path:
        return self.output_dir / "checkpoints" / "phase_b.pt"

    @property
    def evaluation_dir(self) -> Path:
        return self.output_dir / "evaluation"


def prepare_context(config_path: str | Path) -> PipelineContext:
    config = load_config(config_path)
    prices = read_prices(config.data.prices_path)
    itineraries = read_itineraries(config.data.itineraries_path)
    co_bookings = read_co_bookings(config.data.co_bookings_path)
    graph_store = AsOfGraphStore(
        itineraries=itineraries,
        co_bookings=co_bookings,
        prices=prices,
        top_k=config.graph.top_k,
        co_booking_percentile=config.graph.co_booking_percentile,
        jaccard_threshold=config.graph.jaccard_threshold,
    )
    reference_date = dt.date.fromisoformat(config.data.train_end)
    reference_graph = graph_store.for_date(reference_date)
    prepared = prepare_data(prices, reference_graph.itinerary_nodes, config.data)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "config_sha256": config.digest(),
        "splits": {
            "train": len(prepared.train),
            "validation": len(prepared.validation),
            "test": len(prepared.test),
        },
        "graph": {
            "nodes": len(reference_graph.node_ids),
            "directed_edges": int(reference_graph.edge_index.shape[1]),
            "reference_snapshot": reference_graph.snapshot_date.isoformat(),
        },
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return PipelineContext(config, prepared, graph_store, output_dir)


def run_phase_a(context: PipelineContext) -> dict[str, float]:
    metrics = train_phase_a(
        context.data.train,
        context.data.validation,
        context.graph_store,
        context.config,
        context.phase_a_checkpoint,
    )
    (context.output_dir / "phase_a_history.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metrics


def run_materialization(context: PipelineContext) -> TrendCache:
    if not context.phase_a_checkpoint.is_file():
        raise FileNotFoundError("Phase-A checkpoint is required before materialization")
    return materialize_trends(
        [context.data.train, context.data.validation, context.data.test],
        context.graph_store,
        context.config,
        context.phase_a_checkpoint,
        context.trend_cache_path,
    )


def run_phase_b(context: PipelineContext) -> dict[str, float]:
    if not context.trend_cache_path.is_file():
        raise FileNotFoundError("Detached Phase-A trend cache is required before Phase B")
    cache = TrendCache.load(context.trend_cache_path)
    metrics = train_phase_b(
        context.data.train,
        context.data.validation,
        context.graph_store,
        cache,
        context.config,
        context.phase_b_checkpoint,
    )
    (context.output_dir / "phase_b_history.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metrics


def run_evaluation(context: PipelineContext) -> dict[str, float | int]:
    if not context.phase_b_checkpoint.is_file():
        raise FileNotFoundError("Phase-B checkpoint is required before evaluation")
    cache = TrendCache.load(context.trend_cache_path)
    return evaluate_checkpoint(
        context.data.test,
        context.graph_store,
        cache,
        context.config,
        context.phase_b_checkpoint,
        context.evaluation_dir,
    )


def run_all(config_path: str | Path) -> dict[str, float | int]:
    context = prepare_context(config_path)
    run_phase_a(context)
    run_materialization(context)
    run_phase_b(context)
    metrics = run_evaluation(context)
    write_manifest(
        context.config,
        context.output_dir / "manifest.json",
        {
            "prices": context.config.data.prices_path,
            "itineraries": context.config.data.itineraries_path,
            "co_bookings": context.config.data.co_bookings_path,
            "phase_a_checkpoint": context.phase_a_checkpoint,
            "phase_a_trends": context.trend_cache_path,
            "phase_b_checkpoint": context.phase_b_checkpoint,
            "metrics": context.evaluation_dir / "metrics.json",
            "predictions": context.evaluation_dir / "predictions.csv",
        },
    )
    return metrics
