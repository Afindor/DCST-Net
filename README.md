# DCST-Net

DCST-Net is a dual-time forecasting system for airfare price alerts. It models
each itinerary as a booking-date by departure-date surface while preserving the
availability mask seen by the serving system. Route structure, calendar
patterns, and a residual price process are combined to produce trajectories
that can be consumed by a fixed alert policy.

The project is organized around the training pipeline used in our experiments:

1. build masked price surfaces and as-of route graphs;
2. train the residual trajectory model;
3. materialize detached trajectory features;
4. train the decision-aware fusion model; and
5. evaluate forecasts and alerts in rolling replay.

## Quick start

DCST-Net requires Python 3.11 or newer. The example configuration runs on CPU;
CUDA is recommended for larger datasets.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python scripts/generate_example_data.py
dcst run --config configs/demo.toml
```

The run writes checkpoints, detached trajectory features, predictions, and
metrics to `artifacts/demo/`. The generated dataset is deterministic, so it is
also useful for checking changes to preprocessing or model code.

## Training workflow

Each stage can be run separately when developing or comparing model variants:

```bash
dcst prepare --config configs/demo.toml
dcst train-phase-a --config configs/demo.toml
dcst materialize --config configs/demo.toml
dcst train-phase-b --config configs/demo.toml
dcst evaluate --config configs/demo.toml
```

`dcst run` invokes the same stages in order. Phase A learns residual price
trajectories, which are materialized before Phase B trains the dual-time fusion
model. Keeping this boundary explicit prevents gradients from Phase B from
changing the trajectory generator. During rolling replay, graph features are
selected from the latest completed weekly snapshot available at the request
date.

## Repository layout

```text
configs/                         Experiment configurations
data/example/                    Generated example dataset
scripts/                         Data generation, export, and utility commands
src/dcstnet/data/                Schemas, masks, surfaces, splits, route graphs
src/dcstnet/models/              Graph, calendar, diffusion, and fusion modules
src/dcstnet/training/            Phase A, detached cache, and Phase B training
src/dcstnet/evaluation/          Rolling replay, metrics, and user bootstrap
src/dcstnet/serving/             Trajectory cache and alert policy
tests/                           Unit and end-to-end tests
```

## Configurations

`configs/sampled_h15.toml` and `configs/industry_h15.toml` contain the training
profiles for the sampled and industry datasets at the 15-day horizon. Other
horizons use the same structure with the corresponding `data.horizon` value.

Configuration digests are stored with checkpoints and checked when a run is
resumed. Random seeds, input paths, split dates, model sizes, and optimization
settings are all controlled from TOML files.

## Data

The airfare logs used in the industry experiments are not publicly
distributable. [DATA.md](DATA.md) describes the input tables and fields needed
to run the pipeline on another dataset. The generated example under
`data/example/` follows the same schemas on a small synthetic route network.

The data loader applies the serving-time availability mask before filling or
encoding a price surface. Normalization statistics are fitted on the training
split, and route features are read from dated snapshots rather than a graph
constructed over the full observation period.

## Baseline integration

`scripts/export_baseline_arrays.py` exports booking-date-ordered values, masks,
targets, and training-split normalization in a common array format. This keeps
third-party forecasting models in their own environments while sharing the same
inputs and evaluation code. Their predictions can be evaluated with
`dcstnet.evaluation.metrics` after conversion to the CSV layout produced by
`dcst evaluate`.

## Development checks

```bash
python -m unittest discover -s tests -v
ruff check src scripts tests
```

The tests cover temporal masking, as-of graph selection, phase separation,
losses, replay metrics, serving behavior, and an end-to-end run on the example
dataset. [IMPLEMENTATION.md](IMPLEMENTATION.md) describes the implementation
layout, temporal data handling, and saved run state. Additional development
commands are collected in [DEVELOPMENT.md](DEVELOPMENT.md).
