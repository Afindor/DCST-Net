# DCST-Net

This repository contains the reference implementation of **DCST-Net**, a
mask-preserving dual-time forecasting system for Web airfare alerts. The code
follows the experimental order used by the method: construct serving-valid
surfaces and as-of route graphs, train the offline residual forecaster,
materialize its detached trajectories, train decision-aware fusion, and replay
the fixed alert policy after each target horizon closes.

## Quick start

DCST-Net requires Python 3.11 or newer. A CPU is sufficient for the included
example; CUDA is recommended for full-scale experiments.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python scripts/generate_example_data.py
dcst run --config configs/demo.toml
python -m unittest discover -s tests -v
python scripts/audit_repository.py
```

The demo writes checkpoints, the immutable Phase-A trend cache, predictions,
metrics, and a SHA-256 artifact manifest under `artifacts/demo/`. The example
corpus is deterministic and is intended to verify execution rather than
reproduce the proprietary-data measurements in the paper.

## Execution contract

The command-line interface exposes each artifact boundary independently:

```bash
dcst prepare --config configs/demo.toml
dcst train-phase-a --config configs/demo.toml
dcst materialize --config configs/demo.toml
dcst train-phase-b --config configs/demo.toml
dcst evaluate --config configs/demo.toml
```

`dcst run` executes the same commands in order. Phase B refuses to start until
the Phase-A cache exists, and checkpoint loading verifies the configuration
digest. Test replay uses the latest completed weekly graph snapshot no later
than each request date.

## Repository layout

```text
configs/                         Paper and executable example configurations
data/example/                    Deterministic, non-production input sample
scripts/                         End-to-end, baseline-export, and audit commands
src/dcstnet/data/                Schemas, masks, surfaces, splits, route graphs
src/dcstnet/models/              Graph, calendar, diffusion, and fusion modules
src/dcstnet/training/            Phase A, detached cache, and Phase B training
src/dcstnet/evaluation/          Rolling replay, metrics, and user bootstrap
src/dcstnet/serving/             Trajectory cache and fixed alert rule
tests/                           Unit and end-to-end contract tests
```

## Paper configuration

`configs/paper_sampled_h15.toml` and `configs/paper_industry_h15.toml` contain
the reported architecture and optimization settings for the 15-day analysis:
two graph layers, four heads, twelve retained neighbors, 64/32 graph widths,
calendar fields `{3, 7, 14, 30}`, a seven-day residual trend, a 128-unit
two-layer denoiser, 1,000 diffusion steps, 20 deterministic DDIM steps, and the
reported decision-loss weights. For the other horizons, copy the relevant
configuration and change only `data.horizon` to 3, 7, or 30.

The production logs cannot be distributed. To use an authorized corpus, place
files under the paths named by a paper configuration or change those paths to
another local location. The required schemas are documented in [DATA.md](DATA.md).
Normalization is fitted on the training split only; unavailable cells are
masked before any fill or encoding step.

## Baseline contract

The paper uses official implementations of ARIMA, PatchTST, TimesNet,
iTransformer, MTGNN, PDFormer, and STAEformer. Their third-party source trees
are not copied into this repository. `scripts/export_baseline_arrays.py`
exports the booking-date-ordered values, masks, targets, and training-fitted
normalization used by those implementations. Predictions from any baseline can
be evaluated under the same alert policy by converting them to the CSV layout
written by `dcst evaluate` and calling the metric functions in
`dcstnet.evaluation.metrics`.

## Reproducibility boundary

The repository reproduces the processing, model, training, replay, metric, and
serving contracts. It does not contain production records, production model
checkpoints, user-level experiment aggregates, or the paper's reported output
tables. The included example produces its own clearly separated run artifacts.
See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the paper-to-code crosswalk.
