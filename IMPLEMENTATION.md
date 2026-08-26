# Implementation notes

## Main components

| Component | Location |
|---|---|
| Availability masks and valid horizons | `dcstnet.data.masking` |
| Dual-time surface construction | `dcstnet.data.dataset` |
| Dated heterogeneous route graphs | `dcstnet.data.graph.AsOfGraphStore` |
| Directed top-k graph attention | `dcstnet.models.graph` |
| Multi-scale calendar features | `dcstnet.models.calendar` |
| Residual decomposition and diffusion | `dcstnet.models.diffusion` |
| Phase-A trajectory cache | `dcstnet.training.cache.TrendCache` |
| Decision-aware loss | `dcstnet.losses.decision_aware_loss` |
| Dual-time fusion training | `dcstnet.training.phase_b` |
| Rolling replay and metrics | `dcstnet.evaluation` |
| Cached trajectories and alert rule | `dcstnet.serving` |

## Temporal data handling

Surface construction starts from the request date. Observations after that date
are masked before filling, normalization, or feature extraction. Normalization
statistics are fitted on the training split. Batches are grouped by date so
that all examples in a batch use the same completed graph snapshot.

Evaluation follows the same ordering: the graph store selects a snapshot no
later than the request date, the model predicts the valid part of the future
trajectory, and alert labels are computed only after the corresponding horizon
has closed. Raw unmasked surfaces are not passed to model modules.

## Training stages

Phase A trains the residual trajectory model. The `materialize` command writes
its outputs to a separate array together with itinerary, request-date, and
configuration identifiers. Phase B reads this array as a detached feature, so
its optimizer cannot update the Phase-A model.

The command-line stages may be run independently:

```bash
dcst prepare --config configs/demo.toml
dcst train-phase-a --config configs/demo.toml
dcst materialize --config configs/demo.toml
dcst train-phase-b --config configs/demo.toml
dcst evaluate --config configs/demo.toml
```

## Run directory

A completed run has the following layout:

```text
dataset_summary.json
phase_a_history.json
phase_b_history.json
checkpoints/phase_a.pt
checkpoints/phase_b.pt
cache/phase_a_trends.npz
evaluation/predictions.csv
evaluation/metrics.json
manifest.json
```

`manifest.json` stores the configuration digest, seed, file sizes, and hashes.
Checkpoint and cache readers compare configuration identifiers before loading
saved state, which prevents outputs from different runs from being mixed.

## User-level metrics

`scripts/bootstrap_user_metrics.py` reads one row per randomized user with
columns `user_id`, `group`, and the aggregate metric. It draws 1,000 bootstrap
resamples by default. Aggregating sessions before calling the script keeps the
unit of analysis at the randomized-user level.
