# Reproducibility guide

## Paper-to-code crosswalk

| Method element | Implementation |
|---|---|
| Availability condition `d' - j <= b` | `dcstnet.data.masking.is_available` |
| Truncated horizon `min(H, i - 1)` | `dcstnet.data.masking.valid_horizon` |
| Calendar and booking-interval views | `dcstnet.data.dataset` |
| As-of heterogeneous route graph | `dcstnet.data.graph.AsOfGraphStore` |
| Directed top-k graph attention | `dcstnet.models.graph` |
| Multi-scale calendar channel | `dcstnet.models.calendar` |
| Moving-average residual decomposition | `dcstnet.models.diffusion.booking_summary` |
| DDPM training and deterministic DDIM | `dcstnet.models.diffusion` |
| Detached Phase-A artifact | `dcstnet.training.cache.TrendCache` |
| Decision-aware soft-min loss | `dcstnet.losses.decision_aware_loss` |
| Route/calendar/fusion Phase B | `dcstnet.training.phase_b` |
| Horizon-closed alert replay | `dcstnet.evaluation` |
| User-level bootstrap | `dcstnet.evaluation.bootstrap` |
| Cached serving and threshold rule | `dcstnet.serving` |

## Leakage controls

The implementation enforces the following order for every sample:

1. derive the request cutoff from the observed booking date;
2. remove surface cells whose booking dates exceed the cutoff;
3. apply training-fitted per-itinerary normalization;
4. construct the two temporal views;
5. select the latest completed graph snapshot no later than the request;
6. produce a complete valid-horizon trajectory;
7. construct opportunity and timing labels only from the closed horizon.

The dataset never exposes unmasked raw surface arrays to a model. Date-grouped
batch sampling ensures that each training or replay batch uses one as-of graph
snapshot. Phase-A trends are stored in a separate immutable array and are
detached again when loaded by Phase B.

## Artifacts

A complete run produces:

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

`manifest.json` records the configuration digest, random seed, artifact sizes,
and SHA-256 values. Paper tables should be generated from independently retained
run manifests and prediction files rather than manually entered into this code
repository.

## Online analysis

`scripts/bootstrap_user_metrics.py` accepts one row per randomized user with
columns `user_id`, `group`, and the selected aggregate metric. It performs 1,000
bootstrap resamples by default. The script does not accept session-level rows,
which prevents accidental reuse of session counts as independent observations.
