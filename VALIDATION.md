# Validation record

The repository was validated on Python 3.12 with NumPy 2.3.5 and PyTorch
2.13.0. The following checks were run from a clean project working directory:

```bash
python scripts/generate_example_data.py
ruff check src scripts tests
python -m unittest discover -s tests -v
python scripts/audit_repository.py
python -m pip wheel . --no-deps --no-build-isolation
```

The test suite executes the complete Phase-A training, detached trend
materialization, Phase-B training, and test replay on the public example. It
also checks event-time masking, graph-snapshot cutoffs, top-k pruning,
normalized soft-min behavior, the Phase-A gradient boundary, alert metric
denominators, user-level bootstrap inference, cache versioning, and fallbacks.

Validation result: all 16 tests passed; static analysis and repository audit
passed; the package wheel built successfully. The end-to-end example evaluated
312 request-time samples and 864 valid trajectory points. These counts only
describe the included deterministic example and are not paper results.
