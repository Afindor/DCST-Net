# Testing

Install the development dependencies and generate the example dataset before
running the test suite:

```bash
python -m pip install -e ".[dev]"
python scripts/generate_example_data.py
python -m unittest discover -s tests -v
```

The unit tests cover surface masking, graph snapshot selection, top-k pruning,
loss calculations, alert metrics, user-level bootstrap sampling, trajectory
cache versioning, and serving fallbacks. The pipeline test runs Phase A,
materializes its detached trajectories, trains Phase B, and evaluates the
result on the example dataset.

Code style and package builds can be checked separately:

```bash
ruff check src scripts tests
python -m pip wheel . --no-deps --no-build-isolation
```

The workflow in `.github/workflows/ci.yml` generates the example data and runs
lint and unit tests on each push.
