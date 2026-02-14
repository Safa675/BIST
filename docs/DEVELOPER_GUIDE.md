# Developer Guide

## 1. Local Setup

```bash
cd /home/safa/Documents/Markets/BIST
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,data-quality]"
pre-commit install
```

## 2. Repository Workflow

1. Create a branch.
2. Implement changes with tests.
3. Run quality checks.
4. Open PR with test evidence and migration notes.

## 3. Quality Gates

### Lint

```bash
ruff check \
  run_backtest.py \
  Models/__init__.py \
  Models/portfolio_engine.py \
  Models/common \
  Models/signals/factory.py \
  Models/signals/five_factor_rotation_signals.py \
  Models/signals/value_signals.py \
  Models/signals/factor_builders.py \
  Models/signals/factor_axes.py \
  common \
  signals \
  tests
```

### Type-check

```bash
mypy
```

### Tests

```bash
pytest -q tests
```

### Integration tests

```bash
pytest -q tests/integration
```

### Security audit

```bash
pip-audit . --desc
```

### Performance + memory benchmark

```bash
python -m Models.common.benchmarking_cli --repeats 2 --warmup 1
python -m Models.common.benchmarking_cli --write-baseline
python -m Models.common.benchmarking_cli --max-slowdown-pct 15 --max-memory-regression-pct 15
```

## 4. Test Structure

```text
tests/
├── fixtures/       # recorded payloads and deterministic test data
├── integration/    # workflow-level tests (pipeline/backtest/report)
├── property/       # Hypothesis invariants
└── unit/           # pure service/function behavior
```

Guidelines:

- Unit tests should avoid network and filesystem side effects unless explicitly tested.
- Integration tests may use `tmp_path` and mocked clients.
- Property tests should encode mathematical invariants, not exact point values.

## 5. Backtester Development Rules

- Preserve no-lookahead behavior (`signal_lag_days` path).
- Keep rebalance selection deterministic.
- Guard weight-sum and return finiteness invariants.
- Add tests for any new portfolio option flag.

## 6. Data Pipeline Development Rules

- Schema checks must fail loudly with typed exceptions.
- Freshness gates should be configurable but safe-by-default.
- Every persisted dataset should include provenance metadata and checksums.
- Cache invalidation must be deterministic from input fingerprint changes.

## 7. Configuration Changes

Preferred location: `configs/strategies.yaml`.

For each strategy change:

1. Validate `description`, `parameters`, and `portfolio_options` schema.
2. Add/adjust tests for config parsing in `tests/unit/test_config_manager.py`.
3. Ensure CLI execution still works via `run_backtest.py`.

## 8. CI Expectations

CI should fail on:

- Ruff lint violations
- Mypy failures
- Pytest failures
- Dependency vulnerabilities from `pip-audit`

Phase 6 integration gate (recommended):

- Integration test suite (`tests/integration`)
- Benchmark regression checks against `benchmarks/phase6_baseline.json`

## 9. Migration and Operations Docs

- `docs/MIGRATION_GUIDE.md`
- `docs/CONFIG_REFERENCE.md`
- `docs/PERFORMANCE_BENCHMARKING.md`
- `docs/TROUBLESHOOTING.md`
- `docs/FINAL_QA_CHECKLIST.md`

## 10. Troubleshooting

### Tests pass locally but fail in CI
- Verify Python version compatibility (`>=3.8,<3.14`).
- Ensure optional deps used by tests are in `[project.optional-dependencies].dev`.

### Freshness gate failures during strategy runs
- Run diagnostics first:
  - `python scripts/fetch_integrate_fundamentals.py --diagnostics-only`
- Refresh fundamentals if stale.
- Use override only for temporary analysis:
  - `BIST_ALLOW_STALE_FUNDAMENTALS=1`

### Slow tests
- Keep synthetic fixtures small and deterministic.
- Avoid loading full production datasets in unit tests.
