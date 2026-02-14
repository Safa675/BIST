# Troubleshooting

## 1. `Unknown signal` in CLI

Symptoms:

- `run_backtest.py` fails with unknown signal name.

Checks:

1. `python run_backtest.py --list-signals`
2. confirm strategy exists in `configs/strategies.yaml`
3. ensure config `description` is present and valid

## 2. Freshness Gate Blocking Runs

Symptoms:

- backtest or load fails with freshness gate violation.

Resolution:

1. run diagnostics:

```bash
python scripts/fetch_integrate_fundamentals.py --diagnostics-only
```

2. refresh data (`--merge-only` or fetch path).
3. temporary non-production override:

```bash
BIST_ALLOW_STALE_FUNDAMENTALS=1 python run_backtest.py momentum
```

## 3. `No regime outputs directory found`

Symptoms:

- CLI fails before engine starts.

Resolution:

- provide `--regime-outputs <path>` with `regime_features.csv`
- or generate outputs under `Simple Regime Filter/outputs` or `regime_filter/outputs`

## 4. Benchmark Regression Fails

Symptoms:

- `benchmarking_cli` exits with code 2.

Resolution:

1. inspect `benchmarks/phase6_latest.json`
2. compare with `benchmarks/phase6_baseline.json`
3. rerun quick check:

```bash
python -m Models.common.benchmarking_cli --skip-regression-check --repeats 1 --warmup 0
```

4. if changes are intentional and reviewed, refresh baseline with `--write-baseline`

## 5. Import Errors After Refactor

Symptoms:

- `ModuleNotFoundError` from script-style imports.

Resolution:

- use package imports under `Models.*`
- install project in editable mode:

```bash
pip install -e ".[dev]"
```

## 6. Mypy or Ruff Failures in Legacy Modules

Symptoms:

- strict checks fail in untouched legacy signal modules.

Resolution:

- keep Phase 6 gate focused on modernized targets first.
- expand lint/type scope gradually with targeted fixes per module group.
