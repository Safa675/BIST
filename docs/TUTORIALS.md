# Tutorials

## Tutorial 1: Run Your First Strategy

```bash
cd /home/safa/Documents/Markets/BIST
source .venv/bin/activate
python run_backtest.py momentum --start-date 2020-01-01 --end-date 2024-12-31
```

Outputs are written under `Models/results/momentum/`.

Key files:
- `summary.txt`
- `equity_curve.csv`
- `returns.csv`
- `yearly_metrics.csv`

## Tutorial 2: Run a Mini End-to-End Workflow

1. Refresh or validate fundamentals:

```bash
python scripts/fetch_integrate_fundamentals.py --diagnostics-only
```

2. Run a strategy:

```bash
python run_backtest.py five_factor_rotation --start-date 2021-01-01 --end-date 2024-12-31
```

3. Review outputs:

```bash
ls -la Models/results/five_factor_rotation
```

## Tutorial 3: Build a Custom Signal

1. Add a builder in one of:
- `Models/signals/momentum.py`
- `Models/signals/value.py`
- `Models/signals/quality.py`
- `Models/signals/technical.py`
- `Models/signals/composite.py`

2. Register it through that module’s `BUILDERS` mapping.

3. Add strategy config in `configs/strategies.yaml` with signal name and params.

4. Run:

```bash
python run_backtest.py <your_signal_name>
```

5. Add tests under `tests/unit/` and integration tests as needed.

## Tutorial 4: Add a New Portfolio Option Safely

1. Add the default in `Models/common/config_manager.py` (`DEFAULT_PORTFOLIO_OPTIONS`).
2. Wire behavior in backtester/risk services.
3. Add tests covering:
- happy path
- boundary values
- NaN/empty handling
- interaction with existing flags

4. Validate:

```bash
ruff check Models tests
pytest -q tests
```

## Tutorial 5: Debug Freshness Gate Failures

When runs fail due stale fundamentals:

1. Diagnose:

```bash
python scripts/fetch_integrate_fundamentals.py --diagnostics-only
```

2. Refresh from cache or fetch:

```bash
python scripts/fetch_integrate_fundamentals.py --merge-only
python scripts/refresh_fundamentals_q4_2025.py --fetch-only --max-tickers 100
```

3. Rerun diagnostics and ensure thresholds pass.

4. Temporary override (not for production):

```bash
BIST_ALLOW_STALE_FUNDAMENTALS=1 python run_backtest.py momentum
```

## Tutorial 6: Run Phase 6 Integration and Benchmark Gates

1. Run integration tests:

```bash
pytest -q tests/integration
```

2. Run full benchmark:

```bash
python -m Models.common.benchmarking_cli --repeats 2 --warmup 1
```

3. Write or validate baseline:

```bash
python -m Models.common.benchmarking_cli --write-baseline
python -m Models.common.benchmarking_cli --max-slowdown-pct 15 --max-memory-regression-pct 15
```
