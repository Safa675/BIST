# BIST Quantitative Finance Library

Production-oriented quantitative research and backtesting library for Borsa Istanbul (BIST), with modular signal builders, a reusable backtesting engine, and a typed fundamentals reliability pipeline.

## Installation

### Prerequisites
- Python 3.8+
- `pip`

### Install (development mode)

```bash
cd /home/safa/Documents/Markets/BIST
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install -e ".[fetchers]"      # external data fetch clients
pip install -e ".[data-quality]"  # pandera schema validation
```

## Quick Start

### List available strategies

```bash
python run_backtest.py --list-signals
```

### Run one strategy

```bash
python run_backtest.py momentum --start-date 2019-01-01 --end-date 2024-12-31
```

### Run all strategies

```bash
python run_backtest.py all
```

### Run fundamentals diagnostics / refresh pipeline

```bash
python scripts/fetch_integrate_fundamentals.py --diagnostics-only
python scripts/fetch_integrate_fundamentals.py --merge-only
python scripts/refresh_fundamentals_q4_2025.py --fetch-only --max-tickers 25
```

## Project Layout

```text
BIST/
├── Models/
│   ├── common/                 # backtester, risk, report, loaders, utilities
│   ├── signals/                # signal builders + five-factor rotation services
│   ├── data_pipeline/          # unified fundamentals reliability pipeline
│   └── portfolio_engine.py     # orchestration entrypoint
├── configs/                    # YAML strategy configuration
├── scripts/                    # operational CLI wrappers
├── tests/                      # unit/integration/property tests + fixtures
├── docs/                       # API reference, dev guide, tutorials
└── run_backtest.py             # primary CLI
```

## Architecture Highlights

- `Models.common.backtester.Backtester`: orchestrates rebalancing, risk controls, transaction costs, and payload assembly.
- `Models.common.backtest_services`: decomposed services for data prep, selection, returns, transaction costs, and metrics.
- `Models.signals.factory`: stable signal builder registry and parameter resolution.
- `Models.signals.five_factor_rotation_signals`: service-oriented five-factor pipeline with cache + MWU weighting.
- `Models.data_pipeline.FundamentalsPipeline`: typed fetch/validate/normalize/merge/freshness/provenance workflow.

## Data Reliability Controls

The fundamentals pipeline enforces schema and freshness checks, emits audit/provenance metadata, and supports deterministic cache invalidation.

Runtime freshness gate env vars:

- `BIST_ENFORCE_FUNDAMENTAL_FRESHNESS=1|0`
- `BIST_ALLOW_STALE_FUNDAMENTALS=1|0`
- `BIST_MAX_MEDIAN_STALENESS_DAYS`
- `BIST_MAX_PCT_OVER_120_DAYS`
- `BIST_MIN_Q4_2025_COVERAGE_PCT`
- `BIST_MAX_MAX_STALENESS_DAYS`
- `BIST_STALENESS_GRACE_DAYS`

## Testing and Quality

Run full suite:

```bash
pytest -q tests
```

Run integration-only suite:

```bash
pytest -q tests/integration
```

Run quality gates:

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
mypy
pip-audit . --desc
```

Pre-commit:

```bash
pre-commit install
pre-commit run --all-files
```

Performance and memory benchmark:

```bash
python -m Models.common.benchmarking_cli --repeats 2 --warmup 1
python -m Models.common.benchmarking_cli --write-baseline
python -m Models.common.benchmarking_cli --max-slowdown-pct 15 --max-memory-regression-pct 15
```

Single-command Phase 6 validation:

```bash
python scripts/validate_phase6_integration.py
```

## Documentation

- API reference: `docs/API_REFERENCE.md`
- Developer guide: `docs/DEVELOPER_GUIDE.md`
- Tutorials: `docs/TUTORIALS.md`
- Migration guide: `docs/MIGRATION_GUIDE.md`
- Config reference: `docs/CONFIG_REFERENCE.md`
- Performance benchmarking guide: `docs/PERFORMANCE_BENCHMARKING.md`
- Troubleshooting playbook: `docs/TROUBLESHOOTING.md`
- Final QA checklist: `docs/FINAL_QA_CHECKLIST.md`

## Troubleshooting

### `No module named hypothesis`
Install dev dependencies:

```bash
pip install -e ".[dev]"
```

### Freshness gate blocks backtests
Inspect diagnostics and either refresh data or (temporary) override:

```bash
python scripts/fetch_integrate_fundamentals.py --diagnostics-only
BIST_ALLOW_STALE_FUNDAMENTALS=1 python run_backtest.py momentum
```

### Missing regime outputs
Ensure `regime_features.csv` exists under one of:
- `Simple Regime Filter/outputs`
- `regime_filter/outputs`

Or pass `--regime-outputs` explicitly.

### Benchmark regression failed
Check latest report and compare against baseline:

```bash
python -m Models.common.benchmarking_cli --skip-regression-check
cat benchmarks/phase6_latest.json
cat benchmarks/phase6_baseline.json
```

## License

Proprietary. All rights reserved.
