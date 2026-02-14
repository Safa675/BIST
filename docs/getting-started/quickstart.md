# Quick Start

This guide walks you through running your first backtest.

## Prerequisites

1. Install the package: `pip install -e ".[dev]"`
2. Prepare data files in the `data/` directory:
   - `bist_prices_full.parquet` or `bist_prices_full.csv`
   - `xau_try_2013_2026.csv`
   - `xu100_prices.csv`
3. Run the regime pipeline to generate `regime_filter/outputs/regime_features.csv`

## Running a Backtest

### Via CLI

```bash
# Run momentum strategy
bist-backtest momentum --start-date 2020-01-01 --end-date 2024-12-31

# Run all strategies
bist-backtest all

# List available signals
bist-backtest --list-signals

# Dry run (validate inputs)
bist-backtest momentum --dry-run
```

### Via Python

```python
from Models import PortfolioEngine

# Initialize engine
engine = PortfolioEngine(
    data_dir="data/",
    regime_model_dir="regime_filter/outputs",
    start_date="2020-01-01",
    end_date="2024-12-31",
)

# Load data
engine.load_all_data()

# Run single factor
results = engine.run_factor("momentum")

# Or run all factors
engine.run_all_factors()
```

## Understanding Results

Results are saved to `Models/results/<factor_name>/`:

- `summary.txt` - Key metrics (CAGR, Sharpe, max drawdown)
- `equity_curve.csv` - Daily equity values
- `returns.csv` - Daily returns
- `yearly_metrics.csv` - Year-by-year performance
- `holdings_history.csv` - Position history

## Next Steps

- [Signal Configuration](../guide/signals.md) - Customize signal parameters
- [Portfolio Engine](../guide/portfolio-engine.md) - Advanced engine usage
- [Examples](../examples/quick-start.md) - More code examples
