# Examples

This section contains practical examples demonstrating how to use the BIST Quant library.

## Quick Start Examples

### [01 - Quick Start](01-quick-start.md)

Minimal example showing the complete workflow: load data, build a signal, run backtest, view results.

### [02 - Custom Strategy](02-custom-strategy.md)

Build a custom multi-factor strategy combining momentum and value signals.

### [03 - Factor Analysis](03-factor-analysis.md)

Analyze factor performance, correlations, and regime-dependent behavior.

## Running the Examples

All example scripts are available in the `examples/` directory:

```bash
# Run quick start example
python examples/01_quick_start.py

# Run custom strategy example
python examples/02_custom_strategy.py

# Run factor analysis example
python examples/03_factor_analysis.py
```

## Prerequisites

Before running the examples, ensure you have:

1. Installed the package with dev dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

2. Downloaded the required data files to the `data/` directory

3. Run the regime filter pipeline (if using regime-aware features):
   ```bash
   # This generates regime_features.csv
   python regime_filter/run_pipeline.py
   ```

## Data Requirements

The examples expect the following data files:

```
data/
├── bist_prices_full.parquet  # Price data (OHLCV)
├── xu100_prices.csv          # XU100 benchmark
└── xau_try_2013_2026.csv     # Gold benchmark
```
