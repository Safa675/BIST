# 01 - Quick Start

This example demonstrates the basic workflow for running a backtest.

## Overview

1. Initialize the Portfolio Engine
2. Load market data
3. Run a momentum backtest
4. View results

## Full Script

```python
#!/usr/bin/env python3
"""Quick start example: minimal backtest workflow."""

from pathlib import Path
from Models.portfolio_engine import PortfolioEngine

def main():
    # 1. Initialize the engine
    engine = PortfolioEngine(
        data_dir=Path("data"),
        regime_model_dir=Path("regime_filter/outputs"),
        start_date="2020-01-01",
        end_date="2024-12-31",
    )

    # 2. Load all required data
    print("Loading data...")
    engine.load_all_data()

    print(f"Loaded {engine.close_df.shape[1]} tickers")
    print(f"Date range: {engine.close_df.index.min().date()} to {engine.close_df.index.max().date()}")

    # 3. Run a momentum backtest
    print("\nRunning momentum backtest...")
    engine.run_factor("momentum")

    # 4. Results are saved to Models/results/momentum/
    print("\nBacktest complete!")
    print("Results saved to: Models/results/momentum/")

if __name__ == "__main__":
    main()
```

## Step-by-Step Explanation

### 1. Initialize the Engine

```python
engine = PortfolioEngine(
    data_dir=Path("data"),
    regime_model_dir=Path("regime_filter/outputs"),
    start_date="2020-01-01",
    end_date="2024-12-31",
)
```

The `PortfolioEngine` is the main entry point. Configure:
- `data_dir`: Location of price and fundamental data
- `regime_model_dir`: Location of regime classification outputs
- `start_date` / `end_date`: Backtest time range

### 2. Load Data

```python
engine.load_all_data()
```

This loads and preprocesses all required data:
- Price panels (OHLCV)
- Benchmark data (XU100, Gold)
- Regime features
- Signal configurations

### 3. Run Backtest

```python
engine.run_factor("momentum")
```

Executes the backtest for the specified signal. The signal configuration is loaded from `Models/configs/momentum.yaml`.

### 4. View Results

Results are automatically saved to `Models/results/{signal_name}/`. Key files:

- `performance_summary.json`: Key metrics (CAGR, Sharpe, etc.)
- `equity_curve.csv`: Daily portfolio value
- `trades.csv`: Trade log with entry/exit dates

## CLI Alternative

You can also run this via the command line:

```bash
bist-backtest momentum --start-date 2020-01-01 --end-date 2024-12-31
```

## Next Steps

- [02 - Custom Strategy](02-custom-strategy.md): Build multi-factor strategies
- [03 - Factor Analysis](03-factor-analysis.md): Analyze factor performance
