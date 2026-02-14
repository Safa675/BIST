# Running Backtests

This guide covers how to run backtests using the BIST Quant library.

## Command Line Interface

The simplest way to run a backtest is using the CLI:

```bash
# Run a specific signal
bist-backtest momentum

# Run all signals
bist-backtest all

# Custom date range
bist-backtest momentum --start-date 2020-01-01 --end-date 2024-12-31

# Dry run (validate without executing)
bist-backtest momentum --dry-run

# Verbose output
bist-backtest momentum --verbose
```

## Available Options

| Option | Description |
|--------|-------------|
| `--start-date` | Backtest start date (YYYY-MM-DD) |
| `--end-date` | Backtest end date (YYYY-MM-DD) |
| `--data-dir` | Custom data directory path |
| `--regime-outputs` | Path to regime features directory |
| `--list-signals` | List all available signals |
| `--dry-run` | Validate inputs without running |
| `--verbose` / `-v` | Enable debug logging |
| `--quiet` / `-q` | Suppress info logging |
| `--no-color` | Disable colored output |
| `--version` | Show version and exit |

## Programmatic Usage

You can also run backtests programmatically:

```python
from pathlib import Path
from Models.portfolio_engine import PortfolioEngine, load_signal_configs

# Initialize engine
engine = PortfolioEngine(
    data_dir=Path("data"),
    regime_model_dir=Path("regime_filter/outputs"),
    start_date="2020-01-01",
    end_date="2024-12-31",
)

# Load data
engine.load_all_data()

# Run a single factor
engine.run_factor("momentum")

# Or run all factors
engine.run_all_factors()
```

## Understanding Results

Backtest results are saved to the `Models/results/` directory and include:

- **Performance metrics**: CAGR, Sharpe ratio, max drawdown, etc.
- **Trade logs**: Entry/exit dates, positions, returns
- **Equity curves**: Time series of portfolio value
- **Comparison charts**: Strategy vs benchmark performance

## Configuration Files

Signal configurations are stored in `Models/configs/` as YAML files:

```yaml
# Example: momentum.yaml
signal:
  name: momentum
  builder: MomentumSignalBuilder
  params:
    lookback: 60
    skip_days: 5

timeline:
  start_date: "2018-01-01"
  end_date: "2024-12-31"

portfolio:
  top_n: 10
  rebalance_frequency: monthly
```

See [Signal Configuration](signals.md) for more details.
