# Portfolio Engine

The Portfolio Engine is the core backtesting component of BIST Quant. This guide covers its architecture and usage.

## Overview

The `PortfolioEngine` class handles:

- Data loading and preprocessing
- Signal generation and scoring
- Portfolio construction and rebalancing
- Performance calculation and reporting

## Initialization

```python
from pathlib import Path
from Models.portfolio_engine import PortfolioEngine

engine = PortfolioEngine(
    data_dir=Path("data"),
    regime_model_dir=Path("regime_filter/outputs"),
    start_date="2020-01-01",
    end_date="2024-12-31",
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `data_dir` | `Path` | Directory containing price and fundamental data |
| `regime_model_dir` | `Path` | Directory with regime classification outputs |
| `start_date` | `str` | Backtest start date (YYYY-MM-DD) |
| `end_date` | `str` | Backtest end date (YYYY-MM-DD) |

## Data Loading

```python
# Load all required data
engine.load_all_data()

# Access loaded data
print(f"Price panel shape: {engine.close_df.shape}")
print(f"Tickers: {engine.close_df.columns.tolist()[:10]}")
print(f"Date range: {engine.close_df.index.min()} to {engine.close_df.index.max()}")
```

### Available Data Panels

After calling `load_all_data()`:

- `engine.close_df` - Closing prices
- `engine.open_df` - Opening prices
- `engine.high_df` - High prices
- `engine.low_df` - Low prices
- `engine.volume_df` - Trading volume
- `engine.prices` - Raw price DataFrame

## Running Backtests

### Single Signal

```python
# Run with default config
engine.run_factor("momentum")

# Run with custom config override
custom_config = {
    "signal": {
        "name": "momentum",
        "builder": "MomentumSignalBuilder",
        "params": {"lookback": 120},
    },
    "timeline": {
        "start_date": "2021-01-01",
        "end_date": "2024-12-31",
    },
    "portfolio": {
        "top_n": 20,
        "rebalance_frequency": "weekly",
    },
}
engine.run_factor("momentum", override_config=custom_config)
```

### All Signals

```python
# Run all configured signals
engine.run_all_factors()
```

## Risk Management

The engine integrates with the `RiskManager` class:

```python
from Models.common.risk_manager import RiskManager

risk_manager = RiskManager(
    close_df=engine.close_df,
    volume_df=engine.volume_df,
)

# Calculate inverse volatility weights
weights = risk_manager.inverse_downside_vol_weights(
    selected=["TICKER1", "TICKER2", "TICKER3"],
    date=pd.Timestamp("2024-01-15"),
    lookback=60,
    max_weight=0.25,
)
```

## Performance Metrics

Backtest results include standard performance metrics:

```python
# Access results after running backtest
results = engine.latest_results

print(f"CAGR: {results['cagr']:.2%}")
print(f"Sharpe Ratio: {results['sharpe']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
print(f"Win Rate: {results['win_rate']:.2%}")
```

### Available Metrics

- **CAGR**: Compound Annual Growth Rate
- **Sharpe Ratio**: Risk-adjusted returns
- **Sortino Ratio**: Downside-risk adjusted returns
- **Max Drawdown**: Largest peak-to-trough decline
- **Calmar Ratio**: CAGR / Max Drawdown
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profits / Gross losses

## Benchmark Comparison

Results are automatically compared against benchmarks:

- **XU100**: BIST 100 Index
- **XAU/TRY**: Gold in Turkish Lira

```python
# Benchmark metrics are included in results
print(f"Alpha vs XU100: {results['alpha_xu100']:.2%}")
print(f"Beta vs XU100: {results['beta_xu100']:.2f}")
```

## Output Files

Results are saved to `Models/results/{signal_name}/`:

```
Models/results/momentum/
├── performance_summary.json
├── trades.csv
├── equity_curve.csv
├── monthly_returns.csv
└── charts/
    ├── equity_curve.png
    └── drawdown.png
```
