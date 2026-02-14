# BIST Quant

A quantitative finance research and backtesting library for Borsa Istanbul (BIST).

## Features

- **52+ Signal Builders**: Comprehensive coverage of factor investing strategies
  - Momentum: price momentum, consistent momentum, residual momentum, trend following
  - Value: P/E, P/B, dividend yield, composite value metrics
  - Quality: profitability, asset growth, earnings quality, Piotroski F-score
  - Technical: moving averages, RSI, MACD, Bollinger Bands, Ichimoku
  - Composite: multi-factor combinations and rotation strategies

- **Portfolio Engine**: YAML-based strategy configuration with flexible backtesting
  - Multiple rebalancing frequencies (monthly, quarterly)
  - Regime-aware allocation adjustment
  - Signal lag controls for realistic execution

- **Risk Management**: Comprehensive risk controls
  - Stop-loss protection
  - Position size limits
  - Volatility targeting
  - Inverse downside volatility weighting
  - Market-cap aware slippage modeling

- **Data Pipeline**: Robust fundamental data handling
  - Schema validation
  - Freshness gate controls
  - TTM calculations
  - Reporting lag adjustments

- **Benchmarking**: Performance regression testing
  - Automated benchmark suite
  - Memory profiling
  - Baseline comparison

## Quick Start

```python
from Models import PortfolioEngine, build_signal, get_available_signals

# List available signals
print(get_available_signals())

# Build a signal
signal = build_signal("momentum", dates, loader, config)

# Run backtest via CLI
# bist-backtest momentum --start-date 2020-01-01
```

## Installation

```bash
# Install from source
pip install -e ".[dev]"

# Or just core dependencies
pip install -e .
```

## CLI Usage

```bash
# Run a specific signal
bist-backtest momentum

# Run all signals
bist-backtest all --start-date 2020-01-01

# List available signals
bist-backtest --list-signals

# Dry run (validate without executing)
bist-backtest momentum --dry-run

# Verbose output
bist-backtest momentum -v
```

## Documentation

- [Getting Started](getting-started/installation.md) - Installation and setup
- [User Guide](guide/backtests.md) - How to run backtests
- [API Reference](api/index.md) - Detailed API documentation
- [Examples](examples/quick-start.md) - Code examples

## Requirements

- Python 3.10+
- pandas 2.0+
- numpy 1.24+
- PyYAML 6.0+

See [pyproject.toml](https://github.com/OWNER/bist-quant/blob/main/pyproject.toml) for full dependency list.

## License

Proprietary - All rights reserved.
