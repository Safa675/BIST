# API Reference

This section provides detailed API documentation for the BIST Quant library.

## Package Structure

```
Models/
├── __init__.py          # Public API exports
├── portfolio_engine.py  # Main portfolio engine
├── common/              # Core utilities
│   ├── backtester.py    # Backtest execution
│   ├── data_loader.py   # Data loading
│   ├── risk_manager.py  # Risk controls
│   ├── config_manager.py # Configuration
│   └── utils.py         # Utilities
└── signals/             # Signal builders
    ├── factory.py       # Signal factory
    ├── protocol.py      # Type protocols
    ├── momentum.py      # Momentum signals
    ├── value.py         # Value signals
    ├── quality.py       # Quality signals
    ├── technical.py     # Technical signals
    └── composite.py     # Composite signals
```

## Main Exports

The top-level `Models` package exports:

- `PortfolioEngine` - Main portfolio management class
- `build_signal` - Build signals by name
- `get_available_signals` - List available signal names
- `load_signal_configs` - Load signal configurations

```python
from Models import (
    PortfolioEngine,
    build_signal,
    get_available_signals,
    load_signal_configs,
)
```

## Module Documentation

### Common

- [Backtester](common/backtester.md) - Backtest execution engine
- [Data Loader](common/data_loader.md) - Data loading and caching
- [Risk Manager](common/risk_manager.md) - Risk controls and position sizing
- [Config Manager](common/config_manager.md) - Configuration management

### Signals

- [Factory](signals/factory.md) - Signal factory and registry
- [Protocol](signals/protocol.md) - Type protocols
- [Momentum](signals/momentum.md) - Momentum signals
- [Value](signals/value.md) - Value signals
- [Quality](signals/quality.md) - Quality signals
- [Technical](signals/technical.md) - Technical signals
- [Composite](signals/composite.md) - Composite signals

### Portfolio

- [Portfolio Engine](portfolio_engine.md) - Main orchestration class
