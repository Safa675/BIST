# 02 - Custom Strategy

This example shows how to build a custom multi-factor strategy.

## Overview

1. Generate individual factor signals
2. Combine signals using weighted blending
3. Run backtest with the composite signal

## Full Script

```python
#!/usr/bin/env python3
"""Custom strategy example: multi-factor signal combination."""

from pathlib import Path
import pandas as pd

from Models.portfolio_engine import PortfolioEngine
from Models.signals.factory import get_signal_builder
from Models.signals.composite import weighted_sum, zscore_blend

def main():
    # Initialize engine and load data
    engine = PortfolioEngine(
        data_dir=Path("data"),
        regime_model_dir=Path("regime_filter/outputs"),
        start_date="2020-01-01",
        end_date="2024-12-31",
    )
    engine.load_all_data()

    # Get signal builders
    momentum_builder = get_signal_builder("MomentumSignalBuilder", lookback=60)
    value_builder = get_signal_builder("ValueSignalBuilder", metric="earnings_yield")

    # Generate signals
    print("Building momentum signal...")
    momentum_signal = momentum_builder.build(
        close_df=engine.close_df,
        high_df=engine.high_df,
        low_df=engine.low_df,
        volume_df=engine.volume_df,
    )

    print("Building value signal...")
    value_signal = value_builder.build(
        close_df=engine.close_df,
        fundamentals_df=engine.fundamentals_df,
    )

    # Combine signals with weighted sum
    print("Combining signals...")
    combined_signal = weighted_sum(
        panels={"momentum": momentum_signal, "value": value_signal},
        weights={"momentum": 0.6, "value": 0.4},
    )

    # Alternative: z-score normalized blend
    # combined_signal = zscore_blend(
    #     panels={"momentum": momentum_signal, "value": value_signal},
    # )

    # Create custom config for the combined signal
    custom_config = {
        "signal": {
            "name": "momentum_value_combo",
            "builder": "PrecomputedSignalBuilder",
            "params": {},
        },
        "timeline": {
            "start_date": "2020-01-01",
            "end_date": "2024-12-31",
        },
        "portfolio": {
            "top_n": 15,
            "rebalance_frequency": "monthly",
        },
    }

    # Run backtest with precomputed signal
    print("Running backtest...")
    engine.run_factor_with_signal(
        signal_name="momentum_value_combo",
        signal_panel=combined_signal,
        config=custom_config,
    )

    print("\nBacktest complete!")
    print("Results saved to: Models/results/momentum_value_combo/")

if __name__ == "__main__":
    main()
```

## Signal Combination Methods

### Weighted Sum

Simple weighted average of signals:

```python
combined = weighted_sum(
    panels={"a": signal_a, "b": signal_b, "c": signal_c},
    weights={"a": 0.5, "b": 0.3, "c": 0.2},
)
```

Weights are automatically normalized to sum to 1.0.

### Z-Score Blend

Normalizes each signal to z-scores before averaging:

```python
combined = zscore_blend(
    panels={"a": signal_a, "b": signal_b},
)
```

This ensures signals with different scales contribute equally.

## Portfolio Construction Options

### Top N Selection

Select the top N stocks by signal score:

```python
"portfolio": {
    "top_n": 20,
    "rebalance_frequency": "monthly",
}
```

### Rebalance Frequency

Options: `"daily"`, `"weekly"`, `"monthly"`, `"quarterly"`

### Position Sizing

Configure position sizing in the portfolio section:

```python
"portfolio": {
    "top_n": 15,
    "weighting": "inverse_vol",  # or "equal", "signal_weighted"
    "max_position_weight": 0.10,
}
```

## Next Steps

- [03 - Factor Analysis](03-factor-analysis.md): Deep dive into factor performance
