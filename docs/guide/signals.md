# Signal Configuration

This guide explains how to configure and create signals in the BIST Quant library.

## Signal Architecture

Signals in BIST Quant follow a builder pattern:

1. **Signal Builder**: A class that implements the signal generation logic
2. **Configuration**: YAML file specifying parameters and timeline
3. **Signal Factory**: Loads and instantiates builders from configs

## Available Signal Builders

### Momentum Signals

```yaml
signal:
  name: momentum_12m
  builder: MomentumSignalBuilder
  params:
    lookback: 252  # 12 months
    skip_days: 21  # Skip most recent month
```

### Value Signals

```yaml
signal:
  name: book_to_market
  builder: ValueSignalBuilder
  params:
    metric: book_to_market
    winsorize: 0.01
```

### Quality Signals

```yaml
signal:
  name: roe
  builder: QualitySignalBuilder
  params:
    metric: roe
    lookback: 4  # quarters
```

### Technical Signals

```yaml
signal:
  name: rsi_oversold
  builder: TechnicalSignalBuilder
  params:
    indicator: rsi
    period: 14
    threshold: 30
```

## Creating Custom Signals

### Step 1: Implement the Protocol

```python
from Models.signals.protocol import SignalBuilder
import pandas as pd

class MyCustomSignalBuilder(SignalBuilder):
    """Custom signal builder example."""

    def __init__(self, param1: int = 20, param2: float = 0.5):
        self.param1 = param1
        self.param2 = param2

    def build(
        self,
        close_df: pd.DataFrame,
        high_df: pd.DataFrame | None = None,
        low_df: pd.DataFrame | None = None,
        volume_df: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Generate signal panel."""
        # Your signal logic here
        signal = close_df.pct_change(self.param1) * self.param2
        return signal
```

### Step 2: Register the Builder

Add your builder to the factory registry:

```python
from Models.signals.factory import register_builder

register_builder("MyCustomSignalBuilder", MyCustomSignalBuilder)
```

### Step 3: Create Configuration

```yaml
# Models/configs/my_custom_signal.yaml
signal:
  name: my_custom_signal
  builder: MyCustomSignalBuilder
  params:
    param1: 30
    param2: 0.8

timeline:
  start_date: "2020-01-01"
  end_date: "2024-12-31"

portfolio:
  top_n: 15
  rebalance_frequency: weekly
```

## Composite Signals

Combine multiple signals using blending functions:

```python
from Models.signals.composite import weighted_sum, zscore_blend

# Weighted combination
combined = weighted_sum(
    panels={"momentum": mom_signal, "value": val_signal},
    weights={"momentum": 0.6, "value": 0.4},
)

# Z-score normalized blend
blended = zscore_blend(
    panels={"momentum": mom_signal, "value": val_signal},
)
```

## Signal Validation

All signals are validated against a schema:

```python
from Models.common.utils import validate_signal_panel_schema

validated = validate_signal_panel_schema(
    panel=my_signal,
    dates=price_df.index,
    tickers=price_df.columns,
    signal_name="my_signal",
    context="backtest",
)
```

This ensures:

- Correct index alignment with price data
- Proper column (ticker) alignment
- Float dtype for all values
- No invalid infinities
