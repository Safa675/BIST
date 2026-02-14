# 03 - Factor Analysis

This example demonstrates how to analyze factor performance and characteristics.

## Overview

1. Generate multiple factor signals
2. Calculate factor correlations
3. Analyze regime-dependent performance
4. Compute factor decay statistics

## Full Script

```python
#!/usr/bin/env python3
"""Factor analysis example: correlations, regimes, and decay."""

from pathlib import Path
import pandas as pd
import numpy as np

from Models.portfolio_engine import PortfolioEngine
from Models.signals.factory import get_signal_builder
from Models.common.utils import cross_sectional_rank

def main():
    # Initialize and load data
    engine = PortfolioEngine(
        data_dir=Path("data"),
        regime_model_dir=Path("regime_filter/outputs"),
        start_date="2018-01-01",
        end_date="2024-12-31",
    )
    engine.load_all_data()

    # Build multiple factor signals
    factors = {}

    print("Building factor signals...")

    # Momentum factors
    for lookback in [21, 63, 126, 252]:
        name = f"mom_{lookback}d"
        builder = get_signal_builder("MomentumSignalBuilder", lookback=lookback)
        factors[name] = builder.build(close_df=engine.close_df)

    # Volatility factor
    vol_builder = get_signal_builder("VolatilitySignalBuilder", lookback=21)
    factors["vol_21d"] = vol_builder.build(close_df=engine.close_df)

    # Calculate cross-sectional ranks for comparable analysis
    ranked_factors = {
        name: cross_sectional_rank(signal, higher_is_better=True)
        for name, signal in factors.items()
    }

    # 1. Factor Correlations
    print("\n=== Factor Correlations ===")
    analyze_correlations(ranked_factors)

    # 2. Regime Analysis
    print("\n=== Regime Performance ===")
    analyze_regime_performance(factors, engine)

    # 3. Factor Decay
    print("\n=== Factor Decay Analysis ===")
    analyze_factor_decay(factors, engine)

def analyze_correlations(ranked_factors: dict[str, pd.DataFrame]) -> None:
    """Calculate average cross-sectional correlations between factors."""
    factor_names = list(ranked_factors.keys())
    n_factors = len(factor_names)

    # Calculate correlation for each date, then average
    correlations = pd.DataFrame(index=factor_names, columns=factor_names, dtype=float)

    for i, name_i in enumerate(factor_names):
        for j, name_j in enumerate(factor_names):
            if i == j:
                correlations.loc[name_i, name_j] = 1.0
            elif i < j:
                # Stack both factors and calculate correlation
                df_i = ranked_factors[name_i].stack()
                df_j = ranked_factors[name_j].stack()
                common_idx = df_i.index.intersection(df_j.index)
                corr = df_i.loc[common_idx].corr(df_j.loc[common_idx])
                correlations.loc[name_i, name_j] = corr
                correlations.loc[name_j, name_i] = corr

    print(correlations.round(3).to_string())

def analyze_regime_performance(
    factors: dict[str, pd.DataFrame],
    engine: PortfolioEngine,
) -> None:
    """Analyze factor performance across market regimes."""
    # Load regime classifications
    regime_features = pd.read_csv(
        engine.regime_model_dir / "regime_features.csv",
        parse_dates=["date"],
        index_col="date",
    )

    if "regime" not in regime_features.columns:
        print("No regime column found in regime_features.csv")
        return

    # Calculate forward returns
    fwd_returns = engine.close_df.pct_change().shift(-1)

    for factor_name, signal in factors.items():
        # Align dates
        common_dates = signal.index.intersection(regime_features.index).intersection(fwd_returns.index)

        regime_ic = {}
        for regime in regime_features["regime"].unique():
            regime_dates = regime_features.loc[
                regime_features["regime"] == regime
            ].index.intersection(common_dates)

            if len(regime_dates) < 20:
                continue

            # Calculate IC for this regime
            ics = []
            for date in regime_dates:
                sig = signal.loc[date].dropna()
                ret = fwd_returns.loc[date].dropna()
                common = sig.index.intersection(ret.index)
                if len(common) > 5:
                    ics.append(sig[common].corr(ret[common]))

            if ics:
                regime_ic[regime] = np.nanmean(ics)

        print(f"\n{factor_name}:")
        for regime, ic in sorted(regime_ic.items()):
            print(f"  Regime {regime}: IC = {ic:.4f}")

def analyze_factor_decay(
    factors: dict[str, pd.DataFrame],
    engine: PortfolioEngine,
) -> None:
    """Analyze how factor predictive power decays over time."""
    horizons = [1, 5, 10, 21]  # Forward return horizons in days

    for factor_name, signal in list(factors.items())[:2]:  # Just first 2 factors
        print(f"\n{factor_name}:")
        for horizon in horizons:
            # Calculate forward returns
            fwd_ret = engine.close_df.pct_change(horizon).shift(-horizon)

            # Calculate IC across all dates
            ics = []
            for date in signal.index[:-horizon]:
                sig = signal.loc[date].dropna()
                ret = fwd_ret.loc[date].dropna()
                common = sig.index.intersection(ret.index)
                if len(common) > 10:
                    ics.append(sig[common].corr(ret[common]))

            mean_ic = np.nanmean(ics)
            print(f"  {horizon:2d}-day forward IC: {mean_ic:.4f}")

if __name__ == "__main__":
    main()
```

## Understanding the Analysis

### Factor Correlations

High correlations between factors reduce diversification benefits. Look for factors with low or negative correlations to combine.

### Regime Performance

Factors often perform differently in different market regimes:
- **Risk-on regimes**: Momentum and growth factors often outperform
- **Risk-off regimes**: Quality and low-volatility factors tend to be more defensive

### Factor Decay

The decay analysis shows how quickly a factor's predictive power diminishes:
- Fast decay (IC drops quickly): Signal needs frequent rebalancing
- Slow decay: Signal can be traded less frequently

## Extending the Analysis

Additional analyses you might add:

```python
# Turnover analysis
def calculate_turnover(signal: pd.DataFrame, top_n: int = 20) -> float:
    """Calculate average portfolio turnover."""
    positions = signal.rank(axis=1, ascending=False) <= top_n
    turnover = positions.astype(int).diff().abs().sum(axis=1).mean() / (2 * top_n)
    return turnover

# Information coefficient stability
def ic_stability(signal: pd.DataFrame, returns: pd.DataFrame) -> float:
    """Calculate IC stability (% of months with positive IC)."""
    monthly_ics = []
    for month, group in signal.groupby(pd.Grouper(freq="ME")):
        # Calculate IC for this month
        ...
    return np.mean([ic > 0 for ic in monthly_ics])
```
