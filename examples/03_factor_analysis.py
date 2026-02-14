#!/usr/bin/env python3
"""
03 - Factor Analysis Example

Demonstrates factor research and analysis:
1. Generate multiple factor signals
2. Calculate factor correlations
3. Analyze information coefficients
4. Study factor decay

Usage:
    python examples/03_factor_analysis.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Find project root (parent of examples/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def calculate_ic(
    signal: pd.DataFrame,
    forward_returns: pd.DataFrame,
    dates: pd.DatetimeIndex | None = None,
) -> list[float]:
    """Calculate Information Coefficient (rank correlation) for each date."""
    if dates is None:
        dates = signal.index.intersection(forward_returns.index)

    ics = []
    for date in dates:
        sig = signal.loc[date].dropna()
        ret = forward_returns.loc[date].dropna()
        common = sig.index.intersection(ret.index)
        if len(common) > 5:
            # Spearman rank correlation
            ics.append(sig[common].rank().corr(ret[common].rank()))
    return ics


def main() -> None:
    """Analyze factor performance and characteristics."""
    from Models.common.utils import cross_sectional_rank
    from Models.portfolio_engine import PortfolioEngine
    from Models.signals.factory import get_signal_builder

    print("=" * 60)
    print("BIST Quant - Factor Analysis Example")
    print("=" * 60)

    # Initialize and load data
    print("\n[1/5] Loading data...")
    engine = PortfolioEngine(
        data_dir=PROJECT_ROOT / "data",
        regime_model_dir=PROJECT_ROOT / "regime_filter" / "outputs",
        start_date="2018-01-01",
        end_date="2024-12-31",
    )
    engine.load_all_data()

    # Build multiple momentum factors with different lookbacks
    print("[2/5] Building factor signals...")
    factors: dict[str, pd.DataFrame] = {}

    lookbacks = [21, 63, 126, 252]  # 1m, 3m, 6m, 12m
    for lb in lookbacks:
        name = f"mom_{lb}d"
        print(f"       Building {name}...")
        builder = get_signal_builder("MomentumSignalBuilder", lookback=lb, skip_days=5)
        factors[name] = builder.build(close_df=engine.close_df)

    # Calculate forward returns
    print("[3/5] Calculating forward returns...")
    fwd_returns_1d = engine.close_df.pct_change().shift(-1)
    fwd_returns_5d = engine.close_df.pct_change(5).shift(-5)
    fwd_returns_21d = engine.close_df.pct_change(21).shift(-21)

    # Calculate Information Coefficients
    print("[4/5] Calculating Information Coefficients...")
    print()
    print("=" * 60)
    print("INFORMATION COEFFICIENT ANALYSIS")
    print("=" * 60)
    print()
    print(f"{'Factor':<12} {'IC (1d)':<12} {'IC (5d)':<12} {'IC (21d)':<12} {'IC IR':<12}")
    print("-" * 60)

    for name, signal in factors.items():
        ics_1d = calculate_ic(signal, fwd_returns_1d)
        ics_5d = calculate_ic(signal, fwd_returns_5d)
        ics_21d = calculate_ic(signal, fwd_returns_21d)

        mean_ic_1d = np.nanmean(ics_1d)
        mean_ic_5d = np.nanmean(ics_5d)
        mean_ic_21d = np.nanmean(ics_21d)

        # IC Information Ratio = mean(IC) / std(IC)
        ic_ir = mean_ic_21d / np.nanstd(ics_21d) if ics_21d else np.nan

        print(
            f"{name:<12} {mean_ic_1d:>10.4f}   {mean_ic_5d:>10.4f}   "
            f"{mean_ic_21d:>10.4f}   {ic_ir:>10.2f}"
        )

    # Calculate factor correlations
    print()
    print("=" * 60)
    print("FACTOR CORRELATIONS (Cross-sectional)")
    print("=" * 60)
    print()

    # Rank factors for comparable correlation analysis
    ranked_factors = {
        name: cross_sectional_rank(signal, higher_is_better=True)
        for name, signal in factors.items()
    }

    factor_names = list(ranked_factors.keys())
    corr_matrix = pd.DataFrame(
        index=factor_names, columns=factor_names, dtype=float
    )

    for i, name_i in enumerate(factor_names):
        for j, name_j in enumerate(factor_names):
            if i == j:
                corr_matrix.loc[name_i, name_j] = 1.0
            elif i < j:
                # Stack and calculate correlation
                df_i = ranked_factors[name_i].stack()
                df_j = ranked_factors[name_j].stack()
                common = df_i.index.intersection(df_j.index)
                corr = df_i.loc[common].corr(df_j.loc[common])
                corr_matrix.loc[name_i, name_j] = corr
                corr_matrix.loc[name_j, name_i] = corr

    print(corr_matrix.round(3).to_string())

    # Factor decay analysis
    print()
    print("=" * 60)
    print("FACTOR DECAY ANALYSIS")
    print("=" * 60)
    print()
    print("How quickly does predictive power decay over time?")
    print()

    horizons = [1, 5, 10, 21, 42, 63]
    print(f"{'Factor':<12}", end="")
    for h in horizons:
        print(f"{h:>8}d", end="")
    print()
    print("-" * (12 + 9 * len(horizons)))

    for name, signal in list(factors.items())[:2]:  # Just first 2 for brevity
        print(f"{name:<12}", end="")
        for horizon in horizons:
            fwd_ret = engine.close_df.pct_change(horizon).shift(-horizon)
            ics = calculate_ic(signal, fwd_ret)
            mean_ic = np.nanmean(ics)
            print(f"{mean_ic:>9.4f}", end="")
        print()

    print()
    print("[5/5] Analysis complete!")
    print()


if __name__ == "__main__":
    main()
