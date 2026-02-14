#!/usr/bin/env python3
"""
02 - Custom Strategy Example

Demonstrates building a custom multi-factor strategy:
1. Generate individual factor signals
2. Combine signals using weighted blending
3. Run backtest with the composite signal

Usage:
    python examples/02_custom_strategy.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Find project root (parent of examples/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    """Build and backtest a multi-factor strategy."""
    from Models.portfolio_engine import PortfolioEngine
    from Models.signals.composite import weighted_sum
    from Models.signals.factory import get_signal_builder

    print("=" * 60)
    print("BIST Quant - Custom Strategy Example")
    print("=" * 60)

    # Initialize and load data
    print("\n[1/5] Loading data...")
    engine = PortfolioEngine(
        data_dir=PROJECT_ROOT / "data",
        regime_model_dir=PROJECT_ROOT / "regime_filter" / "outputs",
        start_date="2020-01-01",
        end_date="2024-12-31",
    )
    engine.load_all_data()

    # Build momentum signal (12-month lookback, skip last month)
    print("[2/5] Building momentum signal...")
    momentum_builder = get_signal_builder(
        "MomentumSignalBuilder",
        lookback=252,
        skip_days=21,
    )
    momentum_signal = momentum_builder.build(
        close_df=engine.close_df,
        high_df=engine.high_df,
        low_df=engine.low_df,
        volume_df=engine.volume_df,
    )

    # Build short-term reversal signal
    print("[3/5] Building reversal signal...")
    reversal_builder = get_signal_builder(
        "MomentumSignalBuilder",
        lookback=5,
        skip_days=0,
    )
    # Negate for reversal (low momentum = high expected return)
    reversal_signal = -reversal_builder.build(
        close_df=engine.close_df,
        high_df=engine.high_df,
        low_df=engine.low_df,
        volume_df=engine.volume_df,
    )

    # Combine signals
    print("[4/5] Combining signals (60% momentum, 40% reversal)...")
    combined_signal = weighted_sum(
        panels={"momentum": momentum_signal, "reversal": reversal_signal},
        weights={"momentum": 0.6, "reversal": 0.4},
    )

    # Print signal statistics
    print(f"       Signal shape: {combined_signal.shape}")
    print(f"       Non-null values: {combined_signal.notna().sum().sum():,}")

    # Create custom config
    custom_config = {
        "signal": {
            "name": "momentum_reversal_combo",
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

    # Run backtest
    print("[5/5] Running backtest...")
    engine.run_factor_with_signal(
        signal_name="momentum_reversal_combo",
        signal_panel=combined_signal,
        config=custom_config,
    )

    print()
    print("Backtest complete!")
    print("Results saved to: Models/results/momentum_reversal_combo/")
    print()


if __name__ == "__main__":
    main()
