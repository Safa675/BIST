#!/usr/bin/env python3
"""
01 - Quick Start Example

Minimal example demonstrating the complete backtest workflow:
1. Load data
2. Build a signal
3. Run backtest
4. View results

Usage:
    python examples/01_quick_start.py
"""

from __future__ import annotations

from pathlib import Path

# Find project root (parent of examples/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    """Run a minimal momentum backtest."""
    from Models.portfolio_engine import PortfolioEngine

    print("=" * 60)
    print("BIST Quant - Quick Start Example")
    print("=" * 60)

    # 1. Initialize the engine
    print("\n[1/4] Initializing Portfolio Engine...")
    engine = PortfolioEngine(
        data_dir=PROJECT_ROOT / "data",
        regime_model_dir=PROJECT_ROOT / "regime_filter" / "outputs",
        start_date="2020-01-01",
        end_date="2024-12-31",
    )

    # 2. Load all required data
    print("[2/4] Loading market data...")
    engine.load_all_data()

    n_tickers = engine.close_df.shape[1]
    date_min = engine.close_df.index.min().date()
    date_max = engine.close_df.index.max().date()
    print(f"       Loaded {n_tickers} tickers from {date_min} to {date_max}")

    # 3. Run a momentum backtest
    print("[3/4] Running momentum backtest...")
    engine.run_factor("momentum")

    # 4. Report results
    print("[4/4] Backtest complete!")
    print()
    print("Results saved to: Models/results/momentum/")
    print()
    print("Key files:")
    print("  - performance_summary.json  : Performance metrics")
    print("  - equity_curve.csv          : Daily portfolio value")
    print("  - trades.csv                : Trade log")
    print()


if __name__ == "__main__":
    main()
