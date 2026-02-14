#!/usr/bin/env python3
"""
Tiny smoke test for signals.factory.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import pandas as pd

from Models.portfolio_engine import PortfolioEngine, load_signal_configs
from Models.signals.factory import build_signal, get_available_signals

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SMOKE_SIGNALS = ("sma", "ema", "xu100")


def resolve_regime_outputs() -> Path:
    candidates = [
        PROJECT_ROOT / "Simple Regime Filter" / "outputs",
        PROJECT_ROOT / "regime_filter" / "outputs",
    ]
    return next((path for path in candidates if path.exists()), candidates[0])


def build_runtime_config(engine: PortfolioEngine, config: dict) -> dict:
    runtime_config = copy.deepcopy(config) if isinstance(config, dict) else {}
    runtime_config["_runtime_context"] = {
        "prices": engine.prices,
        "close_df": engine.close_df,
        "open_df": engine.open_df,
        "volume_df": engine.volume_df,
        "fundamentals": engine.fundamentals,
        "xu100_prices": engine.xu100_prices,
    }
    return runtime_config


def main() -> int:
    configs = load_signal_configs()
    configured = set(configs.keys())
    available = set(get_available_signals())

    missing_in_factory = sorted(configured - available)
    orphan_builders = sorted(available - configured)

    logger.info(f"Configured signals: {len(configured)}")
    logger.info(f"Factory builders: {len(available)}")

    if missing_in_factory:
        logger.info(f"Missing in factory: {missing_in_factory}")
    if orphan_builders:
        logger.info(f"Builders without config: {orphan_builders}")
    if missing_in_factory:
        return 1

    start_date = pd.Timestamp("2024-01-01")
    end_date = pd.Timestamp("2024-12-31")
    data_dir = PROJECT_ROOT / "data"
    regime_outputs = resolve_regime_outputs()
    logger.info("Loading data for smoke test...")
    engine = PortfolioEngine(
        data_dir=data_dir,
        regime_model_dir=regime_outputs,
        start_date=str(start_date.date()),
        end_date=str(end_date.date()),
    )
    engine.load_all_data()

    date_mask = (engine.close_df.index >= start_date) & (engine.close_df.index <= end_date)
    dates = engine.close_df.index[date_mask]
    if dates.empty:
        raise RuntimeError("No trading dates found in smoke test date range")

    for signal_name in SMOKE_SIGNALS:
        config = configs.get(signal_name)
        if not isinstance(config, dict):
            raise ValueError(f"Missing config for smoke signal '{signal_name}'")

        runtime_config = build_runtime_config(engine, config)
        panel = build_signal(signal_name, dates, engine.loader, runtime_config)

        if not isinstance(panel, pd.DataFrame):
            raise TypeError(f"Signal '{signal_name}' did not return DataFrame")
        if panel.empty:
            raise ValueError(f"Signal '{signal_name}' returned empty DataFrame")
        if panel.shape[0] != len(dates):
            raise ValueError(
                f"Signal '{signal_name}' returned {panel.shape[0]} rows, expected {len(dates)}"
            )
        if panel.shape[1] == 0:
            raise ValueError(f"Signal '{signal_name}' returned zero columns")
        if int(panel.notna().sum().sum()) == 0:
            raise ValueError(f"Signal '{signal_name}' returned only NaN values")

        logger.info(
            f"✅ {signal_name}: shape={panel.shape}, non_null={int(panel.notna().sum().sum())}",
                    )

    logger.info("Factory smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
