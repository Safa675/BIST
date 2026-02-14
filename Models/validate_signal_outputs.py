#!/usr/bin/env python3
"""
Compare legacy signal outputs vs new signals.factory outputs.
"""

from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from Models.portfolio_engine import PortfolioEngine, load_signal_configs
from Models.signals.accrual_signals import build_accrual_signals
from Models.signals.adx_signals import build_adx_signals
from Models.signals.asset_growth_signals import build_asset_growth_signals
from Models.signals.atr_signals import build_atr_signals
from Models.signals.betting_against_beta_signals import build_betting_against_beta_signals
from Models.signals.breakout_value_signals import build_breakout_value_signals
from Models.signals.consistent_momentum_signals import build_consistent_momentum_signals
from Models.signals.dividend_rotation_signals import build_dividend_rotation_signals
from Models.signals.donchian_signals import build_donchian_signals
from Models.signals.earnings_quality_signals import build_earnings_quality_signals
from Models.signals.ema_signals import build_ema_signals
from Models.signals.factory import build_signal
from Models.signals.five_factor_rotation_signals import build_five_factor_rotation_signals
from Models.signals.fscore_reversal_signals import build_fscore_reversal_signals
from Models.signals.ichimoku_signals import build_ichimoku_signals
from Models.signals.investment_signals import build_investment_signals
from Models.signals.low_volatility_signals import build_low_volatility_signals
from Models.signals.macd_signals import build_macd_signals
from Models.signals.macro_hedge_signals import build_macro_hedge_signals
from Models.signals.momentum_asset_growth_signals import build_momentum_asset_growth_signals
from Models.signals.momentum_reversal_volatility_signals import (
    build_momentum_reversal_volatility_signals,
)
from Models.signals.momentum_signals import build_momentum_signals
from Models.signals.obv_signals import build_obv_signals
from Models.signals.pairs_trading_signals import build_pairs_trading_signals
from Models.signals.parabolic_sar_signals import build_parabolic_sar_signals
from Models.signals.profitability_signals import build_profitability_signals
from Models.signals.quality_momentum_signals import build_quality_momentum_signals
from Models.signals.quality_value_signals import build_quality_value_signals
from Models.signals.residual_momentum_signals import build_residual_momentum_signals
from Models.signals.roa_signals import build_roa_signals
from Models.signals.sector_rotation_signals import build_sector_rotation_signals
from Models.signals.size_rotation_momentum_signals import build_size_rotation_momentum_signals
from Models.signals.size_rotation_quality_signals import build_size_rotation_quality_signals
from Models.signals.size_rotation_signals import build_size_rotation_signals
from Models.signals.sma_signals import build_sma_signals
from Models.signals.small_cap_momentum_signals import build_small_cap_momentum_signals
from Models.signals.small_cap_signals import build_small_cap_signals
from Models.signals.supertrend_signals import build_supertrend_signals
from Models.signals.trend_following_signals import build_trend_following_signals
from Models.signals.trend_value_signals import build_trend_value_signals
from Models.signals.value_signals import build_value_signals
from Models.signals.xu100_signals import build_xu100_signals

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SIGNALS = [
    "momentum",
    "value",
    "quality_value",
    "sma",
    "donchian",
    "adx",
    "size_rotation",
    "five_factor_rotation",
]


def _build_high_low(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    high_df = prices.pivot_table(index="Date", columns="Ticker", values="High").sort_index()
    high_df.columns = [column.split(".")[0].upper() for column in high_df.columns]
    low_df = prices.pivot_table(index="Date", columns="Ticker", values="Low").sort_index()
    low_df.columns = [column.split(".")[0].upper() for column in low_df.columns]
    return high_df, low_df


def build_legacy_signal(
    factor_name: str,
    dates: pd.DatetimeIndex,
    config: dict,
    engine: PortfolioEngine,
) -> pd.DataFrame:
    signal_params = config.get("signal_params", {}) if isinstance(config.get("signal_params"), dict) else {}

    if factor_name == "profitability":
        return build_profitability_signals(
            engine.fundamentals,
            dates,
            engine.loader,
            operating_income_weight=float(signal_params.get("operating_income_weight", 0.5)),
            gross_profit_weight=float(signal_params.get("gross_profit_weight", 0.5)),
        )
    if factor_name == "value":
        metric_weights = signal_params.get("metric_weights")
        if not isinstance(metric_weights, dict):
            metric_weights = None
        enabled_metrics = signal_params.get("enabled_metrics")
        if isinstance(enabled_metrics, list):
            enabled_metrics = [metric for metric in enabled_metrics if isinstance(metric, str)]
        else:
            enabled_metrics = None
        return build_value_signals(
            engine.fundamentals,
            engine.close_df,
            dates,
            engine.loader,
            metric_weights=metric_weights,
            enabled_metrics=enabled_metrics,
        )
    if factor_name == "small_cap":
        return build_small_cap_signals(engine.fundamentals, engine.close_df, engine.volume_df, dates, engine.loader)
    if factor_name == "investment":
        return build_investment_signals(engine.fundamentals, engine.close_df, dates, engine.loader)
    if factor_name == "momentum":
        return build_momentum_signals(
            engine.close_df,
            dates,
            engine.loader,
            lookback=int(signal_params.get("lookback", 252)),
            skip=int(signal_params.get("skip", 21)),
            vol_lookback=int(signal_params.get("vol_lookback", 252)),
        )
    if factor_name == "sma":
        return build_sma_signals(engine.close_df, dates, engine.loader)
    if factor_name == "donchian":
        high_df, low_df = _build_high_low(engine.prices)
        return build_donchian_signals(engine.close_df, high_df, low_df, dates, engine.loader)
    if factor_name == "xu100":
        result = build_xu100_signals(engine.close_df, dates, engine.loader)
        if "XU100" not in engine.close_df.columns and engine.xu100_prices is not None:
            engine.close_df["XU100"] = engine.xu100_prices.reindex(engine.close_df.index)
        return result
    if factor_name == "trend_value":
        return build_trend_value_signals(engine.close_df, dates, engine.loader)
    if factor_name == "breakout_value":
        high_df, low_df = _build_high_low(engine.prices)
        return build_breakout_value_signals(engine.close_df, high_df, low_df, dates, engine.loader)
    if factor_name == "dividend_rotation":
        return build_dividend_rotation_signals(engine.close_df, dates, engine.loader)
    if factor_name == "macro_hedge":
        return build_macro_hedge_signals(engine.close_df, dates, engine.loader)
    if factor_name == "quality_momentum":
        return build_quality_momentum_signals(engine.close_df, engine.fundamentals, dates, engine.loader)
    if factor_name == "quality_value":
        return build_quality_value_signals(engine.close_df, engine.fundamentals, dates, engine.loader)
    if factor_name == "small_cap_momentum":
        return build_small_cap_momentum_signals(engine.close_df, dates, engine.loader)
    if factor_name == "size_rotation":
        return build_size_rotation_signals(engine.close_df, dates, engine.loader)
    if factor_name == "size_rotation_momentum":
        return build_size_rotation_momentum_signals(engine.close_df, dates, engine.loader)
    if factor_name == "size_rotation_quality":
        return build_size_rotation_quality_signals(engine.close_df, engine.fundamentals, dates, engine.loader)
    if factor_name == "five_factor_rotation":
        cache_cfg = config.get("construction_cache", {})
        debug_cfg = config.get("debug", {})
        orth_cfg = config.get("axis_orthogonalization", {})
        walk_forward_cfg = config.get("walk_forward", {}) if isinstance(config.get("walk_forward", {}), dict) else {}
        debug_env = False
        debug_enabled = bool(debug_cfg.get("enabled", False) or debug_env)
        signals, _ = build_five_factor_rotation_signals(
            engine.close_df,
            dates,
            engine.loader,
            fundamentals=engine.fundamentals,
            volume_df=engine.volume_df,
            use_construction_cache=cache_cfg.get("enabled", True),
            force_rebuild_construction_cache=cache_cfg.get("force_rebuild", False),
            construction_cache_path=cache_cfg.get("path"),
            mwu_walkforward_config=walk_forward_cfg,
            axis_orthogonalization_config=orth_cfg,
            return_details=True,
            debug=debug_enabled,
        )
        return signals
    if factor_name == "accrual":
        return build_accrual_signals(engine.fundamentals, dates, engine.loader)
    if factor_name == "asset_growth":
        return build_asset_growth_signals(engine.fundamentals, dates, engine.loader)
    if factor_name == "betting_against_beta":
        return build_betting_against_beta_signals(engine.close_df, dates, engine.loader)
    if factor_name == "roa":
        return build_roa_signals(engine.fundamentals, dates, engine.loader)
    if factor_name == "consistent_momentum":
        return build_consistent_momentum_signals(engine.close_df, dates, engine.loader)
    if factor_name == "residual_momentum":
        return build_residual_momentum_signals(engine.close_df, dates, engine.loader)
    if factor_name == "momentum_reversal_volatility":
        return build_momentum_reversal_volatility_signals(engine.close_df, dates, engine.loader)
    if factor_name == "low_volatility":
        return build_low_volatility_signals(engine.close_df, dates, engine.loader)
    if factor_name == "trend_following":
        return build_trend_following_signals(engine.close_df, dates, engine.loader)
    if factor_name == "sector_rotation":
        return build_sector_rotation_signals(engine.close_df, dates, engine.loader)
    if factor_name == "earnings_quality":
        return build_earnings_quality_signals(engine.fundamentals, engine.close_df, dates, engine.loader)
    if factor_name == "fscore_reversal":
        return build_fscore_reversal_signals(engine.fundamentals, engine.close_df, dates, engine.loader)
    if factor_name == "momentum_asset_growth":
        return build_momentum_asset_growth_signals(engine.fundamentals, engine.close_df, dates, engine.loader)
    if factor_name == "pairs_trading":
        return build_pairs_trading_signals(engine.close_df, dates, engine.loader)
    if factor_name == "macd":
        return build_macd_signals(engine.close_df, dates, engine.loader)
    if factor_name == "adx":
        high_df, low_df = _build_high_low(engine.prices)
        return build_adx_signals(engine.close_df, high_df, low_df, dates, engine.loader)
    if factor_name == "supertrend":
        high_df, low_df = _build_high_low(engine.prices)
        return build_supertrend_signals(engine.close_df, high_df, low_df, dates, engine.loader)
    if factor_name == "ema":
        return build_ema_signals(engine.close_df, dates, engine.loader)
    if factor_name == "atr":
        high_df, low_df = _build_high_low(engine.prices)
        return build_atr_signals(engine.close_df, high_df, low_df, dates, engine.loader)
    if factor_name == "obv":
        return build_obv_signals(engine.close_df, engine.volume_df, dates, engine.loader)
    if factor_name == "ichimoku":
        high_df, low_df = _build_high_low(engine.prices)
        return build_ichimoku_signals(engine.close_df, high_df, low_df, dates, engine.loader)
    if factor_name == "parabolic_sar":
        high_df, low_df = _build_high_low(engine.prices)
        return build_parabolic_sar_signals(engine.close_df, high_df, low_df, dates, engine.loader)

    raise ValueError(f"Unsupported signal for legacy compare: {factor_name}")


def compare_panels(
    legacy: pd.DataFrame,
    refactored: pd.DataFrame,
    atol: float,
) -> dict:
    shape_match = legacy.shape == refactored.shape
    index_match = legacy.index.equals(refactored.index)
    columns_match = legacy.columns.equals(refactored.columns)

    dtype_mismatches = None
    if columns_match:
        dtype_mismatches = int((legacy.dtypes.astype(str) != refactored.dtypes.astype(str)).sum())

    if not (shape_match and index_match and columns_match):
        return {
            "shape_match": shape_match,
            "index_match": index_match,
            "columns_match": columns_match,
            "dtype_mismatches": dtype_mismatches,
            "max_abs_diff": np.nan,
            "mismatched_cells": -1,
            "nan_mismatch_cells": -1,
        }

    legacy_num = legacy.astype(float)
    ref_num = refactored.astype(float)

    legacy_nan = legacy_num.isna()
    ref_nan = ref_num.isna()
    nan_mismatch = legacy_nan ^ ref_nan

    diff = (legacy_num - ref_num).abs()
    valid_diff = diff.where(~(legacy_nan | ref_nan))
    max_abs_diff = float(valid_diff.max().max()) if np.isfinite(valid_diff.max().max()) else 0.0

    value_mismatch = (diff > atol) & ~(legacy_nan | ref_nan)
    mismatched_cells = int(value_mismatch.sum().sum())
    nan_mismatch_cells = int(nan_mismatch.sum().sum())

    return {
        "shape_match": shape_match,
        "index_match": index_match,
        "columns_match": columns_match,
        "dtype_mismatches": dtype_mismatches,
        "max_abs_diff": max_abs_diff,
        "mismatched_cells": mismatched_cells,
        "nan_mismatch_cells": nan_mismatch_cells,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate signal parity (legacy vs factory).")
    parser.add_argument("--signals", nargs="+", default=DEFAULT_SIGNALS)
    parser.add_argument("--start-date", default="2018-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--regime-outputs", type=str, default=None)
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else (PROJECT_ROOT / "data").resolve()
    if args.regime_outputs:
        regime_dir = Path(args.regime_outputs).expanduser().resolve()
    else:
        candidates = [
            (PROJECT_ROOT / "Simple Regime Filter" / "outputs").resolve(),
            (PROJECT_ROOT / "regime_filter" / "outputs").resolve(),
        ]
        regime_dir = next((path for path in candidates if path.exists()), candidates[0])
    return data_dir, regime_dir


def main() -> int:
    args = parse_args()
    data_dir, regime_outputs = resolve_paths(args)
    configs = load_signal_configs()
    missing = [signal for signal in args.signals if signal not in configs]
    if missing:
        raise ValueError(f"Signals missing in config: {missing}")

    engine = PortfolioEngine(
        data_dir=data_dir,
        regime_model_dir=regime_outputs,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    engine.load_all_data()
    start_ts = pd.Timestamp(args.start_date)
    end_ts = pd.Timestamp(args.end_date)
    dates = engine.close_df.index[(engine.close_df.index >= start_ts) & (engine.close_df.index <= end_ts)]

    failures = []

    logger.info("=" * 90)
    logger.info("SIGNAL PARITY CHECK (legacy vs factory)")
    logger.info("=" * 90)
    logger.info(f"Signals: {args.signals}")
    logger.info(f"Dates: {dates.min().date()} -> {dates.max().date()} ({len(dates)} days)")
    logger.info(f"Tolerance: {args.atol}")

    for signal_name in args.signals:
        config = copy.deepcopy(configs[signal_name])
        legacy = build_legacy_signal(signal_name, dates, config, engine)

        runtime_config = copy.deepcopy(configs[signal_name])
        runtime_config["_runtime_context"] = {
            "prices": engine.prices,
            "close_df": engine.close_df,
            "open_df": engine.open_df,
            "volume_df": engine.volume_df,
            "fundamentals": engine.fundamentals,
            "xu100_prices": engine.xu100_prices,
        }
        refactored = build_signal(signal_name, dates, engine.loader, runtime_config)

        result = compare_panels(legacy, refactored, atol=args.atol)
        ok = (
            result["shape_match"]
            and result["index_match"]
            and result["columns_match"]
            and result["mismatched_cells"] == 0
            and result["nan_mismatch_cells"] == 0
        )

        logger.info(
            f"[{'OK' if ok else 'FAIL'}] {signal_name:<28} "
            f"shape={result['shape_match']} index={result['index_match']} cols={result['columns_match']} "
            f"dtype_mismatches={result['dtype_mismatches']} "
            f"max_abs_diff={result['max_abs_diff']:.6g} "
            f"value_mismatch={result['mismatched_cells']} "
            f"nan_mismatch={result['nan_mismatch_cells']}"
        )

        if not ok:
            failures.append((signal_name, result))

    if failures:
        logger.info("\nMISMATCH SUMMARY")
        for signal_name, result in failures:
            logger.info(f"- {signal_name}: {result}")
        return 1

    logger.info("\nAll compared signals match within tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
