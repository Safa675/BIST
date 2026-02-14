import os
from typing import Any, Mapping

import numpy as np
import pandas as pd

from Models.signals._context import (
    build_high_low_panels,
    get_runtime_context,
    parse_int_param,
    require_context,
)
from Models.signals.betting_against_beta_signals import build_betting_against_beta_signals
from Models.signals.breakout_value_signals import build_breakout_value_signals
from Models.signals.five_factor_rotation_signals import build_five_factor_rotation_signals
from Models.signals.momentum_asset_growth_signals import build_momentum_asset_growth_signals
from Models.signals.pairs_trading_signals import build_pairs_trading_signals
from Models.signals.quality_momentum_signals import build_quality_momentum_signals
from Models.signals.quality_value_signals import build_quality_value_signals
from Models.signals.size_rotation_momentum_signals import build_size_rotation_momentum_signals
from Models.signals.size_rotation_quality_signals import build_size_rotation_quality_signals
from Models.signals.size_rotation_signals import build_size_rotation_signals
from Models.signals.small_cap_momentum_signals import build_small_cap_momentum_signals
from Models.signals.trend_value_signals import build_trend_value_signals

ConfigDict = dict[str, Any]
SignalParams = dict[str, Any]


def weighted_sum(
    panels: Mapping[str, pd.DataFrame],
    weights: Mapping[str, float],
) -> pd.DataFrame:
    if not panels:
        raise ValueError("weighted_sum requires at least one panel")

    missing = [name for name in panels if name not in weights]
    if missing:
        raise ValueError(f"weighted_sum missing weights for panels: {missing}")

    result = None
    total_weight = 0.0
    for name, panel in panels.items():
        weight = float(weights[name])
        total_weight += weight
        weighted_panel = panel * weight
        result = weighted_panel if result is None else result.add(weighted_panel, fill_value=np.nan)

    if result is None:
        raise ValueError("weighted_sum received no valid panels")
    if total_weight == 0:
        raise ValueError("weighted_sum total weight cannot be zero")
    return result / total_weight


def zscore_blend(
    panels: Mapping[str, pd.DataFrame],
    weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    if not panels:
        raise ValueError("zscore_blend requires at least one panel")

    z_panels = {}
    for name, panel in panels.items():
        mean = panel.mean(axis=1)
        std = panel.std(axis=1).replace(0, np.nan)
        z_panels[name] = panel.sub(mean, axis=0).div(std, axis=0)

    if weights is None:
        equal_weight = 1.0 / len(z_panels)
        weights = {name: equal_weight for name in z_panels}
    return weighted_sum(z_panels, weights)


def rank_blend(
    panels: Mapping[str, pd.DataFrame],
    weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    if not panels:
        raise ValueError("rank_blend requires at least one panel")

    rank_panels = {name: panel.rank(axis=1, pct=True) for name, panel in panels.items()}
    if weights is None:
        equal_weight = 1.0 / len(rank_panels)
        weights = {name: equal_weight for name in rank_panels}
    return weighted_sum(rank_panels, weights)


def build_trend_value_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    close_df = require_context("trend_value", get_runtime_context(config), "close_df")
    return build_trend_value_signals(close_df, dates, loader)


def build_breakout_value_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    context = get_runtime_context(config)
    close_df = require_context("breakout_value", context, "close_df")
    high_df, low_df = build_high_low_panels("breakout_value", context)
    return build_breakout_value_signals(close_df, high_df, low_df, dates, loader)


def build_quality_momentum_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    context = get_runtime_context(config)
    close_df = require_context("quality_momentum", context, "close_df")
    fundamentals = require_context("quality_momentum", context, "fundamentals")
    return build_quality_momentum_signals(close_df, fundamentals, dates, loader)


def build_quality_value_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    context = get_runtime_context(config)
    close_df = require_context("quality_value", context, "close_df")
    fundamentals = require_context("quality_value", context, "fundamentals")
    return build_quality_value_signals(close_df, fundamentals, dates, loader)


def build_small_cap_momentum_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    close_df = require_context(
        "small_cap_momentum",
        get_runtime_context(config),
        "close_df",
    )
    return build_small_cap_momentum_signals(close_df, dates, loader)


def build_momentum_asset_growth_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    context = get_runtime_context(config)
    fundamentals = require_context("momentum_asset_growth", context, "fundamentals")
    close_df = require_context("momentum_asset_growth", context, "close_df")
    return build_momentum_asset_growth_signals(fundamentals, close_df, dates, loader)


def build_pairs_trading_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    close_df = require_context(
        "pairs_trading",
        get_runtime_context(config),
        "close_df",
    )
    return build_pairs_trading_signals(close_df, dates, loader)


def build_betting_against_beta_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    close_df = require_context(
        "betting_against_beta",
        get_runtime_context(config),
        "close_df",
    )
    beta_window = parse_int_param(
        "betting_against_beta",
        signal_params,
        "beta_window",
        252,
    )
    return build_betting_against_beta_signals(close_df, dates, loader, beta_window=beta_window)


def build_size_rotation_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    close_df = require_context("size_rotation", get_runtime_context(config), "close_df")
    return build_size_rotation_signals(close_df, dates, loader)


def build_size_rotation_momentum_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    close_df = require_context(
        "size_rotation_momentum",
        get_runtime_context(config),
        "close_df",
    )
    return build_size_rotation_momentum_signals(close_df, dates, loader)


def build_size_rotation_quality_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    context = get_runtime_context(config)
    close_df = require_context("size_rotation_quality", context, "close_df")
    fundamentals = require_context("size_rotation_quality", context, "fundamentals")
    return build_size_rotation_quality_signals(close_df, fundamentals, dates, loader)


def build_five_factor_rotation_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
    signal_params: SignalParams,
) -> pd.DataFrame:
    context = get_runtime_context(config)
    close_df = require_context("five_factor_rotation", context, "close_df")
    fundamentals = require_context("five_factor_rotation", context, "fundamentals")
    volume_df = require_context("five_factor_rotation", context, "volume_df")

    cache_cfg = config.get("construction_cache", {})
    if not isinstance(cache_cfg, dict):
        cache_cfg = {}

    debug_cfg = config.get("debug", {})
    if not isinstance(debug_cfg, dict):
        debug_cfg = {}

    orth_cfg = config.get("axis_orthogonalization", {})
    if not isinstance(orth_cfg, dict):
        orth_cfg = {}

    walk_forward_cfg = config.get("walk_forward", {})
    if not isinstance(walk_forward_cfg, dict):
        walk_forward_cfg = {}

    debug_env = os.getenv("FIVE_FACTOR_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
    debug_enabled = bool(debug_cfg.get("enabled", False) or debug_env)

    signals, factor_details = build_five_factor_rotation_signals(
        close_df,
        dates,
        loader,
        fundamentals=fundamentals,
        volume_df=volume_df,
        use_construction_cache=cache_cfg.get("enabled", True),
        force_rebuild_construction_cache=cache_cfg.get("force_rebuild", False),
        construction_cache_path=cache_cfg.get("path"),
        mwu_walkforward_config=walk_forward_cfg,
        axis_orthogonalization_config=orth_cfg,
        return_details=True,
        debug=debug_enabled,
    )

    config["_factor_details"] = factor_details
    return signals


BUILDERS = {
    "trend_value": build_trend_value_from_config,
    "breakout_value": build_breakout_value_from_config,
    "quality_momentum": build_quality_momentum_from_config,
    "quality_value": build_quality_value_from_config,
    "small_cap_momentum": build_small_cap_momentum_from_config,
    "momentum_asset_growth": build_momentum_asset_growth_from_config,
    "pairs_trading": build_pairs_trading_from_config,
    "betting_against_beta": build_betting_against_beta_from_config,
    "size_rotation": build_size_rotation_from_config,
    "size_rotation_momentum": build_size_rotation_momentum_from_config,
    "size_rotation_quality": build_size_rotation_quality_from_config,
    "five_factor_rotation": build_five_factor_rotation_from_config,
}
