"""Thin orchestrator for five-factor rotation signal construction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from Models.common.utils import (
    assert_has_cross_section,
    assert_panel_not_constant,
    raise_signal_data_error,
    validate_signal_panel_schema,
)
from Models.signals.axis_cache import AXIS_PANEL_NAMES, _resolve_axis_cache_path
from Models.signals.debug_utils import _debug_log, _debug_panel_stats
from Models.signals.factor_axes import ENSEMBLE_LOOKBACK_WINDOWS
from Models.signals.five_factor_pipeline import (
    OPTIONAL_EMPTY_CACHE_PANELS,
    USE_QUINTILE_BUCKETS,
    AxisCacheContract,
    AxisCacheManager,
    AxisComponentsBundle,
    AxisComponentService,
    FactorConstructionPipeline,
    FiveFactorBuildContext,
    FiveFactorDataPreparationService,
    MWUService,
    OrthogonalizationService,
    _build_axis_construction_panels,
    _build_yearly_axis_winner_report,
)

logger = logging.getLogger(__name__)


def build_five_factor_rotation_signals(
    close_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader=None,
    fundamentals: Dict | None = None,
    volume_df: pd.DataFrame | None = None,
    use_construction_cache: bool = True,
    force_rebuild_construction_cache: bool = False,
    construction_cache_path: Path | str | None = None,
    mwu_walkforward_config: dict | None = None,
    axis_orthogonalization_config: dict | None = None,
    return_details: bool = False,
    debug: bool = False,
    include_debug_artifacts: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """Build multi-factor rotation score panel (dates x tickers, 0-100)."""
    logger.info("\n🔧 Building multi-factor rotation signals (13 axes, exponential weighting)...")
    logger.info(f"  Multi-lookback ensemble: {ENSEMBLE_LOOKBACK_WINDOWS} days")
    logger.info("  Original: Size / Value / Profitability / Investment / Momentum / Risk")
    logger.info("  New: Quality / Liquidity / TradingIntensity / Sentiment / FundMom / Carry / Defensive")
    logger.info("  Axis weighting: Exponentially-weighted factor selection (6mo half-life)")
    _debug_log(debug, "Detailed line-by-line debug tracing is enabled")

    context = FiveFactorDataPreparationService().prepare(
        close_df=close_df,
        dates=dates,
        data_loader=data_loader,
        fundamentals=fundamentals,
        volume_df=volume_df,
    )

    axis_panels = AxisCacheManager(
        data_loader=data_loader,
        contract=AxisCacheContract(
            panel_names=AXIS_PANEL_NAMES,
            optional_empty_panels=frozenset(OPTIONAL_EMPTY_CACHE_PANELS),
        ),
    ).load_or_build(
        context=context,
        use_cache=use_construction_cache,
        force_rebuild_cache=force_rebuild_construction_cache,
        save_cache=True,
        cache_path=construction_cache_path,
    )
    for panel_name in AXIS_PANEL_NAMES:
        _debug_panel_stats(debug, f"panel:{panel_name}", axis_panels.get(panel_name))

    axis_specs = FactorConstructionPipeline(debug=debug).build_axis_specs(
        context=context,
        axis_panels=axis_panels,
    )
    n_total = len(axis_specs)

    axis_specs, orth_enabled, orth_details = OrthogonalizationService(debug=debug).maybe_orthogonalize(
        axis_specs=axis_specs,
        axis_orthogonalization_config=axis_orthogonalization_config,
    )

    components_bundle = AxisComponentService(debug=debug).build(
        axis_specs=axis_specs,
        daily_returns=context.daily_returns,
    )

    axis_weights = MWUService.compute_axis_weights(
        axis_daily_returns=components_bundle.axis_daily_returns,
        signal_dates=context.signal_dates,
        mwu_walkforward_config=mwu_walkforward_config,
        debug=debug,
    )

    weighted_bundle = MWUService.combine_axis_components(
        axis_components=components_bundle.axis_components,
        axis_weights=axis_weights,
        signal_dates=context.signal_dates,
        tickers=context.tickers,
    )
    aligned_components = weighted_bundle["aligned_components"]
    final_scores = weighted_bundle["final_scores"]

    final_scores = final_scores.clip(0.0, 100.0)
    final_scores = validate_signal_panel_schema(
        final_scores,
        dates=context.signal_dates,
        tickers=context.tickers,
        signal_name="five_factor_rotation",
        context="final score panel",
        dtype=np.float32,
    )
    assert_has_cross_section(
        final_scores,
        "five_factor_rotation",
        "final score panel",
        min_valid_tickers=5,
    )
    assert_panel_not_constant(final_scores, "five_factor_rotation", "final score panel")
    latest_valid = int(final_scores.iloc[-1].notna().sum()) if len(final_scores.index) else 0
    if latest_valid < 5:
        raise_signal_data_error(
            "five_factor_rotation",
            f"latest date has insufficient coverage: {latest_valid} valid names (< 5)",
        )
    _debug_panel_stats(debug, "final_scores", final_scores)

    latest_date = final_scores.index[-1]
    latest_scores = final_scores.loc[latest_date].dropna()

    logger.info(f"  Latest date: {latest_date.date()}")
    if len(latest_scores) > 0:
        logger.info(f"  Latest scores - Mean: {latest_scores.mean():.1f}, Std: {latest_scores.std():.1f}")
        top_5 = latest_scores.nlargest(5)
        logger.info(f"  Top 5 stocks: {', '.join(top_5.index.tolist())}")

    latest_w = pd.Series(dtype=float)
    if not axis_weights.empty:
        latest_w = axis_weights.loc[latest_date] if latest_date in axis_weights.index else axis_weights.iloc[-1]
    if debug and not axis_weights.empty:
        sample_indices = [0, len(axis_weights) // 2, len(axis_weights) - 1]
        seen_dates = set()
        for sample_idx in sample_indices:
            sample_date = axis_weights.index[sample_idx]
            if sample_date in seen_dates:
                continue
            seen_dates.add(sample_date)
            sample_w = axis_weights.loc[sample_date].sort_values(ascending=False).head(5)
            sample_str = ", ".join([f"{k}={v:.1%}" for k, v in sample_w.items()])
            _debug_log(debug, f"axis_weights@{sample_date.date()}: {sample_str}")

    logger.info("  MWU axis weights:")
    if latest_w.empty:
        logger.info("    unavailable")
    else:
        for name in sorted(latest_w.index, key=lambda n: -latest_w[n]):
            logger.info(f"    {name:<16}: {latest_w[name]:.1%}")

    yearly_report = _build_yearly_axis_winner_report(
        components_bundle.axis_summary,
        components_bundle.axis_daily_returns,
        components_bundle.axis_bucket_returns,
        USE_QUINTILE_BUCKETS,
    )

    logger.info(
        f"  Multi-factor rotation signals: {final_scores.shape[0]} days x {final_scores.shape[1]} tickers ({n_total} axes)"
    )

    if return_details:
        details = {
            "yearly_axis_winners": yearly_report,
            "axis_weights": axis_weights,
            "axis_components": aligned_components,
            "active_axes": list(axis_specs.keys()),
        }
        if include_debug_artifacts:
            details["axis_raw_scores"] = {
                axis_name: axis_raw.reindex(index=context.signal_dates, columns=context.tickers)
                for axis_name, (axis_raw, _, _) in axis_specs.items()
            }
            details["axis_bucket_returns"] = dict(components_bundle.axis_bucket_returns)
            details["axis_winning_side"] = {
                axis_name: components_bundle.axis_summary[axis_name][0]
                for axis_name in axis_specs.keys()
            }
        if orth_enabled:
            details["axis_orthogonalization"] = {
                "method": "cross_sectional_residualization",
                "axis_order": orth_details.get("axis_order", list(axis_specs.keys())),
                "raw_mean_abs_corr": orth_details.get("raw_mean_abs_corr", np.nan),
                "orth_mean_abs_corr": orth_details.get("orth_mean_abs_corr", np.nan),
                "raw_daily_mean_abs_corr": orth_details.get("raw_daily_mean_abs_corr"),
                "orth_daily_mean_abs_corr": orth_details.get("orth_daily_mean_abs_corr"),
            }
        return final_scores, details

    return final_scores


def build_five_factor_rotation_axis_cache(
    close_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader,
    fundamentals: Dict | None = None,
    volume_df: pd.DataFrame | None = None,
    cache_path: Path | str | None = None,
    force_rebuild: bool = True,
) -> Path:
    """Precompute heavy axis-construction inputs and persist to parquet."""
    if fundamentals is None:
        fundamentals = data_loader.load_fundamentals() if data_loader is not None else {}

    if volume_df is None:
        volume_df = getattr(data_loader, "_volume_df", None)
    if volume_df is None and data_loader is not None:
        prices_file = data_loader.data_dir / "bist_prices_full.csv"
        prices = data_loader.load_prices(prices_file)
        volume_df = data_loader.build_volume_panel(prices)
    if volume_df is None:
        volume_df = pd.DataFrame(np.nan, index=dates, columns=close_df.columns)

    close = close_df.reindex(dates).astype(float)
    tickers = close.columns

    _build_axis_construction_panels(
        close=close,
        dates=dates,
        tickers=tickers,
        data_loader=data_loader,
        fundamentals=fundamentals,
        volume_df=volume_df,
        use_cache=True,
        force_rebuild_cache=force_rebuild,
        save_cache=True,
        cache_path=cache_path,
    )
    return _resolve_axis_cache_path(data_loader, cache_path)


__all__ = [
    "AxisCacheContract",
    "AxisCacheManager",
    "AxisComponentService",
    "AxisComponentsBundle",
    "FiveFactorBuildContext",
    "FiveFactorDataPreparationService",
    "FactorConstructionPipeline",
    "MWUService",
    "OrthogonalizationService",
    "build_five_factor_rotation_axis_cache",
    "build_five_factor_rotation_signals",
]
