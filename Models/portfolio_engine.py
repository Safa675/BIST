#!/usr/bin/env python3
"""
Config-Based Portfolio Engine

Orchestrates all factor models with:
- Centralized data loading (load once, use multiple times)
- Config-based signal integration
- Comprehensive reporting

Usage:
    python portfolio_engine.py --factor profitability
    python portfolio_engine.py --factor momentum
    python portfolio_engine.py --factor all
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from Models.common.backtester import Backtester, identify_quarterly_rebalance_days
from Models.common.config_manager import (
    LIQUIDITY_QUANTILE,
    REGIME_ALLOCATIONS,
    TARGET_DOWNSIDE_VOL,
    VOL_CAP,
    VOL_FLOOR,
    VOL_LOOKBACK,
)
from Models.common.config_manager import (
    load_signal_configs as load_configs_from_manager,
)
from Models.common.data_loader import DataLoader
from Models.common.data_manager import DataManager
from Models.common.panel_cache import PanelCache
from Models.common.report_generator import (
    ReportGenerator,
)
from Models.common.risk_manager import RiskManager
from Models.signals.factory import build_signal
from Models.signals.size_rotation_signals import (
    build_market_cap_panel as build_size_market_cap_panel,
)

logger = logging.getLogger(__name__)

def apply_downside_vol_targeting(
    returns: pd.Series,
    target_vol: float = TARGET_DOWNSIDE_VOL,
    lookback: int = VOL_LOOKBACK,
    vol_floor: float = VOL_FLOOR,
    vol_cap: float = VOL_CAP,
) -> pd.Series:
    """
    Apply downside volatility targeting to scale returns.
    
    Scales position sizes to target a constant annualized downside volatility.
    When realized vol is low, increase exposure; when high, reduce it.
    
    Args:
        returns: Daily portfolio returns
        target_vol: Target annualized downside volatility (default 20%)
        lookback: Days to look back for realized vol calculation
        vol_floor: Minimum scaling factor (default 0.10 = 10% leverage min)
        vol_cap: Maximum scaling factor (default 1.0 = 100% leverage max)
    
    Returns:
        pd.Series: Volatility-targeted returns
    """
    if len(returns) < lookback:
        return returns
    
    # Vectorized rolling downside volatility with legacy >2 negative-return guard.
    min_periods = lookback // 2
    negative_only = returns.where(returns < 0.0)
    total_counts = returns.rolling(lookback, min_periods=min_periods).count()
    negative_counts = negative_only.rolling(lookback, min_periods=1).count()
    rolling_downside_vol = negative_only.rolling(lookback, min_periods=1).std() * np.sqrt(252)
    rolling_downside_vol = rolling_downside_vol.where(
        (total_counts >= min_periods) & (negative_counts > 2)
    )
    
    # Calculate leverage factor: target_vol / realized_vol
    # Shift by 1 to avoid lookahead bias (use yesterday's vol for today's sizing)
    leverage = target_vol / rolling_downside_vol.shift(1)
    
    # Clip leverage to reasonable bounds
    leverage = leverage.clip(lower=vol_floor, upper=vol_cap)
    
    # Fill NaN (early period) with 1.0 (no scaling)
    leverage = leverage.fillna(1.0)
    
    # Apply leverage to returns
    targeted_returns = returns * leverage
    
    return targeted_returns
# ============================================================================
# CONFIG LOADING
# ============================================================================

def load_signal_configs(prefer_yaml: bool = True) -> dict:
    """
    Backward-compatible signal config loader shim.

    Delegates to common.config_manager.ConfigManager.
    """
    return load_configs_from_manager(prefer_yaml=prefer_yaml)


# ============================================================================
# PORTFOLIO ENGINE
# ============================================================================

class PortfolioEngine:
    """Config-based portfolio engine"""
    
    def __init__(self, data_dir: Path, regime_model_dir: Path, start_date: str, end_date: str):
        self.data_dir = Path(data_dir)
        self.regime_model_dir = Path(regime_model_dir)
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        
        # Load signal configurations
        self.signal_configs = load_signal_configs()
        logger.info(f"\n📋 Loaded {len(self.signal_configs)} signal configurations:")
        for name, config in self.signal_configs.items():
            status = "✅ Enabled" if config.get('enabled', True) else "⚠️  Disabled"
            rebal = config.get('rebalance_frequency', 'quarterly')
            logger.info(f"   {name}: {status} ({rebal})")
        
        # Initialize data loader
        self.loader = DataLoader(data_dir, regime_model_dir)
        if not hasattr(self.loader, "panel_cache") or self.loader.panel_cache is None:
            self.loader.panel_cache = PanelCache()
        self.panel_cache = self.loader.panel_cache
        self.data_manager = DataManager(
            data_loader=self.loader,
            data_dir=self.data_dir,
            base_regime_allocations=REGIME_ALLOCATIONS,
        )
        self.risk_manager = RiskManager()
        self.backtester = Backtester(
            loader=self.loader,
            data_dir=self.data_dir,
            risk_manager=self.risk_manager,
            build_size_market_cap_panel=build_size_market_cap_panel,
        )
        self.report_generator = ReportGenerator(
            models_dir=Path(__file__).parent,
            data_dir=self.data_dir,
            loader=self.loader,
        )
        
        # Cached data
        self.prices = None
        self.close_df = None
        self.open_df = None
        self.volume_df = None
        self.regime_series = None
        self.regime_allocations = REGIME_ALLOCATIONS.copy()
        self.xautry_prices = None
        self.xu100_prices = None
        self.fundamentals = None
        
        # Store factor returns for correlation analysis
        self.factor_returns = {}
        self.factor_capm = {}
        self.factor_yearly_rolling_beta = {}

    def _apply_loaded_data(self, loaded_data) -> None:
        self.prices = loaded_data.prices
        self.close_df = loaded_data.close_df
        self.open_df = loaded_data.open_df
        self.volume_df = loaded_data.volume_df
        self.fundamentals = loaded_data.fundamentals
        self.regime_series = loaded_data.regime_series
        self.regime_allocations = loaded_data.regime_allocations
        self.xautry_prices = loaded_data.xautry_prices
        self.xu100_prices = loaded_data.xu100_prices
        self.backtester.update_data(
            prices=self.prices,
            close_df=self.close_df,
            volume_df=self.volume_df,
            regime_series=self.regime_series,
            regime_allocations=self.regime_allocations,
            xu100_prices=self.xu100_prices,
            xautry_prices=self.xautry_prices,
        )

    def load_all_data(self, use_cache: bool = True):
        """Load all required datasets and cache prepared data panels."""
        loaded_data = self.data_manager.load_all(use_cache=use_cache)
        self._apply_loaded_data(loaded_data)

    def _build_runtime_config(self, config: dict | None) -> dict:
        runtime_config = dict(config) if isinstance(config, dict) else {}
        runtime_config["_runtime_context"] = self.data_manager.build_runtime_context()
        return runtime_config

    def _resolve_factor_config(self, factor_name: str, override_config: dict | None) -> dict:
        if override_config:
            if not isinstance(override_config, dict):
                raise TypeError(
                    f"override_config must be dict or None, got {type(override_config).__name__}"
                )
            return override_config

        config = self.signal_configs.get(factor_name)
        if not isinstance(config, dict):
            raise ValueError(f"No config found for factor: {factor_name}")
        return config

    def _resolve_factor_timeline(self, factor_name: str, config: dict) -> tuple[pd.Timestamp, pd.Timestamp, bool]:
        timeline = config.get('timeline', {})
        if not isinstance(timeline, dict):
            timeline = {}

        custom_start = timeline.get('start_date')
        custom_end = timeline.get('end_date')
        factor_start_date = pd.Timestamp(custom_start) if custom_start else self.start_date
        factor_end_date = pd.Timestamp(custom_end) if custom_end else self.end_date

        # Optional walk-forward timeline clamp (used by five_factor_rotation)
        walk_forward_cfg = config.get("walk_forward", {})
        if not isinstance(walk_forward_cfg, dict):
            walk_forward_cfg = {}
        if factor_name == "five_factor_rotation" and walk_forward_cfg.get("enabled", False):
            first_test_year = walk_forward_cfg.get("first_test_year")
            last_test_year = walk_forward_cfg.get("last_test_year")
            if first_test_year is not None:
                wf_start = pd.Timestamp(year=int(first_test_year), month=1, day=1)
                if factor_start_date < wf_start:
                    logger.info(f"Walk-forward start clamp: {factor_start_date.date()} -> {wf_start.date()}")
                    factor_start_date = wf_start
            if last_test_year is not None:
                wf_end = pd.Timestamp(year=int(last_test_year), month=12, day=31)
                if factor_end_date > wf_end:
                    logger.info(f"Walk-forward end clamp: {factor_end_date.date()} -> {wf_end.date()}")
                    factor_end_date = wf_end

        has_custom_timeline = bool(custom_start or custom_end)
        return factor_start_date, factor_end_date, has_custom_timeline

    def _resolve_portfolio_options(self, portfolio_options: dict | None) -> dict:
        return self.risk_manager.resolve_options(portfolio_options)

    def _print_portfolio_settings(self, opts: dict) -> None:
        self.risk_manager.print_settings(opts)

    def _build_signals_for_factor(self, factor_name: str, dates: pd.DatetimeIndex, config: dict):
        """Build factor signal panel using signals.factory dispatch."""
        runtime_config = self._build_runtime_config(config)
        signals = build_signal(factor_name, dates, self.loader, runtime_config)
        factor_details = runtime_config.get("_factor_details", {})
        return signals, factor_details
        
    def run_factor(self, factor_name: str, override_config: dict = None):
        """Run backtest for a single factor using its config"""
        logger.info("\n" + "="*70)
        logger.info(f"RUNNING {factor_name.upper()} FACTOR")
        logger.info("="*70)
        
        config = self._resolve_factor_config(factor_name, override_config)
        
        # Check if enabled
        if not config.get('enabled', True):
            logger.warning(f"⚠️  {factor_name.upper()} is disabled in config")
            return None
        
        # Get rebalancing frequency from config
        rebalance_freq = config.get('rebalance_frequency', 'quarterly')
        logger.info(f"Rebalancing frequency: {rebalance_freq}")
        
        factor_start_date, factor_end_date, has_custom_timeline = self._resolve_factor_timeline(
            factor_name,
            config,
        )
        
        # Display timeline
        if has_custom_timeline:
            logger.info(f"Custom timeline: {factor_start_date.date()} to {factor_end_date.date()}")
        
        start_time = time.time()
        dates = self.close_df.index
        signals, factor_details = self._build_signals_for_factor(factor_name, dates, config)

        # Get portfolio options from config
        portfolio_options = config.get('portfolio_options', {})

        # Run backtest with custom timeline and portfolio options
        results = self._run_backtest(
            signals,
            factor_name,
            rebalance_freq,
            factor_start_date,
            factor_end_date,
            portfolio_options,
        )
        if factor_details:
            results.update(factor_details)
            if 'yearly_axis_winners' in results and isinstance(results['yearly_axis_winners'], pd.DataFrame):
                yearly_axis = results['yearly_axis_winners']
                if not yearly_axis.empty and 'Year' in yearly_axis.columns:
                    start_year = int(factor_start_date.year)
                    end_year = int(factor_end_date.year)
                    results['yearly_axis_winners'] = yearly_axis[
                        (yearly_axis['Year'] >= start_year) & (yearly_axis['Year'] <= end_year)
                    ].copy()
        
        # Save results
        self.save_results(results, factor_name)
        
        # Store returns for correlation analysis
        self.factor_returns[factor_name] = results['returns']
        
        elapsed = time.time() - start_time
        logger.info(f"\n✅ {factor_name.upper()} completed in {elapsed:.1f} seconds")
        
        return results
    
    def _run_backtest(self, signals: pd.DataFrame, factor_name: str, rebalance_freq: str = 'quarterly',
                     start_date: pd.Timestamp = None, end_date: pd.Timestamp = None,
                     portfolio_options: dict = None):
        """Run backtest through the modular Backtester service."""
        return self.backtester.run(
            signals=signals,
            factor_name=factor_name,
            rebalance_freq=rebalance_freq,
            start_date=start_date if start_date is not None else self.start_date,
            end_date=end_date if end_date is not None else self.end_date,
            portfolio_options=portfolio_options,
        )

    def _identify_quarterly_rebalance_days(self, trading_days: pd.DatetimeIndex) -> set:
        """Backward-compatible wrapper around modular quarterly rebalance calendar."""
        return identify_quarterly_rebalance_days(trading_days)
    
    def _filter_by_liquidity(self, tickers, date, liquidity_quantile=LIQUIDITY_QUANTILE):
        """Backward-compatible wrapper around RiskManager liquidity filter."""
        return self.risk_manager.filter_by_liquidity(tickers, date, liquidity_quantile)
    
    def save_results(self, results, factor_name, output_dir=None):
        """Save backtest results via modular ReportGenerator."""
        self.report_generator.save_results(
            results=results,
            factor_name=factor_name,
            xu100_prices=self.xu100_prices,
            xautry_prices=self.xautry_prices,
            factor_capm_store=self.factor_capm,
            factor_yearly_rolling_beta_store=self.factor_yearly_rolling_beta,
            output_dir=output_dir,
        )
    
    def save_correlation_matrix(self, output_dir=None):
        """Save full return-correlation matrix via modular ReportGenerator."""
        return self.report_generator.save_correlation_matrix(
            factor_returns=self.factor_returns,
            xautry_prices=self.xautry_prices,
            output_dir=output_dir,
        )
    
    def run_all_factors(self):
        """Run all enabled factors"""
        results = {}
        
        for factor_name, config in self.signal_configs.items():
            if config.get('enabled', True):
                results[factor_name] = self.run_factor(factor_name)
            else:
                logger.warning(f"\n⚠️  Skipping {factor_name} (disabled in config)")
        
        # Save correlation matrix after all factors complete
        if self.factor_returns:
            self.save_correlation_matrix()
        if self.factor_capm:
            self.save_capm_summary()
        if self.factor_yearly_rolling_beta:
            self.save_yearly_rolling_beta_summary()
        
        return results

    def save_capm_summary(self, output_dir=None):
        """Save CAPM summary across all factors via modular ReportGenerator."""
        self.report_generator.save_capm_summary(
            factor_capm=self.factor_capm,
            output_dir=output_dir,
            models_dir=Path(__file__).parent,
        )

    def save_yearly_rolling_beta_summary(self, output_dir=None):
        """Save yearly rolling-beta summary via modular ReportGenerator."""
        self.report_generator.save_yearly_rolling_beta_summary(
            factor_yearly_rolling_beta=self.factor_yearly_rolling_beta,
            output_dir=output_dir,
            models_dir=Path(__file__).parent,
        )


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Load available signals dynamically from configs
    available_signals = load_signal_configs()
    signal_names = list(available_signals.keys())
    
    parser = argparse.ArgumentParser(
        description='Config-Based Portfolio Engine - Automatically detects signals from configs/',
        epilog=f'Available signals: {", ".join(signal_names)}'
    )
    
    # Support both positional and --factor argument
    parser.add_argument('signal', nargs='?', type=str, default=None,
                       help=f'Signal to run: {", ".join(signal_names)}, or "all"')
    parser.add_argument('--factor', type=str, default=None,
                       help='Alternative way to specify signal (deprecated, use positional arg)')
    parser.add_argument('--start-date', type=str, default='2018-01-01',
                       help='Start date (default: 2018-01-01)')
    parser.add_argument('--end-date', type=str, default='2024-12-31',
                       help='End date (default: 2024-12-31)')
    
    args = parser.parse_args()
    
    # Determine which signal to run (positional takes precedence)
    signal_to_run = args.signal or args.factor or 'all'
    
    # Validate signal name
    if signal_to_run != 'all' and signal_to_run not in signal_names:
        logger.error(f"❌ Unknown signal: {signal_to_run}")
        logger.info(f"Available signals: {', '.join(signal_names)}, all")
        sys.exit(1)
    
    # Setup paths
    script_dir = Path(__file__).parent  # Models/
    bist_root = script_dir.parent        # BIST/
    data_dir = bist_root / "data"
    regime_model_dir_candidates = [
        bist_root / "Simple Regime Filter" / "outputs",
        bist_root / "regime_filter" / "outputs",
    ]
    regime_model_dir = next((p for p in regime_model_dir_candidates if p.exists()), regime_model_dir_candidates[0])



    
    # Initialize engine
    engine = PortfolioEngine(data_dir, regime_model_dir, args.start_date, args.end_date)
    engine.load_all_data()
    
    # Run signal(s)
    if signal_to_run == 'all':
        engine.run_all_factors()
    else:
        engine.run_factor(signal_to_run)


if __name__ == "__main__":
    total_start = time.time()
    main()
    total_elapsed = time.time() - total_start
    logger.info("\n" + "="*70)
    logger.info(f"✅ TOTAL RUNTIME: {total_elapsed:.1f} seconds ({total_elapsed/60:.1f} minutes)")
    logger.info("="*70)
