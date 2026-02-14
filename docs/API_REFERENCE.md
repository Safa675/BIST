# API Reference

## Core Engine

### `Models.common.backtester`

#### `Backtester`
Main backtesting orchestrator.

- `update_data(prices, close_df, volume_df, regime_series, regime_allocations, xu100_prices, xautry_prices=None)`
  - Updates runtime data panels and resets cached market-cap panel.
- `run(signals, factor_name, rebalance_freq='quarterly', start_date=None, end_date=None, portfolio_options=None) -> dict`
  - Runs strategy simulation and returns backward-compatible payload (`returns`, `equity`, `returns_df`, `holdings_history`, `sanity_checks`, etc.).

#### Helpers
- `identify_monthly_rebalance_days(trading_days)`
- `identify_quarterly_rebalance_days(trading_days)`

### `Models.common.backtest_services`

Service decomposition used by `Backtester`:

- `DataPreparationService`
  - Aligns prices/signals/regime and enforces lagged execution.
- `RebalancingSelectionService`
  - Rebalance-day stock selection and turnover updates.
- `DailyReturnService`
  - Vectorized weighted daily return computation.
- `HoldingsHistoryAggregator`
  - Buffers holdings into stable output rows.
- `TransactionCostModel`
  - Applies slippage costs (flat or market-cap bucketed).
- `BacktestMetricsService`
  - Computes return/equity/sharpe/sortino/drawdown/win-rate.
- `BacktestPayloadAssembler`
  - Builds output payload contract.

### `Models.common.risk_manager`

#### `RiskManager`
Risk control and sizing utilities.

- `resolve_options(portfolio_options)`
- `filter_by_liquidity(tickers, date, liquidity_quantile)`
- `inverse_downside_vol_weights(selected, date, lookback, max_weight)`
- `apply_downside_vol_targeting(returns, target_vol, lookback, vol_floor, vol_cap)`
- `apply_stop_loss(current_holdings, stopped_out, entry_prices, open_df, date, stop_loss_threshold)`
- `slippage_cost_bps(...)`

### `Models.common.report_generator`

Reporting and analytics.

- `compute_yearly_metrics(returns, benchmark_returns=None, xautry_returns=None)`
- `compute_capm_metrics(strategy_returns, market_returns, risk_free_daily=0.0)`
- `compute_rolling_beta_series(strategy_returns, market_returns, window=252, min_periods=126, risk_free_daily=0.0)`
- `compute_yearly_rolling_beta_metrics(rolling_beta)`

#### `ReportGenerator`
- `save_results(results, factor_name, xu100_prices, xautry_prices, factor_capm_store, factor_yearly_rolling_beta_store, output_dir=None)`
- `save_correlation_matrix(factor_returns, xautry_prices, output_dir=None)`
- `save_capm_summary(factor_capm, output_dir=None, models_dir=None)`
- `save_yearly_rolling_beta_summary(factor_yearly_rolling_beta, output_dir=None, models_dir=None)`

## Signal System

### `Models.signals.factory`

- `get_available_signals() -> list[str]`
- `build_signal(name, dates, loader, config) -> pd.DataFrame`

`build_signal` validates input types, resolves `parameters` + `signal_params`, dispatches to registered builder, and enforces DataFrame output contract.

### `Models.signals.five_factor_rotation_signals`

Main entrypoint:

- `build_five_factor_rotation_signals(close_df, dates, data_loader=None, fundamentals=None, volume_df=None, use_construction_cache=True, force_rebuild_construction_cache=False, construction_cache_path=None, mwu_walkforward_config=None, axis_orthogonalization_config=None, return_details=False, debug=False, include_debug_artifacts=False)`

Supporting services:

- `FiveFactorDataPreparationService`
- `AxisCacheManager`
- `FactorConstructionPipeline`
- `OrthogonalizationService`
- `AxisComponentService`
- `MWUService`

## Data Reliability Pipeline

### `Models.data_pipeline.pipeline`

#### `FundamentalsPipeline`
Unified fundamentals refresh pipeline.

- `fetch_data(...) -> RawDataBundle`
- `validate_schema(data) -> ValidatedDataBundle`
- `normalize_data(data) -> NormalizedDataBundle`
- `merge_data(data, force_merge=False) -> MergedDataBundle`
- `validate_freshness(data) -> bool`
- `save_data(normalized, merged) -> dict[str, Path]`
- `run(...) -> PipelineRunResult`
- `run_diagnostics() -> PipelineRunResult`

Helpers:

- `build_default_paths(base_dir=None) -> PipelinePaths`
- `build_default_config(...) -> PipelineConfig`
- `compute_default_periods(count=5, as_of=None)`

### `Models.data_pipeline.schemas`
- `validate_raw_payload_structure`
- `validate_flat_normalized`
- `validate_consolidated_panel`
- `validate_staleness_report`

### `Models.data_pipeline.freshness`
- `compute_staleness_report`
- `summarize_quality_metrics`
- `evaluate_freshness`
- `enforce_freshness_gate`

### `Models.data_pipeline.provenance`
- `dataframe_checksum_sha256`
- `file_checksum_sha256`
- `stable_json_sha256`
- `write_dataset_provenance`

## Config API

### `Models.common.config_manager`

- `ConfigManager.from_default_paths()`
- `ConfigManager.load_signal_configs(prefer_yaml=True)`
- `load_signal_configs(prefer_yaml=True)`

Validation errors raise `ConfigError`.

## Benchmarking API

### `Models.common.benchmarking`

- `BenchmarkConfig`
- `run_backtester_benchmark(days, tickers, top_n, seed=42)`
- `run_pipeline_benchmark(raw_payload, workdir)`
- `run_benchmark_suite(raw_payload, config=None, tmp_root=None)`
- `compare_with_baseline(current, baseline, max_slowdown_pct=20.0, max_memory_regression_pct=20.0)`
- `save_benchmark_report(report, path)`
- `load_benchmark_report(path)`

### `Models.common.benchmarking_cli`

- `main()`
  - CLI entrypoint for integrated performance/memory benchmarking and regression checks against a baseline JSON report.
