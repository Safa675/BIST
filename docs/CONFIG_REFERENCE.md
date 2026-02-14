# Configuration Reference

## 1. Strategy Config (`configs/strategies.yaml`)

Top-level keys:

- `defaults`: base strategy options
- `strategies`: per-strategy overrides

Example:

```yaml
defaults:
  description: Base strategy
  timeline:
    start_date: "2018-01-01"
    end_date: "2024-12-31"
  rebalance_frequency: quarterly
  portfolio_options:
    top_n: 20
    signal_lag_days: 1
strategies:
  momentum:
    description: Price momentum
    signal_params:
      lookback_days: 252
```

Validation:

- `description` is required and must be non-empty.
- `timeline`, `parameters`, `signal_params`, `portfolio_options` must be mappings if present.

## 2. Portfolio Options

Common options:

- `use_regime_filter` (bool)
- `use_vol_targeting` (bool)
- `target_downside_vol` (float)
- `vol_lookback` (int)
- `vol_floor` / `vol_cap` (float)
- `use_inverse_vol_sizing` (bool)
- `inverse_vol_lookback` (int)
- `max_position_weight` (float)
- `use_stop_loss` (bool)
- `stop_loss_threshold` (float)
- `use_liquidity_filter` (bool)
- `liquidity_quantile` (float)
- `use_slippage` (bool)
- `use_mcap_slippage` (bool)
- `slippage_bps` (float)
- `small_cap_slippage_bps` / `mid_cap_slippage_bps` (float)
- `top_n` (int)
- `signal_lag_days` (int)

Defaults are defined in `Models/common/config_manager.py`.

## 3. Freshness Gate Environment Variables

- `BIST_ENFORCE_FUNDAMENTAL_FRESHNESS=1|0`
- `BIST_ALLOW_STALE_FUNDAMENTALS=1|0`
- `BIST_MAX_MEDIAN_STALENESS_DAYS`
- `BIST_MAX_PCT_OVER_120_DAYS`
- `BIST_MIN_Q4_2025_COVERAGE_PCT`
- `BIST_MAX_MAX_STALENESS_DAYS`
- `BIST_STALENESS_GRACE_DAYS`

## 4. Backtest CLI (`run_backtest.py`)

Primary usage:

```bash
python run_backtest.py <signal|all> --start-date YYYY-MM-DD --end-date YYYY-MM-DD
```

Useful flags:

- `--list-signals`
- `--data-dir`
- `--regime-outputs`
- `--use-config-timeline`

## 5. Benchmark CLI

```bash
python -m Models.common.benchmarking_cli --repeats 3 --warmup 1
```

Key flags:

- `--output`
- `--baseline`
- `--write-baseline`
- `--max-slowdown-pct`
- `--max-memory-regression-pct`

