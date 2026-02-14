from __future__ import annotations

from pathlib import Path

import pandas as pd

from Models.common.report_generator import ReportGenerator


def _workflow_options() -> dict:
    return {
        "use_regime_filter": True,
        "use_vol_targeting": False,
        "use_inverse_vol_sizing": False,
        "use_stop_loss": False,
        "use_liquidity_filter": False,
        "use_slippage": False,
        "use_mcap_slippage": False,
        "top_n": 2,
        "signal_lag_days": 1,
        "slippage_bps": 5.0,
        "small_cap_slippage_bps": 20.0,
        "mid_cap_slippage_bps": 10.0,
        "stop_loss_threshold": 0.15,
        "liquidity_quantile": 0.25,
        "inverse_vol_lookback": 20,
        "max_position_weight": 0.5,
        "target_downside_vol": 0.2,
        "vol_lookback": 20,
        "vol_floor": 0.1,
        "vol_cap": 1.0,
    }


def test_backtest_workflow_payload_contract(backtester, signals_df: pd.DataFrame) -> None:
    result = backtester.run(
        signals=signals_df,
        factor_name="momentum",
        rebalance_freq="monthly",
        portfolio_options=_workflow_options(),
    )

    required_keys = {
        "returns",
        "equity",
        "total_return",
        "cagr",
        "sharpe",
        "sortino",
        "max_drawdown",
        "win_rate",
        "xautry_returns",
        "regime_performance",
        "returns_df",
        "holdings_history",
        "sanity_checks",
        "signal_lag_days",
        "trade_events",
    }
    assert required_keys.issubset(result.keys())
    assert len(result["returns"]) == len(result["returns_df"])
    assert result["signal_lag_days"] == 1
    assert not result["sanity_checks"].empty


def test_backtest_workflow_report_generation(
    tmp_path: Path,
    backtester,
    signals_df: pd.DataFrame,
    xu100_prices: pd.Series,
    dummy_loader,
) -> None:
    results = backtester.run(
        signals=signals_df,
        factor_name="momentum",
        rebalance_freq="monthly",
        portfolio_options=_workflow_options(),
    )

    out_dir = tmp_path / "results" / "momentum_test"
    capm_store: dict[str, dict] = {}
    yearly_beta_store: dict[str, pd.DataFrame] = {}

    generator = ReportGenerator(models_dir=tmp_path, data_dir=tmp_path, loader=dummy_loader)
    generator.save_results(
        results=results,
        factor_name="momentum_test",
        xu100_prices=xu100_prices,
        xautry_prices=None,
        factor_capm_store=capm_store,
        factor_yearly_rolling_beta_store=yearly_beta_store,
        output_dir=out_dir,
    )

    assert (out_dir / "summary.txt").exists()
    assert (out_dir / "equity_curve.csv").exists()
    assert (out_dir / "returns.csv").exists()
    assert (out_dir / "yearly_metrics.csv").exists()
    assert "momentum_test" in capm_store
    assert "momentum_test" in yearly_beta_store
