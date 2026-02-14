from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from Models.common.report_generator import (
    ReportGenerator,
    compute_capm_metrics,
    compute_rolling_beta_series,
    compute_yearly_rolling_beta_metrics,
)


class _LoaderStub:
    def load_xautry_prices(self, _path: Path) -> pd.Series:
        dates = pd.bdate_range("2025-01-02", periods=10)
        return pd.Series(np.linspace(2000, 2010, len(dates)), index=dates)


def test_compute_capm_metrics_recovers_beta() -> None:
    dates = pd.bdate_range("2025-01-02", periods=300)
    market = pd.Series(np.linspace(-0.01, 0.02, len(dates)), index=dates)
    strategy = 0.0005 + 1.5 * market

    metrics = compute_capm_metrics(strategy, market)

    assert metrics["n_obs"] == len(dates)
    assert np.isclose(metrics["beta"], 1.5, atol=1e-3)
    assert metrics["r_squared"] > 0.999


def test_compute_capm_metrics_handles_small_samples() -> None:
    dates = pd.bdate_range("2025-01-02", periods=20)
    market = pd.Series(np.random.default_rng(42).normal(0, 0.01, len(dates)), index=dates)
    strategy = market * 0.8

    metrics = compute_capm_metrics(strategy, market)
    assert np.isnan(metrics["beta"])
    assert metrics["n_obs"] == 20


def test_rolling_beta_and_yearly_rollup_contract() -> None:
    dates = pd.bdate_range("2024-01-02", periods=320)
    market = pd.Series(np.random.default_rng(7).normal(0, 0.01, len(dates)), index=dates)
    strategy = 0.0002 + 1.2 * market + np.random.default_rng(8).normal(0, 0.002, len(dates))

    beta = compute_rolling_beta_series(strategy, market, window=63, min_periods=30)
    yearly = compute_yearly_rolling_beta_metrics(beta)

    assert beta.name == "Rolling_Beta"
    assert isinstance(yearly, pd.DataFrame)
    assert {"Year", "Beta_Mean", "Beta_Change"}.issubset(yearly.columns)


def test_report_generator_save_results_writes_expected_outputs(tmp_path: Path) -> None:
    dates = pd.bdate_range("2025-01-02", periods=120)
    returns = pd.Series(np.random.default_rng(1).normal(0.0008, 0.01, len(dates)), index=dates)
    equity = (1 + returns).cumprod()
    xautry_returns = pd.Series(np.random.default_rng(2).normal(0.0004, 0.004, len(dates)), index=dates)
    xu100_prices = pd.Series(10_000 + np.arange(len(dates)) * 20.0, index=dates)

    results = {
        "returns": returns,
        "equity": equity,
        "xautry_returns": xautry_returns,
        "total_return": float(equity.iloc[-1] - 1.0),
        "cagr": 0.12,
        "max_drawdown": -0.18,
        "sharpe": 1.3,
        "sortino": 2.1,
        "win_rate": 0.56,
        "rebalance_count": 10,
        "trade_count": 42,
        "regime_performance": {
            "Bull": {"count": 60, "mean_return": 0.15, "total_return": 0.20, "win_rate": 0.6},
            "Bear": {"count": 60, "mean_return": 0.05, "total_return": 0.08, "win_rate": 0.52},
        },
        "returns_df": pd.DataFrame(
            {
                "return": returns,
                "xautry_return": xautry_returns,
                "regime": ["Bull"] * len(dates),
                "n_stocks": [2] * len(dates),
                "allocation": [1.0] * len(dates),
            }
        ),
        "holdings_history": [
            {"date": dates[0], "ticker": "AAA", "weight": 0.5, "regime": "Bull", "allocation": 1.0},
            {"date": dates[0], "ticker": "BBB", "weight": 0.5, "regime": "Bull", "allocation": 1.0},
        ],
        "sanity_checks": pd.DataFrame(
            {
                "regime": ["Bull"],
                "allocation": [1.0],
                "is_rebalance_day": [True],
                "signal_count": [3],
                "n_active_holdings": [2],
                "weight_sum_raw": [1.0],
                "effective_weight_sum": [1.0],
                "rebalance_turnover": [0.5],
                "portfolio_return": [0.01],
            },
            index=[dates[0]],
        ),
    }

    factor_capm_store: dict[str, dict] = {}
    yearly_roll_beta_store: dict[str, pd.DataFrame] = {}
    out_dir = tmp_path / "results" / "demo_factor"

    generator = ReportGenerator(models_dir=tmp_path, data_dir=tmp_path, loader=_LoaderStub())
    generator.save_results(
        results=results,
        factor_name="demo_factor",
        xu100_prices=xu100_prices,
        xautry_prices=None,
        factor_capm_store=factor_capm_store,
        factor_yearly_rolling_beta_store=yearly_roll_beta_store,
        output_dir=out_dir,
    )

    assert (out_dir / "summary.txt").exists()
    assert (out_dir / "equity_curve.csv").exists()
    assert (out_dir / "returns.csv").exists()
    assert (out_dir / "yearly_metrics.csv").exists()
    assert "demo_factor" in factor_capm_store
    assert "demo_factor" in yearly_roll_beta_store
