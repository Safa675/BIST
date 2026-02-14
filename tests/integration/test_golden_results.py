"""Golden-file regression tests for backtest results.

These tests ensure that backtest outputs remain deterministic across code changes.
They use fixed synthetic data and compare results against known-good values.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from Models.common.backtester import Backtester
from Models.common.risk_manager import RiskManager

SEED = 12345
FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures"
GOLDEN_RESULTS_FILE = FIXTURE_DIR / "golden_results.json"


def _create_golden_data() -> dict:
    """Create fixed synthetic data for golden regression tests."""
    rng = np.random.default_rng(SEED)

    dates = pd.bdate_range("2024-01-02", periods=120)
    tickers = ["GOLD01", "GOLD02", "GOLD03", "GOLD04", "GOLD05"]

    base_prices = np.array([100.0, 80.0, 120.0, 90.0, 110.0])
    daily_rets = rng.normal(loc=0.0005, scale=0.015, size=(len(dates), len(tickers)))

    close_df = pd.DataFrame(index=dates, columns=tickers, dtype=float)
    for idx, ticker in enumerate(tickers):
        close_df[ticker] = base_prices[idx] * np.cumprod(1.0 + daily_rets[:, idx])

    open_df = close_df.shift(1).fillna(close_df.iloc[0] * 0.999)
    volume_df = pd.DataFrame(
        {ticker: 1_000_000.0 + 50_000.0 * idx for idx, ticker in enumerate(tickers)},
        index=dates,
        dtype=float,
    )

    prices_records: list[dict] = []
    for date in dates:
        for ticker in tickers:
            prices_records.append(
                {
                    "Date": date,
                    "Ticker": f"{ticker}.IS",
                    "Open": float(open_df.loc[date, ticker]),
                    "Close": float(close_df.loc[date, ticker]),
                    "Volume": float(volume_df.loc[date, ticker]),
                }
            )
    prices_long = pd.DataFrame(prices_records)

    regime_series = pd.Series("Bull", index=dates)
    xu100_prices = close_df.mean(axis=1).rename("XU100")
    xautry_prices = pd.Series(2000.0 + np.arange(len(dates)) * 2.0, index=dates, name="XAUTRY")

    return {
        "dates": dates,
        "tickers": tickers,
        "close_df": close_df,
        "open_df": open_df,
        "volume_df": volume_df,
        "prices_long": prices_long,
        "regime_series": regime_series,
        "xu100_prices": xu100_prices,
        "xautry_prices": xautry_prices,
    }


def _create_momentum_signals(close_df: pd.DataFrame, lookback: int = 20, skip: int = 5) -> pd.DataFrame:
    """Create simple momentum signals (price momentum over lookback days, skipping recent skip days)."""
    returns = close_df.pct_change(lookback).shift(skip)
    return returns.rank(axis=1, pct=True) * 100.0


def _create_value_signals(close_df: pd.DataFrame) -> pd.DataFrame:
    """Create simple value signals (inverse of price level as a proxy)."""
    inverse_price = 1.0 / close_df
    return inverse_price.rank(axis=1, pct=True) * 100.0


def _backtest_options() -> dict:
    """Standard backtest options for golden tests."""
    return {
        "use_regime_filter": False,
        "use_vol_targeting": False,
        "use_inverse_vol_sizing": False,
        "use_stop_loss": False,
        "use_liquidity_filter": False,
        "use_slippage": False,
        "use_mcap_slippage": False,
        "top_n": 2,
        "signal_lag_days": 0,
        "slippage_bps": 0.0,
        "stop_loss_threshold": 0.15,
        "liquidity_quantile": 0.25,
        "inverse_vol_lookback": 20,
        "max_position_weight": 0.5,
        "target_downside_vol": 0.2,
        "vol_lookback": 20,
        "vol_floor": 0.1,
        "vol_cap": 1.0,
        "debug": False,
    }


class DummyLoaderGolden:
    def __init__(self, xautry_prices: pd.Series, data_dir: Path) -> None:
        self._xautry_prices = xautry_prices
        self.data_dir = data_dir

    def load_xautry_prices(
        self,
        _path: Path,
        start_date: pd.Timestamp | None = None,
        end_date: pd.Timestamp | None = None,
    ) -> pd.Series:
        series = self._xautry_prices.copy()
        if start_date is not None:
            series = series[series.index >= pd.Timestamp(start_date)]
        if end_date is not None:
            series = series[series.index <= pd.Timestamp(end_date)]
        return series


def _build_market_cap_panel_stub(close_df: pd.DataFrame, dates: pd.DatetimeIndex, _loader) -> pd.DataFrame:
    base = pd.DataFrame(index=dates, columns=close_df.columns, dtype=float)
    for idx, ticker in enumerate(close_df.columns):
        base[ticker] = 1_000_000_000.0 + idx * 200_000_000.0
    return base


@pytest.fixture
def golden_data(tmp_path: Path) -> dict:
    """Fixture providing golden test data."""
    data = _create_golden_data()
    data["tmp_path"] = tmp_path
    return data


@pytest.fixture
def golden_backtester(golden_data: dict) -> Backtester:
    """Fixture providing a backtester configured with golden data."""
    data_dir = golden_data["tmp_path"] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    loader = DummyLoaderGolden(
        xautry_prices=golden_data["xautry_prices"],
        data_dir=data_dir,
    )

    risk_manager = RiskManager(
        close_df=golden_data["close_df"],
        volume_df=golden_data["volume_df"],
    )

    bt = Backtester(
        loader=loader,
        data_dir=data_dir,
        risk_manager=risk_manager,
        build_size_market_cap_panel=_build_market_cap_panel_stub,
    )

    bt.update_data(
        prices=golden_data["prices_long"],
        close_df=golden_data["close_df"],
        volume_df=golden_data["volume_df"],
        regime_series=golden_data["regime_series"],
        regime_allocations={"Bull": 1.0, "Bear": 0.0, "Recovery": 0.5, "Stress": 0.0},
        xu100_prices=golden_data["xu100_prices"],
        xautry_prices=golden_data["xautry_prices"],
    )

    return bt


def _load_golden_results() -> dict:
    """Load expected golden results from fixture file."""
    if not GOLDEN_RESULTS_FILE.exists():
        return {}
    return json.loads(GOLDEN_RESULTS_FILE.read_text(encoding="utf-8"))


def _save_golden_results(results: dict) -> None:
    """Save golden results to fixture file."""
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    GOLDEN_RESULTS_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")


def test_golden_momentum_backtest(golden_data: dict, golden_backtester: Backtester) -> None:
    """Test momentum strategy produces reproducible results."""
    signals = _create_momentum_signals(golden_data["close_df"])

    result = golden_backtester.run(
        signals=signals,
        factor_name="golden_momentum",
        rebalance_freq="monthly",
        portfolio_options=_backtest_options(),
    )

    cagr = float(result["cagr"])
    sharpe = float(result["sharpe"])
    max_dd = float(result["max_drawdown"])
    final_equity = float(result["equity"].iloc[-1])

    golden = _load_golden_results()
    expected = golden.get("momentum", {})

    if not expected:
        pytest.skip(
            "No golden baseline for momentum strategy. "
            "Run with --update-golden to create baseline."
        )

    assert np.isclose(cagr, expected["cagr"], rtol=1e-6), f"CAGR mismatch: {cagr} vs {expected['cagr']}"
    assert np.isclose(sharpe, expected["sharpe"], rtol=1e-4), f"Sharpe mismatch: {sharpe} vs {expected['sharpe']}"
    assert np.isclose(max_dd, expected["max_drawdown"], rtol=1e-6), f"Max DD mismatch: {max_dd} vs {expected['max_drawdown']}"
    assert np.isclose(final_equity, expected["final_equity"], rtol=1e-6), f"Equity mismatch: {final_equity} vs {expected['final_equity']}"


def test_golden_value_backtest(golden_data: dict, golden_backtester: Backtester) -> None:
    """Test value strategy produces reproducible results."""
    signals = _create_value_signals(golden_data["close_df"])

    result = golden_backtester.run(
        signals=signals,
        factor_name="golden_value",
        rebalance_freq="monthly",
        portfolio_options=_backtest_options(),
    )

    cagr = float(result["cagr"])
    sharpe = float(result["sharpe"])
    max_dd = float(result["max_drawdown"])
    final_equity = float(result["equity"].iloc[-1])

    golden = _load_golden_results()
    expected = golden.get("value", {})

    if not expected:
        pytest.skip(
            "No golden baseline for value strategy. "
            "Run with --update-golden to create baseline."
        )

    assert np.isclose(cagr, expected["cagr"], rtol=1e-6), f"CAGR mismatch: {cagr} vs {expected['cagr']}"
    assert np.isclose(sharpe, expected["sharpe"], rtol=1e-4), f"Sharpe mismatch: {sharpe} vs {expected['sharpe']}"
    assert np.isclose(max_dd, expected["max_drawdown"], rtol=1e-6), f"Max DD mismatch: {max_dd} vs {expected['max_drawdown']}"
    assert np.isclose(final_equity, expected["final_equity"], rtol=1e-6), f"Equity mismatch: {final_equity} vs {expected['final_equity']}"


def test_golden_backtest_determinism(golden_data: dict, golden_backtester: Backtester) -> None:
    """Test that running the same backtest twice produces identical results."""
    signals = _create_momentum_signals(golden_data["close_df"])
    options = _backtest_options()

    result1 = golden_backtester.run(
        signals=signals,
        factor_name="momentum_run1",
        rebalance_freq="monthly",
        portfolio_options=options,
    )

    result2 = golden_backtester.run(
        signals=signals,
        factor_name="momentum_run2",
        rebalance_freq="monthly",
        portfolio_options=options,
    )

    assert result1["cagr"] == result2["cagr"]
    assert result1["sharpe"] == result2["sharpe"]
    assert result1["max_drawdown"] == result2["max_drawdown"]
    pd.testing.assert_series_equal(result1["equity"], result2["equity"])
    pd.testing.assert_series_equal(result1["returns"], result2["returns"])


def test_update_golden_baseline(golden_data: dict, golden_backtester: Backtester, request) -> None:
    """Update golden baseline files when --update-golden is passed."""
    if not request.config.getoption("--update-golden", default=False):
        pytest.skip("Pass --update-golden to update baseline files")

    momentum_signals = _create_momentum_signals(golden_data["close_df"])
    value_signals = _create_value_signals(golden_data["close_df"])
    options = _backtest_options()

    momentum_result = golden_backtester.run(
        signals=momentum_signals,
        factor_name="golden_momentum",
        rebalance_freq="monthly",
        portfolio_options=options,
    )

    value_result = golden_backtester.run(
        signals=value_signals,
        factor_name="golden_value",
        rebalance_freq="monthly",
        portfolio_options=options,
    )

    golden_results = {
        "momentum": {
            "cagr": float(momentum_result["cagr"]),
            "sharpe": float(momentum_result["sharpe"]),
            "max_drawdown": float(momentum_result["max_drawdown"]),
            "final_equity": float(momentum_result["equity"].iloc[-1]),
        },
        "value": {
            "cagr": float(value_result["cagr"]),
            "sharpe": float(value_result["sharpe"]),
            "max_drawdown": float(value_result["max_drawdown"]),
            "final_equity": float(value_result["equity"].iloc[-1]),
        },
    }

    _save_golden_results(golden_results)
    print(f"\nUpdated golden results: {GOLDEN_RESULTS_FILE}")
    print(json.dumps(golden_results, indent=2))
