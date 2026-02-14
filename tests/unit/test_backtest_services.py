from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from Models.common.backtest_services import (
    BacktestMetricsService,
    DailyReturnService,
    DataPreparationService,
    HoldingsHistoryAggregator,
    TransactionCostModel,
)


class _RiskStub:
    def __init__(self, avg_bps: float = 12.0) -> None:
        self._avg_bps = avg_bps

    def slippage_cost_bps(self, **_kwargs) -> float:
        return float(self._avg_bps)

    def apply_downside_vol_targeting(self, returns: pd.Series, **_kwargs) -> pd.Series:
        return returns


class _LoaderStub:
    def __init__(self, xautry_prices: pd.Series) -> None:
        self._xautry_prices = xautry_prices
        self.data_dir = pd.Timestamp("2025-01-01")  # not used in this test path

    def load_xautry_prices(self, _path, start_date=None, end_date=None) -> pd.Series:
        s = self._xautry_prices.copy()
        if start_date is not None:
            s = s[s.index >= pd.Timestamp(start_date)]
        if end_date is not None:
            s = s[s.index <= pd.Timestamp(end_date)]
        return s


def test_data_preparation_applies_signal_lag_and_split_neutralization() -> None:
    dates = pd.bdate_range("2025-01-02", periods=15)
    prices = pd.DataFrame(
        {
            "Date": np.repeat(dates, 2),
            "Ticker": [ticker for _ in dates for ticker in ("AAA.IS", "BBB.IS")],
            "Open": [100.0 + i for i in range(len(dates)) for _ in (0, 1)],
        }
    )
    # Inject a split-like jump (>100%) in AAA open forward return.
    prices.loc[(prices["Date"] == dates[8]) & (prices["Ticker"] == "AAA.IS"), "Open"] = 350.0

    signals = pd.DataFrame(
        {
            "AAA": np.linspace(10.0, 20.0, len(dates)),
            "BBB": np.linspace(20.0, 10.0, len(dates)),
        },
        index=dates,
    )
    regime = pd.Series(["Bull"] * len(dates), index=dates)
    xautry = pd.Series(np.linspace(2_000.0, 2_030.0, len(dates)), index=dates)

    prep = DataPreparationService(
        loader=_LoaderStub(xautry_prices=xautry),
        data_dir=Path("."),
        prices=prices,
        regime_series=regime,
        xu100_prices=None,
        xautry_prices=xautry,
    ).prepare(
        signals=signals,
        factor_name="momentum",
        rebalance_freq="monthly",
        start_date=dates.min(),
        end_date=dates.max(),
        signal_lag_days=1,
    )

    assert prep.signals_exec.iloc[0].isna().all()
    assert prep.signals_exec.iloc[1].notna().all()
    assert float(prep.open_fwd_ret.abs().max().max()) <= 1.0


def test_daily_return_service_handles_missing_values() -> None:
    dates = pd.bdate_range("2025-01-02", periods=3)
    open_fwd_ret = pd.DataFrame(
        {
            "AAA": [0.01, np.nan, 0.02],
            "BBB": [0.03, 0.01, -0.01],
        },
        index=dates,
    )
    service = DailyReturnService(open_fwd_ret)
    weights = pd.Series({"AAA": 0.5, "BBB": 0.5})

    ret = service.compute_weighted_return(dates[1], ["AAA", "BBB"], weights)
    # AAA NaN -> treated as 0 by the vectorized engine.
    assert np.isclose(ret, 0.005)


def test_holdings_history_aggregator_falls_back_to_gold_when_uninvested() -> None:
    agg = HoldingsHistoryAggregator()
    date = pd.Timestamp("2025-01-03")

    agg.add(
        date=date,
        regime="Bear",
        allocation=0.0,
        active_holdings=[],
        weights=None,
    )

    records = agg.to_records()
    assert len(records) == 1
    assert records[0]["ticker"] == "XAU/TRY"
    assert np.isclose(records[0]["weight"], 1.0)


def test_transaction_cost_model_applies_expected_slippage_math() -> None:
    model = TransactionCostModel(_RiskStub(avg_bps=15.0))
    base_return = 0.02

    adjusted_flat = model.apply_rebalance_slippage(
        base_return,
        date=pd.Timestamp("2025-01-10"),
        is_rebalance_day=True,
        old_selected={"AAA"},
        active_holdings=["BBB"],
        rebalance_turnover=0.5,
        opts={"use_slippage": True, "slippage_bps": 5.0},
        slippage_factor=5.0 / 10000.0,
        use_mcap_slippage=False,
        mcap_slippage_panel=None,
        mcap_slippage_liquidity=None,
        small_cap_slippage_bps=20.0,
        mid_cap_slippage_bps=10.0,
    )
    expected_flat = base_return - 0.5 * (5.0 / 10000.0) * 2
    assert np.isclose(adjusted_flat, expected_flat)

    panel = pd.DataFrame({"AAA": [1.0], "BBB": [1.0]}, index=[pd.Timestamp("2025-01-10")])
    adjusted_mcap = model.apply_rebalance_slippage(
        base_return,
        date=pd.Timestamp("2025-01-10"),
        is_rebalance_day=True,
        old_selected={"AAA"},
        active_holdings=["BBB"],
        rebalance_turnover=0.5,
        opts={"use_slippage": True, "slippage_bps": 5.0},
        slippage_factor=5.0 / 10000.0,
        use_mcap_slippage=True,
        mcap_slippage_panel=panel,
        mcap_slippage_liquidity=panel,
        small_cap_slippage_bps=20.0,
        mid_cap_slippage_bps=10.0,
    )
    expected_mcap = base_return - 0.5 * (15.0 / 10000.0) * 2
    assert np.isclose(adjusted_mcap, expected_mcap)


def test_backtest_metrics_service_produces_contract() -> None:
    dates = pd.bdate_range("2025-01-02", periods=10)
    returns_df = pd.DataFrame(
        {
            "return": [0.01, -0.005, 0.004, 0.003, -0.002, 0.002, 0.001, 0.0, 0.003, -0.001],
            "regime": ["Bull", "Bull", "Bear", "Bear", "Recovery", "Recovery", "Bull", "Stress", "Bull", "Bear"],
        },
        index=dates,
    )

    service = BacktestMetricsService(_RiskStub())
    metrics = service.compute(
        returns_df=returns_df,
        opts={"use_vol_targeting": False},
        regime_allocations={"Bull": 1.0, "Bear": 0.0, "Recovery": 0.5, "Stress": 0.0},
    )

    assert isinstance(metrics.total_return, float)
    assert len(metrics.returns) == len(returns_df)
    assert len(metrics.equity) == len(returns_df)
    assert "Bull" in metrics.regime_performance
