from __future__ import annotations

import numpy as np
import pandas as pd

from Models.common.backtester import (
    identify_monthly_rebalance_days,
    identify_quarterly_rebalance_days,
)


def _base_options(**overrides: object) -> dict:
    opts = {
        "use_regime_filter": False,
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
        "debug": False,
    }
    opts.update(overrides)
    return opts


def test_identify_rebalance_days_contract() -> None:
    trading_days = pd.bdate_range("2025-01-01", "2025-04-30")

    monthly = identify_monthly_rebalance_days(trading_days)
    expected_monthly = {
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2025-02-03"),
        pd.Timestamp("2025-03-03"),
        pd.Timestamp("2025-04-01"),
    }
    assert expected_monthly.issubset(monthly)

    quarterly = identify_quarterly_rebalance_days(trading_days)
    # Targets are Mar/May/Aug/Nov 15th, advanced to first trading day >= target.
    assert pd.Timestamp("2025-03-17") in quarterly


def test_backtester_run_respects_signal_lag(backtester, signals_df: pd.DataFrame) -> None:
    lagged = backtester.run(
        signals=signals_df,
        factor_name="momentum",
        rebalance_freq="monthly",
        portfolio_options=_base_options(signal_lag_days=1),
    )
    no_lag = backtester.run(
        signals=signals_df,
        factor_name="momentum",
        rebalance_freq="monthly",
        portfolio_options=_base_options(signal_lag_days=0),
    )

    lagged_sanity = lagged["sanity_checks"]
    no_lag_sanity = no_lag["sanity_checks"]

    first_day = lagged_sanity.index.min()
    assert lagged_sanity.loc[first_day, "signal_count"] == 0
    assert lagged_sanity.loc[first_day, "n_active_holdings"] == 0

    assert no_lag_sanity.loc[first_day, "signal_count"] > 0
    assert no_lag_sanity.loc[first_day, "n_active_holdings"] > 0


def test_backtester_regime_blending_uses_gold_leg(backtester, signals_df: pd.DataFrame) -> None:
    result = backtester.run(
        signals=signals_df,
        factor_name="momentum",
        rebalance_freq="monthly",
        portfolio_options=_base_options(signal_lag_days=1, use_regime_filter=True),
    )

    returns_df = result["returns_df"]
    zero_alloc = returns_df[returns_df["allocation"] == 0.0]
    assert not zero_alloc.empty
    assert np.allclose(
        zero_alloc["return"].to_numpy(),
        zero_alloc["xautry_return"].to_numpy(),
        atol=1e-12,
        equal_nan=True,
    )


def test_backtester_sanity_weight_invariants(backtester, signals_df: pd.DataFrame) -> None:
    result = backtester.run(
        signals=signals_df,
        factor_name="momentum",
        rebalance_freq="monthly",
        portfolio_options=_base_options(signal_lag_days=1),
    )

    sanity = result["sanity_checks"]
    invested = sanity[(sanity["allocation"] > 0) & (sanity["n_active_holdings"] > 0)]
    assert not invested.empty
    assert invested["weight_sum_raw"].sub(1.0).abs().max() <= 1e-6
