from __future__ import annotations

import numpy as np
import pandas as pd


def _base_options(**overrides: object) -> dict:
    options = {
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
    options.update(overrides)
    return options


def test_backtester_all_nan_signals_keeps_equity_flat(backtester, business_dates: pd.DatetimeIndex, tickers: list[str]) -> None:
    nan_signals = pd.DataFrame(np.nan, index=business_dates, columns=tickers)

    result = backtester.run(
        signals=nan_signals,
        factor_name="momentum",
        rebalance_freq="monthly",
        portfolio_options=_base_options(use_regime_filter=False),
    )

    returns = result["returns"]
    equity = result["equity"]

    assert np.allclose(returns.to_numpy(dtype=float), 0.0, atol=1e-12)
    assert np.allclose(equity.to_numpy(dtype=float), 1.0, atol=1e-12)


def test_backtester_single_ticker_universe_runs(backtester, close_df: pd.DataFrame, volume_df: pd.DataFrame, prices_long: pd.DataFrame, regime_series: pd.Series, xu100_prices: pd.Series, xautry_prices: pd.Series, signals_df: pd.DataFrame) -> None:
    single_close = close_df[["AAA"]]
    single_volume = volume_df[["AAA"]]
    single_prices = prices_long[prices_long["Ticker"] == "AAA.IS"].copy()
    single_signals = signals_df[["AAA"]]

    backtester.update_data(
        prices=single_prices,
        close_df=single_close,
        volume_df=single_volume,
        regime_series=regime_series,
        regime_allocations={"Bull": 1.0, "Bear": 0.0, "Recovery": 0.5, "Stress": 0.0},
        xu100_prices=xu100_prices,
        xautry_prices=xautry_prices,
    )

    result = backtester.run(
        signals=single_signals,
        factor_name="momentum",
        rebalance_freq="monthly",
        portfolio_options=_base_options(top_n=1, signal_lag_days=0),
    )

    sanity = result["sanity_checks"]
    assert not sanity.empty
    assert int(sanity["n_active_holdings"].max()) <= 1
    assert np.isfinite(result["returns"].to_numpy(dtype=float)).all()


def test_backtester_stress_regime_zero_allocation_produces_near_zero_returns(backtester, prices_long: pd.DataFrame, close_df: pd.DataFrame, volume_df: pd.DataFrame, business_dates: pd.DatetimeIndex, signals_df: pd.DataFrame, xu100_prices: pd.Series) -> None:
    stress_regime = pd.Series("Stress", index=business_dates)
    flat_xautry = pd.Series(2_000.0, index=business_dates, name="XAU_TRY")

    backtester.update_data(
        prices=prices_long,
        close_df=close_df,
        volume_df=volume_df,
        regime_series=stress_regime,
        regime_allocations={"Stress": 0.0},
        xu100_prices=xu100_prices,
        xautry_prices=flat_xautry,
    )

    result = backtester.run(
        signals=signals_df,
        factor_name="momentum",
        rebalance_freq="monthly",
        portfolio_options=_base_options(use_regime_filter=True, signal_lag_days=0),
    )

    returns_df = result["returns_df"]
    assert (returns_df["allocation"] == 0.0).all()
    assert float(returns_df["return"].abs().max()) <= 1e-12


def test_backtester_very_short_date_range_still_runs(backtester, signals_df: pd.DataFrame, business_dates: pd.DatetimeIndex) -> None:
    start_date = business_dates[0]
    end_date = business_dates[4]

    result = backtester.run(
        signals=signals_df,
        factor_name="momentum",
        rebalance_freq="monthly",
        start_date=start_date,
        end_date=end_date,
        portfolio_options=_base_options(signal_lag_days=0),
    )

    assert len(result["returns_df"]) == 4
    assert len(result["equity"]) == 4
    assert np.isfinite(result["returns"].to_numpy(dtype=float)).all()


def test_backtester_stop_loss_trigger_all_holdings(backtester) -> None:
    dates = pd.bdate_range("2025-01-02", periods=12)
    tickers = ["AAA", "BBB", "CCC"]

    open_df = pd.DataFrame(
        {
            "AAA": [100.0, 70.0, 69.0, 69.5, 70.0, 69.8, 70.2, 70.1, 69.9, 70.0, 70.0, 70.0],
            "BBB": [100.0, 72.0, 71.0, 71.2, 71.5, 71.4, 71.3, 71.0, 71.1, 71.0, 71.0, 71.0],
            "CCC": [100.0, 85.0, 84.0, 84.5, 84.6, 84.7, 84.8, 84.7, 84.6, 84.6, 84.6, 84.6],
        },
        index=dates,
    )
    close_df = open_df * 1.001
    volume_df = pd.DataFrame(1_000_000.0, index=dates, columns=tickers)

    prices_records: list[dict[str, object]] = []
    for date in dates:
        for ticker in tickers:
            open_px = float(open_df.loc[date, ticker])
            close_px = float(close_df.loc[date, ticker])
            prices_records.append(
                {
                    "Date": date,
                    "Ticker": f"{ticker}.IS",
                    "Open": open_px,
                    "High": close_px * 1.01,
                    "Low": open_px * 0.99,
                    "Close": close_px,
                    "Volume": 1_000_000.0,
                }
            )

    prices = pd.DataFrame(prices_records)
    regime_series = pd.Series("Bull", index=dates)
    xu100_prices = close_df.mean(axis=1).rename("XU100")
    xautry_prices = pd.Series(2_000.0, index=dates, name="XAU_TRY")

    backtester.update_data(
        prices=prices,
        close_df=close_df,
        volume_df=volume_df,
        regime_series=regime_series,
        regime_allocations={"Bull": 1.0},
        xu100_prices=xu100_prices,
        xautry_prices=xautry_prices,
    )

    signals = pd.DataFrame(
        {
            "AAA": 95.0,
            "BBB": 90.0,
            "CCC": 80.0,
        },
        index=dates,
    )

    result = backtester.run(
        signals=signals,
        factor_name="momentum",
        rebalance_freq="monthly",
        portfolio_options=_base_options(
            top_n=2,
            signal_lag_days=0,
            use_stop_loss=True,
            stop_loss_threshold=0.10,
        ),
    )

    sanity = result["sanity_checks"]
    assert sanity.loc[dates[1], "n_active_holdings"] == 0
    assert np.isfinite(result["returns"].to_numpy(dtype=float)).all()


def test_backtester_empty_rebalance_days_keeps_equity_flat(backtester, signals_df: pd.DataFrame, business_dates: pd.DatetimeIndex) -> None:
    start_date = business_dates[0]
    end_date = business_dates[14]

    result = backtester.run(
        signals=signals_df,
        factor_name="momentum",
        rebalance_freq="quarterly",
        start_date=start_date,
        end_date=end_date,
        portfolio_options=_base_options(use_regime_filter=False, signal_lag_days=0),
    )

    assert result["rebalance_count"] == 0
    assert result["trade_count"] == 0
    assert np.allclose(result["equity"].to_numpy(dtype=float), 1.0, atol=1e-12)
