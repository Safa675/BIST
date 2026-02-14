from __future__ import annotations

import numpy as np
import pandas as pd

from Models.common.risk_manager import RiskManager


def test_filter_by_liquidity_applies_quantile_cutoff() -> None:
    dates = pd.bdate_range("2025-01-02", periods=1)
    volume = pd.DataFrame(
        {"AAA": [3_000_000.0], "BBB": [2_000_000.0], "CCC": [1_000_000.0]},
        index=dates,
    )
    manager = RiskManager(close_df=None, volume_df=volume)

    filtered = manager.filter_by_liquidity(["AAA", "BBB", "CCC"], dates[0], liquidity_quantile=0.5)
    assert filtered == ["AAA", "BBB"]


def test_inverse_downside_vol_weights_sum_to_one_and_respect_cap() -> None:
    dates = pd.bdate_range("2025-01-02", periods=90)
    close = pd.DataFrame(
        {
            "AAA": 100 + np.cumsum(np.sin(np.arange(90) / 7.0) + 0.3),
            "BBB": 90 + np.cumsum(np.sin(np.arange(90) / 5.0) + 0.2),
            "CCC": 80 + np.cumsum(np.sin(np.arange(90) / 4.0) + 0.1),
        },
        index=dates,
    )

    manager = RiskManager(close_df=close, volume_df=None)
    weights = manager.inverse_downside_vol_weights(
        selected=["AAA", "BBB", "CCC"],
        date=dates[-1],
        lookback=60,
        max_weight=0.60,
    )

    assert np.isclose(weights.sum(), 1.0)
    assert (weights <= 0.60 + 1e-12).all()
    assert (weights >= 0.0).all()


def test_apply_stop_loss_removes_breached_positions() -> None:
    date = pd.Timestamp("2025-01-10")
    open_df = pd.DataFrame(
        {
            "AAA": [80.0],
            "BBB": [95.0],
        },
        index=[date],
    )

    kept = RiskManager.apply_stop_loss(
        current_holdings=["AAA", "BBB"],
        stopped_out=set(),
        entry_prices={"AAA": 100.0, "BBB": 100.0},
        open_df=open_df,
        date=date,
        stop_loss_threshold=0.15,
    )

    assert kept == ["BBB"]


def test_apply_downside_vol_targeting_preserves_shape_and_finiteness() -> None:
    dates = pd.bdate_range("2025-01-02", periods=120)
    returns = pd.Series(np.sin(np.arange(120) / 6.0) * 0.01, index=dates)

    targeted = RiskManager.apply_downside_vol_targeting(
        returns,
        target_vol=0.20,
        lookback=30,
        vol_floor=0.10,
        vol_cap=1.5,
    )

    assert len(targeted) == len(returns)
    assert targeted.index.equals(returns.index)
    assert np.isfinite(targeted.to_numpy()).all()
