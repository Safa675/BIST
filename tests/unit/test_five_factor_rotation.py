from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import Models.signals.five_factor_rotation_signals as ff
from Models.common.utils import SignalDataError


def test_mwu_combine_axis_components_weighted_average_contract() -> None:
    dates = pd.bdate_range("2025-01-02", periods=4)
    tickers = pd.Index(["A", "B", "C", "D", "E"])

    axis_components = {
        "size": pd.DataFrame(10.0, index=dates, columns=tickers),
        "value": pd.DataFrame(50.0, index=dates, columns=tickers),
    }
    axis_weights = pd.DataFrame(
        {
            "size": [0.25, 0.25, 0.25, 0.25],
            "value": [0.75, 0.75, 0.75, 0.75],
        },
        index=dates,
    )

    bundle = ff.MWUService.combine_axis_components(
        axis_components=axis_components,
        axis_weights=axis_weights,
        signal_dates=dates,
        tickers=tickers,
    )
    final_scores = bundle["final_scores"]

    assert final_scores.shape == (len(dates), len(tickers))
    assert np.isclose(float(final_scores.iloc[0, 0]), 40.0)


def test_data_preparation_requires_loader_for_fundamentals(close_df: pd.DataFrame, business_dates: pd.DatetimeIndex) -> None:
    with pytest.raises(SignalDataError, match="no data_loader provided"):
        ff.FiveFactorDataPreparationService().prepare(
            close_df=close_df,
            dates=business_dates,
            data_loader=None,
            fundamentals=None,
            volume_df=None,
        )


def test_build_five_factor_rotation_orchestrates_services(monkeypatch: pytest.MonkeyPatch) -> None:
    dates = pd.bdate_range("2025-01-02", periods=6)
    tickers = pd.Index(["A", "B", "C", "D", "E", "F"])
    close_df = pd.DataFrame(
        np.linspace(100.0, 110.0, len(dates))[:, None] + np.arange(len(tickers))[None, :],
        index=dates,
        columns=tickers,
    )

    context = ff.FiveFactorBuildContext(
        close=close_df,
        full_dates=dates,
        signal_dates=dates,
        tickers=tickers,
        daily_returns=close_df.pct_change(),
        fundamentals={"dummy": {"path": None}},
        volume_df=pd.DataFrame(1_000_000.0, index=dates, columns=tickers),
        metrics_df=pd.DataFrame(),
    )

    class _Prep:
        def prepare(self, **_kwargs):
            return context

    class _CacheManager:
        def __init__(self, *args, **kwargs):
            pass

        def load_or_build(self, **_kwargs):
            return {}

    class _FactorPipeline:
        def __init__(self, debug: bool = False):
            self.debug = debug

        def build_axis_specs(self, **_kwargs):
            raw = pd.DataFrame(
                np.linspace(0.0, 100.0, len(dates) * len(tickers)).reshape(len(dates), len(tickers)),
                index=dates,
                columns=tickers,
            )
            return {"size": (raw, "High", "Low")}

    class _Ortho:
        def __init__(self, debug: bool = False):
            self.debug = debug

        def maybe_orthogonalize(self, axis_specs, axis_orthogonalization_config):
            return axis_specs, False, {}

    class _AxisComponent:
        def __init__(self, debug: bool = False):
            self.debug = debug

        def build(self, axis_specs, daily_returns):
            component = axis_specs["size"][0]
            high_daily = pd.Series(0.001, index=dates)
            low_daily = pd.Series(0.0005, index=dates)
            bucket = pd.DataFrame({"Q1": [1.0] * len(dates)}, index=dates)
            return ff.AxisComponentsBundle(
                axis_components={"size": component},
                axis_summary={"size": ("high", bucket, "High", "Low")},
                axis_daily_returns={"size": (high_daily, low_daily)},
                axis_bucket_returns={"size": bucket},
            )

    class _MWU:
        @staticmethod
        def compute_axis_weights(*_args, **_kwargs):
            return pd.DataFrame({"size": [1.0] * len(dates)}, index=dates)

        @staticmethod
        def combine_axis_components(axis_components, axis_weights, signal_dates, tickers):
            # Intentionally include out-of-range values to verify clipping.
            raw = pd.DataFrame(
                [[120.0, -10.0, 30.0, 40.0, 60.0, 80.0] for _ in signal_dates],
                index=signal_dates,
                columns=tickers,
            )
            return {
                "aligned_components": axis_components,
                "final_scores": raw,
            }

    monkeypatch.setattr(ff, "FiveFactorDataPreparationService", _Prep)
    monkeypatch.setattr(ff, "AxisCacheManager", _CacheManager)
    monkeypatch.setattr(ff, "FactorConstructionPipeline", _FactorPipeline)
    monkeypatch.setattr(ff, "OrthogonalizationService", _Ortho)
    monkeypatch.setattr(ff, "AxisComponentService", _AxisComponent)
    monkeypatch.setattr(ff, "MWUService", _MWU)
    monkeypatch.setattr(
        ff,
        "_build_yearly_axis_winner_report",
        lambda *args, **kwargs: pd.DataFrame({"Year": [2025], "Axis": ["size"], "Winner": ["High"]}),
    )

    scores, details = ff.build_five_factor_rotation_signals(
        close_df=close_df,
        dates=dates,
        data_loader=object(),
        fundamentals={"dummy": {"path": None}},
        volume_df=pd.DataFrame(1_000_000.0, index=dates, columns=tickers),
        return_details=True,
        include_debug_artifacts=True,
    )

    assert isinstance(scores, pd.DataFrame)
    assert float(scores.max().max()) <= 100.0
    assert float(scores.min().min()) >= 0.0
    assert "yearly_axis_winners" in details
    assert "axis_weights" in details
    assert "axis_components" in details
