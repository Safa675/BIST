from __future__ import annotations

import numpy as np
import pandas as pd

import Models.portfolio_engine as legacy_engine
from Models.common import report_generator as common_report
from Models.common.config_manager import load_signal_configs as load_configs_modern
from Models.common.report_generator import compute_capm_metrics, compute_yearly_metrics


def test_report_metric_shims_match_common_implementations() -> None:
    dates = pd.bdate_range("2024-01-02", periods=280)
    strategy = pd.Series(np.random.default_rng(1).normal(0.0007, 0.01, len(dates)), index=dates)
    market = pd.Series(np.random.default_rng(2).normal(0.0005, 0.009, len(dates)), index=dates)
    xautry = pd.Series(np.random.default_rng(3).normal(0.0003, 0.004, len(dates)), index=dates)

    yearly_direct = compute_yearly_metrics(strategy, market, xautry)
    yearly_module = common_report.compute_yearly_metrics(strategy, market, xautry)
    pd.testing.assert_frame_equal(yearly_direct.reset_index(drop=True), yearly_module.reset_index(drop=True))

    capm_direct = compute_capm_metrics(strategy, market)
    capm_module = common_report.compute_capm_metrics(strategy, market)
    assert capm_direct.keys() == capm_module.keys()
    for key in capm_direct:
        a = capm_direct[key]
        b = capm_module[key]
        if isinstance(a, float):
            assert np.isclose(float(a), float(b), equal_nan=True)
        else:
            assert a == b


def test_signal_config_loader_shim_matches_config_manager() -> None:
    legacy = legacy_engine.load_signal_configs()
    modern = load_configs_modern()

    assert set(legacy.keys()) == set(modern.keys())
    assert legacy
