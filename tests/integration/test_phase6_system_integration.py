from __future__ import annotations

import importlib
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from Models.common.config_manager import ConfigManager
from Models.common.data_loader import DataLoader
from Models.common.report_generator import ReportGenerator
from Models.data_pipeline.pipeline import (
    FundamentalsPipeline,
    build_default_config,
    build_default_paths,
)
from Models.data_pipeline.types import RawDataBundle
from Models.signals import factory


def test_complete_config_signal_backtest_report_workflow(
    tmp_path: Path,
    monkeypatch,
    backtester,
    close_df: pd.DataFrame,
    xu100_prices: pd.Series,
    dummy_loader,
) -> None:
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "strategies.yaml").write_text(
        """
defaults:
  description: Base
  timeline:
    start_date: "2025-01-02"
    end_date: "2025-03-31"
  portfolio_options:
    top_n: 2
    signal_lag_days: 1
    use_regime_filter: true
    use_vol_targeting: false
    use_inverse_vol_sizing: false
    use_stop_loss: false
    use_liquidity_filter: false
    use_slippage: false
strategies:
  phase6_demo:
    description: Integration demo signal
    signal_params:
      slope: 1.0
""".strip(),
        encoding="utf-8",
    )
    manager = ConfigManager(project_root=tmp_path, models_dir=tmp_path / "Models")
    configs = manager.load_signal_configs(prefer_yaml=True)
    config = configs["phase6_demo"]

    def _demo_builder(*, dates, loader, config, signal_params):
        del loader, config
        slope = float(signal_params.get("slope", 1.0))
        base = np.linspace(0.0, 100.0, len(dates), dtype=float) * slope
        panel = pd.DataFrame(index=dates, columns=close_df.columns, dtype=float)
        for idx, ticker in enumerate(close_df.columns):
            panel[ticker] = base + idx
        return panel

    monkeypatch.setattr(factory, "BUILDERS", {"phase6_demo": _demo_builder})
    signals = factory.build_signal(
        name="phase6_demo",
        dates=close_df.index,
        loader=dummy_loader,
        config=config,
    )

    result = backtester.run(
        signals=signals,
        factor_name="phase6_demo",
        rebalance_freq="monthly",
        portfolio_options=config["portfolio_options"],
    )
    assert len(result["returns"]) == len(result["returns_df"])
    assert not result["sanity_checks"].empty

    output_dir = tmp_path / "results" / "phase6_demo"
    capm_store: dict[str, dict] = {}
    yearly_store: dict[str, pd.DataFrame] = {}
    generator = ReportGenerator(models_dir=tmp_path, data_dir=tmp_path, loader=dummy_loader)
    generator.save_results(
        results=result,
        factor_name="phase6_demo",
        xu100_prices=xu100_prices,
        xautry_prices=None,
        factor_capm_store=capm_store,
        factor_yearly_rolling_beta_store=yearly_store,
        output_dir=output_dir,
    )
    assert (output_dir / "summary.txt").exists()
    assert (output_dir / "returns.csv").exists()
    assert "phase6_demo" in capm_store


def test_pipeline_outputs_are_loader_compatible(
    tmp_path: Path,
    raw_fundamentals_payload: dict,
    monkeypatch,
) -> None:
    monkeypatch.setenv("BIST_ENFORCE_FUNDAMENTAL_FRESHNESS", "0")
    paths = build_default_paths(base_dir=tmp_path)
    config = build_default_config(enforce_freshness_gate=False, allow_stale_override=True)
    config = replace(config, request_delay_seconds=0.0, max_retries=1)
    pipeline = FundamentalsPipeline(paths=paths, config=config)

    raw_bundle = RawDataBundle(
        raw_by_ticker=raw_fundamentals_payload,
        errors=[],
        source_name="phase6_fixture",
        fetched_at=datetime.now(timezone.utc),
    )
    run_result = pipeline.process_raw_bundle(raw_bundle=raw_bundle)
    assert run_result.merged_bundle is not None
    assert paths.consolidated_parquet.exists()

    regime_dir = tmp_path / "regime_filter" / "outputs"
    regime_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(data_dir=paths.data_dir, regime_model_dir=regime_dir)
    panel = loader.load_fundamentals_parquet()

    assert panel is not None
    assert not panel.empty
    assert panel.index.nlevels == 3
    assert list(panel.index.names) == ["ticker", "sheet_name", "row_name"]


def test_run_backtest_cli_compatibility(monkeypatch, tmp_path: Path) -> None:
    module = importlib.import_module("run_backtest")

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    regime_outputs = tmp_path / "regime_filter" / "outputs"
    regime_outputs.mkdir(parents=True, exist_ok=True)

    (data_dir / "bist_prices_full.csv").write_text("Date,Ticker,Open,Close,Volume\n", encoding="utf-8")
    (data_dir / "xau_try_2013_2026.csv").write_text("Date,XAU_TRY\n2025-01-02,2000\n", encoding="utf-8")
    (data_dir / "xu100_prices.csv").write_text("Date,Open\n2025-01-02,10000\n", encoding="utf-8")
    (regime_outputs / "regime_features.csv").write_text(
        "Date,regime_label\n2025-01-02,Bull\n2025-01-03,Bull\n2025-01-06,Bear\n",
        encoding="utf-8",
    )

    class EngineStub:
        last_instance = None

        def __init__(self, data_dir, regime_model_dir, start_date, end_date):
            del data_dir, regime_model_dir, start_date, end_date
            EngineStub.last_instance = self
            self.signal_configs = {"momentum": {"description": "Momentum strategy"}}
            self.prices = pd.DataFrame()
            self.open_df = pd.DataFrame()
            self.close_df = pd.DataFrame()
            self.calls: list[tuple[str, dict | None]] = []

        def load_all_data(self):
            dates = pd.bdate_range("2025-01-02", periods=6)
            self.prices = pd.DataFrame(
                {
                    "Date": np.repeat(dates, 2),
                    "Ticker": [ticker for _ in dates for ticker in ("AAA.IS", "BBB.IS")],
                    "Open": 100.0,
                    "Close": 100.0,
                    "Volume": 1_000_000.0,
                }
            )
            self.open_df = pd.DataFrame({"AAA": 100.0, "BBB": 100.0}, index=dates)
            self.close_df = self.open_df.copy()

        def run_all_factors(self):
            self.calls.append(("all", None))

        def run_factor(self, factor_name, override_config=None):
            self.calls.append((str(factor_name), override_config))

    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "load_signal_configs", lambda: {"momentum": {"description": "Momentum"}})
    monkeypatch.setattr(module, "PortfolioEngine", EngineStub)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "run_backtest.py",
            "momentum",
            "--start-date",
            "2025-01-02",
            "--end-date",
            "2025-01-10",
            "--data-dir",
            str(data_dir),
            "--regime-outputs",
            str(regime_outputs),
        ],
    )

    rc = module.main()
    assert rc == 0

    engine = EngineStub.last_instance
    assert engine is not None
    assert len(engine.calls) == 1
    factor_name, override_config = engine.calls[0]
    assert factor_name == "momentum"
    assert override_config is not None
    assert override_config["timeline"]["start_date"] == "2025-01-02"
    assert override_config["timeline"]["end_date"] == "2025-01-10"
