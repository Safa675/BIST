from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import Models.common.data_loader as data_loader_module
from Models.common.data_loader import DataLoader


def _stale_panel() -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(
        [
            ("AAA", "Bilanço", "Toplam Varlıklar"),
            ("BBB", "Bilanço", "Toplam Varlıklar"),
        ],
        names=["ticker", "sheet_name", "row_name"],
    )
    # Deliberately stale (no recent quarter columns).
    return pd.DataFrame({"2020/12": [100.0, 200.0]}, index=index)


def test_freshness_gate_blocks_stale_fundamentals(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BIST_ENFORCE_FUNDAMENTAL_FRESHNESS", "1")
    monkeypatch.setenv("BIST_ALLOW_STALE_FUNDAMENTALS", "0")

    loader = DataLoader(data_dir=tmp_path, regime_model_dir=tmp_path)
    with pytest.raises(ValueError, match="freshness gate"):
        loader._enforce_fundamentals_freshness_gate(_stale_panel())


def test_freshness_gate_override_allows_stale_fundamentals(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BIST_ENFORCE_FUNDAMENTAL_FRESHNESS", "1")
    monkeypatch.setenv("BIST_ALLOW_STALE_FUNDAMENTALS", "1")

    loader = DataLoader(data_dir=tmp_path, regime_model_dir=tmp_path)
    # Should not raise because override is enabled.
    loader._enforce_fundamentals_freshness_gate(_stale_panel())


def test_load_fundamentals_parquet_invokes_gate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BIST_ENFORCE_FUNDAMENTAL_FRESHNESS", "0")

    panel = _stale_panel()
    parquet = tmp_path / "fundamental_data_consolidated.parquet"
    panel.to_parquet(parquet)

    loader = DataLoader(data_dir=tmp_path, regime_model_dir=tmp_path)

    seen = {"called": False}

    def _gate_spy(frame: pd.DataFrame) -> None:
        seen["called"] = True
        assert isinstance(frame, pd.DataFrame)

    monkeypatch.setattr(loader, "_enforce_fundamentals_freshness_gate", _gate_spy)
    out = loader.load_fundamentals_parquet()

    assert isinstance(out, pd.DataFrame)
    assert seen["called"] is True


def test_load_prices_parquet_and_csv_match(tmp_path: Path) -> None:
    price_rows = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-01-02", "2025-01-02", "2025-01-03", "2025-01-03"]),
            "Ticker": ["AAA.IS", "BBB.IS", "AAA.IS", "BBB.IS"],
            "Open": [10.0, 20.0, 10.5, 20.2],
            "High": [10.2, 20.5, 10.6, 20.4],
            "Low": [9.8, 19.8, 10.2, 20.0],
            "Close": [10.1, 20.1, 10.4, 20.3],
            "Volume": [1_000_000, 2_000_000, 1_100_000, 2_100_000],
        }
    )

    csv_path = tmp_path / "prices.csv"
    parquet_path = csv_path.with_suffix(".parquet")
    price_rows.to_csv(csv_path, index=False)
    price_rows.to_parquet(parquet_path, index=False)

    parquet_loader = DataLoader(data_dir=tmp_path, regime_model_dir=tmp_path)
    csv_loader = DataLoader(data_dir=tmp_path, regime_model_dir=tmp_path)

    loaded_from_parquet = parquet_loader.load_prices(csv_path)
    parquet_path.unlink()
    loaded_from_csv = csv_loader.load_prices(csv_path)

    pd.testing.assert_frame_equal(
        loaded_from_parquet.sort_values(["Date", "Ticker"]).reset_index(drop=True),
        loaded_from_csv.sort_values(["Date", "Ticker"]).reset_index(drop=True),
        check_dtype=False,
    )


def test_load_prices_missing_file_raises_helpful_error(tmp_path: Path) -> None:
    loader = DataLoader(data_dir=tmp_path, regime_model_dir=tmp_path)
    missing_csv = tmp_path / "missing_prices.csv"

    with pytest.raises(FileNotFoundError, match="missing_prices\\.csv"):
        loader.load_prices(missing_csv)


def test_load_fundamentals_empty_directory_returns_empty_map(tmp_path: Path) -> None:
    loader = DataLoader(data_dir=tmp_path, regime_model_dir=tmp_path)

    out = loader.load_fundamentals()

    assert isinstance(out, dict)
    assert out == {}


def test_build_close_panel_normalizes_is_suffix(tmp_path: Path) -> None:
    loader = DataLoader(data_dir=tmp_path, regime_model_dir=tmp_path)
    prices = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-01-02", "2025-01-02"]),
            "Ticker": ["abc.IS", "XYZ.is"],
            "Close": [100.0, 200.0],
        }
    )

    panel = loader.build_close_panel(prices)

    # Column order may vary, so check as set
    assert set(panel.columns.tolist()) == {"ABC", "XYZ"}
    assert isinstance(panel.index, pd.DatetimeIndex)


def test_load_regime_predictions_missing_regime_column_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    regime_root = tmp_path / "regime_filter"
    outputs = regime_root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "Date": ["2025-01-02", "2025-01-03"],
            "unexpected_col": ["foo", "bar"],
        }
    ).to_csv(outputs / "regime_features.csv", index=False)

    monkeypatch.setattr(data_loader_module, "REGIME_DIR_CANDIDATES", [regime_root])
    loader = DataLoader(data_dir=tmp_path, regime_model_dir=tmp_path)

    with pytest.raises(ValueError, match="No regime column found in regime file"):
        loader.load_regime_predictions()
