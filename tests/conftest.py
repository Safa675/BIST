from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from Models.common.backtester import Backtester
from Models.common.risk_manager import RiskManager


def pytest_addoption(parser):
    """Add custom pytest options."""
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Update golden results baseline files",
    )


@pytest.fixture
def fixture_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def raw_fundamentals_payload(fixture_dir: Path) -> dict:
    return json.loads((fixture_dir / "raw_fundamentals_payload.json").read_text(encoding="utf-8"))


@pytest.fixture
def api_value_response(fixture_dir: Path) -> dict:
    return json.loads((fixture_dir / "api_value_response.json").read_text(encoding="utf-8"))


@pytest.fixture
def business_dates() -> pd.DatetimeIndex:
    return pd.bdate_range("2025-01-02", periods=60)


@pytest.fixture
def tickers() -> list[str]:
    return ["AAA", "BBB", "CCC"]


@pytest.fixture
def close_df(business_dates: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
    t = np.arange(len(business_dates), dtype=float)
    data = {
        "AAA": 100.0 + 0.90 * t,
        "BBB": 90.0 + 0.45 * t,
        "CCC": 80.0 + 0.20 * t + np.sin(t / 6.0),
    }
    frame = pd.DataFrame(data, index=business_dates)
    return frame[tickers]


@pytest.fixture
def open_df(close_df: pd.DataFrame) -> pd.DataFrame:
    # Keep open close to previous close to produce stable forward returns.
    return close_df.shift(1).fillna(close_df.iloc[0]) * 1.001


@pytest.fixture
def prices_long(open_df: pd.DataFrame, close_df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for date in close_df.index:
        for ticker in close_df.columns:
            records.append(
                {
                    "Date": date,
                    "Ticker": f"{ticker}.IS",
                    "Open": float(open_df.loc[date, ticker]),
                    "Close": float(close_df.loc[date, ticker]),
                    "Volume": float(1_000_000 + (hash(ticker) % 500_000)),
                }
            )
    return pd.DataFrame(records)


@pytest.fixture
def volume_df(close_df: pd.DataFrame) -> pd.DataFrame:
    data = {
        "AAA": np.full(len(close_df.index), 2_500_000.0),
        "BBB": np.full(len(close_df.index), 1_500_000.0),
        "CCC": np.full(len(close_df.index), 750_000.0),
    }
    return pd.DataFrame(data, index=close_df.index)


@pytest.fixture
def regime_series(business_dates: pd.DatetimeIndex) -> pd.Series:
    regimes = ["Bull", "Bear", "Recovery", "Stress"]
    values = [regimes[i % len(regimes)] for i in range(len(business_dates))]
    return pd.Series(values, index=business_dates)


@pytest.fixture
def xu100_prices(business_dates: pd.DatetimeIndex) -> pd.Series:
    vals = 5_000.0 + np.arange(len(business_dates), dtype=float) * 10.0
    return pd.Series(vals, index=business_dates, name="XU100")


@pytest.fixture
def xautry_prices(business_dates: pd.DatetimeIndex) -> pd.Series:
    vals = 2_000.0 + np.arange(len(business_dates), dtype=float) * 2.0
    return pd.Series(vals, index=business_dates, name="XAUTRY")


@pytest.fixture
def signals_df(business_dates: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
    # AAA is strongest except occasional reversals to test rebalancing behavior.
    frame = pd.DataFrame(index=business_dates, columns=tickers, dtype=float)
    frame.loc[:, "AAA"] = 90.0
    frame.loc[:, "BBB"] = 70.0
    frame.loc[:, "CCC"] = 50.0
    frame.iloc[20:25, :] = frame.iloc[20:25, ::-1].to_numpy()
    return frame


class DummyLoader:
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


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    out = tmp_path / "data"
    out.mkdir(parents=True, exist_ok=True)
    return out


@pytest.fixture
def dummy_loader(xautry_prices: pd.Series, data_dir: Path) -> DummyLoader:
    return DummyLoader(xautry_prices=xautry_prices, data_dir=data_dir)


@pytest.fixture
def build_size_market_cap_panel_stub():
    def _builder(close_df: pd.DataFrame, dates: pd.DatetimeIndex, _loader) -> pd.DataFrame:
        base = pd.DataFrame(index=dates, columns=close_df.columns, dtype=float)
        if "AAA" in base.columns:
            base["AAA"] = 4_000_000_000.0
        if "BBB" in base.columns:
            base["BBB"] = 1_800_000_000.0
        if "CCC" in base.columns:
            base["CCC"] = 400_000_000.0
        return base

    return _builder


@pytest.fixture
def backtester(
    dummy_loader: DummyLoader,
    data_dir: Path,
    prices_long: pd.DataFrame,
    close_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    regime_series: pd.Series,
    xu100_prices: pd.Series,
    xautry_prices: pd.Series,
    build_size_market_cap_panel_stub,
) -> Backtester:
    manager = RiskManager(close_df=close_df, volume_df=volume_df)
    bt = Backtester(
        loader=dummy_loader,
        data_dir=data_dir,
        risk_manager=manager,
        build_size_market_cap_panel=build_size_market_cap_panel_stub,
    )
    bt.update_data(
        prices=prices_long,
        close_df=close_df,
        volume_df=volume_df,
        regime_series=regime_series,
        regime_allocations={"Bull": 1.0, "Bear": 0.0, "Recovery": 0.5, "Stress": 0.0},
        xu100_prices=xu100_prices,
        xautry_prices=xautry_prices,
    )
    return bt
