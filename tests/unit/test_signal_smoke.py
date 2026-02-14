from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from Models.signals.factory import build_signal, get_available_signals

SEED = 42

_SIGNAL_XFAIL_REASONS: dict[str, str] = {
    "breakout_value": "requires richer value/fundamental overlap than synthetic 60-day panel provides",
    "consistent_momentum": "requires >126-day lookbacks; 60-day smoke data is intentionally short",
    "dividend_rotation": "requires long volatility lookback and fully populated metric history",
    "five_factor_rotation": "requires dense multi-axis construction data not covered by minimal smoke inputs",
    "fscore_reversal": "piotroski + reversal blend is sparse under minimal synthetic fundamentals",
    "investment": "investment signal is sparse under minimal quarter alignment in 60-day window",
    "momentum_asset_growth": "requires richer asset-growth history than minimal smoke fundamentals",
    "momentum_reversal_volatility": "default 252-day momentum/volatility windows exceed smoke horizon",
    "residual_momentum": "default residual momentum windows exceed 60-day smoke horizon",
    "sector_rotation": "depends on sector classification data unavailable in smoke fixture",
    "size_rotation": "requires broader cross-section than 10-ticker smoke universe",
    "size_rotation_momentum": "inherits size-rotation universe constraints in smoke fixture",
    "size_rotation_quality": "inherits size-rotation + quality data constraints in smoke fixture",
    "trend_following": "trend windows exceed smoke horizon causing structurally sparse outputs",
    "trend_value": "requires mature trend/value history beyond minimal smoke data",
    "value": "value ratios are sparse under minimal quarter/date overlap in smoke fixture",
    "xu100": "benchmark signal intentionally returns only XU100 column, not full ticker universe",
}


class _NullPanelCache:
    def make_key(self, *args: object, **kwargs: object) -> tuple[tuple[object, ...], tuple[tuple[str, object], ...]]:
        return tuple(args), tuple(sorted((str(k), v) for k, v in kwargs.items()))

    def get(self, _key: object) -> None:
        return None

    def set(self, _key: object, _value: object) -> None:
        return None


class DummyLoader:
    def __init__(
        self,
        *,
        dates: pd.DatetimeIndex,
        tickers: list[str],
        prices: pd.DataFrame,
        volume_df: pd.DataFrame,
        fundamentals_parquet: pd.DataFrame,
        metrics_df: pd.DataFrame,
        xu100_prices: pd.Series,
    ) -> None:
        self.data_dir = Path(".")
        self.panel_cache = _NullPanelCache()
        self._dates = dates
        self._tickers = tickers
        self._prices = prices
        self._volume_df = volume_df
        self._fundamentals_parquet = fundamentals_parquet
        self._metrics_df = metrics_df
        self._xu100_prices = xu100_prices

    def load_fundamentals(self) -> dict[str, dict[str, Path | None]]:
        return {ticker: {"path": None} for ticker in self._tickers}

    def load_fundamentals_parquet(self) -> pd.DataFrame:
        return self._fundamentals_parquet

    def load_shares_outstanding(self, ticker: str) -> pd.Series:
        ticker_idx = self._tickers.index(ticker)
        shares = 40_000_000.0 + ticker_idx * 3_000_000.0
        return pd.Series(shares, index=self._dates, dtype=float)

    def load_shares_outstanding_panel(self) -> pd.DataFrame:
        panel = pd.DataFrame(index=self._dates, columns=self._tickers, dtype=float)
        for ticker_idx, ticker in enumerate(self._tickers):
            panel[ticker] = 40_000_000.0 + ticker_idx * 3_000_000.0
        return panel

    def load_fundamental_metrics(self) -> pd.DataFrame:
        return self._metrics_df

    def load_prices(self, _prices_file: Path) -> pd.DataFrame:
        return self._prices

    def build_volume_panel(self, _prices: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
        del lookback
        return self._volume_df

    def load_xu100_prices(self, _csv_path: Path) -> pd.Series:
        return self._xu100_prices


@dataclass(frozen=True)
class SignalSmokeInputs:
    dates: pd.DatetimeIndex
    tickers: list[str]
    prices: pd.DataFrame
    close_df: pd.DataFrame
    open_df: pd.DataFrame
    high_df: pd.DataFrame
    low_df: pd.DataFrame
    volume_df: pd.DataFrame
    fundamentals: dict[str, dict[str, Path | None]]
    loader: DummyLoader
    xu100_prices: pd.Series

    def config(self) -> dict[str, object]:
        return {
            "_runtime_context": {
                "prices": self.prices,
                "close_df": self.close_df,
                "open_df": self.open_df,
                "high_df": self.high_df,
                "low_df": self.low_df,
                "volume_df": self.volume_df,
                "fundamentals": self.fundamentals,
                "xu100_prices": self.xu100_prices,
            },
            # Keep minimal shared overrides so short smoke horizon can exercise more builders.
            "signal_params": {
                "lookback": 20,
                "skip": 5,
                "vol_lookback": 20,
                "beta_window": 30,
                "short_period": 5,
                "long_period": 20,
                "period": 10,
                "momentum_lookback": 10,
                "conversion_period": 9,
                "base_period": 20,
                "span_b_period": 40,
            },
        }


def _build_fundamentals_parquet(tickers: list[str]) -> pd.DataFrame:
    quarters = pd.period_range("2023Q1", "2025Q4", freq="Q")
    quarter_cols = [f"{quarter.year}/{quarter.quarter * 3:02d}" for quarter in quarters]

    row_names = [
        "Dönem Net Karı (Zararı)",
        "Dönem Karı (Zararı)",
        "Satış Gelirleri",
        "FAVÖK",
        "Faaliyetlerden Elde Edilen Nakit Akışları",
        "İşletme Faaliyetlerinden Nakit Akışları",
        "Maddi ve Maddi Olmayan Duran Varlıkların Alımından Kaynaklanan Nakit Çıkışları",
        "Finansal Borçlar",
        "Nakit ve Nakit Benzerleri",
        "Toplam Varlıklar",
        "Özkaynaklar",
        "Dönen Varlıklar",
        "Kısa Vadeli Yükümlülükler",
        "Uzun Vadeli Yükümlülükler",
        "Toplam Yükümlülükler",
        "Brüt Kar (Zarar)",
        "Faaliyet Karı (Zararı)",
        "Ödenen Temettüler",
    ]
    sheet_names = ["Gelir Tablosu (Çeyreklik)", "Bilanço", "Nakit Akış (Çeyreklik)"]

    index_tuples: list[tuple[str, str, str]] = []
    values: list[list[float]] = []

    rng = np.random.default_rng(SEED)
    for ticker_idx, ticker in enumerate(tickers):
        for sheet in sheet_names:
            for row_idx, row_name in enumerate(row_names):
                base = 100.0 + 10.0 * ticker_idx + 2.5 * row_idx
                drift = rng.normal(loc=1.5 + 0.05 * ticker_idx, scale=0.15, size=len(quarter_cols))
                series = np.cumsum(np.maximum(drift, 0.1)) + base
                index_tuples.append((ticker, sheet, row_name))
                values.append(series.tolist())

    return pd.DataFrame(
        values,
        index=pd.MultiIndex.from_tuples(
            index_tuples,
            names=["ticker", "sheet_name", "row_name"],
        ),
        columns=quarter_cols,
    )


def _build_metrics_panel(tickers: list[str], dates: pd.DatetimeIndex) -> pd.DataFrame:
    index = pd.MultiIndex.from_product([tickers, dates], names=["ticker", "date"])
    metrics = pd.DataFrame(index=index)

    for ticker_idx, ticker in enumerate(tickers):
        date_pos = np.arange(len(dates), dtype=float)
        base_shift = float(ticker_idx) * 0.01

        metrics.loc[(ticker,), "debt_to_equity"] = 0.35 + base_shift + 0.0005 * date_pos
        metrics.loc[(ticker,), "cash_ratio"] = 1.20 + base_shift + 0.002 * np.sin(date_pos / 5.0)
        metrics.loc[(ticker,), "current_ratio"] = 1.40 + base_shift + 0.002 * np.cos(date_pos / 7.0)
        metrics.loc[(ticker,), "operating_cash_flow"] = 800_000 + 25_000 * ticker_idx + 350 * date_pos
        metrics.loc[(ticker,), "dividend_payout_ratio"] = 0.22 + base_shift + 0.002 * np.sin(date_pos / 4.0)
        metrics.loc[(ticker,), "earnings_growth_yoy"] = 0.05 + base_shift + 0.002 * np.cos(date_pos / 6.0)

    return metrics.astype(float)


@pytest.fixture
def signal_smoke_inputs() -> SignalSmokeInputs:
    dates = pd.bdate_range("2025-01-02", periods=60)
    tickers = [f"TK{i:02d}" for i in range(10)]

    rng = np.random.default_rng(SEED)

    base_prices = np.linspace(20.0, 200.0, len(tickers))
    daily_rets = rng.normal(loc=0.001, scale=0.02, size=(len(dates), len(tickers)))

    close_df = pd.DataFrame(index=dates, columns=tickers, dtype=float)
    for idx, ticker in enumerate(tickers):
        close_df[ticker] = base_prices[idx] * np.cumprod(1.0 + daily_rets[:, idx])

    open_df = close_df.shift(1).fillna(close_df.iloc[0] * 0.999)
    high_df = pd.DataFrame(
        np.maximum(open_df, close_df) * (1.0 + rng.uniform(0.003, 0.015, size=close_df.shape)),
        index=dates,
        columns=tickers,
    )
    low_df = pd.DataFrame(
        np.minimum(open_df, close_df) * (1.0 - rng.uniform(0.003, 0.015, size=close_df.shape)),
        index=dates,
        columns=tickers,
    )
    volume_df = pd.DataFrame(
        {
            ticker: 800_000.0 + 90_000.0 * idx + np.arange(len(dates), dtype=float) * 500.0
            for idx, ticker in enumerate(tickers)
        },
        index=dates,
        dtype=float,
    )

    prices = pd.DataFrame(
        {
            "Date": np.repeat(dates.values, len(tickers)),
            "Ticker": [f"{ticker}.IS" for _ in dates for ticker in tickers],
            "Open": open_df.stack().to_numpy(dtype=float),
            "High": high_df.stack().to_numpy(dtype=float),
            "Low": low_df.stack().to_numpy(dtype=float),
            "Close": close_df.stack().to_numpy(dtype=float),
            "Volume": volume_df.stack().to_numpy(dtype=float),
        }
    )

    fundamentals_parquet = _build_fundamentals_parquet(tickers)
    metrics_df = _build_metrics_panel(tickers, dates)
    xu100_prices = close_df.mean(axis=1).rename("XU100")

    loader = DummyLoader(
        dates=dates,
        tickers=tickers,
        prices=prices,
        volume_df=volume_df,
        fundamentals_parquet=fundamentals_parquet,
        metrics_df=metrics_df,
        xu100_prices=xu100_prices,
    )
    fundamentals = loader.load_fundamentals()

    return SignalSmokeInputs(
        dates=dates,
        tickers=tickers,
        prices=prices,
        close_df=close_df,
        open_df=open_df,
        high_df=high_df,
        low_df=low_df,
        volume_df=volume_df,
        fundamentals=fundamentals,
        loader=loader,
        xu100_prices=xu100_prices,
    )


def _signal_param(name: str) -> pytest.ParamSpec:
    reason = _SIGNAL_XFAIL_REASONS.get(name)
    if reason is None:
        return pytest.param(name)
    return pytest.param(name, marks=pytest.mark.xfail(reason=reason))


_SIGNAL_PARAMS = [_signal_param(name) for name in get_available_signals()]


@pytest.mark.parametrize("signal_name", _SIGNAL_PARAMS)
def test_signal_factory_smoke_all_registered_signals(
    signal_name: str,
    signal_smoke_inputs: SignalSmokeInputs,
) -> None:
    config = signal_smoke_inputs.config()

    signal = build_signal(
        signal_name,
        signal_smoke_inputs.dates,
        signal_smoke_inputs.loader,
        config,
    )

    assert isinstance(signal, pd.DataFrame)
    assert signal.shape == (len(signal_smoke_inputs.dates), len(signal_smoke_inputs.tickers))
    assert isinstance(signal.index, pd.DatetimeIndex)
    assert signal.index.equals(signal_smoke_inputs.dates)
    assert signal.columns.tolist() == signal_smoke_inputs.tickers

    # Every ticker column should have at least some signal data in the smoke horizon.
    assert signal.notna().sum(axis=0).gt(0).all()

    values = signal.to_numpy(dtype=float, copy=False)
    assert not np.isinf(values).any()
    assert all(pd.api.types.is_float_dtype(dtype) for dtype in signal.dtypes)
