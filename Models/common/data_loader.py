"""
Common Data Loader - Centralized data loading to eliminate redundant I/O

This module loads all fundamental data, price data, and regime predictions ONCE
and caches them in memory for use by all factor models.

Supports multiple data sources:
- Local parquet/CSV files (primary)
- Borsapy API (alternative/supplement)
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from Models.common.borsapy_adapter import BorsapyAdapter, StockData
from Models.common.enums import RegimeLabel
from Models.common.macro_adapter import MacroAdapter
from Models.common.panel_cache import PanelCache
from Models.common.portfolio_analytics import PortfolioAnalyticsAdapter

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FETCHER_DIR = PROJECT_ROOT / "data" / "Fetcher-Scrapper"
BORSAPY_CLIENT_PATH = FETCHER_DIR / "borsapy_client.py"
MACRO_EVENTS_PATH = FETCHER_DIR / "macro_events.py"
REGIME_DIR_CANDIDATES = [
    PROJECT_ROOT / "Simple Regime Filter",
    PROJECT_ROOT / "regime_filter",
]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


class DataLoader:
    """Centralized data loader with caching and multi-source support"""

    def __init__(self, data_dir: Path, regime_model_dir: Path):
        self.data_dir = Path(data_dir)
        self.regime_model_dir = Path(regime_model_dir)
        self.fundamental_dir = self.data_dir / "fundamental_data"
        self.isyatirim_dir = self.data_dir / "price" / "isyatirim_prices"

        # Cache
        self._fundamentals = None
        self._prices = None
        self._close_df = None
        self._open_df = None
        self._volume_df = None
        self._volume_lookback = None
        self._regime_series = None
        self._regime_allocations = None
        self._xautry_prices = None
        self._xu100_prices = None
        self._fundamentals_parquet = None
        self._isyatirim_parquet = None
        self._shares_consolidated = None
        self.panel_cache = PanelCache(
            max_entries=_env_int("BIST_PANEL_CACHE_MAX_ENTRIES", 32),
        )

        self._borsapy_adapter = BorsapyAdapter(self, client_path=BORSAPY_CLIENT_PATH)
        self._macro_adapter = MacroAdapter(self, macro_events_path=MACRO_EVENTS_PATH)
        self._portfolio_analytics_adapter = PortfolioAnalyticsAdapter(self)

        # Freshness gate controls (defaults are strict for production safety).
        self._fundamentals_freshness_gate_enabled = os.getenv(
            "BIST_ENFORCE_FUNDAMENTAL_FRESHNESS",
            "1",
        ).strip().lower() not in {"0", "false", "no", "off"}
        self._allow_stale_fundamentals = os.getenv(
            "BIST_ALLOW_STALE_FUNDAMENTALS",
            "0",
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._freshness_threshold_overrides = {
            "max_median_staleness_days": _env_int("BIST_MAX_MEDIAN_STALENESS_DAYS", 120),
            "max_pct_over_120_days": _env_float("BIST_MAX_PCT_OVER_120_DAYS", 0.90),
            "min_q4_coverage_pct": _env_float("BIST_MIN_Q4_2025_COVERAGE_PCT", 0.10),
            "max_max_staleness_days": _env_int("BIST_MAX_MAX_STALENESS_DAYS", 500),
            "grace_days": _env_int("BIST_STALENESS_GRACE_DAYS", 0),
        }

    # -------------------------------------------------------------------------
    # Adapter Facades
    # -------------------------------------------------------------------------

    @property
    def borsapy_adapter(self) -> BorsapyAdapter:
        return self._borsapy_adapter

    @property
    def macro_adapter(self) -> MacroAdapter:
        return self._macro_adapter

    @property
    def portfolio_analytics(self) -> PortfolioAnalyticsAdapter:
        return self._portfolio_analytics_adapter

    @property
    def borsapy(self):
        return self.borsapy_adapter.client

    @property
    def macro(self):
        return self.macro_adapter.client

    def load_prices_borsapy(
        self,
        symbols: list[str] | None = None,
        period: str = "5y",
        index: str = "XU100",
    ) -> pd.DataFrame:
        return self.borsapy_adapter.load_prices(symbols=symbols, period=period, index=index)

    def get_index_components_borsapy(self, index: str = "XU100") -> list[str]:
        return self.borsapy_adapter.get_index_components(index=index)

    def get_financials_borsapy(self, symbol: str) -> dict[str, pd.DataFrame]:
        return self.borsapy_adapter.get_financials(symbol=symbol)

    def get_financial_ratios_borsapy(self, symbol: str) -> pd.DataFrame:
        return self.borsapy_adapter.get_financial_ratios(symbol=symbol)

    def get_dividends_borsapy(self, symbol: str) -> pd.DataFrame:
        return self.borsapy_adapter.get_dividends(symbol=symbol)

    def get_fast_info_borsapy(self, symbol: str) -> dict:
        return self.borsapy_adapter.get_fast_info(symbol=symbol)

    def screen_stocks_borsapy(
        self,
        template: str | None = None,
        filters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        merged_filters = dict(filters or {})
        merged_filters.update(kwargs)
        return self.borsapy_adapter.screen_stocks(template=template, filters=merged_filters)

    def get_stock_data_borsapy(
        self,
        symbol: str,
        period: str = "1mo",
        indicators: list[str] | None = None,
    ) -> StockData | None:
        return self.borsapy_adapter.get_stock_data(
            symbol=symbol,
            period=period,
            indicators=indicators,
        )

    def get_history_with_indicators_borsapy(
        self,
        symbol: str,
        indicators: list[str] | None = None,
        period: str = "2y",
    ) -> pd.DataFrame:
        return self.borsapy_adapter.get_history_with_indicators(
            symbol=symbol,
            indicators=indicators,
            period=period,
        )

    def create_portfolio_analytics(
        self,
        holdings: dict[str, float] | None = None,
        weights: dict[str, float] | None = None,
        returns: pd.Series | None = None,
        benchmark: str = "XU100",
        name: str = "Portfolio",
    ):
        return self.portfolio_analytics.create_portfolio_analytics(
            holdings=holdings,
            weights=weights,
            returns=returns,
            benchmark=benchmark,
            name=name,
        )

    def get_economic_calendar(
        self,
        days_ahead: int = 7,
        countries: list[str] | None = None,
    ) -> pd.DataFrame:
        return self.macro_adapter.get_economic_calendar(
            days_ahead=days_ahead,
            countries=countries,
        )

    def analyze_strategy_performance(
        self,
        equity_curve: pd.Series,
        benchmark_curve: pd.Series | None = None,
        name: str = "Strategy",
    ):
        return self.portfolio_analytics.analyze_strategy_performance(
            equity_curve=equity_curve,
            benchmark_curve=benchmark_curve,
            name=name,
        )

    def get_inflation_data(self, periods: int = 24) -> pd.DataFrame:
        return self.macro_adapter.get_inflation_data(periods=periods)

    def get_bond_yields(self) -> dict:
        return self.macro_adapter.get_bond_yields()

    def get_stock_news(self, symbol: str, limit: int = 10) -> list[dict]:
        return self.macro_adapter.get_stock_news(symbol=symbol, limit=limit)

    def get_macro_summary(self) -> dict:
        return self.macro_adapter.get_macro_summary()

    def load_prices(self, prices_file: Path) -> pd.DataFrame:
        """Load stock prices"""
        if self._prices is None:
            logger.info("\n📊 Loading price data...")
            parquet_file = prices_file.with_suffix(".parquet")
            if parquet_file.exists():
                logger.info(f"  📦 Using Parquet: {parquet_file.name}")
                self._prices = pd.read_parquet(
                    parquet_file,
                    columns=["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"],
                )
            else:
                logger.info(f"  📄 Using CSV: {prices_file.name}")
                self._prices = pd.read_csv(
                    prices_file,
                    usecols=["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"],
                )
            if "Date" in self._prices.columns:
                self._prices["Date"] = pd.to_datetime(self._prices["Date"], errors="coerce")
            logger.info(f"  ✅ Loaded {len(self._prices)} price records")
        return self._prices
    
    def build_close_panel(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Build close price panel (Date x Ticker)"""
        if self._close_df is None:
            logger.info("  Building close price panel...")
            close_df = prices.pivot_table(index='Date', columns='Ticker', values='Close').sort_index()
            close_df.columns = [c.split('.')[0].upper() for c in close_df.columns]
            self._close_df = close_df
            logger.info(f"  ✅ Close panel: {close_df.shape[0]} days × {close_df.shape[1]} tickers")
        return self._close_df
    
    def build_open_panel(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Build open price panel (Date x Ticker)"""
        if self._open_df is None:
            logger.info("  Building open price panel...")
            open_df = prices.pivot_table(index='Date', columns='Ticker', values='Open').sort_index()
            open_df.columns = [c.split('.')[0].upper() for c in open_df.columns]
            self._open_df = open_df
            logger.info(f"  ✅ Open panel: {open_df.shape[0]} days × {open_df.shape[1]} tickers")
        return self._open_df
    
    def build_volume_panel(self, prices: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
        """Build rolling median volume panel"""
        panel_cache = getattr(self, "panel_cache", None)
        cache_key = None
        if panel_cache is not None:
            cache_key = panel_cache.make_key(
                "volume",
                lookback=int(lookback),
                rows=int(len(prices)),
                date_start=(
                    str(pd.to_datetime(prices["Date"], errors="coerce").min())
                    if "Date" in prices.columns
                    else None
                ),
                date_end=(
                    str(pd.to_datetime(prices["Date"], errors="coerce").max())
                    if "Date" in prices.columns
                    else None
                ),
                ticker_count=int(prices["Ticker"].nunique()) if "Ticker" in prices.columns else None,
            )
            cached = panel_cache.get(cache_key)
            if isinstance(cached, pd.DataFrame):
                self._volume_df = cached
                self._volume_lookback = int(lookback)
                return cached

        if self._volume_df is None or self._volume_lookback != int(lookback):
            logger.info(f"  Building volume panel (lookback={lookback})...")
            vol_pivot = prices.pivot_table(index="Date", columns="Ticker", values="Volume").sort_index()
            vol_pivot.columns = [c.split('.')[0].upper() for c in vol_pivot.columns]
            
            # Drop holiday rows
            valid_pct = vol_pivot.notna().mean(axis=1)
            holiday_mask = valid_pct < 0.5
            if holiday_mask.any():
                vol_clean = vol_pivot.loc[~holiday_mask]
            else:
                vol_clean = vol_pivot
            
            median_adv = vol_clean.rolling(lookback, min_periods=lookback).median()
            median_adv = median_adv.reindex(vol_pivot.index).ffill()
            self._volume_df = median_adv
            self._volume_lookback = int(lookback)
            if panel_cache is not None and cache_key is not None:
                panel_cache.set(cache_key, median_adv)
            logger.info(f"  ✅ Volume panel: {median_adv.shape[0]} days × {median_adv.shape[1]} tickers")
        return self._volume_df
    
    def load_fundamentals(self) -> Dict:
        """Load all fundamental data from Excel files"""
        if self._fundamentals is None:
            logger.info("\n📈 Loading fundamental data...")
            fundamentals = {}
            parquet_file = self.data_dir / "fundamental_data_consolidated.parquet"
            if parquet_file.exists():
                logger.info("  📦 Loading consolidated fundamentals (Parquet)...")
                self._fundamentals_parquet = pd.read_parquet(parquet_file)
                self._enforce_fundamentals_freshness_gate(self._fundamentals_parquet)
                tickers = (
                    self._fundamentals_parquet.index.get_level_values("ticker")
                    .unique()
                    .tolist()
                )
                for ticker in tickers:
                    fundamentals[ticker] = {'path': None}
                logger.info(f"  ✅ Loaded consolidated fundamentals for {len(tickers)} tickers")
            else:
                count = 0
                for file_path in self.fundamental_dir.rglob("*.xlsx"):
                    ticker = file_path.stem.split('.')[0].upper()
                    try:
                        fundamentals[ticker] = {
                            'path': file_path,
                            'income': None,  # Lazy load
                            'balance': None,
                            'cashflow': None,
                        }
                        count += 1
                        if count % 100 == 0:
                            logger.info(f"  Indexed {count} tickers...")
                    except Exception:
                        continue
                logger.info(f"  ✅ Indexed {count} fundamental data files")
            
            self._fundamentals = fundamentals
        return self._fundamentals

    def load_fundamentals_parquet(self) -> pd.DataFrame | None:
        """Load consolidated fundamentals parquet if available"""
        if self._fundamentals_parquet is None:
            parquet_file = self.data_dir / "fundamental_data_consolidated.parquet"
            if parquet_file.exists():
                logger.info("  📦 Loading consolidated fundamentals (Parquet)...")
                self._fundamentals_parquet = pd.read_parquet(parquet_file)
                self._enforce_fundamentals_freshness_gate(self._fundamentals_parquet)
        return self._fundamentals_parquet

    def _enforce_fundamentals_freshness_gate(self, panel: pd.DataFrame) -> None:
        if panel is None or panel.empty:
            return
        if not self._fundamentals_freshness_gate_enabled:
            return
        try:
            from Models.data_pipeline.freshness import (
                compute_staleness_report,
                evaluate_freshness,
                summarize_quality_metrics,
            )
            from Models.data_pipeline.types import FreshnessThresholds
        except Exception as exc:
            logger.warning(f"  ⚠️  Freshness gate dependencies unavailable, skipping gate: {exc}")
            return

        thresholds = FreshnessThresholds(**self._freshness_threshold_overrides)
        staleness = compute_staleness_report(panel)
        quality = summarize_quality_metrics(staleness)
        violations = evaluate_freshness(quality, thresholds)
        if not violations:
            return

        details = "; ".join(violations)
        message = (
            "Fundamental freshness gate violated. "
            f"Set BIST_ALLOW_STALE_FUNDAMENTALS=1 to override. Details: {details}"
        )
        if self._allow_stale_fundamentals:
            logger.warning(f"  ⚠️  {message}")
            return
        raise ValueError(message)

    def load_shares_outstanding_panel(self) -> pd.DataFrame:
        """Load Date x Ticker shares outstanding panel."""
        panel_cache = getattr(self, "panel_cache", None)
        cache_key = None
        if panel_cache is not None:
            cache_key = panel_cache.make_key(
                "shares_outstanding",
                data_dir=str(self.data_dir),
            )
            cached = panel_cache.get(cache_key)
            if isinstance(cached, pd.DataFrame):
                self._shares_consolidated = cached

        if self._shares_consolidated is None:
            shares_file = self.data_dir / "shares_outstanding_consolidated.csv"
            if shares_file.exists():
                logger.info("  📊 Loading consolidated shares file...")
                panel = pd.read_csv(shares_file, index_col=0, parse_dates=True)
                panel.index = pd.to_datetime(panel.index, errors="coerce")
                panel = panel.sort_index()
                panel.columns = [str(c).upper() for c in panel.columns]
                self._shares_consolidated = panel
                logger.info(f"  ✅ Loaded shares for {panel.shape[1]} tickers")
            else:
                # Fallback: build panel from consolidated isyatirim parquet
                isy = self._load_isyatirim_parquet()
                if isy is not None and not isy.empty:
                    try:
                        daily = isy[isy["sheet_type"] == "daily"]
                        required_cols = {"ticker", "HGDG_TARIH", "SERMAYE"}
                        if not daily.empty and required_cols.issubset(daily.columns):
                            panel = daily.pivot_table(
                                index="HGDG_TARIH",
                                columns="ticker",
                                values="SERMAYE",
                                aggfunc="last",
                            )
                            panel.index = pd.to_datetime(panel.index, errors="coerce")
                            panel = panel.sort_index()
                            panel.columns = [str(c).upper() for c in panel.columns]
                            self._shares_consolidated = panel
                            logger.info(f"  ✅ Built shares panel from isyatirim parquet for {panel.shape[1]} tickers")
                        else:
                            self._shares_consolidated = None
                    except Exception as exc:
                        logger.warning(f"  ⚠️  Failed to build shares panel from isyatirim parquet: {exc}")
                        self._shares_consolidated = None
                else:
                    logger.warning("  ⚠️  Consolidated shares file not found")
                    self._shares_consolidated = None

        if self._shares_consolidated is None:
            return pd.DataFrame()
        if panel_cache is not None and cache_key is not None:
            panel_cache.set(cache_key, self._shares_consolidated)
        return self._shares_consolidated
    
    def load_shares_outstanding(self, ticker: str) -> pd.Series:
        """Load shares outstanding from consolidated file (fast!)"""
        shares_panel = self.load_shares_outstanding_panel()
        if not shares_panel.empty and ticker in shares_panel.columns:
            return shares_panel[ticker].dropna()

        # Fallback: consolidated isyatirim parquet (if available)
        isy = self._load_isyatirim_parquet()
        if isy is not None:
            try:
                daily = isy[(isy["ticker"] == ticker) & (isy["sheet_type"] == "daily")]
                if not daily.empty and "HGDG_TARIH" in daily.columns and "SERMAYE" in daily.columns:
                    series = daily.set_index("HGDG_TARIH")["SERMAYE"].dropna()
                    return series
            except Exception:
                pass
        
        # Fallback to individual Excel file (slow)
        excel_path = self.isyatirim_dir / f"{ticker}_2016_2026_daily_and_quarterly.xlsx"
        
        if not excel_path.exists():
            return pd.Series(dtype=float)
        
        try:
            df = pd.read_excel(excel_path, sheet_name='daily')
            if 'HGDG_TARIH' not in df.columns or 'SERMAYE' not in df.columns:
                return pd.Series(dtype=float)
            
            df['HGDG_TARIH'] = pd.to_datetime(df['HGDG_TARIH'])
            df = df.set_index('HGDG_TARIH').sort_index()
            return df['SERMAYE'].dropna()
        except Exception:
            return pd.Series(dtype=float)

    def _load_isyatirim_parquet(self) -> pd.DataFrame | None:
        """Load consolidated isyatirim prices parquet (used for shares fallback)"""
        if self._isyatirim_parquet is None:
            parquet_file = self.data_dir / "isyatirim_prices_consolidated.parquet"
            if parquet_file.exists():
                logger.info("  📦 Loading consolidated isyatirim prices (Parquet)...")
                self._isyatirim_parquet = pd.read_parquet(
                    parquet_file,
                    columns=["ticker", "sheet_type", "HGDG_TARIH", "SERMAYE"],
                )
        return self._isyatirim_parquet
    
    def load_regime_predictions(self, features: pd.DataFrame | None = None) -> pd.Series:
        """
        Load regime labels from regime filter outputs.

        Args:
            features: Unused legacy argument kept for backward compatibility.
        """
        del features  # Backward compatibility placeholder

        if self._regime_series is None:
            logger.info("\n🎯 Loading regime labels...")
            candidate_files = [p / "outputs" / "regime_features.csv" for p in REGIME_DIR_CANDIDATES]
            regime_file = next((f for f in candidate_files if f.exists()), candidate_files[0])

            if not regime_file.exists():
                candidate_dirs = ", ".join(str(p / "outputs") for p in REGIME_DIR_CANDIDATES)
                raise FileNotFoundError(
                    f"Regime file not found in expected locations: {candidate_dirs}\n"
                    "Run the simplified regime pipeline to generate outputs."
                )

            regime_df = pd.read_csv(regime_file)
            if regime_df.empty:
                raise ValueError(f"Regime file is empty: {regime_file}")

            date_col = next((c for c in ("Date", "date", "DATE") if c in regime_df.columns), regime_df.columns[0])
            regime_df[date_col] = pd.to_datetime(regime_df[date_col], errors="coerce")
            regime_df = regime_df.dropna(subset=[date_col]).set_index(date_col).sort_index()

            regime_col = next(
                (c for c in ("regime_label", "simplified_regime", "regime", "detailed_regime") if c in regime_df.columns),
                None,
            )
            if regime_col is None:
                raise ValueError(
                    "No regime column found in regime file. "
                    "Expected one of: regime_label, simplified_regime, regime, detailed_regime."
                )

            raw_regimes = regime_df[regime_col].dropna()
            coerced = raw_regimes.map(RegimeLabel.coerce)
            coerced = coerced[coerced.notna()]
            self._regime_series = coerced.astype(object)
            if self._regime_series.empty:
                raise ValueError(f"No valid regime rows found in: {regime_file}")

            # Load regime->allocation mapping from simplified regime export.
            # This keeps portfolio sizing aligned with whichever regime config was last exported.
            self._regime_allocations = {}
            labels_file = regime_file.parent / "regime_labels.json"
            if labels_file.exists():
                try:
                    labels = json.loads(labels_file.read_text(encoding="utf-8"))
                    for payload in labels.values():
                        if not isinstance(payload, dict):
                            continue
                            regime = RegimeLabel.coerce(payload.get("regime"))
                            alloc = payload.get("allocation")
                            if regime is not None and alloc is not None:
                                try:
                                    self._regime_allocations[regime] = float(alloc)
                                except (TypeError, ValueError):
                                    continue
                except Exception as exc:
                    logger.warning(f"  ⚠️  Could not parse regime allocations from {labels_file.name}: {exc}")

            logger.info(f"  ✅ Loaded {len(self._regime_series)} regime labels")
            logger.info("\n  Regime distribution:")
            for regime, count in self._regime_series.astype(str).value_counts().items():
                pct = count / len(self._regime_series) * 100
                logger.info(f"    {regime}: {count} days ({pct:.1f}%)")
            if self._regime_allocations:
                logger.info("  Regime allocations:")
                for regime, alloc in sorted(
                    self._regime_allocations.items(),
                    key=lambda item: item[0].value if hasattr(item[0], "value") else str(item[0]),
                ):
                    logger.info(f"    {regime}: {alloc:.2f}")

        return self._regime_series

    def load_regime_allocations(self) -> Dict[RegimeLabel, float]:
        """Get regime allocation mapping loaded from regime_labels.json when available."""
        if self._regime_series is None:
            self.load_regime_predictions()
        return dict(self._regime_allocations or {})
    
    def load_xautry_prices(
        self,
        csv_path: Path,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.Series:
        """Load XAU/TRY prices"""
        if self._xautry_prices is None:
            logger.info("\n💰 Loading XAU/TRY prices...")
            df = pd.read_csv(csv_path, parse_dates=["Date"])
            if "XAU_TRY" not in df.columns:
                raise ValueError("XAU_TRY column not found in CSV.")
            df = df.sort_values("Date")
            series = df.set_index("Date")["XAU_TRY"].astype(float)
            series.name = "XAU_TRY"
            self._xautry_prices = series
            logger.info(f"  ✅ Loaded {len(series)} XAU/TRY observations")

        series = self._xautry_prices
        if start_date is not None:
            series = series.loc[series.index >= start_date]
        if end_date is not None:
            series = series.loc[series.index <= end_date]
        return series
    
    def load_xu100_prices(self, csv_path: Path) -> pd.Series:
        """Load XU100 benchmark prices"""
        if self._xu100_prices is None:
            logger.info("\n📊 Loading XU100 benchmark...")
            df = pd.read_csv(csv_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date').sort_index()
            self._xu100_prices = df['Open'] if 'Open' in df.columns else df.iloc[:, 0]
            logger.info(f"  ✅ Loaded {len(self._xu100_prices)} XU100 observations")
        return self._xu100_prices
    
    def load_usdtry(self) -> pd.DataFrame:
        """Load USD/TRY exchange rate data"""
        logger.info("\n💱 Loading USD/TRY data...")
        usdtry_file = self.data_dir / "usdtry_data.csv"
        
        if not usdtry_file.exists():
            logger.warning(f"  ⚠️  USD/TRY file not found: {usdtry_file}")
            return pd.DataFrame()
        
        df = pd.read_csv(usdtry_file, parse_dates=['Date'])
        df = df.set_index('Date').sort_index()
        
        # Rename column to 'Close' for consistency
        if 'USDTRY' in df.columns:
            df = df.rename(columns={'USDTRY': 'Close'})
        
        logger.info(f"  ✅ Loaded {len(df)} USD/TRY observations")
        return df
    
    def load_fundamental_metrics(self) -> pd.DataFrame:
        """Load pre-calculated fundamental metrics"""
        logger.info("\n📊 Loading fundamental metrics...")
        metrics_file = self.data_dir / "fundamental_metrics.parquet"
        
        if not metrics_file.exists():
            logger.warning(f"  ⚠️  Fundamental metrics file not found: {metrics_file}")
            logger.info("  Run calculate_fundamental_metrics.py to generate this file")
            return pd.DataFrame()
        
        df = pd.read_parquet(metrics_file)
        logger.info(f"  ✅ Loaded {len(df)} metric observations")
        logger.info(f"  Metrics: {df.columns.tolist()}")
        return df
