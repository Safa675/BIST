from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib import util
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from Models.common.data_loader import DataLoader

logger = logging.getLogger(__name__)


def _load_module_from_path(module_name: str, module_path: Path) -> ModuleType:
    """Import a module from an explicit file path without mutating sys.path."""
    spec = util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec for {module_path}")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class StockData:
    """Container for per-symbol data pulled via borsapy integrations."""

    symbol: str
    quote: pd.Series
    history: pd.DataFrame
    technical_indicators: pd.DataFrame
    financials: dict[str, pd.DataFrame]


class BorsapyAdapter:
    """Encapsulates borsapy-related DataLoader integrations."""

    def __init__(self, loader: "DataLoader", client_path: Path) -> None:
        self._loader = loader
        self._client_path = client_path
        self._config_path = self._client_path.parents[2] / "configs" / "borsapy_config.yaml"
        self._client: Any | None = None

    @property
    def client(self) -> Any | None:
        if self._client is None:
            try:
                from borsapy_client import BorsapyClient
            except ImportError:
                if not self._client_path.exists():
                    logger.warning(f"  ⚠️  Borsapy client file not found: {self._client_path}")
                    logger.info("     Install with: pip install borsapy")
                    return None
                try:
                    module = _load_module_from_path(
                        "bist_fetcher_borsapy_client",
                        self._client_path,
                    )
                    BorsapyClient = getattr(module, "BorsapyClient")
                except Exception as exc:
                    logger.warning(f"  ⚠️  Borsapy not available: {exc}")
                    logger.info("     Install with: pip install borsapy")
                    return None

            try:
                self._client = BorsapyClient(
                    cache_dir=self._loader.data_dir / "borsapy_cache",
                    use_mcp_fallback=True,
                    config_path=self._config_path,
                )
            except Exception as exc:
                logger.warning(f"  ⚠️  Borsapy client initialization failed: {exc}")
                return None
            logger.info("  ✅ Borsapy client initialized")
        return self._client

    def load_prices(
        self,
        symbols: list[str] | None = None,
        period: str = "5y",
        index: str = "XU100",
    ) -> pd.DataFrame:
        if self.client is None:
            logger.warning("  ⚠️  Borsapy not available, cannot load prices")
            return pd.DataFrame()

        logger.info(f"\n📊 Loading prices via borsapy (period={period})...")

        resolved_symbols = symbols
        if resolved_symbols is None:
            resolved_symbols = self.client.get_index_components(index)
            logger.info(f"  Using {len(resolved_symbols)} symbols from {index}")

        result = self.client.batch_download_to_long(
            symbols=resolved_symbols,
            period=period,
            group_by="ticker",
            add_is_suffix=False,
        )

        if result.empty:
            logger.warning("  ⚠️  No data returned from borsapy")
            return pd.DataFrame()

        loaded = result["Ticker"].dropna().nunique() if "Ticker" in result.columns else 0
        logger.info(
            f"  ✅ Loaded {len(result)} price records for {loaded}/{len(resolved_symbols)} tickers"
        )
        return result

    def get_index_components(self, index: str = "XU100") -> list[str]:
        if self.client is None:
            return []
        result = self.client.get_index_components(index)
        if not isinstance(result, list):
            return []
        return [str(symbol) for symbol in result]

    def get_financials(self, symbol: str) -> dict[str, pd.DataFrame]:
        if self.client is None:
            return {}
        try:
            if hasattr(self.client, "get_financial_statements"):
                result = self.client.get_financial_statements(symbol)
            else:
                result = self.client.get_financials(symbol)
        except Exception as exc:
            logger.warning(f"  ⚠️  Failed to load financials for {symbol}: {exc}")
            return {}

        if not isinstance(result, dict):
            return {}

        normalized = {
            "balance_sheet": result.get("balance_sheet", pd.DataFrame()),
            "income_stmt": result.get("income_stmt", pd.DataFrame()),
        }
        cash_flow = result.get("cash_flow", result.get("cashflow", pd.DataFrame()))
        normalized["cashflow"] = cash_flow
        normalized["cash_flow"] = cash_flow
        return normalized

    def get_financial_ratios(self, symbol: str) -> pd.DataFrame:
        if self.client is None or not hasattr(self.client, "get_financial_ratios"):
            return pd.DataFrame()
        try:
            result = self.client.get_financial_ratios(symbol)
        except Exception as exc:
            logger.warning(f"  ⚠️  Failed to load financial ratios for {symbol}: {exc}")
            return pd.DataFrame()
        return result if isinstance(result, pd.DataFrame) else pd.DataFrame()

    def get_dividends(self, symbol: str) -> pd.DataFrame:
        if self.client is None:
            return pd.DataFrame()
        return self.client.get_dividends(symbol)

    def get_fast_info(self, symbol: str) -> dict[str, Any]:
        if self.client is None:
            return {}
        result = self.client.get_fast_info(symbol)
        return result if isinstance(result, dict) else {}

    def screen_stocks(
        self,
        template: str | None = None,
        filters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if self.client is None:
            return pd.DataFrame()
        merged_filters = dict(filters or {})
        merged_filters.update(kwargs)
        return self.client.screen_stocks(template=template, filters=merged_filters)

    def get_history_with_indicators(
        self,
        symbol: str,
        indicators: list[str] | None = None,
        period: str = "2y",
    ) -> pd.DataFrame:
        if self.client is None:
            return pd.DataFrame()
        return self.client.get_history_with_indicators(
            symbol,
            indicators=indicators,
            period=period,
        )

    def get_stock_data(
        self,
        symbol: str,
        period: str = "1mo",
        indicators: list[str] | None = None,
    ) -> StockData | None:
        """Get quote/history/technicals/financials in one call."""
        if self.client is None:
            return None

        history = self.client.get_history(symbol, period=period)
        technical = self._calculate_technical_indicators(
            symbol=symbol,
            history=history,
            period=period,
            indicators=indicators,
        )
        financials = self.get_financials(symbol)

        quote_payload = self.get_fast_info(symbol)
        quote_series = pd.Series(quote_payload if isinstance(quote_payload, dict) else {}, dtype="object")

        return StockData(
            symbol=symbol,
            quote=quote_series,
            history=history if isinstance(history, pd.DataFrame) else pd.DataFrame(),
            technical_indicators=technical,
            financials=financials,
        )

    def _calculate_technical_indicators(
        self,
        symbol: str,
        history: pd.DataFrame,
        period: str,
        indicators: list[str] | None = None,
    ) -> pd.DataFrame:
        if self.client is None:
            return pd.DataFrame()
        if indicators is None:
            indicators = ["rsi", "macd", "bb"]
        try:
            return self.client.get_history_with_indicators(
                symbol,
                indicators=indicators,
                period=period,
            )
        except Exception:
            return history if isinstance(history, pd.DataFrame) else pd.DataFrame()
