from __future__ import annotations

import logging
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


class MacroAdapter:
    """Encapsulates macro-events related DataLoader integrations."""

    def __init__(self, loader: "DataLoader", macro_events_path: Path) -> None:
        self._loader = loader
        self._macro_events_path = macro_events_path
        self._client: Any | None = None

    @property
    def client(self) -> Any | None:
        if self._client is None:
            try:
                from macro_events import MacroEventsClient
            except ImportError:
                if not self._macro_events_path.exists():
                    logger.warning(f"  ⚠️  Macro events module not found: {self._macro_events_path}")
                    return None
                try:
                    module = _load_module_from_path(
                        "bist_fetcher_macro_events",
                        self._macro_events_path,
                    )
                    MacroEventsClient = getattr(module, "MacroEventsClient")
                except Exception as exc:
                    logger.warning(f"  ⚠️  Macro events not available: {exc}")
                    return None

            self._client = MacroEventsClient()
            logger.info("  ✅ Macro events client initialized")
        return self._client

    def get_economic_calendar(
        self,
        days_ahead: int = 7,
        countries: list[str] | None = None,
    ) -> pd.DataFrame:
        if self.client is None:
            return pd.DataFrame()
        return self.client.get_economic_calendar(days_ahead=days_ahead, countries=countries)

    def get_inflation_data(self, periods: int = 24) -> pd.DataFrame:
        if self.client is None:
            return pd.DataFrame()
        return self.client.get_inflation_data(periods=periods)

    def get_bond_yields(self) -> dict[str, Any]:
        if self.client is None:
            return {}
        result = self.client.get_bond_yields()
        return result if isinstance(result, dict) else {}

    def get_stock_news(self, symbol: str, limit: int = 10) -> list[dict[str, Any]]:
        if self.client is None:
            return []
        result = self.client.get_stock_news(symbol, limit=limit)
        return result if isinstance(result, list) else []

    def get_macro_summary(self) -> dict[str, Any]:
        if self.client is None:
            return {}
        result = self.client.get_macro_summary()
        return result if isinstance(result, dict) else {}
