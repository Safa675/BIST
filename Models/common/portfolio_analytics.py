from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pandas as pd

if TYPE_CHECKING:
    from Models.common.data_loader import DataLoader


class PortfolioAnalyticsAdapter:
    """Adapter for PortfolioAnalytics helper methods used by DataLoader."""

    def __init__(self, loader: "DataLoader") -> None:
        self._loader = loader

    @staticmethod
    def _portfolio_analytics_cls() -> type[Any]:
        try:
            from Models.analytics import PortfolioAnalytics
        except ImportError:
            from analytics import PortfolioAnalytics
        return cast(type[Any], PortfolioAnalytics)

    def create_portfolio_analytics(
        self,
        holdings: dict[str, float] | None = None,
        weights: dict[str, float] | None = None,
        returns: pd.Series | None = None,
        benchmark: str = "XU100",
        name: str = "Portfolio",
    ) -> Any:
        portfolio_analytics_cls = self._portfolio_analytics_cls()
        close_df = getattr(self._loader, "_close_df", None)

        if returns is not None:
            benchmark_returns = None
            if (
                benchmark
                and isinstance(close_df, pd.DataFrame)
                and benchmark in close_df.columns
            ):
                benchmark_returns = close_df[benchmark].pct_change().dropna()
            return portfolio_analytics_cls(
                returns=returns,
                benchmark_returns=benchmark_returns,
                name=name,
            )

        if not isinstance(close_df, pd.DataFrame):
            raise ValueError("Price data not loaded. Call load_prices() first.")

        benchmark_returns = None
        if benchmark and benchmark in close_df.columns:
            benchmark_returns = close_df[benchmark].pct_change().dropna()

        if holdings:
            return portfolio_analytics_cls.from_holdings(
                holdings=holdings,
                close_df=close_df,
                benchmark_col=benchmark if benchmark in close_df.columns else None,
                weights=weights,
                name=name,
            )

        if weights:
            synthetic_holdings = {symbol: 1.0 for symbol in weights.keys()}
            return portfolio_analytics_cls.from_holdings(
                holdings=synthetic_holdings,
                close_df=close_df,
                benchmark_col=benchmark if benchmark in close_df.columns else None,
                weights=weights,
                name=name,
            )

        raise ValueError("Either holdings, weights, or returns must be provided")

    def analyze_strategy_performance(
        self,
        equity_curve: pd.Series,
        benchmark_curve: pd.Series | None = None,
        name: str = "Strategy",
    ) -> Any:
        portfolio_analytics_cls = self._portfolio_analytics_cls()
        return portfolio_analytics_cls.from_equity_curve(
            equity_curve=equity_curve,
            benchmark_curve=benchmark_curve,
            name=name,
        )
