"""
Signal Service - Bridge between API and Portfolio Engine

Wraps the existing quant engine to provide signal data to the API.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime, timedelta

# Add Models to path
PROJECT_ROOT = Path(__file__).resolve().parents[5]
MODELS_DIR = PROJECT_ROOT / "Models"
sys.path.insert(0, str(MODELS_DIR))


class SignalService:
    """Service to fetch and process trading signals from the quant engine."""

    def __init__(self):
        self._engine = None
        self._loader = None
        self._last_signals: Dict[str, pd.DataFrame] = {}
        self._cache_time: Optional[datetime] = None
        self._cache_duration = timedelta(hours=1)

    def _get_engine(self):
        """Lazy load the portfolio engine."""
        if self._engine is None:
            try:
                from portfolio_engine import PortfolioEngine
                self._engine = PortfolioEngine()
            except ImportError as e:
                print(f"Warning: Could not import PortfolioEngine: {e}")
                return None
        return self._engine

    def _get_loader(self):
        """Lazy load the data loader."""
        if self._loader is None:
            try:
                from common.data_loader import DataLoader
                self._loader = DataLoader()
            except ImportError as e:
                print(f"Warning: Could not import DataLoader: {e}")
                return None
        return self._loader

    def get_available_signals(self) -> List[Dict[str, str]]:
        """Get list of available trading signals."""
        return [
            {"name": "five_factor_rotation", "category": "multi_factor", "description": "5-faktör rotasyon stratejisi"},
            {"name": "momentum", "category": "momentum", "description": "12-1 momentum anomalisi"},
            {"name": "short_term_reversal", "category": "reversal", "description": "Haftalık geri dönüş"},
            {"name": "consistent_momentum", "category": "momentum", "description": "Tutarlı momentum"},
            {"name": "residual_momentum", "category": "momentum", "description": "Rezidüel momentum"},
            {"name": "low_volatility", "category": "defensive", "description": "Düşük volatilite anomalisi"},
            {"name": "trend_following", "category": "trend", "description": "Trend takip stratejisi"},
            {"name": "sector_rotation", "category": "sector", "description": "Sektör rotasyonu"},
            {"name": "earnings_quality", "category": "quality", "description": "Kazanç kalitesi"},
            {"name": "fscore_reversal", "category": "value", "description": "F-Score + Geri dönüş"},
            {"name": "momentum_asset_growth", "category": "multi_factor", "description": "Momentum + Varlık büyümesi"},
            {"name": "pairs_trading", "category": "arbitrage", "description": "Pairs trading"},
            {"name": "accrual", "category": "quality", "description": "Tahakkuk anomalisi"},
            {"name": "asset_growth", "category": "value", "description": "Varlık büyümesi faktörü"},
            {"name": "betting_against_beta", "category": "defensive", "description": "Beta karşıtı strateji"},
            {"name": "roa", "category": "quality", "description": "Aktif karlılığı"},
        ]

    def get_signal_data(self, signal_name: str, top_n: int = 10) -> Dict[str, Any]:
        """
        Get current signal values for a specific factor.

        Returns top and bottom picks based on signal strength.
        """
        engine = self._get_engine()

        if engine is None:
            # Return mock data if engine not available
            return self._get_mock_signal_data(signal_name, top_n)

        try:
            # Check cache
            if self._is_cache_valid(signal_name):
                signals_df = self._last_signals[signal_name]
            else:
                # Run the signal
                signals_df = engine.run_factor(signal_name)
                self._last_signals[signal_name] = signals_df
                self._cache_time = datetime.now()

            if signals_df is None or signals_df.empty:
                return self._get_mock_signal_data(signal_name, top_n)

            # Get latest row
            latest = signals_df.iloc[-1].dropna().sort_values(ascending=False)

            top_picks = [
                {"ticker": t, "score": round(s, 3)}
                for t, s in latest.head(top_n).items()
            ]

            bottom_picks = [
                {"ticker": t, "score": round(s, 3)}
                for t, s in latest.tail(top_n).items()
            ]

            return {
                "signal_name": signal_name,
                "last_updated": datetime.now().isoformat(),
                "total_stocks": len(latest),
                "top_picks": top_picks,
                "bottom_picks": bottom_picks,
            }

        except Exception as e:
            print(f"Error getting signal data: {e}")
            return self._get_mock_signal_data(signal_name, top_n)

    def _is_cache_valid(self, signal_name: str) -> bool:
        """Check if cached signal data is still valid."""
        if signal_name not in self._last_signals:
            return False
        if self._cache_time is None:
            return False
        return datetime.now() - self._cache_time < self._cache_duration

    def _get_mock_signal_data(self, signal_name: str, top_n: int) -> Dict[str, Any]:
        """Return mock data for demo purposes."""
        mock_stocks = {
            "momentum": [
                ("THYAO", 0.92), ("ASELS", 0.88), ("TUPRS", 0.82),
                ("SISE", 0.78), ("EREGL", 0.72), ("BIMAS", 0.68),
            ],
            "quality": [
                ("ASELS", 0.95), ("BIMAS", 0.89), ("THYAO", 0.85),
                ("KCHOL", 0.80), ("SAHOL", 0.76), ("TCELL", 0.71),
            ],
            "low_volatility": [
                ("BIMAS", 0.90), ("TCELL", 0.85), ("TTKOM", 0.82),
                ("AKBNK", 0.78), ("GARAN", 0.74), ("YKBNK", 0.70),
            ],
        }

        stocks = mock_stocks.get(signal_name, mock_stocks["momentum"])

        return {
            "signal_name": signal_name,
            "last_updated": datetime.now().isoformat(),
            "total_stocks": 100,
            "top_picks": [{"ticker": t, "score": s} for t, s in stocks[:top_n]],
            "bottom_picks": [{"ticker": t, "score": -s} for t, s in stocks[-3:]],
        }

    def get_composite_recommendations(
        self,
        factors: List[str],
        weights: Optional[Dict[str, float]] = None,
        top_n: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get composite recommendations by combining multiple factors.

        Args:
            factors: List of factor names to combine
            weights: Optional weights for each factor (default: equal)
            top_n: Number of recommendations to return

        Returns:
            List of stock recommendations with composite scores
        """
        if weights is None:
            weights = {f: 1.0 / len(factors) for f in factors}

        engine = self._get_engine()
        loader = self._get_loader()

        if engine is None or loader is None:
            return self._get_mock_recommendations(top_n)

        try:
            # Get signals for each factor
            composite = None

            for factor, weight in weights.items():
                signal_data = self.get_signal_data(factor, top_n=100)

                if "top_picks" not in signal_data:
                    continue

                # Convert to series
                scores = pd.Series({
                    p["ticker"]: p["score"]
                    for p in signal_data["top_picks"]
                })

                if composite is None:
                    composite = scores * weight
                else:
                    composite = composite.add(scores * weight, fill_value=0)

            if composite is None:
                return self._get_mock_recommendations(top_n)

            # Sort and get top picks
            top = composite.sort_values(ascending=False).head(top_n)

            # Get company names (mock for now)
            company_names = self._get_company_names()

            recommendations = []
            for ticker, score in top.items():
                recommendations.append({
                    "ticker": ticker,
                    "name": company_names.get(ticker, ticker),
                    "signal_strength": round(score, 3),
                    "factors": factors,
                    "target_weight": round(1.0 / top_n, 3),
                })

            return recommendations

        except Exception as e:
            print(f"Error getting composite recommendations: {e}")
            return self._get_mock_recommendations(top_n)

    def _get_mock_recommendations(self, top_n: int) -> List[Dict[str, Any]]:
        """Return mock recommendations for demo."""
        names = self._get_company_names()

        stocks = [
            ("THYAO", 0.85), ("ASELS", 0.78), ("TUPRS", 0.72),
            ("SISE", 0.68), ("EREGL", 0.65), ("BIMAS", 0.62),
            ("KCHOL", 0.58), ("SAHOL", 0.55), ("TCELL", 0.52),
            ("AKBNK", 0.48),
        ]

        return [
            {
                "ticker": t,
                "name": names.get(t, t),
                "signal_strength": s,
                "factors": ["momentum", "quality"],
                "target_weight": round(1.0 / top_n, 3),
            }
            for t, s in stocks[:top_n]
        ]

    def _get_company_names(self) -> Dict[str, str]:
        """Get company name mapping."""
        return {
            "THYAO": "Türk Hava Yolları",
            "ASELS": "Aselsan",
            "TUPRS": "Tüpraş",
            "SISE": "Şişe Cam",
            "EREGL": "Ereğli Demir Çelik",
            "BIMAS": "BİM Mağazaları",
            "KCHOL": "Koç Holding",
            "SAHOL": "Sabancı Holding",
            "TCELL": "Turkcell",
            "AKBNK": "Akbank",
            "GARAN": "Garanti BBVA",
            "YKBNK": "Yapı Kredi",
            "TTKOM": "Türk Telekom",
            "PETKM": "Petkim",
            "KOZAL": "Koza Altın",
            "EKGYO": "Emlak Konut GYO",
            "PGSUS": "Pegasus",
            "TAVHL": "TAV Havalimanları",
            "FROTO": "Ford Otosan",
            "TOASO": "Tofaş",
        }


# Singleton instance
signal_service = SignalService()
