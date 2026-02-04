"""
Pydantic models for API request/response schemas
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime, date
from enum import Enum


class RegimeEnum(str, Enum):
    """Possible regime states"""
    BULL = "Bull"
    BEAR = "Bear"
    STRESS = "Stress"
    CHOPPY = "Choppy"
    RECOVERY = "Recovery"
    UNKNOWN = "Unknown"


class ModelTypeEnum(str, Enum):
    """Available model types"""
    ENSEMBLE = "ensemble"
    XGBOOST = "xgboost"
    LSTM = "lstm"
    HMM = "hmm"


class RegimeProbabilities(BaseModel):
    """Regime probability distribution"""
    Bull: float = Field(ge=0, le=1)
    Bear: float = Field(ge=0, le=1)
    Stress: float = Field(ge=0, le=1)
    Choppy: float = Field(ge=0, le=1)
    Recovery: float = Field(ge=0, le=1)


class ModelAgreement(BaseModel):
    """Per-model predictions"""
    xgboost: Optional[RegimeEnum] = None
    lstm: Optional[RegimeEnum] = None
    hmm: Optional[RegimeEnum] = None


class RegimeResponse(BaseModel):
    """Current regime prediction response"""
    date: datetime
    regime: RegimeEnum
    confidence: float = Field(ge=0, le=1, description="Prediction confidence (0-1)")
    probabilities: RegimeProbabilities
    model_agreement: Optional[ModelAgreement] = None
    disagreement: float = Field(ge=0, le=1, description="Model disagreement indicator")
    recommendation: str = Field(description="Trading recommendation based on regime")

    class Config:
        json_schema_extra = {
            "example": {
                "date": "2026-01-28T00:00:00",
                "regime": "Bull",
                "confidence": 0.82,
                "probabilities": {
                    "Bull": 0.82,
                    "Bear": 0.05,
                    "Stress": 0.03,
                    "Choppy": 0.08,
                    "Recovery": 0.02
                },
                "model_agreement": {
                    "xgboost": "Bull",
                    "lstm": "Bull",
                    "hmm": "Bull"
                },
                "disagreement": 0.0,
                "recommendation": "Increase exposure, trend-following strategies"
            }
        }


class RegimeHistoryItem(BaseModel):
    """Single item in regime history"""
    date: datetime
    regime: RegimeEnum
    confidence: float
    volatility_regime: Optional[str] = None
    trend_regime: Optional[str] = None
    risk_regime: Optional[str] = None


class RegimeHistoryResponse(BaseModel):
    """Regime history response"""
    start_date: datetime
    end_date: datetime
    count: int
    history: List[RegimeHistoryItem]


class PredictionResponse(BaseModel):
    """N-day ahead prediction response"""
    prediction_date: datetime
    target_date: datetime
    forecast_horizon: int = Field(description="Days ahead predicted")
    regime: RegimeEnum
    confidence: float
    probabilities: RegimeProbabilities


class FeatureValue(BaseModel):
    """Single feature value"""
    name: str
    value: float
    percentile: Optional[float] = None
    description: Optional[str] = None


class FeatureResponse(BaseModel):
    """Current feature values response"""
    date: datetime
    features: List[FeatureValue]

    class Config:
        json_schema_extra = {
            "example": {
                "date": "2026-01-28T00:00:00",
                "features": [
                    {"name": "realized_vol_20d", "value": 0.22, "percentile": 45.2,
                     "description": "20-day realized volatility"},
                    {"name": "return_20d", "value": 0.05, "percentile": 72.1,
                     "description": "20-day return"},
                    {"name": "usdtry_momentum_20d", "value": 0.02, "percentile": 55.0,
                     "description": "USD/TRY 20-day momentum"}
                ]
            }
        }


class BacktestRequest(BaseModel):
    """Backtest request parameters"""
    start_date: date
    end_date: Optional[date] = None
    strategy: str = Field(default="regime_filter", description="Strategy type")
    avoid_regimes: List[RegimeEnum] = Field(
        default=[RegimeEnum.STRESS, RegimeEnum.BEAR],
        description="Regimes to avoid"
    )
    allocation_weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Position size per regime (e.g., {'Bull': 1.5, 'Choppy': 0.5})"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "start_date": "2020-01-01",
                "end_date": "2025-12-31",
                "strategy": "regime_rotation",
                "avoid_regimes": ["Stress"],
                "allocation_weights": {
                    "Bull": 1.5,
                    "Bear": 0.2,
                    "Stress": 0.0,
                    "Choppy": 0.5,
                    "Recovery": 1.0
                }
            }
        }


class BacktestMetrics(BaseModel):
    """Backtest performance metrics"""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: Optional[float] = None
    max_drawdown: float
    win_rate: float
    n_trades: int


class BacktestResponse(BaseModel):
    """Backtest results response"""
    strategy: str
    start_date: date
    end_date: date
    metrics: BacktestMetrics
    regime_performance: Dict[str, BacktestMetrics]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    models_loaded: Dict[str, bool]
    last_update: Optional[datetime] = None
    uptime_seconds: float


class AlertConfig(BaseModel):
    """Alert configuration"""
    email: Optional[str] = None
    regime_change: bool = True
    stress_warning: bool = True
    high_disagreement: bool = True
    min_confidence: float = Field(default=0.7, ge=0, le=1)


class WebSocketMessage(BaseModel):
    """WebSocket message format"""
    type: str = Field(description="Message type: regime_update, alert, heartbeat")
    timestamp: datetime
    data: Dict


class RegimeChangeAlert(BaseModel):
    """Regime change alert"""
    previous_regime: RegimeEnum
    new_regime: RegimeEnum
    confidence: float
    timestamp: datetime
    message: str
