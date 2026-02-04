"""
Configuration parameters for the BIST Regime Filter System
"""

from pathlib import Path

# Get absolute paths (works from any directory)
_CONFIG_FILE = Path(__file__).resolve()
_REGIME_FILTER_DIR = _CONFIG_FILE.parent
_PROJECT_ROOT = _REGIME_FILTER_DIR.parent

# Data paths (absolute)
DATA_DIR = str(_REGIME_FILTER_DIR / "data")
XU100_FILE = "xu100_prices.csv"
BIST_PRICES_FILE = "bist_prices_full.csv"

# Yahoo Finance tickers
XU100_TICKER = "XU100.IS"
USDTRY_TICKER = "TRY=X"

# Feature calculation windows
VOLATILITY_WINDOWS = {
    'short': 20,  # 20 days
    'long': 60    # 60 days
}

MOMENTUM_WINDOWS = {
    'short': [20, 60],      # 1-3 months
    'long': [120, 252]      # 6-12 months
}

DRAWDOWN_WINDOWS = [20, 60]

# Percentile thresholds for regime classification
# RECALIBRATED (2026-01-29) to fix regime imbalance and improve Bear detection
# Historical volatility percentiles: 20th=15.95%, 50th=21.24%, 80th=28.73%, 95th=43.03%
VOLATILITY_PERCENTILES = {
    'low': 25,      # 0-25%: Low volatility (below ~17% annualized)
    'mid': 75,      # 25-75%: Mid volatility (17-28% annualized)
    'high': 92      # 75-92%: High volatility (28-40% annualized), 92%+: Stress (>40%)
}

TREND_THRESHOLDS = {
    'up': 0.015,       # 1.5% threshold for uptrend (more sensitive)
    'down': -0.015,    # -1.5% threshold for downtrend (more sensitive)
    'ma_slope_days': 20  # Days for MA slope calculation
}

RISK_PERCENTILES = {
    'risk_on': 35,     # Below 35th percentile = risk-on
    'risk_off': 65     # Above 65th percentile = risk-off (more sensitive)
}

# Liquidity thresholds (PERCENTILE-BASED for adaptive behavior)
# Using percentiles like volatility does - adapts to changing market conditions
LIQUIDITY_PERCENTILES = {
    'very_low': 20,     # 0-20%: Very low liquidity
    'low': 40,          # 20-40%: Low liquidity  
    'normal': 40        # 40%+: Normal liquidity
}

# Fallback static thresholds (used if percentile calculation fails)
LIQUIDITY_THRESHOLDS_STATIC = {
    'min_turnover': 1.5e13,     # Minimum daily turnover (~20th percentile historically)
    'low_turnover': 3.0e13,     # Below this is "low liquidity" (~40th percentile)
    'normal_turnover': 5.0e13   # Above this is "normal liquidity" (~55th percentile)
}

# USD/TRY data
USDTRY_TICKER = "TRY=X"  # Yahoo Finance ticker
USDTRY_WINDOWS = {
    'trend': 20,
    'vol': 20
}

# Lookback period for percentile calculations (in days)
PERCENTILE_LOOKBACK = 252  # 1 year

# Output settings (absolute paths)
OUTPUT_DIR = str(_REGIME_FILTER_DIR / "outputs")
DASHBOARD_FILE = "regime_dashboard.html"
REGIME_JSON = "regime_labels.json"
FEATURES_CSV = "regime_features.csv"

# ============================================================
# ENHANCED CONFIGURATION (v2.0)
# ============================================================

# TCMB EVDS API Settings
TCMB_EVDS_API_KEY = None  # Set via environment: TCMB_EVDS_API_KEY
TCMB_CACHE_DIR = str(_REGIME_FILTER_DIR / "data" / "tcmb_cache")

# TCMB Series Codes (for reference)
TCMB_SERIES = {
    'yield_2y': 'TP.DK.TRE.YSKG.02',
    'yield_5y': 'TP.DK.TRE.YSKG.05',
    'yield_10y': 'TP.DK.TRE.YSKG.10',
    'policy_rate': 'TP.PH.S01',
    'cpi_annual': 'TP.FG.J0',
    'inflation_exp': 'TP.BEK.S01.A',
}

# LSTM Model Settings
LSTM_CONFIG = {
    'sequence_length': 20,       # Days of history for each prediction
    'forecast_horizon': 5,       # Days ahead to predict
    'hidden_size': 64,           # LSTM hidden layer size
    'num_layers': 2,             # Number of LSTM layers
    'dropout': 0.2,              # Dropout rate
    'learning_rate': 0.001,      # Adam learning rate
    'batch_size': 32,            # Training batch size
    'epochs': 50,                # Maximum training epochs
    'early_stopping_patience': 10  # Early stopping patience
}

# Ensemble Model Settings
ENSEMBLE_CONFIG = {
    'weights': {
        'xgboost': 0.40,         # XGBoost weight
        'lstm': 0.35,            # LSTM weight
        'hmm': 0.25              # HMM weight
    },
    'dynamic_weights': True,     # Adjust weights based on rolling accuracy
    'weight_update_window': 60   # Rolling window for accuracy calculation
}

# API Settings
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'reload': False,             # Set True for development
    'cors_origins': ['*'],       # Allowed CORS origins
    'docs_url': '/docs',         # Swagger docs URL
    'redoc_url': '/redoc'        # ReDoc URL
}

# Scheduler Settings (Istanbul timezone)
SCHEDULER_CONFIG = {
    'timezone': 'Europe/Istanbul',
    'daily_update_hour': 18,     # Run at 18:00 (after market close)
    'daily_update_minute': 0,
    'heartbeat_interval': 5,     # Minutes between heartbeats
    'market_hours': {
        'start': 9,              # Market opens at 09:00
        'end': 18                # Market closes at 18:00
    }
}

# Alert Settings
ALERT_CONFIG = {
    'enabled': True,
    'alert_on_regime_change': True,
    'alert_on_stress': True,
    'alert_on_high_disagreement': True,
    'alert_on_volatility_spike': True,
    'stress_confidence_threshold': 0.70,
    'disagreement_threshold': 0.30,
    'volatility_percentile_threshold': 95.0,
    'min_alert_interval_minutes': 60,
    'max_alerts_per_day': 10
}

# Model Paths
MODEL_PATHS = {
    'ensemble': str(_REGIME_FILTER_DIR / "outputs" / "ensemble_model"),
    'lstm': str(_REGIME_FILTER_DIR / "outputs" / "lstm_model.pt"),
    'xgboost': str(_REGIME_FILTER_DIR / "outputs" / "xgboost_model.pkl"),
    'hmm': str(_REGIME_FILTER_DIR / "outputs" / "hmm_model.pkl")
}

# Simplified Regime Mapping
REGIME_RECOMMENDATIONS = {
    'Bull': 'Increase exposure, trend-following strategies work well',
    'Bear': 'Reduce exposure, consider defensive positioning',
    'Stress': 'Minimize exposure, use wide stops, avoid illiquid names',
    'Choppy': 'Range trading, mean reversion strategies, reduce position size',
    'Recovery': 'Gradual re-entry, focus on quality names'
}

# Feature Groups for Different Models
FEATURE_GROUPS = {
    'lstm_features': [
        'return_20d', 'realized_vol_20d', 'max_drawdown_20d',
        'usdtry_momentum_20d', 'volume_ratio',
        'vix', 'vix_change_20d', 'dxy_change_20d', 'spx_change_20d',
        'viop30_proxy', 'cds_proxy', 'yield_curve_slope', 'real_rate', 'iv_rv_spread'
    ],
    'xgboost_features': 'all',  # Use all available features
    'hmm_features': [
        'return_20d', 'realized_vol_20d', 'max_drawdown_20d',
        'usdtry_momentum_20d', 'volume_ratio'
    ]
}
