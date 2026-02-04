"""
Alert system for regime notifications

Provides email alerts for:
- Regime changes
- Stress warnings
- Model disagreement
- Custom triggers
"""

from .email_alerts import EmailAlertSender, send_regime_alert
from .alert_config import AlertConfig, load_alert_config

__all__ = ['EmailAlertSender', 'send_regime_alert', 'AlertConfig', 'load_alert_config']
