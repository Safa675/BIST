"""
Alert configuration management

Supports configuration via:
1. Environment variables
2. Config file (JSON/YAML)
3. Direct initialization
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import warnings


@dataclass
class SMTPConfig:
    """SMTP server configuration"""
    host: str = "smtp.gmail.com"
    port: int = 587
    username: str = ""
    password: str = ""
    use_tls: bool = True
    use_ssl: bool = False

    @classmethod
    def from_env(cls) -> 'SMTPConfig':
        """Load from environment variables"""
        return cls(
            host=os.environ.get('SMTP_HOST', 'smtp.gmail.com'),
            port=int(os.environ.get('SMTP_PORT', 587)),
            username=os.environ.get('SMTP_USERNAME', ''),
            password=os.environ.get('SMTP_PASSWORD', ''),
            use_tls=os.environ.get('SMTP_USE_TLS', 'true').lower() == 'true',
            use_ssl=os.environ.get('SMTP_USE_SSL', 'false').lower() == 'true'
        )


@dataclass
class AlertConfig:
    """
    Complete alert configuration

    Attributes:
        enabled: Whether alerts are enabled
        smtp: SMTP server configuration
        sender_email: Email address to send from
        recipients: List of recipient email addresses
        alert_on_regime_change: Alert when regime changes
        alert_on_stress: Alert when entering Stress regime
        alert_on_high_disagreement: Alert when models disagree
        stress_confidence_threshold: Min confidence to trigger stress alert
        disagreement_threshold: Min disagreement to trigger alert
        include_features: Include key features in alert
        include_model_details: Include per-model predictions
    """
    enabled: bool = True
    smtp: SMTPConfig = field(default_factory=SMTPConfig)
    sender_email: str = ""
    recipients: List[str] = field(default_factory=list)

    # Alert triggers
    alert_on_regime_change: bool = True
    alert_on_stress: bool = True
    alert_on_high_disagreement: bool = True
    alert_on_volatility_spike: bool = True

    # Thresholds
    stress_confidence_threshold: float = 0.7
    disagreement_threshold: float = 0.3
    volatility_percentile_threshold: float = 95.0

    # Content options
    include_features: bool = True
    include_model_details: bool = True
    include_recommendation: bool = True

    # Rate limiting
    min_alert_interval_minutes: int = 60
    max_alerts_per_day: int = 10

    @classmethod
    def from_env(cls) -> 'AlertConfig':
        """Load configuration from environment variables"""
        recipients_str = os.environ.get('ALERT_RECIPIENTS', '')
        recipients = [r.strip() for r in recipients_str.split(',') if r.strip()]

        return cls(
            enabled=os.environ.get('ALERTS_ENABLED', 'true').lower() == 'true',
            smtp=SMTPConfig.from_env(),
            sender_email=os.environ.get('ALERT_SENDER_EMAIL', ''),
            recipients=recipients,
            alert_on_regime_change=os.environ.get('ALERT_ON_REGIME_CHANGE', 'true').lower() == 'true',
            alert_on_stress=os.environ.get('ALERT_ON_STRESS', 'true').lower() == 'true',
            alert_on_high_disagreement=os.environ.get('ALERT_ON_DISAGREEMENT', 'true').lower() == 'true',
            stress_confidence_threshold=float(os.environ.get('STRESS_CONFIDENCE_THRESHOLD', 0.7)),
            disagreement_threshold=float(os.environ.get('DISAGREEMENT_THRESHOLD', 0.3))
        )

    @classmethod
    def from_file(cls, filepath: str) -> 'AlertConfig':
        """Load configuration from JSON file"""
        filepath = Path(filepath)

        if not filepath.exists():
            warnings.warn(f"Config file not found: {filepath}, using defaults")
            return cls()

        with open(filepath, 'r') as f:
            data = json.load(f)

        # Parse SMTP config
        smtp_data = data.pop('smtp', {})
        smtp = SMTPConfig(**smtp_data)

        return cls(smtp=smtp, **data)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'enabled': self.enabled,
            'smtp': {
                'host': self.smtp.host,
                'port': self.smtp.port,
                'username': self.smtp.username,
                'use_tls': self.smtp.use_tls,
                'use_ssl': self.smtp.use_ssl
            },
            'sender_email': self.sender_email,
            'recipients': self.recipients,
            'alert_on_regime_change': self.alert_on_regime_change,
            'alert_on_stress': self.alert_on_stress,
            'alert_on_high_disagreement': self.alert_on_high_disagreement,
            'alert_on_volatility_spike': self.alert_on_volatility_spike,
            'stress_confidence_threshold': self.stress_confidence_threshold,
            'disagreement_threshold': self.disagreement_threshold,
            'volatility_percentile_threshold': self.volatility_percentile_threshold,
            'include_features': self.include_features,
            'include_model_details': self.include_model_details,
            'include_recommendation': self.include_recommendation,
            'min_alert_interval_minutes': self.min_alert_interval_minutes,
            'max_alerts_per_day': self.max_alerts_per_day
        }

    def save(self, filepath: str):
        """Save configuration to JSON file"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        print(f"Alert config saved to {filepath}")

    def validate(self) -> List[str]:
        """Validate configuration, return list of issues"""
        issues = []

        if self.enabled:
            if not self.smtp.username:
                issues.append("SMTP username not configured")
            if not self.smtp.password:
                issues.append("SMTP password not configured")
            if not self.sender_email:
                issues.append("Sender email not configured")
            if not self.recipients:
                issues.append("No recipients configured")

        return issues

    def is_valid(self) -> bool:
        """Check if configuration is valid for sending alerts"""
        return len(self.validate()) == 0


def load_alert_config() -> AlertConfig:
    """
    Load alert configuration from multiple sources

    Priority:
    1. Environment variables
    2. Config file (if exists)
    3. Defaults
    """
    # Try environment variables first
    config = AlertConfig.from_env()

    # Check if valid
    if config.is_valid():
        return config

    # Try config file
    config_paths = [
        Path(__file__).parent.parent / "alert_config.json",
        Path(__file__).parent / "config.json",
        Path.home() / ".config" / "regime_filter" / "alerts.json"
    ]

    for path in config_paths:
        if path.exists():
            try:
                config = AlertConfig.from_file(path)
                if config.is_valid():
                    return config
            except Exception as e:
                warnings.warn(f"Could not load config from {path}: {e}")

    # Return default (may not be valid for sending)
    return config


# Default config templates
DEFAULT_GMAIL_CONFIG = {
    "enabled": True,
    "smtp": {
        "host": "smtp.gmail.com",
        "port": 587,
        "username": "your-email@gmail.com",
        "password": "your-app-password",
        "use_tls": True,
        "use_ssl": False
    },
    "sender_email": "your-email@gmail.com",
    "recipients": ["recipient@example.com"],
    "alert_on_regime_change": True,
    "alert_on_stress": True,
    "alert_on_high_disagreement": True,
    "stress_confidence_threshold": 0.7,
    "disagreement_threshold": 0.3
}


if __name__ == "__main__":
    print("="*60)
    print("ALERT CONFIGURATION")
    print("="*60)

    # Show environment variable names
    print("\nEnvironment Variables:")
    print("  ALERTS_ENABLED=true")
    print("  SMTP_HOST=smtp.gmail.com")
    print("  SMTP_PORT=587")
    print("  SMTP_USERNAME=your-email@gmail.com")
    print("  SMTP_PASSWORD=your-app-password")
    print("  SMTP_USE_TLS=true")
    print("  ALERT_SENDER_EMAIL=your-email@gmail.com")
    print("  ALERT_RECIPIENTS=recipient1@example.com,recipient2@example.com")

    print("\nFor Gmail:")
    print("  1. Enable 2-factor authentication")
    print("  2. Generate an App Password: https://myaccount.google.com/apppasswords")
    print("  3. Use the App Password as SMTP_PASSWORD")

    # Load current config
    config = load_alert_config()
    issues = config.validate()

    if issues:
        print("\nConfiguration Issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nConfiguration valid!")

    print("\n" + "="*60)
