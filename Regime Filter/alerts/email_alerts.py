"""
Email alert system for regime notifications

Sends email alerts when:
1. Regime changes (e.g., Bull -> Bear)
2. High-confidence stress prediction
3. Model disagreement exceeds threshold
4. Volatility spike detected
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import asyncio
from pathlib import Path

try:
    import aiosmtplib
    ASYNC_SMTP_AVAILABLE = True
except ImportError:
    ASYNC_SMTP_AVAILABLE = False

from .alert_config import AlertConfig, load_alert_config


class EmailAlertSender:
    """
    Email alert sender for regime notifications

    Features:
    - Sync and async sending
    - Rate limiting
    - HTML formatted emails
    - Alert history tracking
    """

    def __init__(self, config: AlertConfig = None):
        """
        Args:
            config: Alert configuration (loads from env/file if not provided)
        """
        self.config = config or load_alert_config()
        self.alert_history: List[Dict] = []
        self._last_alert_time: Optional[datetime] = None
        self._alerts_today = 0
        self._today = datetime.now().date()

    def _check_rate_limit(self) -> bool:
        """Check if we can send an alert (rate limiting)"""
        now = datetime.now()

        # Reset daily counter
        if now.date() != self._today:
            self._today = now.date()
            self._alerts_today = 0

        # Check daily limit
        if self._alerts_today >= self.config.max_alerts_per_day:
            print(f"Daily alert limit reached ({self.config.max_alerts_per_day})")
            return False

        # Check interval
        if self._last_alert_time:
            elapsed = (now - self._last_alert_time).total_seconds() / 60
            if elapsed < self.config.min_alert_interval_minutes:
                print(f"Alert rate limited (wait {self.config.min_alert_interval_minutes - elapsed:.0f} more minutes)")
                return False

        return True

    def _update_rate_limit(self):
        """Update rate limiting counters after sending"""
        self._last_alert_time = datetime.now()
        self._alerts_today += 1

    def _build_regime_change_email(
        self,
        previous_regime: str,
        new_regime: str,
        confidence: float,
        model_agreement: Optional[Dict[str, str]] = None,
        features: Optional[Dict[str, float]] = None,
        recommendation: Optional[str] = None
    ) -> tuple:
        """Build email subject and body for regime change alert"""

        subject = f"[BIST Regime Alert] Regime Changed: {previous_regime} → {new_regime}"

        # Build HTML body
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .header {{ background-color: #1a73e8; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; }}
                .alert-box {{ background-color: #fef3cd; border: 1px solid #ffc107; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .metrics {{ background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .model-agreement {{ margin: 10px 0; }}
                .recommendation {{ background-color: #d4edda; border: 1px solid #28a745; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .regime-bull {{ color: #28a745; font-weight: bold; }}
                .regime-bear {{ color: #dc3545; font-weight: bold; }}
                .regime-stress {{ color: #dc3545; font-weight: bold; background-color: #f8d7da; }}
                .regime-choppy {{ color: #ffc107; font-weight: bold; }}
                .regime-recovery {{ color: #17a2b8; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>BIST Regime Alert</h1>
                <p>Market Regime Change Detected</p>
            </div>

            <div class="content">
                <div class="alert-box">
                    <h2>Regime Change</h2>
                    <p><strong>Previous Regime:</strong> <span class="regime-{previous_regime.lower()}">{previous_regime}</span></p>
                    <p><strong>New Regime:</strong> <span class="regime-{new_regime.lower()}">{new_regime}</span></p>
                    <p><strong>Confidence:</strong> {confidence:.1%}</p>
                    <p><strong>Timestamp:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
        """

        # Add model agreement if available
        if model_agreement and self.config.include_model_details:
            html += """
                <div class="model-agreement">
                    <h3>Model Agreement</h3>
                    <table>
                        <tr><th>Model</th><th>Prediction</th></tr>
            """
            for model, prediction in model_agreement.items():
                html += f"<tr><td>{model.title()}</td><td class='regime-{prediction.lower()}'>{prediction}</td></tr>"
            html += "</table></div>"

        # Add key features if available
        if features and self.config.include_features:
            html += """
                <div class="metrics">
                    <h3>Key Metrics</h3>
                    <table>
                        <tr><th>Metric</th><th>Value</th></tr>
            """
            key_features = [
                ('realized_vol_20d', 'Volatility (20d)', '{:.2%}'),
                ('usdtry_momentum_20d', 'USD/TRY Momentum', '{:.2%}'),
                ('return_20d', 'XU100 Return (20d)', '{:.2%}'),
                ('viop30_proxy', 'VIOP30 (Turkish VIX)', '{:.1f}'),
                ('cds_proxy', 'CDS Proxy', '{:.2f}'),
                ('yield_curve_slope', 'Yield Curve Slope', '{:.2%}'),
            ]
            for key, label, fmt in key_features:
                if key in features and features[key] is not None:
                    try:
                        value = fmt.format(features[key])
                        html += f"<tr><td>{label}</td><td>{value}</td></tr>"
                    except (ValueError, TypeError):
                        pass
            html += "</table></div>"

        # Add recommendation
        if recommendation and self.config.include_recommendation:
            html += f"""
                <div class="recommendation">
                    <h3>Recommended Action</h3>
                    <p>{recommendation}</p>
                </div>
            """

        html += """
                <hr>
                <p style="color: #666; font-size: 12px;">
                    This is an automated alert from the BIST Regime Filter system.<br>
                    To unsubscribe, update your alert configuration.
                </p>
            </div>
        </body>
        </html>
        """

        # Plain text version
        text = f"""
BIST Regime Alert - Regime Change Detected

Previous Regime: {previous_regime}
New Regime: {new_regime}
Confidence: {confidence:.1%}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
        if recommendation:
            text += f"Recommended Action: {recommendation}\n"

        return subject, text, html

    def _build_stress_warning_email(
        self,
        confidence: float,
        features: Optional[Dict[str, float]] = None
    ) -> tuple:
        """Build email for high-confidence stress warning"""

        subject = f"[BIST Regime Alert] STRESS WARNING - Confidence {confidence:.0%}"

        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .warning {{ background-color: #f8d7da; border: 2px solid #dc3545; padding: 20px; text-align: center; }}
            </style>
        </head>
        <body>
            <div class="warning">
                <h1 style="color: #dc3545;">⚠️ STRESS REGIME WARNING ⚠️</h1>
                <p><strong>High-confidence stress regime prediction detected</strong></p>
                <p>Confidence: <strong>{confidence:.1%}</strong></p>
                <p>Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <hr>
                <p><strong>Recommended Action:</strong> Minimize exposure, use wide stops, avoid illiquid names</p>
            </div>
        </body>
        </html>
        """

        text = f"""
BIST Regime Alert - STRESS WARNING

High-confidence stress regime prediction detected!
Confidence: {confidence:.1%}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Recommended Action: Minimize exposure, use wide stops, avoid illiquid names
        """

        return subject, text, html

    def send_email(self, subject: str, text_body: str, html_body: str = None) -> bool:
        """
        Send email (synchronous)

        Args:
            subject: Email subject
            text_body: Plain text body
            html_body: HTML body (optional)

        Returns:
            True if sent successfully
        """
        if not self.config.enabled:
            print("Alerts disabled")
            return False

        if not self.config.is_valid():
            print(f"Invalid alert config: {self.config.validate()}")
            return False

        if not self._check_rate_limit():
            return False

        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.config.sender_email
            msg['To'] = ', '.join(self.config.recipients)

            # Attach text and HTML parts
            msg.attach(MIMEText(text_body, 'plain'))
            if html_body:
                msg.attach(MIMEText(html_body, 'html'))

            # Connect and send
            if self.config.smtp.use_ssl:
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL(
                    self.config.smtp.host,
                    self.config.smtp.port,
                    context=context
                ) as server:
                    server.login(self.config.smtp.username, self.config.smtp.password)
                    server.sendmail(
                        self.config.sender_email,
                        self.config.recipients,
                        msg.as_string()
                    )
            else:
                with smtplib.SMTP(self.config.smtp.host, self.config.smtp.port) as server:
                    if self.config.smtp.use_tls:
                        server.starttls()
                    server.login(self.config.smtp.username, self.config.smtp.password)
                    server.sendmail(
                        self.config.sender_email,
                        self.config.recipients,
                        msg.as_string()
                    )

            self._update_rate_limit()
            self.alert_history.append({
                'type': 'email',
                'subject': subject,
                'timestamp': datetime.now(),
                'recipients': self.config.recipients,
                'success': True
            })

            print(f"Alert sent to {len(self.config.recipients)} recipients")
            return True

        except Exception as e:
            print(f"Failed to send alert: {e}")
            self.alert_history.append({
                'type': 'email',
                'subject': subject,
                'timestamp': datetime.now(),
                'recipients': self.config.recipients,
                'success': False,
                'error': str(e)
            })
            return False

    async def send_email_async(self, subject: str, text_body: str, html_body: str = None) -> bool:
        """
        Send email (asynchronous)

        Requires aiosmtplib package.
        """
        if not ASYNC_SMTP_AVAILABLE:
            # Fall back to sync
            return self.send_email(subject, text_body, html_body)

        if not self.config.enabled or not self.config.is_valid():
            return False

        if not self._check_rate_limit():
            return False

        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.config.sender_email
            msg['To'] = ', '.join(self.config.recipients)

            msg.attach(MIMEText(text_body, 'plain'))
            if html_body:
                msg.attach(MIMEText(html_body, 'html'))

            # Send async
            await aiosmtplib.send(
                msg,
                hostname=self.config.smtp.host,
                port=self.config.smtp.port,
                username=self.config.smtp.username,
                password=self.config.smtp.password,
                start_tls=self.config.smtp.use_tls
            )

            self._update_rate_limit()
            print(f"Alert sent (async) to {len(self.config.recipients)} recipients")
            return True

        except Exception as e:
            print(f"Failed to send async alert: {e}")
            return False

    def send_regime_change_alert(
        self,
        previous_regime: str,
        new_regime: str,
        confidence: float,
        model_agreement: Optional[Dict[str, str]] = None,
        features: Optional[Dict[str, float]] = None,
        recommendation: Optional[str] = None
    ) -> bool:
        """Send regime change alert"""

        if not self.config.alert_on_regime_change:
            return False

        subject, text, html = self._build_regime_change_email(
            previous_regime, new_regime, confidence,
            model_agreement, features, recommendation
        )

        return self.send_email(subject, text, html)

    def send_stress_warning(
        self,
        confidence: float,
        features: Optional[Dict[str, float]] = None
    ) -> bool:
        """Send high-confidence stress warning"""

        if not self.config.alert_on_stress:
            return False

        if confidence < self.config.stress_confidence_threshold:
            return False

        subject, text, html = self._build_stress_warning_email(confidence, features)
        return self.send_email(subject, text, html)

    def send_disagreement_alert(
        self,
        disagreement: float,
        model_predictions: Dict[str, str]
    ) -> bool:
        """Send model disagreement alert"""

        if not self.config.alert_on_high_disagreement:
            return False

        if disagreement < self.config.disagreement_threshold:
            return False

        subject = f"[BIST Regime Alert] High Model Disagreement ({disagreement:.0%})"

        text = f"""
BIST Regime Alert - Model Disagreement

The ensemble models are showing significant disagreement.
Disagreement Level: {disagreement:.1%}

Model Predictions:
"""
        for model, pred in model_predictions.items():
            text += f"  {model}: {pred}\n"

        text += "\nThis may indicate an uncertain market environment. Consider reducing position sizes."

        return self.send_email(subject, text)


# Convenience function
def send_regime_alert(
    alert_type: str,
    **kwargs
) -> bool:
    """
    Convenience function to send alerts

    Args:
        alert_type: 'regime_change', 'stress', 'disagreement'
        **kwargs: Alert-specific parameters

    Returns:
        True if sent successfully
    """
    sender = EmailAlertSender()

    if alert_type == 'regime_change':
        return sender.send_regime_change_alert(**kwargs)
    elif alert_type == 'stress':
        return sender.send_stress_warning(**kwargs)
    elif alert_type == 'disagreement':
        return sender.send_disagreement_alert(**kwargs)
    else:
        print(f"Unknown alert type: {alert_type}")
        return False


if __name__ == "__main__":
    print("="*60)
    print("EMAIL ALERT SYSTEM - TEST")
    print("="*60)

    # Load config
    config = load_alert_config()
    issues = config.validate()

    if issues:
        print("\nConfiguration issues:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nSet environment variables or create config file to enable alerts.")
    else:
        print("\nConfiguration valid!")

        # Test send (uncomment to actually send)
        # sender = EmailAlertSender(config)
        # sender.send_regime_change_alert(
        #     previous_regime='Bull',
        #     new_regime='Stress',
        #     confidence=0.85,
        #     recommendation='Minimize exposure, use wide stops'
        # )

    print("\nUsage:")
    print("  from alerts.email_alerts import send_regime_alert")
    print("  ")
    print("  send_regime_alert('regime_change',")
    print("      previous_regime='Bull',")
    print("      new_regime='Stress',")
    print("      confidence=0.85")
    print("  )")

    print("="*60)
