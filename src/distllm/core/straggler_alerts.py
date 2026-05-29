"""Straggler Alerting — webhook/email/Slack notifications.

Sends alerts when stragglers are detected. Integrates with
StragglerDetector via callback.

Usage::

    alerter = StragglerAlerter(webhook_url="https://hooks.slack.com/...")
    detector = StragglerDetector(on_straggler_cb=alerter.alert)
"""

from __future__ import annotations

import json
import time
import threading
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class AlertRecord:
    """Record of a sent alert."""
    timestamp: float = field(default_factory=time.time)
    node_id: str = ""
    severity: str = ""
    channel: str = ""  # "webhook", "email", "log"
    success: bool = True
    error: str = ""


class StragglerAlerter:
    """Sends straggler detection alerts via configured channels.

    Usage::

        alerter = StragglerAlerter(
            webhook_url="https://hooks.slack.com/services/...",
            email_smtp="smtp.gmail.com",
            email_to="team@example.com",
        )
        detector = StragglerDetector(on_straggler_cb=alerter.alert)
    """

    def __init__(
        self,
        webhook_url: str = "",
        email_smtp: str = "",
        email_to: str = "",
        email_from: str = "distllm@localhost",
        smtp_port: int = 587,
        smtp_user: str = "",
        smtp_password: str = "",
        alert_cooldown_s: float = 300.0,
        min_severity: str = "mild",
    ):
        self._webhook_url = webhook_url
        self._email_smtp = email_smtp
        self._email_to = email_to
        self._email_from = email_from
        self._smtp_port = smtp_port
        self._smtp_user = smtp_user
        self._smtp_password = smtp_password
        self._alert_cooldown_s = alert_cooldown_s
        self._min_severity = min_severity
        self._last_alerts: dict[str, float] = {}
        self._records: list[AlertRecord] = []
        self._lock = threading.Lock()

    def alert(self, report: Any) -> None:
        """Send alert for a straggler report.

        Args:
            report: StragglerReport with node_id, severity, etc.
        """
        severity = getattr(report, "severity", None)
        severity_val = severity.value if severity else "unknown"
        node_id = getattr(report, "node_id", "unknown")

        # Check minimum severity
        severity_order = {"none": 0, "mild": 1, "moderate": 2, "severe": 3}
        if severity_order.get(severity_val, 0) < severity_order.get(self._min_severity, 0):
            return

        # Check cooldown
        with self._lock:
            now = time.time()
            last = self._last_alerts.get(node_id, 0)
            if now - last < self._alert_cooldown_s:
                return
            self._last_alerts[node_id] = now

        message = self._format_message(report)

        # Send via webhook
        if self._webhook_url:
            self._send_webhook(message, node_id, severity_val)

        # Send via email
        if self._email_smtp and self._email_to:
            self._send_email(message, node_id, severity_val)

        # Always log
        logger.warning(f"Straggler alert: {message}")
        self._record(node_id, severity_val, "log", True)

    def _format_message(self, report: Any) -> str:
        node_id = getattr(report, "node_id", "?")
        severity = getattr(report, "severity", None)
        severity_val = severity.value if severity else "?"
        slowdown = getattr(report, "slowdown_factor", 0)
        action = getattr(report, "recommended_action", "?")
        avg_lat = getattr(report, "avg_latency", 0)
        p95_lat = getattr(report, "p95_latency", 0)
        return (
            f"Straggler detected: {node_id}\n"
            f"Severity: {severity_val}\n"
            f"Slowdown: {slowdown}x\n"
            f"Avg latency: {avg_lat}ms, P95: {p95_lat}ms\n"
            f"Recommended action: {action}"
        )

    def _send_webhook(self, message: str, node_id: str, severity: str) -> None:
        try:
            import httpx
            payload = {"text": message, "node_id": node_id, "severity": severity}
            resp = httpx.post(self._webhook_url, json=payload, timeout=10.0)
            resp.raise_for_status()
            self._record(node_id, severity, "webhook", True)
        except Exception as e:
            logger.debug(f"Webhook alert failed: {e}")
            self._record(node_id, severity, "webhook", False, str(e))

    def _send_email(self, message: str, node_id: str, severity: str) -> None:
        try:
            import smtplib
            from email.mime.text import MIMEText
            msg = MIMEText(message)
            msg["Subject"] = f"DistLLM Straggler Alert: {node_id} ({severity})"
            msg["From"] = self._email_from
            msg["To"] = self._email_to
            with smtplib.SMTP(self._email_smtp, self._smtp_port) as server:
                if self._smtp_user:
                    server.starttls()
                    server.login(self._smtp_user, self._smtp_password)
                server.sendmail(self._email_from, [self._email_to], msg.as_string())
            self._record(node_id, severity, "email", True)
        except Exception as e:
            logger.debug(f"Email alert failed: {e}")
            self._record(node_id, severity, "email", False, str(e))

    def _record(self, node_id: str, severity: str, channel: str, success: bool, error: str = "") -> None:
        with self._lock:
            self._records.append(AlertRecord(
                node_id=node_id, severity=severity,
                channel=channel, success=success, error=error,
            ))
            if len(self._records) > 1000:
                self._records = self._records[-1000:]

    def get_records(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "timestamp": r.timestamp,
                    "node_id": r.node_id,
                    "severity": r.severity,
                    "channel": r.channel,
                    "success": r.success,
                }
                for r in self._records[-limit:]
            ]
