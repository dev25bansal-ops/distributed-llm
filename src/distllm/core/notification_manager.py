"""Notification System — Slack, Discord, email, HTTP notifications.

Provides a unified interface for sending notifications through multiple
channels. Supports templating, severity levels, and rate limiting to
prevent alert storms.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger


class NotificationSeverity(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class NotificationChannel(Enum):
    SLACK = "slack"
    DISCORD = "discord"
    EMAIL = "email"
    HTTP = "http"
    CONSOLE = "console"


@dataclass
class Notification:
    """A single notification message."""
    title: str
    message: str
    severity: NotificationSeverity = NotificationSeverity.INFO
    channel: NotificationChannel = NotificationChannel.CONSOLE
    source: str = ""
    timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NotificationChannelConfig:
    """Configuration for a notification channel."""
    channel: NotificationChannel
    enabled: bool = True
    webhook_url: str = ""
    api_key: str = ""
    from_address: str = ""
    to_addresses: list[str] = field(default_factory=list)
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    rate_limit_per_minute: int = 10
    min_severity: NotificationSeverity = NotificationSeverity.INFO


class NotificationManager:
    """Multi-channel notification dispatcher.

    Usage:
        nm = NotificationManager()
        nm.configure_slack("https://hooks.slack.com/...")
        nm.configure_discord("https://discord.com/api/webhooks/...")
        nm.send(Notification(
            title="Node Failed",
            message="worker-3 went offline",
            severity=NotificationSeverity.ERROR,
        ))
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._channels: dict[NotificationChannel, NotificationChannelConfig] = {
            NotificationChannel.CONSOLE: NotificationChannelConfig(
                channel=NotificationChannel.CONSOLE,
            ),
        }
        self._sent: list[Notification] = []
        self._rate_limiters: dict[str, list[float]] = {}

    # ── Channel configuration ───────────────────────────────────────────

    def configure_slack(
        self, webhook_url: str,
        min_severity: NotificationSeverity = NotificationSeverity.WARNING,
    ) -> None:
        self._channels[NotificationChannel.SLACK] = NotificationChannelConfig(
            channel=NotificationChannel.SLACK,
            webhook_url=webhook_url,
            min_severity=min_severity,
        )

    def configure_discord(
        self, webhook_url: str,
        min_severity: NotificationSeverity = NotificationSeverity.WARNING,
    ) -> None:
        self._channels[NotificationChannel.DISCORD] = NotificationChannelConfig(
            channel=NotificationChannel.DISCORD,
            webhook_url=webhook_url,
            min_severity=min_severity,
        )

    def configure_email(
        self, from_address: str, to_addresses: list[str],
        smtp_host: str, smtp_port: int = 587,
        smtp_user: str = "", smtp_password: str = "",
        min_severity: NotificationSeverity = NotificationSeverity.ERROR,
    ) -> None:
        self._channels[NotificationChannel.EMAIL] = NotificationChannelConfig(
            channel=NotificationChannel.EMAIL,
            from_address=from_address,
            to_addresses=to_addresses,
            smtp_host=smtp_host,
            smtp_port=smtp_port,
            smtp_user=smtp_user,
            smtp_password=smtp_password,
            min_severity=min_severity,
        )

    def configure_http(
        self, webhook_url: str,
        min_severity: NotificationSeverity = NotificationSeverity.INFO,
    ) -> None:
        self._channels[NotificationChannel.HTTP] = NotificationChannelConfig(
            channel=NotificationChannel.HTTP,
            webhook_url=webhook_url,
            min_severity=min_severity,
        )

    def enable_channel(self, channel: NotificationChannel) -> None:
        if channel in self._channels:
            self._channels[channel].enabled = True

    def disable_channel(self, channel: NotificationChannel) -> None:
        if channel in self._channels:
            self._channels[channel].enabled = False

    # ── Sending ─────────────────────────────────────────────────────────

    def send(self, notification: Notification) -> bool:
        """Send a notification through its configured channel.

        Applies rate limiting and severity filtering.

        Returns:
            True if the notification was sent (or queued).
        """
        channel = notification.channel
        config = self._channels.get(channel)
        if config is None or not config.enabled:
            return False

        # Severity check
        severity_order = [s.value for s in NotificationSeverity]
        if severity_order.index(notification.severity.value) < severity_order.index(config.min_severity.value):
            return False

        # Rate limit check
        if not self._check_rate_limit(config):
            logger.warning(f"Rate limit exceeded for {channel.value}")
            return False

        notification.timestamp = time.time()
        self._sent.append(notification)

        try:
            if channel == NotificationChannel.SLACK:
                return self._send_slack(notification, config)
            elif channel == NotificationChannel.DISCORD:
                return self._send_discord(notification, config)
            elif channel == NotificationChannel.EMAIL:
                return self._send_email(notification, config)
            elif channel == NotificationChannel.HTTP:
                return self._send_http(notification, config)
            elif channel == NotificationChannel.CONSOLE:
                self._send_console(notification)
                return True
        except Exception as e:
            logger.error(f"Failed to send {channel.value} notification: {e}")
            return False
        return False

    def send_alert(
        self,
        title: str,
        message: str,
        severity: NotificationSeverity = NotificationSeverity.ERROR,
        channel: NotificationChannel = NotificationChannel.SLACK,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Convenience: create and send a notification in one call."""
        return self.send(Notification(
            title=title, message=message, severity=severity,
            channel=channel, metadata=metadata or {},
        ))

    # ── Queries ─────────────────────────────────────────────────────────

    def recent(self, limit: int = 50, severity: NotificationSeverity | None = None) -> list[Notification]:
        """Return recent notifications, optionally filtered by severity."""
        result = self._sent[-limit:]
        if severity:
            result = [n for n in result if n.severity == severity]
        return result

    def is_channel_healthy(self, channel: NotificationChannel) -> bool:
        config = self._channels.get(channel)
        if config is None:
            return False
        return config.enabled

    # ── Channel implementations ─────────────────────────────────────────

    def _send_slack(self, notification: Notification, config: NotificationChannelConfig) -> bool:
        import httpx
        color_map = {
            "debug": "#808080", "info": "#2196F3",
            "warning": "#FF9800", "error": "#F44336", "critical": "#9C27B0",
        }
        payload = {
            "attachments": [{
                "color": color_map.get(notification.severity.value, "#808080"),
                "title": notification.title,
                "text": notification.message,
                "fields": [
                    {"title": "Severity", "value": notification.severity.value, "short": True},
                    {"title": "Source", "value": notification.source, "short": True},
                ],
                "footer": "DistLLM Notification",
                "ts": int(notification.timestamp or time.time()),
            }]
        }
        resp = httpx.post(config.webhook_url, json=payload, timeout=5.0)
        return resp.status_code < 500

    def _send_discord(self, notification: Notification, config: NotificationChannelConfig) -> bool:
        import httpx
        color_map = {
            "debug": 0x808080, "info": 0x2196F3,
            "warning": 0xFF9800, "error": 0xF44336, "critical": 0x9C27B0,
        }
        payload = {
            "embeds": [{
                "title": notification.title,
                "description": notification.message,
                "color": color_map.get(notification.severity.value, 0x808080),
                "footer": {"text": f"DistLLM · {notification.severity.value}"},
                "timestamp": datetime.utcnow().isoformat() if notification.timestamp else None,
            }]
        }
        resp = httpx.post(config.webhook_url, json=payload, timeout=5.0)
        return resp.status_code < 500

    def _send_email(self, notification: Notification, config: NotificationChannelConfig) -> bool:
        import smtplib
        from email.mime.text import MIMEText
        msg = MIMEText(
            f"Severity: {notification.severity.value}\n"
            f"Source: {notification.source}\n"
            f"Time: {datetime.fromtimestamp(notification.timestamp or time.time()).isoformat()}\n\n"
            f"{notification.message}"
        )
        msg["Subject"] = f"[DistLLM {notification.severity.value.upper()}] {notification.title}"
        msg["From"] = config.from_address
        msg["To"] = ", ".join(config.to_addresses)
        with smtplib.SMTP(config.smtp_host, config.smtp_port) as server:
            if config.smtp_user:
                server.starttls()
                server.login(config.smtp_user, config.smtp_password)
            server.send_message(msg)
        return True

    def _send_http(self, notification: Notification, config: NotificationChannelConfig) -> bool:
        import httpx
        payload = {
            "title": notification.title,
            "message": notification.message,
            "severity": notification.severity.value,
            "source": notification.source,
            "timestamp": notification.timestamp or time.time(),
            "metadata": notification.metadata,
        }
        resp = httpx.post(config.webhook_url, json=payload, timeout=5.0)
        return resp.status_code < 500

    def _send_console(self, notification: Notification) -> None:
        level = notification.severity.value.upper()
        logger.log(level, f"[{notification.source}] {notification.title}: {notification.message}")

    # ── Rate limiting ───────────────────────────────────────────────────

    def _check_rate_limit(self, config: NotificationChannelConfig) -> bool:
        key = config.channel.value
        now = time.time()
        with self._lock:
            timestamps = self._rate_limiters.get(key, [])
            timestamps = [t for t in timestamps if now - t < 60.0]
            if len(timestamps) >= config.rate_limit_per_minute:
                return False
            timestamps.append(now)
            self._rate_limiters[key] = timestamps
            return True


# Local import for email timestamp
from datetime import datetime
