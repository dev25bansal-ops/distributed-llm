"""Budget alerter for cost-optimized cloud inference.

Monitors spend against budget limits and sends alerts through
multiple channels (log, webhook, Slack) with hysteresis.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from loguru import logger


class AlertLevel(str, Enum):
    WARNING = "warning"       # 80% of budget
    CRITICAL = "critical"     # 90% of budget
    EXCEEDED = "exceeded"     # 100%+ of budget


class AlertChannel(str, Enum):
    LOG = "log"
    WEBHOOK = "webhook"
    SLACK = "slack"
    EMAIL = "email"


@dataclass
class BudgetAlert:
    """A budget alert event."""
    level: AlertLevel
    current_cost: float
    budget_limit: float
    percent_used: float
    timestamp: float = 0.0
    message: str = ""

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if not self.message:
            self.message = (
                f"Budget {self.level.value}: ${self.current_cost:.2f} / "
                f"${self.budget_limit:.2f} ({self.percent_used:.1f}%)"
            )


class BudgetAlerter:
    """Monitors spend and sends alerts on budget threshold breach.

    Uses hysteresis to avoid alert storms:
    - WARNING at 80%
    - CRITICAL at 90%
    - EXCEEDED at 100%+
    Each level fires only once per budget period.
    """

    THRESHOLDS = {
        AlertLevel.WARNING: 0.80,
        AlertLevel.CRITICAL: 0.90,
        AlertLevel.EXCEEDED: 1.00,
    }

    def __init__(
        self,
        channels: list[AlertChannel] | None = None,
        webhook_url: str = "",
    ) -> None:
        self.channels = channels or [AlertChannel.LOG]
        self.webhook_url = webhook_url
        self._fired_alerts: set[AlertLevel] = set()
        self._alert_history: list[BudgetAlert] = []
        self._budget_limit: float = 0.0

    def set_budget(self, budget_per_hour: float) -> None:
        """Set the hourly budget limit."""
        self._budget_limit = budget_per_hour
        self._fired_alerts.clear()  # Reset on budget change

    def check_budget(self, current_cost: float) -> BudgetAlert | None:
        """Check if current cost triggers any alert levels.

        Args:
            current_cost: Current accumulated cost.

        Returns:
            BudgetAlert if a threshold was crossed, None otherwise.
        """
        if self._budget_limit <= 0:
            return None

        percent = current_cost / self._budget_limit

        # Check thresholds in order (highest first)
        for level in [AlertLevel.EXCEEDED, AlertLevel.CRITICAL, AlertLevel.WARNING]:
            threshold = self.THRESHOLDS[level]
            if percent >= threshold and level not in self._fired_alerts:
                alert = BudgetAlert(
                    level=level,
                    current_cost=current_cost,
                    budget_limit=self._budget_limit,
                    percent_used=percent * 100,
                )
                self._fire_alert(alert)
                self._fired_alerts.add(level)
                return alert

        return None

    def reset(self) -> None:
        """Reset alert state (e.g., for new budget period)."""
        self._fired_alerts.clear()

    def _fire_alert(self, alert: BudgetAlert) -> None:
        """Send alert through configured channels."""
        self._alert_history.append(alert)

        for channel in self.channels:
            if channel == AlertChannel.LOG:
                self._send_log(alert)
            elif channel == AlertChannel.WEBHOOK:
                self._send_webhook(alert)
            elif channel == AlertChannel.SLACK:
                self._send_slack(alert)
            elif channel == AlertChannel.EMAIL:
                self._send_email(alert)

    def _send_log(self, alert: BudgetAlert) -> None:
        level_map = {
            AlertLevel.WARNING: logger.warning,
            AlertLevel.CRITICAL: logger.error,
            AlertLevel.EXCEEDED: logger.critical,
        }
        level_map.get(alert.level, logger.warning)(alert.message)

    def _send_webhook(self, alert: BudgetAlert) -> None:
        if not self.webhook_url:
            return
        import urllib.request
        import json

        payload = json.dumps({
            "level": alert.level.value,
            "message": alert.message,
            "current_cost": alert.current_cost,
            "budget_limit": alert.budget_limit,
            "percent_used": alert.percent_used,
        }).encode()

        try:
            req = urllib.request.Request(
                self.webhook_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:
            logger.debug(f"Webhook alert failed: {e}")

    def _send_slack(self, alert: BudgetAlert) -> None:
        # Slack uses webhook format
        self._send_webhook(alert)

    def _send_email(self, alert: BudgetAlert) -> None:
        logger.debug(f"Email alert (not implemented): {alert.message}")

    def get_alert_history(self, limit: int = 20) -> list[BudgetAlert]:
        """Return recent alerts."""
        return self._alert_history[-limit:]
