"""Alertmanager routing configuration generator."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml


@dataclass
class RouteConfig:
    """A single routing rule in alertmanager."""
    match: Dict[str, str] = field(default_factory=dict)
    match_re: Dict[str, str] = field(default_factory=dict)
    receiver: str = "default"
    continue_: bool = False
    group_by: List[str] = field(default_factory=lambda: ["alertname", "node_id"])
    group_wait: str = "30s"
    group_interval: str = "5m"
    repeat_interval: str = "4h"

    def to_dict(self) -> dict:
        result = {"receiver": self.receiver}
        if self.match:
            result["match"] = self.match
        if self.match_re:
            result["match_re"] = self.match_re
        result["continue"] = self.continue_
        result["group_by"] = self.group_by
        result["group_wait"] = self.group_wait
        result["group_interval"] = self.group_interval
        result["repeat_interval"] = self.repeat_interval
        return result


@dataclass
class ReceiverConfig:
    """A notification receiver in alertmanager."""
    name: str
    email_configs: List[Dict] = field(default_factory=list)
    slack_configs: List[Dict] = field(default_factory=list)
    webhook_configs: List[Dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        result = {"name": self.name}
        if self.email_configs:
            result["email_configs"] = self.email_configs
        if self.slack_configs:
            result["slack_configs"] = self.slack_configs
        if self.webhook_configs:
            result["webhook_configs"] = self.webhook_configs
        return result


def generate_alertmanager_config(
    smtp_host: str = "localhost",
    smtp_port: int = 25,
    smtp_from: str = "alertmanager@distllm.local",
    slack_webhook_url: str = "",
    webhook_url: str = "",
    email_recipients: Optional[List[str]] = None,
) -> str:
    """Generate an alertmanager YAML configuration.

    Routes critical alerts to email/slack immediately, warnings with delay,
    and info-level alerts to a catch-all receiver.
    """
    receivers = [
        ReceiverConfig(name="default").to_dict(),
    ]

    if email_recipients:
        receivers.append(ReceiverConfig(
            name="email-critical",
            email_configs=[{
                "to": ", ".join(email_recipients),
                "from": smtp_from,
                "smarthost": f"{smtp_host}:{smtp_port}",
            }],
        ).to_dict())

    if slack_webhook_url:
        receivers.append(ReceiverConfig(
            name="slack-critical",
            slack_configs=[{
                "api_url": slack_webhook_url,
                "channel": "#distllm-alerts",
            }],
        ).to_dict())

    if webhook_url:
        receivers.append(ReceiverConfig(
            name="webhook-all",
            webhook_configs=[{"url": webhook_url}],
        ).to_dict())

    routes = [
        RouteConfig(
            match={"severity": "critical"},
            receiver="email-critical" if email_recipients else "default",
            group_wait="10s",
            repeat_interval="1h",
        ).to_dict(),
        RouteConfig(
            match={"severity": "warning"},
            receiver="slack-critical" if slack_webhook_url else "default",
            group_wait="5m",
            repeat_interval="4h",
        ).to_dict(),
    ]

    config = {
        "global": {
            "resolve_timeout": "5m",
            "smtp_from": smtp_from,
            "smtp_smarthost": f"{smtp_host}:{smtp_port}",
        },
        "route": {
            "receiver": "default",
            "group_by": ["alertname", "node_id"],
            "group_wait": "30s",
            "group_interval": "5m",
            "repeat_interval": "4h",
            "routes": routes,
        },
        "receivers": receivers,
    }

    return yaml.dump(config, default_flow_style=False, sort_keys=False)
