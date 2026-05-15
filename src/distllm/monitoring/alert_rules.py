"""Prometheus alerting and recording rule definitions."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml


@dataclass
class AlertRule:
    """A single Prometheus alerting rule."""
    alert: str
    expr: str
    for_duration: str  # e.g., "1m", "5m"
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "alert": self.alert,
            "expr": self.expr,
            "for": self.for_duration,
            "labels": self.labels,
            "annotations": self.annotations,
        }


@dataclass
class RecordingRule:
    """A Prometheus recording rule for pre-computed metrics."""
    record: str
    expr: str
    labels: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        result = {"record": self.record, "expr": self.expr}
        if self.labels:
            result["labels"] = self.labels
        return result


@dataclass
class RuleGroup:
    """A group of alerting or recording rules."""
    name: str
    interval: Optional[str] = None
    rules: List = field(default_factory=list)  # AlertRule or RecordingRule

    def to_dict(self) -> dict:
        result = {"name": self.name, "rules": [r.to_dict() for r in self.rules]}
        if self.interval:
            result["interval"] = self.interval
        return result


def get_default_alerts() -> List[AlertRule]:
    """Return the default set of alerting rules."""
    return [
        AlertRule(
            alert="NodeDown",
            expr="distllm_node_health == -1",
            for_duration="1m",
            labels={"severity": "critical"},
            annotations={
                "summary": "Node {{ $labels.node_id }} is offline",
                "description": "Node {{ $labels.node_id }} has been offline for more than 1 minute.",
            },
        ),
        AlertRule(
            alert="NodeDegraded",
            expr="distllm_node_health == 1",
            for_duration="5m",
            labels={"severity": "warning"},
            annotations={
                "summary": "Node {{ $labels.node_id }} is degraded",
                "description": "Node {{ $labels.node_id }} has been in degraded state for more than 5 minutes.",
            },
        ),
        AlertRule(
            alert="HighP99Latency",
            expr="histogram_quantile(0.99, rate(distllm_request_duration_seconds_bucket[5m])) > 2",
            for_duration="5m",
            labels={"severity": "warning"},
            annotations={
                "summary": "P99 request latency is above 2 seconds",
                "description": "The 99th percentile of request latency is {{ $value }}s.",
            },
        ),
        AlertRule(
            alert="OOMRisk",
            expr="distllm_kv_cache_usage_ratio > 0.85",
            for_duration="2m",
            labels={"severity": "warning"},
            annotations={
                "summary": "KV cache usage is above 85%",
                "description": "KV cache usage ratio is {{ $value | humanizePercentage }}.",
            },
        ),
        AlertRule(
            alert="HighErrorRate",
            expr="rate(distllm_errors_total[5m]) / rate(distllm_request_duration_seconds_count[5m]) > 0.05",
            for_duration="5m",
            labels={"severity": "critical"},
            annotations={
                "summary": "Error rate is above 5%",
                "description": "Error rate is {{ $value | humanizePercentage }}.",
            },
        ),
        AlertRule(
            alert="ThroughputDrop",
            expr="rate(distllm_tokens_generated_total[5m]) < 0.5 * avg_over_time(rate(distllm_tokens_generated_total[5m])[1h:])",
            for_duration="10m",
            labels={"severity": "warning"},
            annotations={
                "summary": "Token generation throughput dropped below 50% of baseline",
                "description": "Current throughput is {{ $value }} tokens/s, less than half the 1-hour average.",
            },
        ),
    ]


def get_default_recording_rules() -> List[RecordingRule]:
    """Return the default set of recording rules."""
    return [
        RecordingRule(
            record="node_health_avg:1m",
            expr="avg_over_time(distllm_node_health[1m])",
        ),
        RecordingRule(
            record="request_rate:5m",
            expr="rate(distllm_request_duration_seconds_count[5m])",
        ),
        RecordingRule(
            record="error_ratio:5m",
            expr="rate(distllm_errors_total[5m]) / rate(distllm_request_duration_seconds_count[5m])",
        ),
    ]


def rules_to_yaml(alerts: List[AlertRule], recording: List[RecordingRule]) -> str:
    """Serialize alert and recording rules to Prometheus YAML format."""
    groups = []
    if recording:
        groups.append(RuleGroup(
            name="distllm_recording_rules",
            rules=recording,
        ))
    if alerts:
        groups.append(RuleGroup(
            name="distllm_alerting_rules",
            rules=alerts,
        ))

    doc = {"groups": [g.to_dict() for g in groups]}
    return yaml.dump(doc, default_flow_style=False, sort_keys=False)
