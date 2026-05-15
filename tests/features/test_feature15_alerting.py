"""Tests for Feature 15: Prometheus Alerting Rules."""

import pytest
import yaml

from distllm.monitoring.alert_rules import (
    AlertRule,
    RecordingRule,
    RuleGroup,
    get_default_alerts,
    get_default_recording_rules,
    rules_to_yaml,
)
from distllm.monitoring.rule_applier import RuleApplier, validate_promql
from distllm.monitoring.alertmanager_config import (
    generate_alertmanager_config,
    RouteConfig,
    ReceiverConfig,
)


class TestAlertRule:
    def test_alert_rule_to_dict(self):
        rule = AlertRule(
            alert="TestAlert",
            expr="up == 0",
            for_duration="5m",
            labels={"severity": "critical"},
            annotations={"summary": "Test"},
        )
        d = rule.to_dict()
        assert d["alert"] == "TestAlert"
        assert d["expr"] == "up == 0"
        assert d["for"] == "5m"
        assert d["labels"]["severity"] == "critical"

    def test_alert_rule_defaults(self):
        rule = AlertRule(alert="Minimal", expr="up == 1", for_duration="1m")
        d = rule.to_dict()
        assert d["labels"] == {}
        assert d["annotations"] == {}


class TestRecordingRule:
    def test_recording_rule_to_dict(self):
        rule = RecordingRule(record="job:requests:rate5m", expr="rate(requests_total[5m])")
        d = rule.to_dict()
        assert d["record"] == "job:requests:rate5m"
        assert d["expr"] == "rate(requests_total[5m])"

    def test_recording_rule_with_labels(self):
        rule = RecordingRule(
            record="test:metric",
            expr="sum(metric)",
            labels={"env": "prod"},
        )
        d = rule.to_dict()
        assert d["labels"]["env"] == "prod"


class TestRuleGroup:
    def test_rule_group_to_dict(self):
        group = RuleGroup(
            name="test_group",
            interval="1m",
            rules=[AlertRule(alert="A", expr="up == 0", for_duration="1m")],
        )
        d = group.to_dict()
        assert d["name"] == "test_group"
        assert d["interval"] == "1m"
        assert len(d["rules"]) == 1


class TestDefaultAlerts:
    def test_six_default_alerts(self):
        alerts = get_default_alerts()
        assert len(alerts) == 6
        names = {a.alert for a in alerts}
        expected = {"NodeDown", "NodeDegraded", "HighP99Latency", "OOMRisk", "HighErrorRate", "ThroughputDrop"}
        assert names == expected

    def test_all_alerts_have_severity(self):
        alerts = get_default_alerts()
        for alert in alerts:
            assert "severity" in alert.labels

    def test_all_alerts_have_annotations(self):
        alerts = get_default_alerts()
        for alert in alerts:
            assert "summary" in alert.annotations
            assert "description" in alert.annotations


class TestDefaultRecordingRules:
    def test_three_default_recording_rules(self):
        rules = get_default_recording_rules()
        assert len(rules) == 3
        names = {r.record for r in rules}
        expected = {"node_health_avg:1m", "request_rate:5m", "error_ratio:5m"}
        assert names == expected


class TestRulesToYAML:
    def test_yaml_roundtrip(self):
        alerts = get_default_alerts()
        recording = get_default_recording_rules()
        yaml_str = rules_to_yaml(alerts, recording)
        data = yaml.safe_load(yaml_str)
        assert "groups" in data
        assert len(data["groups"]) == 2

    def test_yaml_contains_recording_and_alerting_groups(self):
        yaml_str = rules_to_yaml(get_default_alerts(), get_default_recording_rules())
        data = yaml.safe_load(yaml_str)
        group_names = {g["name"] for g in data["groups"]}
        assert "distllm_recording_rules" in group_names
        assert "distllm_alerting_rules" in group_names


class TestPromQLValidation:
    def test_empty_expr(self):
        assert len(validate_promql("")) > 0

    def test_unbalanced_parens(self):
        assert len(validate_promql("rate(metric[5m])")) == 0
        assert len(validate_promql("rate(metric[5m]")) > 0

    def test_valid_expressions(self):
        exprs = [
            "distllm_node_health == -1",
            "rate(distllm_errors_total[5m]) / rate(distllm_request_duration_seconds_count[5m]) > 0.05",
            "histogram_quantile(0.99, rate(distllm_request_duration_seconds_bucket[5m])) > 2",
        ]
        for expr in exprs:
            assert len(validate_promql(expr)) == 0, f"Failed for: {expr}"

    def test_unmatched_quote(self):
        assert len(validate_promql('metric{job="test}')) > 0


class TestRuleApplier:
    def test_load_defaults(self):
        applier = RuleApplier("http://localhost:9090")
        alerts, recording, yaml_content = applier.load_and_validate(use_defaults=True)
        assert len(alerts) == 6
        assert len(recording) == 3
        assert isinstance(yaml_content, str)

    def test_get_rules_yaml(self):
        applier = RuleApplier("http://localhost:9090")
        yaml_str = applier.get_rules_yaml()
        data = yaml.safe_load(yaml_str)
        assert "groups" in data

    def test_apply_rules_connection_refused(self):
        """Should gracefully handle unreachable Prometheus without raising."""
        applier = RuleApplier("http://localhost:19999")
        result = applier.apply_rules()
        assert result is False


class TestAlertmanagerConfig:
    def test_default_config(self):
        config_str = generate_alertmanager_config()
        data = yaml.safe_load(config_str)
        assert "route" in data
        assert "receivers" in data
        assert data["route"]["receiver"] == "default"

    def test_config_with_email(self):
        config_str = generate_alertmanager_config(
            email_recipients=["ops@example.com"],
        )
        data = yaml.safe_load(config_str)
        receiver_names = [r["name"] for r in data["receivers"]]
        assert "email-critical" in receiver_names

    def test_config_with_slack(self):
        config_str = generate_alertmanager_config(
            slack_webhook_url="https://hooks.slack.com/test",
        )
        data = yaml.safe_load(config_str)
        receiver_names = [r["name"] for r in data["receivers"]]
        assert "slack-critical" in receiver_names

    def test_route_config_defaults(self):
        route = RouteConfig()
        d = route.to_dict()
        assert d["receiver"] == "default"
        assert d["continue"] is False

    def test_receiver_config(self):
        receiver = ReceiverConfig(name="test")
        d = receiver.to_dict()
        assert d["name"] == "test"
