"""Tests: HPA integration, webhook validation, Grafana dashboards, Prometheus rules."""

import json
from pathlib import Path

import pytest
import yaml

HELM_DIR = Path(__file__).parent.parent.parent / "deploy" / "helm"
WEBHOOK_DIR = Path(__file__).parent.parent.parent / "deploy" / "webhook"


try:
    import kopf  # noqa: F401
    _HAVE_KOPF = True
except ImportError:
    _HAVE_KOPF = False


# ===========================================================================
# HPA — CustomHPA Integration
# ===========================================================================


@pytest.mark.skipif(not _HAVE_KOPF, reason="kopf not installed")
class TestCustomHPA:
    def test_init_defaults(self):
        from distllm.operator.hpa import CustomHPA
        hpa = CustomHPA()
        assert hpa._metric == "tokens_per_second"
        assert hpa._min_replicas == 1
        assert hpa._max_replicas == 10
        assert hpa._target_value == 100.0

    def test_init_custom_params(self):
        from distllm.operator.hpa import CustomHPA
        hpa = CustomHPA(metric="queue_depth", target_value=50, min_replicas=2, max_replicas=8)
        assert hpa._metric == "queue_depth"
        assert hpa._target_value == 50.0
        assert hpa._min_replicas == 2

    def test_prometheus_url_validation_valid(self):
        from distllm.operator.hpa import _validate_prometheus_url
        url = _validate_prometheus_url("http://prometheus:9090")
        assert url == "http://prometheus:9090"

    def test_prometheus_url_validation_invalid_scheme(self):
        from distllm.operator.hpa import _validate_prometheus_url
        from distllm.errors import ConfigValidationError
        with pytest.raises(ConfigValidationError):
            _validate_prometheus_url("ftp://prometheus:9090")

    def test_metric_name_mapping_tokens(self):
        from distllm.operator.hpa import CustomHPA
        hpa = CustomHPA()
        assert hpa._metric == "tokens_per_second"

    def test_hpa_templates_in_helm(self):
        tmpl_path = HELM_DIR / "templates" / "hpa.yaml"
        assert tmpl_path.exists()
        content = tmpl_path.read_text()
        assert "HorizontalPodAutoscaler" in content
        assert ".Values.hpa" in content


# ===========================================================================
# Webhook — Validation
# ===========================================================================


class TestWebhookConfig:
    def test_webhook_deployment_exists(self):
        assert (WEBHOOK_DIR / "deployment.yaml").exists()

    def test_webhook_service_exists(self):
        assert (WEBHOOK_DIR / "service.yaml").exists()

    def test_webhook_validating_config_exists(self):
        assert (WEBHOOK_DIR / "validatingwebhookconfiguration.yaml").exists()

    def test_webhook_image_tag(self):
        dep = yaml.safe_load((WEBHOOK_DIR / "deployment.yaml").read_text())
        img = dep["spec"]["template"]["spec"]["containers"][0]["image"]
        assert "operator:0.4.0" in img

    def test_webhook_two_replicas(self):
        dep = yaml.safe_load((WEBHOOK_DIR / "deployment.yaml").read_text())
        assert dep["spec"]["replicas"] == 2

    def test_webhook_https_port(self):
        dep = yaml.safe_load((WEBHOOK_DIR / "deployment.yaml").read_text())
        ports = [c["ports"] for c in dep["spec"]["template"]["spec"]["containers"]]
        assert any(p["containerPort"] == 8443 for port_list in ports for p in port_list)

    def test_webhook_tls_mount(self):
        dep = yaml.safe_load((WEBHOOK_DIR / "deployment.yaml").read_text())
        vols = dep["spec"]["template"]["spec"]["volumes"]
        assert any("distllm-webhook-tls" in str(v) for v in vols)

    def test_webhook_two_validate_endpoints(self):
        wh = yaml.safe_load((WEBHOOK_DIR / "validatingwebhookconfiguration.yaml").read_text())
        assert len(wh["webhooks"]) == 2
        paths = [w["clientConfig"]["service"]["path"] for w in wh["webhooks"]]
        assert "/validate-crd" in paths
        assert "/validate-cluster" in paths

    def test_webhook_cert_manager_annotation(self):
        wh = yaml.safe_load((WEBHOOK_DIR / "validatingwebhookconfiguration.yaml").read_text())
        ann = wh["metadata"].get("annotations", {})
        assert "cert-manager.io/inject-ca-from" in ann

    def test_webhook_timeout_10s(self):
        wh = yaml.safe_load((WEBHOOK_DIR / "validatingwebhookconfiguration.yaml").read_text())
        for w in wh["webhooks"]:
            assert w["timeoutSeconds"] == 10

    def test_webhook_failure_policy_ignore(self):
        wh = yaml.safe_load((WEBHOOK_DIR / "validatingwebhookconfiguration.yaml").read_text())
        for w in wh["webhooks"]:
            assert w["failurePolicy"] == "Ignore"


class TestWebhookServer:
    def test_validate_crd_no_changes(self):
        from distllm.deploy.webhook_server import _validate_crd_compatibility
        old = {"openAPIV3Schema": {"properties": {"a": {"type": "string"}}}}
        new = {"openAPIV3Schema": {"properties": {"a": {"type": "string"}}}}
        valid, msg = _validate_crd_compatibility(old, new)
        assert valid is True

    def test_health_endpoint(self):
        from distllm.deploy.webhook_server import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_health_endpoint(self):
        from distllm.deploy.webhook_server import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ===========================================================================
# Grafana Dashboards — metric names
# ===========================================================================


class TestGrafanaDashboards:
    def test_gpu_dashboard_exists(self):
        assert (HELM_DIR / "grafana-dashboards" / "distllm-gpu.json").exists()

    def test_performance_dashboard_exists(self):
        assert (HELM_DIR / "grafana-dashboards" / "distllm-performance.json").exists()

    def _extract_metrics(self, exprs):
        names = set()
        for expr in exprs:
            import re
            for m in re.finditer(r'distllm_[a-zA-Z_]+', expr):
                names.add(m.group())
        return names

    def test_gpu_dashboard_metrics(self):
        dash = json.loads((HELM_DIR / "grafana-dashboards" / "distllm-gpu.json").read_text())
        exprs = [t.get("expr", "") for p in dash.get("panels", []) for t in p.get("targets", [])]
        names = self._extract_metrics(exprs)
        assert "distllm_gpu_memory_used_bytes" in names
        assert "distllm_gpu_compute_time_ms" in names
        assert "distllm_kv_cache_used_bytes" in names

    def test_performance_dashboard_metrics(self):
        dash = json.loads((HELM_DIR / "grafana-dashboards" / "distllm-performance.json").read_text())
        exprs = [t.get("expr", "") for p in dash.get("panels", []) for t in p.get("targets", [])]
        names = self._extract_metrics(exprs)
        expected = {"distllm_total_requests", "distllm_tokens_per_second", "distllm_token_latency_bucket",
                    "distllm_errors_total", "distllm_cache_hits_total", "distllm_cache_misses_total",
                    "distllm_gpu_memory_used_bytes", "distllm_active_batch_size"}
        missing = expected - names
        assert not missing, f"Missing metrics: {missing}"
        expected = {"distllm_total_requests", "distllm_tokens_per_second", "distllm_token_latency_bucket",
                    "distllm_errors_total", "distllm_cache_hits_total", "distllm_cache_misses_total",
                    "distllm_gpu_memory_used_bytes", "distllm_active_batch_size"}
        assert expected.issubset(names), f"Missing: {expected - names}"


# ===========================================================================
# Prometheus Rules — alert names and metric names
# ===========================================================================


class TestPrometheusRules:
    def test_rules_template_exists(self):
        assert (HELM_DIR / "templates" / "prometheusrules.yaml").exists()

    def test_alert_names_match_definitions(self):
        content = (HELM_DIR / "templates" / "prometheusrules.yaml").read_text()
        expected_alerts = ["HighErrorRate", "HighP99Latency", "NodeDown", "LowThroughput",
                          "CacheHitRateDrop", "DegradationActive", "GPUMemoryPressure", "RateLimitViolations"]
        for alert in expected_alerts:
            assert alert in content, f"Missing alert: {alert}"

    def test_all_alert_promql_uses_valid_metrics(self):
        content = (HELM_DIR / "templates" / "prometheusrules.yaml").read_text()
        known_metrics = {"distllm_errors_total", "distllm_total_requests", "distllm_token_latency_bucket",
                        "distllm_node_healthy", "distllm_tokens_per_second", "distllm_cache_hits_total",
                        "distllm_cache_misses_total", "distllm_degradation_level", "distllm_gpu_memory_free",
                        "distllm_gpu_memory_total", "distllm_rate_limited_total"}
        for metric in known_metrics:
            assert metric in content, f"Metric {metric} not found in Prometheus rules"

    def test_service_monitor_template_exists(self):
        assert (HELM_DIR / "templates" / "servicemonitor.yaml").exists()

    def test_service_monitor_refs_both_coordinator_and_worker(self):
        content = (HELM_DIR / "templates" / "servicemonitor.yaml").read_text()
        assert "coordinator" in content
        assert "worker" in content
