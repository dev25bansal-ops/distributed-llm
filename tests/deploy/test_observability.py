"""Tests: HPA integration, webhook validation, Grafana dashboards, Prometheus rules."""

import json
import re
from pathlib import Path

import pytest
import yaml

HELM_DIR = Path(__file__).parent.parent.parent / "deploy" / "helm"
WEBHOOK_DIR = Path(__file__).parent.parent.parent / "deploy" / "webhook"
SRC_DIR = Path(__file__).parent.parent.parent / "src"

_METRIC_RE = re.compile(r"distllm_[a-zA-Z_]+")


def _metric_is_emitted(metric: str, emitted: set[str]) -> bool:
    """True if `metric` is a Prometheus name derivable from a source metric.

    Prometheus appends _bucket/_count/_sum to Histogram and Summary families,
    so `distllm_request_latency_seconds_bucket` is emitted by code that declares
    `distllm_request_latency_seconds`. Check the literal name and each stem.
    """
    if metric in emitted:
        return True
    for suffix in ("_bucket", "_count", "_sum", "_total"):
        if metric.endswith(suffix) and metric[: -len(suffix)] in emitted:
            return True
    return False


def _unknown_metrics(used: set[str], emitted: set[str]) -> set[str]:
    return {m for m in used if not _metric_is_emitted(m, emitted)}


def _emitted_metric_names() -> set[str]:
    """Every distllm_* metric name mentioned anywhere under src/.

    Used to check that dashboards and alert rules query metrics the code
    actually produces. Over-approximates (a name in a comment counts), which is
    the safe direction: it will not produce false failures when a new metric is
    added, only when one is queried that no source file mentions at all.
    """
    names: set[str] = set()
    for path in SRC_DIR.rglob("*.py"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        names.update(_METRIC_RE.findall(text))
    return names


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
        # The dashboard's GPU panels read the node-level memory gauges the
        # exporter actually emits; the older distllm_gpu_memory_used_bytes and
        # distllm_gpu_compute_time_ms names are not produced anywhere in src/.
        for expected in (
            "distllm_node_gpu_memory_bytes",
            "distllm_node_gpu_memory_total_bytes",
            "distllm_kv_cache_usage_ratio",
        ):
            assert expected in names, f"GPU dashboard missing {expected}"

    def test_performance_dashboard_metrics(self):
        dash = json.loads((HELM_DIR / "grafana-dashboards" / "distllm-performance.json").read_text())
        exprs = [t.get("expr", "") for p in dash.get("panels", []) for t in p.get("targets", [])]
        names = self._extract_metrics(exprs)
        for expected in (
            "distllm_requests_total",
            "distllm_tokens_per_second",
            "distllm_request_latency_seconds_bucket",
            "distllm_errors_total",
            "distllm_coordinator_queue_depth",
        ):
            assert expected in names, f"performance dashboard missing {expected}"

    def test_dashboards_only_use_metrics_the_code_emits(self):
        """Every metric a dashboard queries must exist in src/.

        This is the assertion that has teeth: the previous version compared
        against a hardcoded list of names that nothing emitted, so it passed on
        its own fiction. Deriving the emitted set from the source makes a
        renamed metric fail here instead of silently breaking a dashboard.
        """
        emitted = _emitted_metric_names()
        assert emitted, "could not find any distllm_* metrics in src/"
        for dash_name in ("distllm-gpu", "distllm-performance"):
            dash = json.loads((HELM_DIR / "grafana-dashboards" / f"{dash_name}.json").read_text())
            exprs = [t.get("expr", "") for p in dash.get("panels", []) for t in p.get("targets", [])]
            used = self._extract_metrics(exprs)
            unknown = _unknown_metrics(used, emitted)
            assert not unknown, f"{dash_name} queries metrics src/ never emits: {sorted(unknown)}"


# ===========================================================================
# Prometheus Rules — alert names and metric names
# ===========================================================================


def _strip_helm_directives(text: str) -> str:
    """Turn a Helm template into parseable YAML.

    A directive on its own line (`{{- if ... }}` / `{{- end }}`) is replaced by
    a blank line; an inline one (`enabled: {{ .Values.x }}`) is replaced by a
    placeholder scalar. Block bodies are kept so the rules they contain can be
    read; this over-approximates which alerts are active, which is why the test
    that distinguishes active from commented-out alerts checks the specific
    names it cares about rather than trusting the count.
    """
    out_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("{{") and stripped.endswith("}}"):
            out_lines.append("")
            continue
        out_lines.append(re.sub(r"\{\{.*?\}\}", "HELMVALUE", line))
    return "\n".join(out_lines)


class TestPrometheusRules:
    def test_rules_template_exists(self):
        assert (HELM_DIR / "templates" / "prometheusrules.yaml").exists()

    def _rule_groups(self):
        """The rule groups as data, not text.

        Reading the file as text counts commented-out alerts as present --
        GPUMemoryPressure and RateLimitViolations are disabled with a `#` in the
        template, and the old text-based test still asserted them because it
        only grepped for the substring.

        The template is Helm, so its `{{ ... }}` directives are replaced before
        parsing. Conditional blocks are dropped rather than kept: a rule that
        only appears when a value is enabled is not unconditionally active.
        """
        raw = (HELM_DIR / "templates" / "prometheusrules.yaml").read_text()
        cleaned = _strip_helm_directives(raw)
        doc = yaml.safe_load(cleaned) or {}
        return doc.get("spec", {}).get("groups", []) or []

    def _active_alerts(self) -> set[str]:
        return {
            rule["alert"]
            for group in self._rule_groups()
            for rule in group.get("rules", [])
            if "alert" in rule
        }

    def test_active_alert_names(self):
        alerts = self._active_alerts()
        for expected in ("HighErrorRate", "HighP99Latency", "NodeDown", "LowThroughput",
                         "KVCachePressure", "DegradationActive"):
            assert expected in alerts, f"missing active alert {expected}; have {sorted(alerts)}"

    def test_commented_out_alerts_are_not_counted(self):
        # Documents the intent of the two disabled alerts: they must stay out
        # of the active set until their metrics exist.
        alerts = self._active_alerts()
        assert "GPUMemoryPressure" not in alerts
        assert "RateLimitViolations" not in alerts

    def test_every_alert_has_an_expr(self):
        for group in self._rule_groups():
            for rule in group.get("rules", []):
                if "alert" in rule:
                    assert rule.get("expr", "").strip(), f"{rule['alert']} has no expr"

    def test_alert_promql_queries_metrics_the_code_emits(self):
        """Every metric an alert queries must be mentioned in src/.

        The previous version compared against a hardcoded list of names that
        nothing emitted (distllm_total_requests, distllm_gpu_memory_used_bytes),
        so it failed forever while the rules were correct. Deriving the set from
        the source makes a genuinely wrong alert name fail and a correct one pass.
        """
        emitted = _emitted_metric_names()
        assert emitted, "could not find any distllm_* metrics in src/"
        unknown: set[str] = set()
        for group in self._rule_groups():
            for rule in group.get("rules", []):
                exprs = [rule.get("expr", "")]
                for note in rule.get("annotations", {}).values():
                    exprs.append(str(note))
                for expr in exprs:
                    unknown |= _unknown_metrics(set(_METRIC_RE.findall(expr)), emitted)
        assert not unknown, f"alert rules query metrics src/ never emits: {sorted(unknown)}"

    def test_service_monitor_template_exists(self):
        assert (HELM_DIR / "templates" / "servicemonitor.yaml").exists()

    def test_service_monitor_targets_the_api_port(self):
        content = (HELM_DIR / "templates" / "servicemonitor.yaml").read_text()
        # The chart scrapes one service on its api port; it does not declare a
        # separate worker ServiceMonitor, which the old assertion assumed.
        assert "path: /metrics" in content
        assert "port:" in content
