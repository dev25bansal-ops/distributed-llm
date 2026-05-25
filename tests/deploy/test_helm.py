"""Tests: Helm chart template validation — rendering, GPU, probes, image tags, features."""

import os
from pathlib import Path

import pytest
import yaml


HELM_DIR = Path(__file__).parent.parent.parent / "deploy" / "helm"
VALUES_PATH = HELM_DIR / "values.yaml"
TEMPLATES_DIR = HELM_DIR / "templates"
CHART_PATH = HELM_DIR / "Chart.yaml"


def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _load_all_templates():
    """Load all template files as raw text for inspection."""
    templates = {}
    for f in sorted(TEMPLATES_DIR.glob("*.yaml")):
        with open(f) as fh:
            templates[f.name] = fh.read()
    return templates


# ===========================================================================
# Chart metadata
# ===========================================================================


class TestChartMetadata:
    def test_chart_yaml_exists(self):
        assert CHART_PATH.exists()

    def test_chart_version_matches(self):
        chart = _load_yaml(CHART_PATH)
        assert chart["version"] == "0.1.0"
        assert chart["appVersion"] == "0.4.0"
        assert chart["type"] == "application"
        assert chart["apiVersion"] == "v2"

    def test_values_yaml_exists(self):
        assert VALUES_PATH.exists()
        values = _load_yaml(VALUES_PATH)
        assert "image" in values
        assert "coordinator" in values
        assert "nodePools" in values


# ===========================================================================
# Image tag
# ===========================================================================


class TestImageTag:
    def test_image_tag_defaults_to_chart_version(self):
        chart = _load_yaml(CHART_PATH)
        values = _load_yaml(VALUES_PATH)
        expected_tag = values.get("image", {}).get("tag", chart["appVersion"])
        assert expected_tag == "0.4.0"

    def test_image_repository_defined(self):
        values = _load_yaml(VALUES_PATH)
        repo = values["image"]["repository"]
        assert repo == "ghcr.io/distributed-llm/coordinator"

    def test_image_pull_policy_defined(self):
        values = _load_yaml(VALUES_PATH)
        assert values["image"]["pullPolicy"] in ("Always", "IfNotPresent", "Never")


# ===========================================================================
# GPU resources — worker templates
# ===========================================================================


class TestGPUResources:
    def test_worker_gpu_resource_in_values(self):
        values = _load_yaml(VALUES_PATH)
        for pool in values["nodePools"]:
            assert "gpuType" in pool
            assert pool["gpuType"] == "nvidia.com/gpu"
            assert pool["gpuCount"] >= 1

    def test_worker_statefulset_templates_exist(self):
        tmpl_path = TEMPLATES_DIR / "worker-statefulset.yaml"
        assert tmpl_path.exists()

    def test_template_refers_to_nvidia_gpu(self):
        tmpl = (TEMPLATES_DIR / "worker-statefulset.yaml").read_text()
        assert "gpuResourceName" in tmpl or "nvidia.com/gpu" in tmpl

    def test_coordinator_has_no_gpu(self):
        coord = (TEMPLATES_DIR / "coordinator-deployment.yaml").read_text()
        assert "nvidia.com/gpu" not in coord

    def test_worker_tolerations_for_gpu(self):
        values = _load_yaml(VALUES_PATH)
        tols = values.get("tolerations", [])
        assert any("nvidia.com/gpu" in str(t) for t in tols)


# ===========================================================================
# Probes — readiness + liveness
# ===========================================================================


class TestProbes:
    def test_coordinator_readiness_probe(self):
        tmpl = (TEMPLATES_DIR / "coordinator-deployment.yaml").read_text()
        assert "/ready" in tmpl
        assert "initialDelaySeconds: 10" in tmpl

    def test_coordinator_liveness_probe(self):
        tmpl = (TEMPLATES_DIR / "coordinator-deployment.yaml").read_text()
        assert "/live" in tmpl
        assert "initialDelaySeconds: 60" in tmpl

    def test_worker_readiness_probe(self):
        tmpl = (TEMPLATES_DIR / "worker-statefulset.yaml").read_text()
        assert "tcpSocket" in tmpl
        assert "initialDelaySeconds: 5" in tmpl

    def test_worker_liveness_probe(self):
        tmpl = (TEMPLATES_DIR / "worker-statefulset.yaml").read_text()
        assert "tcpSocket" in tmpl
        assert "failureThreshold: 3" in tmpl

    def test_coordinator_ha_readiness(self):
        tmpl = (TEMPLATES_DIR / "coordinator-ha.yaml").read_text()
        assert "/ready" in tmpl
        assert "initialDelaySeconds: 10" in tmpl

    def test_coordinator_ha_liveness(self):
        tmpl = (TEMPLATES_DIR / "coordinator-ha.yaml").read_text()
        assert "/live" in tmpl
        assert "initialDelaySeconds: 30" in tmpl


# ===========================================================================
# Core resources always present
# ===========================================================================


class TestCoreResources:
    def test_coordinator_deployment_exists(self):
        assert (TEMPLATES_DIR / "coordinator-deployment.yaml").exists()

    def test_coordinator_service_exists(self):
        assert (TEMPLATES_DIR / "coordinator-service.yaml").exists()

    def test_worker_statefulset_exists(self):
        assert (TEMPLATES_DIR / "worker-statefulset.yaml").exists()

    def test_worker_service_exists(self):
        assert (TEMPLATES_DIR / "worker-service.yaml").exists()

    def test_configmap_exists(self):
        assert (TEMPLATES_DIR / "configmap.yaml").exists()

    def test_helpers_tpl_exists(self):
        assert (TEMPLATES_DIR / "_helpers.tpl").exists()


# ===========================================================================
# Optional features — template files exist when enabled
# ===========================================================================


class TestOptionalFeatures:
    def test_ingress_template_exists(self):
        assert (TEMPLATES_DIR / "ingress.yaml").exists()

    def test_hpa_template_exists(self):
        assert (TEMPLATES_DIR / "hpa.yaml").exists()

    def test_pdb_coordinator_template_exists(self):
        assert (TEMPLATES_DIR / "pdb-coordinator.yaml").exists()

    def test_pdb_worker_template_exists(self):
        assert (TEMPLATES_DIR / "pdb-worker.yaml").exists()

    def test_network_policy_template_exists(self):
        assert (TEMPLATES_DIR / "networkpolicy.yaml").exists()

    def test_service_monitor_template_exists(self):
        assert (TEMPLATES_DIR / "servicemonitor.yaml").exists()

    def test_prometheus_rules_template_exists(self):
        assert (TEMPLATES_DIR / "prometheusrules.yaml").exists()

    def test_grafana_dashboard_template_exists(self):
        assert (TEMPLATES_DIR / "grafana-dashboard.yaml").exists()

    def test_fluentbit_template_exists(self):
        assert (TEMPLATES_DIR / "fluentbit-daemonset.yaml").exists()

    def test_coordinator_ha_template_exists(self):
        assert (TEMPLATES_DIR / "coordinator-ha.yaml").exists()


# ===========================================================================
# Template content patterns
# ===========================================================================


class TestTemplatePatterns:
    def test_templates_use_go_template_syntax(self):
        templates = _load_all_templates()
        for name, content in templates.items():
            assert "{{" in content, f"{name} has no Go template syntax"

    def test_all_templates_have_metadata(self):
        templates = _load_all_templates()
        for name, content in templates.items():
            assert any(marker in content for marker in [
                "apiVersion", "kind", "metadata",
            ]), f"{name} missing K8s resource markers"

    def test_helpers_defined(self):
        helpers = (TEMPLATES_DIR / "_helpers.tpl").read_text()
        assert "distllm.name" in helpers
        assert "distllm.fullname" in helpers
        assert "distllm.labels" in helpers
        assert "distllm.selectorLabels" in helpers
        assert "distllm.gpuResourceName" in helpers
        assert "distllm.coordinatorLabels" in helpers
        assert "distllm.workerLabels" in helpers

    def test_image_repository_in_all_workloads(self):
        templates = _load_all_templates()
        for name in ("coordinator-deployment.yaml", "worker-statefulset.yaml", "coordinator-ha.yaml"):
            assert '.Values.image.repository' in templates[name]


# ===========================================================================
# Security context
# ===========================================================================


class TestSecurityContext:
    def test_coordinator_security_context(self):
        tmpl = (TEMPLATES_DIR / "coordinator-deployment.yaml").read_text()
        assert "runAsNonRoot: true" in tmpl
        assert "allowPrivilegeEscalation: false" in tmpl

    def test_worker_security_context(self):
        tmpl = (TEMPLATES_DIR / "worker-statefulset.yaml").read_text()
        assert "runAsNonRoot: true" in tmpl
        assert "readOnlyRootFilesystem: true" in tmpl


# ===========================================================================
# PVC permissions
# ===========================================================================


class TestPVCPermissions:
    def test_pvc_template_exists(self):
        assert (TEMPLATES_DIR / "pvc.yaml").exists()

    def test_pvc_access_mode_read_write_many(self):
        tmpl = (TEMPLATES_DIR / "pvc.yaml").read_text()
        assert "ReadWriteMany" in tmpl

    def test_cache_pvc_size_in_values(self):
        values = _load_yaml(VALUES_PATH)
        assert values["persistence"]["cache"]["size"] == "100Gi"

    def test_audit_pvc_size_in_values(self):
        values = _load_yaml(VALUES_PATH)
        assert values["persistence"]["audit"]["size"] == "10Gi"


# ===========================================================================
# Network policy
# ===========================================================================


class TestNetworkPolicy:
    def test_network_policy_template_exists(self):
        assert (TEMPLATES_DIR / "networkpolicy.yaml").exists()

    def test_coordinator_ingress_from_workers(self):
        tmpl = (TEMPLATES_DIR / "networkpolicy.yaml").read_text()
        assert "component: worker" in tmpl

    def test_worker_ingress_from_coordinator(self):
        tmpl = (TEMPLATES_DIR / "networkpolicy.yaml").read_text()
        matches = tmpl.count("component: coordinator")
        assert matches >= 1

    def test_both_policy_types_present(self):
        tmpl = (TEMPLATES_DIR / "networkpolicy.yaml").read_text()
        assert "Ingress" in tmpl
        assert "Egress" in tmpl
