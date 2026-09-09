"""Tests: Helm chart + deploy manifest validation — rendering, GPU, probes, image tags, features.

The ``deploy/helm`` chart is the ``distllm-operator`` chart (operator pattern): it
deploys the operator itself plus coordinator/worker workload templates.
Workload-level deployment details live in the static manifests under
``deploy/inference/`` (probes, security contexts, PVCs, network policies,
PDBs), autoscaling under ``deploy/kustomize/autoscaling.yaml``, monitoring
under ``deploy/monitoring/`` and log-based alerting under ``deploy/logging/``.
Tests pin those current locations.
"""

import json
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).parent.parent.parent
HELM_DIR = REPO_ROOT / "deploy" / "helm" / "distllm-operator"
VALUES_PATH = HELM_DIR / "values.yaml"
TEMPLATES_DIR = HELM_DIR / "templates"
CHART_PATH = HELM_DIR / "Chart.yaml"
CRDS_PATH = HELM_DIR / "crds" / "crds.yaml"
INFERENCE_DIR = REPO_ROOT / "deploy" / "inference"
AUTOSCALING_PATH = REPO_ROOT / "deploy" / "kustomize" / "autoscaling.yaml"
MONITORING_DIR = REPO_ROOT / "deploy" / "monitoring"
LOGGING_DIR = REPO_ROOT / "deploy" / "logging"


def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _load_yaml_docs(path):
    """Load a (possibly multi-document) YAML file."""
    with open(path) as f:
        return [doc for doc in yaml.safe_load_all(f) if doc]


def _load_all_templates():
    """Load all chart template files as raw text for inspection."""
    templates = {}
    for f in sorted(TEMPLATES_DIR.glob("*.yaml")):
        with open(f) as fh:
            templates[f.name] = fh.read()
    return templates


def _netpol_docs():
    """Network policy manifests indexed by the component they select."""
    docs = _load_yaml_docs(INFERENCE_DIR / "network-policy.yaml")
    return {
        doc["spec"]["podSelector"]["matchLabels"]["app.kubernetes.io/component"]: doc
        for doc in docs
    }


# ===========================================================================
# Chart metadata
# ===========================================================================


class TestChartMetadata:
    def test_chart_yaml_exists(self):
        assert CHART_PATH.exists()

    def test_chart_version_matches(self):
        chart = _load_yaml(CHART_PATH)
        assert chart["name"] == "distllm-operator"
        assert chart["version"] == "0.4.0"
        assert chart["appVersion"] == "0.4.0"
        assert chart["type"] == "application"
        assert chart["apiVersion"] == "v2"

    def test_values_yaml_exists(self):
        assert VALUES_PATH.exists()
        values = _load_yaml(VALUES_PATH)
        assert "operator" in values
        assert "controller" in values
        assert "defaultResources" in values


# ===========================================================================
# Image tag
# ===========================================================================


class TestImageTag:
    def test_image_tag_defaults_to_chart_version(self):
        chart = _load_yaml(CHART_PATH)
        values = _load_yaml(VALUES_PATH)
        expected_tag = values["operator"]["image"]["tag"]
        assert expected_tag == chart["appVersion"] == "0.4.0"
        # Workload images track the chart version too.
        assert values["controller"]["defaultCoordinatorImage"].endswith(":0.4.0")
        assert values["controller"]["defaultWorkerImage"].endswith(":0.4.0")

    def test_image_repository_defined(self):
        values = _load_yaml(VALUES_PATH)
        assert values["operator"]["image"]["repository"] == "ghcr.io/distributed-llm/operator"
        assert values["controller"]["defaultCoordinatorImage"].startswith(
            "ghcr.io/distributed-llm/coordinator:"
        )
        assert values["controller"]["defaultWorkerImage"].startswith(
            "ghcr.io/distributed-llm/worker:"
        )

    def test_image_pull_policy_defined(self):
        values = _load_yaml(VALUES_PATH)
        assert values["operator"]["image"]["pullPolicy"] in ("Always", "IfNotPresent", "Never")


# ===========================================================================
# GPU resources — worker templates
# ===========================================================================


class TestGPUResources:
    def test_worker_gpu_resource_in_values(self):
        values = _load_yaml(VALUES_PATH)
        worker = values["defaultResources"]["worker"]
        assert worker["requests"]["nvidia.com/gpu"] == "1"
        assert worker["limits"]["nvidia.com/gpu"] == "1"

    def test_worker_statefulset_templates_exist(self):
        tmpl_path = TEMPLATES_DIR / "worker.yaml"
        assert tmpl_path.exists()
        assert "kind: StatefulSet" in tmpl_path.read_text()

    def test_template_refers_to_nvidia_gpu(self):
        tmpl = (TEMPLATES_DIR / "worker.yaml").read_text()
        assert "nvidia.com/gpu" in tmpl

    def test_coordinator_has_no_gpu(self):
        coord = (TEMPLATES_DIR / "coordinator.yaml").read_text()
        assert "nvidia.com/gpu" not in coord
        values = _load_yaml(VALUES_PATH)
        coordinator = values["defaultResources"]["coordinator"]
        assert "nvidia.com/gpu" not in coordinator["requests"]
        assert "nvidia.com/gpu" not in coordinator["limits"]

    def test_worker_tolerations_for_gpu(self):
        # Chart worker template tolerates GPU (tainted) nodes.
        tmpl = (TEMPLATES_DIR / "worker.yaml").read_text()
        assert "nvidia.com/gpu" in tmpl
        assert "effect: NoSchedule" in tmpl
        # Static worker manifest carries the same toleration.
        sts = _load_yaml_docs(INFERENCE_DIR / "worker-statefulset.yaml")[0]
        tols = sts["spec"]["template"]["spec"].get("tolerations", [])
        assert any(t.get("key") == "nvidia.com/gpu" for t in tols)


# ===========================================================================
# Probes — readiness + liveness
# ===========================================================================


class TestProbes:
    def test_coordinator_readiness_probe(self):
        tmpl = (TEMPLATES_DIR / "coordinator.yaml").read_text()
        assert "readinessProbe" in tmpl
        assert "path: /ready" in tmpl
        assert "httpGet" in tmpl
        # Probe delays are configured in the static inference manifest.
        dep = _load_yaml_docs(INFERENCE_DIR / "coordinator-deployment.yaml")[0]
        probe = dep["spec"]["template"]["spec"]["containers"][0]["readinessProbe"]
        assert probe["httpGet"]["path"] == "/ready"
        assert probe["initialDelaySeconds"] == 5

    def test_coordinator_liveness_probe(self):
        tmpl = (TEMPLATES_DIR / "coordinator.yaml").read_text()
        assert "livenessProbe" in tmpl
        assert "path: /live" in tmpl
        assert "httpGet" in tmpl
        dep = _load_yaml_docs(INFERENCE_DIR / "coordinator-deployment.yaml")[0]
        probe = dep["spec"]["template"]["spec"]["containers"][0]["livenessProbe"]
        assert probe["httpGet"]["path"] == "/live"
        assert probe["initialDelaySeconds"] == 10

    def test_worker_readiness_probe(self):
        tmpl = (TEMPLATES_DIR / "worker.yaml").read_text()
        assert "readinessProbe" in tmpl
        assert "tcpSocket" in tmpl
        assert ".Values.network.workerGrpcPort" in tmpl
        sts = _load_yaml_docs(INFERENCE_DIR / "worker-statefulset.yaml")[0]
        probe = sts["spec"]["template"]["spec"]["containers"][0]["readinessProbe"]
        assert "tcpSocket" in probe
        assert probe["initialDelaySeconds"] == 5

    def test_worker_liveness_probe(self):
        tmpl = (TEMPLATES_DIR / "worker.yaml").read_text()
        assert "livenessProbe" in tmpl
        assert "tcpSocket" in tmpl
        sts = _load_yaml_docs(INFERENCE_DIR / "worker-statefulset.yaml")[0]
        probe = sts["spec"]["template"]["spec"]["containers"][0]["livenessProbe"]
        assert "tcpSocket" in probe
        assert probe["initialDelaySeconds"] == 15

    def test_coordinator_deploys_as_statefulset(self):
        # coordinator-ha.yaml was replaced by the operator chart's coordinator
        # StatefulSet — the HA-capable form — with a headless serviceName and
        # a readiness probe.
        tmpl = (TEMPLATES_DIR / "coordinator.yaml").read_text()
        assert "kind: StatefulSet" in tmpl
        assert "serviceName:" in tmpl
        assert "path: /ready" in tmpl

    def test_coordinator_graceful_lifecycle(self):
        # HA-specific liveness timing was replaced by graceful-drain wiring:
        # liveness probe + termination grace period from values.
        tmpl = (TEMPLATES_DIR / "coordinator.yaml").read_text()
        assert "path: /live" in tmpl
        assert "terminationGracePeriodSeconds" in tmpl
        assert ".Values.controller.coordinatorGracePeriodSeconds" in tmpl


# ===========================================================================
# Core resources always present
# ===========================================================================


class TestCoreResources:
    def test_coordinator_deployment_exists(self):
        # Coordinator workload + Service now ship as one template.
        assert (TEMPLATES_DIR / "coordinator.yaml").exists()

    def test_coordinator_service_exists(self):
        tmpl = (TEMPLATES_DIR / "coordinator.yaml").read_text()
        assert "kind: StatefulSet" in tmpl
        assert "kind: Service" in tmpl

    def test_worker_statefulset_exists(self):
        tmpl = (TEMPLATES_DIR / "worker.yaml").read_text()
        assert "kind: StatefulSet" in tmpl

    def test_worker_service_exists(self):
        tmpl = (TEMPLATES_DIR / "worker.yaml").read_text()
        assert "kind: Service" in tmpl
        assert "clusterIP: None" in tmpl  # headless service for stable pod DNS

    def test_crd_exists(self):
        # Cluster configuration ships as the DistributedLLMCluster + NodePool
        # CRDs (replaces the old chart configmap).
        assert CRDS_PATH.exists()
        docs = _load_yaml_docs(CRDS_PATH)
        kinds = {doc["metadata"]["name"] for doc in docs if doc["kind"] == "CustomResourceDefinition"}
        assert "distributedllmclusters.distllm.zeroroute.ai" in kinds
        assert "nodepools.distllm.zeroroute.ai" in kinds
        assert "required: [name, layers]" in CRDS_PATH.read_text()

    def test_helpers_tpl_exists(self):
        assert (TEMPLATES_DIR / "_helpers.tpl").exists()


# ===========================================================================
# Optional features — current homes (chart RBAC, kustomize autoscaling,
# inference manifests, monitoring, logging)
# ===========================================================================


class TestOptionalFeatures:
    def test_hpa_template_exists(self):
        # HPA moved from the chart to deploy/kustomize/autoscaling.yaml.
        assert AUTOSCALING_PATH.exists()
        assert "HorizontalPodAutoscaler" in AUTOSCALING_PATH.read_text()

    def test_pdb_coordinator_template_exists(self):
        path = INFERENCE_DIR / "coordinator-pdb.yaml"
        assert path.exists()
        pdb = _load_yaml(path)
        assert pdb["kind"] == "PodDisruptionBudget"
        assert pdb["spec"]["minAvailable"] == 1

    def test_pdb_template_in_community_chart(self):
        # The chart's pdb-coordinator.yaml/pdb-worker.yaml moved to the
        # community distllm chart (integrations/kubernetes/helm), now a
        # single configurable PDB.
        tmpl = (
            REPO_ROOT / "integrations" / "kubernetes" / "helm" / "distllm"
            / "templates" / "pdb.yaml"
        )
        assert tmpl.exists()
        content = tmpl.read_text()
        assert "kind: PodDisruptionBudget" in content
        assert ".Values.podDisruptionBudget" in content

    def test_ingress_template_in_community_chart(self):
        # ingress.yaml moved to the community distllm chart.
        tmpl = (
            REPO_ROOT / "integrations" / "kubernetes" / "helm" / "distllm"
            / "templates" / "ingress.yaml"
        )
        assert tmpl.exists()
        content = tmpl.read_text()
        assert "kind: Ingress" in content
        assert ".Values.ingress.enabled" in content

    def test_network_policy_template_exists(self):
        assert (INFERENCE_DIR / "network-policy.yaml").exists()

    def test_service_monitor_template_exists(self):
        content = AUTOSCALING_PATH.read_text()
        assert "ServiceMonitor" in content

    def test_prometheus_rules_template_exists(self):
        rules = MONITORING_DIR / "prometheus" / "alert-rules.yml"
        assert rules.exists()
        content = rules.read_text()
        assert "alert:" in content
        assert "DistLLMNodeDown" in content
        assert "DistLLMHighErrorRate" in content

    def test_grafana_dashboard_template_exists(self):
        dashboards = MONITORING_DIR / "grafana" / "dashboards"
        for name in ("coordinator.json", "inference.json", "node-pool.json"):
            path = dashboards / name
            assert path.exists(), f"missing dashboard {name}"
            dash = json.loads(path.read_text())
            assert dash.get("panels"), f"{name} has no panels"

    def test_loki_log_alerts_exist(self):
        # The chart's fluentbit daemonset was replaced by the Loki-based
        # logging pipeline; its alert rules live in deploy/logging/loki/.
        rules = LOGGING_DIR / "loki" / "alert-rules.yaml"
        assert rules.exists()
        content = rules.read_text()
        assert "NodeUnreachable" in content
        assert "GPUOutOfMemory" in content

    def test_serviceaccount_template_exists(self):
        tmpl = (TEMPLATES_DIR / "serviceaccount.yaml").read_text()
        assert "kind: ServiceAccount" in tmpl
        assert "kind: ClusterRole" in tmpl
        assert "kind: ClusterRoleBinding" in tmpl
        # RBAC scope: only the DistLLM CRDs and child resources.
        assert "distllm.zeroroute.ai" in tmpl


# ===========================================================================
# Template content patterns
# ===========================================================================


class TestTemplatePatterns:
    def test_templates_use_go_template_syntax(self):
        templates = _load_all_templates()
        assert templates, "no chart templates found"
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
        for helper in (
            "distllm-operator.name",
            "distllm-operator.fullname",
            "distllm-operator.labels",
            "distllm-operator.coordinator-selector-labels",
            "distllm-operator.worker-selector-labels",
        ):
            assert helper in helpers, f"missing helper define: {helper}"

    def test_image_repository_in_all_workloads(self):
        templates = _load_all_templates()
        assert ".Values.controller.defaultCoordinatorImage" in templates["coordinator.yaml"]
        assert ".Values.controller.defaultWorkerImage" in templates["worker.yaml"]
        assert ".Values.operator.image.repository" in templates["operator.yaml"]


# ===========================================================================
# Security context
# ===========================================================================


class TestSecurityContext:
    def test_coordinator_security_context(self):
        # Chart template: pod-level non-root security context.
        tmpl = (TEMPLATES_DIR / "coordinator.yaml").read_text()
        assert "runAsNonRoot: true" in tmpl
        assert "runAsUser: 1000" in tmpl
        # Static manifest: full container-level hardening.
        dep = _load_yaml_docs(INFERENCE_DIR / "coordinator-deployment.yaml")[0]
        pod = dep["spec"]["template"]["spec"]
        assert pod["securityContext"]["runAsNonRoot"] is True
        container = pod["containers"][0]
        sec = container["securityContext"]
        assert sec["allowPrivilegeEscalation"] is False
        assert sec["readOnlyRootFilesystem"] is True
        assert sec["capabilities"]["drop"] == ["ALL"]

    def test_worker_security_context(self):
        tmpl = (TEMPLATES_DIR / "worker.yaml").read_text()
        assert "runAsNonRoot: true" in tmpl
        assert "runAsUser: 1000" in tmpl
        sts = _load_yaml_docs(INFERENCE_DIR / "worker-statefulset.yaml")[0]
        pod = sts["spec"]["template"]["spec"]
        assert pod["securityContext"]["runAsNonRoot"] is True
        container = pod["containers"][0]
        sec = container["securityContext"]
        assert sec["allowPrivilegeEscalation"] is False
        assert sec["readOnlyRootFilesystem"] is True
        assert sec["capabilities"]["drop"] == ["ALL"]


# ===========================================================================
# PVC permissions
# ===========================================================================


class TestPVCPermissions:
    def test_pvc_template_exists(self):
        assert (INFERENCE_DIR / "pvc.yaml").exists()

    def test_pvc_access_mode_read_write_many(self):
        for doc in _load_yaml_docs(INFERENCE_DIR / "pvc.yaml"):
            assert doc["spec"]["accessModes"] == ["ReadWriteMany"]

    def test_cache_pvc_size(self):
        docs = {
            d["metadata"]["name"]: d for d in _load_yaml_docs(INFERENCE_DIR / "pvc.yaml")
        }
        assert docs["distllm-cache"]["spec"]["resources"]["requests"]["storage"] == "100Gi"

    def test_audit_pvc_size(self):
        docs = {
            d["metadata"]["name"]: d for d in _load_yaml_docs(INFERENCE_DIR / "pvc.yaml")
        }
        assert docs["distllm-audit"]["spec"]["resources"]["requests"]["storage"] == "10Gi"


# ===========================================================================
# Network policy
# ===========================================================================


class TestNetworkPolicy:
    def test_network_policy_template_exists(self):
        assert (INFERENCE_DIR / "network-policy.yaml").exists()

    def test_coordinator_ingress_from_workers(self):
        coord = _netpol_docs()["coordinator"]
        worker_sources = [
            src
            for rule in coord["spec"]["ingress"]
            for src in rule.get("from", [])
            if src.get("podSelector", {}).get("matchLabels", {}).get(
                "app.kubernetes.io/component"
            ) == "worker"
        ]
        assert worker_sources, "coordinator NetworkPolicy does not allow ingress from workers"
        ports = [p["port"] for rule in coord["spec"]["ingress"] for p in rule.get("ports", [])]
        assert 50050 in ports  # worker gRPC

    def test_worker_ingress_from_coordinator(self):
        worker = _netpol_docs()["worker"]
        coord_sources = [
            src
            for rule in worker["spec"]["ingress"]
            for src in rule.get("from", [])
            if src.get("podSelector", {}).get("matchLabels", {}).get(
                "app.kubernetes.io/component"
            ) == "coordinator"
        ]
        assert coord_sources, "worker NetworkPolicy does not allow ingress from the coordinator"
        ports = [p["port"] for rule in worker["spec"]["ingress"] for p in rule.get("ports", [])]
        assert 50051 in ports  # worker gRPC

    def test_both_policy_types_present(self):
        for doc in _load_yaml_docs(INFERENCE_DIR / "network-policy.yaml"):
            assert set(doc["spec"]["policyTypes"]) == {"Ingress", "Egress"}
