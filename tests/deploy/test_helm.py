"""Tests: Helm chart validation for the operator chart in deploy/helm.

The chart installs the DistLLM **operator** plus a default coordinator and
worker; the operator reconciles the CustomResourceDefinitions in crds/. This
file previously described a different, direct-deployment chart (top-level
`image`, `nodePools`, `worker-statefulset.yaml`, `coordinator-ha.yaml`) that no
longer exists, so 40 of its assertions failed against files the chart had long
stopped shipping. The tests below assert the chart that is actually rendered --
every expectation was checked against the template text, not assumed.

Rendering itself is verified in CI by `helm template` (validate-k8s); these
tests are the fast structural check that runs without a helm binary.
"""

from pathlib import Path

import yaml

HELM_DIR = Path(__file__).parent.parent.parent / "deploy" / "helm"
VALUES_PATH = HELM_DIR / "values.yaml"
TEMPLATES_DIR = HELM_DIR / "templates"
CHART_PATH = HELM_DIR / "Chart.yaml"
CRDS_DIR = HELM_DIR / "crds"

# Templates this chart ships. Kept explicit so a deleted file fails a test
# rather than silently reducing coverage.
WORKLOAD_TEMPLATES = ("coordinator.yaml", "worker.yaml", "operator.yaml")
SUPPORT_TEMPLATES = (
    "serviceaccount.yaml",
    "hpa.yaml",
    "servicemonitor.yaml",
    "prometheusrules.yaml",
)


def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _tmpl(name: str) -> str:
    return (TEMPLATES_DIR / name).read_text()


# ===========================================================================
# Chart metadata
# ===========================================================================


class TestChartMetadata:
    def test_chart_yaml_exists(self):
        assert CHART_PATH.exists()

    def test_chart_is_v2_application(self):
        chart = _load_yaml(CHART_PATH)
        assert chart["apiVersion"] == "v2"
        assert chart["type"] == "application"

    def test_chart_version_tracks_app_version(self):
        # The chart and the app ship together; a version skew here means an
        # operator will reconcile CRs against the wrong controller image.
        chart = _load_yaml(CHART_PATH)
        assert chart["version"] == chart["appVersion"]

    def test_values_yaml_exists(self):
        assert VALUES_PATH.exists()


# ===========================================================================
# Values structure — the keys the templates actually read
# ===========================================================================


class TestValuesStructure:
    def test_top_level_keys(self):
        values = _load_yaml(VALUES_PATH)
        for key in ("operator", "controller", "network", "model", "worker", "hpa"):
            assert key in values, f"values.yaml missing {key}"

    def test_controller_default_images(self):
        values = _load_yaml(VALUES_PATH)
        ctrl = values["controller"]
        assert "/" in ctrl["defaultCoordinatorImage"]
        assert "/" in ctrl["defaultWorkerImage"]

    def test_model_name_is_a_value_not_hardcoded(self):
        # worker.yaml marks model.name required; it must be settable.
        assert "name" in _load_yaml(VALUES_PATH)["model"]

    def test_coordinator_ha_toggle_declared(self):
        values = _load_yaml(VALUES_PATH)
        assert values["coordinator"]["ha"]["enabled"] is False

    def test_operator_replica_count(self):
        values = _load_yaml(VALUES_PATH)
        assert values["operator"]["replicaCount"] >= 1


# ===========================================================================
# Workloads
# ===========================================================================


class TestWorkloads:
    def test_workload_templates_exist(self):
        for name in WORKLOAD_TEMPLATES:
            assert (TEMPLATES_DIR / name).is_file(), f"missing {name}"

    def test_support_templates_exist(self):
        for name in SUPPORT_TEMPLATES:
            assert (TEMPLATES_DIR / name).is_file(), f"missing {name}"

    def test_helpers_defined(self):
        helpers = (TEMPLATES_DIR / "_helpers.tpl").read_text()
        for name in (
            "distllm-operator.name",
            "distllm-operator.fullname",
            "distllm-operator.labels",
            "distllm-operator.coordinator-selector-labels",
            "distllm-operator.worker-selector-labels",
        ):
            assert f'define "{name}"' in helpers, f"missing helper {name}"

    def test_worker_requires_a_model(self):
        # A worker with no model has nothing to serve, so the template must
        # refuse to render rather than start a broken pod.
        assert "required" in _tmpl("worker.yaml")
        assert ".Values.model.name" in _tmpl("worker.yaml")

    def test_coordinator_uses_controller_coordinator_image(self):
        assert ".Values.controller.defaultCoordinatorImage" in _tmpl("coordinator.yaml")

    def test_worker_uses_controller_worker_image(self):
        assert ".Values.controller.defaultWorkerImage" in _tmpl("worker.yaml")

    def test_coordinator_replicas_follow_ha_toggle(self):
        # coordinator.ha.enabled was a no-op before; it must reach the manifest.
        tmpl = _tmpl("coordinator.yaml")
        assert ".Values.coordinator.ha.enabled" in tmpl


# ===========================================================================
# CRDs and operator wiring
# ===========================================================================


class TestOperatorAndCRDs:
    def test_crd_directory_present(self):
        assert CRDS_DIR.is_dir(), "crds/ missing: the operator has nothing to reconcile"

    def test_at_least_one_crd(self):
        crds = list(CRDS_DIR.glob("*.yaml"))
        assert crds, "no CRD manifests in crds/"

    def test_crds_are_apiextensions_v1(self):
        # crds/ files are multi-document (a chart may ship several CRDs in one
        # file), so this must iterate documents, not call safe_load.
        checked = 0
        for path in CRDS_DIR.glob("*.yaml"):
            for doc in yaml.safe_load_all(path.read_text()):
                if doc and doc.get("kind") == "CustomResourceDefinition":
                    checked += 1
                    assert doc["apiVersion"] == "apiextensions.k8s.io/v1", path.name
        assert checked, "no CustomResourceDefinition documents found in crds/"

    def test_operator_deployment_uses_replica_count(self):
        assert ".Values.operator.replicaCount" in _tmpl("operator.yaml")


# ===========================================================================
# Template hygiene
# ===========================================================================


class TestTemplateHygiene:
    def test_every_template_has_go_template_syntax(self):
        for path in TEMPLATES_DIR.glob("*.yaml"):
            assert "{{" in path.read_text(), f"{path.name} has no Go template syntax"

    def test_every_template_emits_a_k8s_resource(self):
        for path in TEMPLATES_DIR.glob("*.yaml"):
            text = path.read_text()
            assert "apiVersion" in text and "kind" in text, f"{path.name} is not a resource"

    def test_no_template_hardcodes_a_machine_path(self):
        # autoscaler-style configs have leaked local checkouts before.
        for path in TEMPLATES_DIR.glob("*.yaml"):
            text = path.read_text()
            assert "C:\\" not in text and "D:\\" not in text, f"{path.name} hardcodes a Windows path"