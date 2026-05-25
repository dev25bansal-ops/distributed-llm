"""Tests: Kustomize overlays — dev, production, base consistency."""

from pathlib import Path

import pytest
import yaml


KUSTOMIZE_DIR = Path(__file__).parent.parent.parent / "deploy" / "kustomize"


def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


class TestKustomizeBase:
    def test_base_kustomization_exists(self):
        assert (KUSTOMIZE_DIR / "base" / "kustomization.yaml").exists()

    def test_base_has_namespace(self):
        k = _load_yaml(KUSTOMIZE_DIR / "base" / "kustomization.yaml")
        assert k["namespace"] == "distllm"

    def test_base_has_resources(self):
        k = _load_yaml(KUSTOMIZE_DIR / "base" / "kustomization.yaml")
        assert len(k["resources"]) >= 1

    def test_base_has_configmap_generator(self):
        k = _load_yaml(KUSTOMIZE_DIR / "base" / "kustomization.yaml")
        assert "configMapGenerator" in k
        assert len(k["configMapGenerator"]) >= 1

    def test_base_namespace_yaml(self):
        ns = _load_yaml(KUSTOMIZE_DIR / "base" / "namespace.yaml")
        assert ns["kind"] == "Namespace"
        assert ns["metadata"]["name"] == "distllm"

    def test_base_operator_patch_exists(self):
        assert (KUSTOMIZE_DIR / "base" / "operator-patch.yaml").exists()

    def test_base_patch_has_coordinator_resources(self):
        patch = (KUSTOMIZE_DIR / "base" / "operator-patch.yaml").read_text()
        assert "distllm-coordinator" in patch
        assert "cpu" in patch
        assert "memory" in patch


class TestKustomizeDevOverlay:
    def test_dev_kustomization_exists(self):
        assert (KUSTOMIZE_DIR / "dev" / "kustomization.yaml").exists()

    def test_dev_namespace_is_dev(self):
        k = _load_yaml(KUSTOMIZE_DIR / "dev" / "kustomization.yaml")
        assert k["namespace"] == "distllm-dev"

    def test_dev_inherits_base(self):
        k = _load_yaml(KUSTOMIZE_DIR / "dev" / "kustomization.yaml")
        assert "../base" in k.get("resources", [])

    def test_dev_patches_exist(self):
        assert (KUSTOMIZE_DIR / "dev" / "resource-patch.yaml").exists()

    def test_dev_coordinator_single_replica(self):
        patch = (KUSTOMIZE_DIR / "dev" / "resource-patch.yaml").read_text()
        assert "replicas: 1" in patch

    def test_dev_worker_single_replica(self):
        patch = (KUSTOMIZE_DIR / "dev" / "resource-patch.yaml").read_text()
        count = patch.count("replicas: 1")
        assert count >= 1

    def test_dev_no_gpu(self):
        patch = (KUSTOMIZE_DIR / "dev" / "resource-patch.yaml").read_text()
        assert "gpu" not in patch.lower()

    def test_dev_environment_label(self):
        k = _load_yaml(KUSTOMIZE_DIR / "dev" / "kustomization.yaml")
        labels = k.get("commonLabels", {})
        assert labels.get("app.kubernetes.io/environment") == "dev"


class TestKustomizeProductionOverlay:
    def test_prod_kustomization_exists(self):
        assert (KUSTOMIZE_DIR / "production" / "kustomization.yaml").exists()

    def test_prod_namespace_is_production(self):
        k = _load_yaml(KUSTOMIZE_DIR / "production" / "kustomization.yaml")
        assert k["namespace"] == "distllm-production"

    def test_prod_inherits_base(self):
        k = _load_yaml(KUSTOMIZE_DIR / "production" / "kustomization.yaml")
        assert "../base" in k.get("resources", [])

    def test_prod_image_tags_pinned(self):
        k = _load_yaml(KUSTOMIZE_DIR / "production" / "kustomization.yaml")
        assert "images" in k
        for img in k["images"]:
            assert img.get("newTag") == "0.4.0"

    def test_prod_coordinator_three_replicas(self):
        patch = (KUSTOMIZE_DIR / "production" / "resource-patch.yaml").read_text()
        assert "replicas: 3" in patch

    def test_prod_worker_four_replicas(self):
        patch = (KUSTOMIZE_DIR / "production" / "resource-patch.yaml").read_text()
        assert "replicas: 4" in patch

    def test_prod_rolling_update_strategy(self):
        patch = (KUSTOMIZE_DIR / "production" / "resource-patch.yaml").read_text()
        assert "RollingUpdate" in patch
        assert "maxUnavailable: 0" in patch

    def test_prod_gpu_requests(self):
        patch = (KUSTOMIZE_DIR / "production" / "resource-patch.yaml").read_text()
        assert "nvidia.com/gpu" in patch

    def test_prod_worker_large_memory(self):
        patch = (KUSTOMIZE_DIR / "production" / "resource-patch.yaml").read_text()
        assert "32Gi" in patch

    def test_prod_environment_label(self):
        k = _load_yaml(KUSTOMIZE_DIR / "production" / "kustomization.yaml")
        labels = k.get("commonLabels", {})
        assert labels.get("app.kubernetes.io/environment") == "production"


class TestKustomizeStagingOverlay:
    def test_staging_kustomization_exists(self):
        assert (KUSTOMIZE_DIR / "staging" / "kustomization.yaml").exists()

    def test_staging_coordinator_three_replicas(self):
        patch = (KUSTOMIZE_DIR / "staging" / "resource-patch.yaml").read_text()
        assert "replicas: 3" in patch

    def test_staging_worker_two_replicas(self):
        patch = (KUSTOMIZE_DIR / "staging" / "resource-patch.yaml").read_text()
        assert "replicas: 2" in patch
