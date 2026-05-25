"""Tests: Karpenter EC2NodeClass/NodePool, ArgoCD Application/ApplicationSet, Flux GitRepository/Kustomization."""

from pathlib import Path

import pytest
import yaml

KARPENTER_DIR = Path(__file__).parent.parent.parent / "deploy" / "karpenter"
ARGOCD_DIR = Path(__file__).parent.parent.parent / "deploy" / "gitops" / "argocd"
FLUX_DIR = Path(__file__).parent.parent.parent / "deploy" / "gitops" / "flux"


def _load(path):
    with open(path) as f:
        return yaml.safe_load(f)


# ===========================================================================
# Karpenter — EC2NodeClass
# ===========================================================================


class TestKarpenterEC2NodeClass:
    def test_ec2nodeclass_exists(self):
        assert (KARPENTER_DIR / "ec2nodeclass-gpu.yaml").exists()

    def test_ec2nodeclass_kind(self):
        obj = _load(KARPENTER_DIR / "ec2nodeclass-gpu.yaml")
        assert obj["kind"] == "EC2NodeClass"
        assert obj["apiVersion"] == "karpenter.k8s.aws/v1beta1"

    def test_ami_family_bottlerocket(self):
        obj = _load(KARPENTER_DIR / "ec2nodeclass-gpu.yaml")
        assert obj["spec"]["amiFamily"] == "Bottlerocket"

    def test_block_device_encrypted(self):
        obj = _load(KARPENTER_DIR / "ec2nodeclass-gpu.yaml")
        bdm = obj["spec"]["blockDeviceMappings"][0]
        assert bdm["ebs"]["encrypted"] is True
        assert bdm["ebs"]["volumeSize"] == "100Gi"

    def test_subnet_selector_exists(self):
        obj = _load(KARPENTER_DIR / "ec2nodeclass-gpu.yaml")
        assert len(obj["spec"]["subnetSelectorTerms"]) >= 1

    def test_security_group_selector_exists(self):
        obj = _load(KARPENTER_DIR / "ec2nodeclass-gpu.yaml")
        assert len(obj["spec"]["securityGroupSelectorTerms"]) >= 1


# ===========================================================================
# Karpenter — NodePool spot/ondemand
# ===========================================================================


class TestKarpenterNodePoolSpot:
    def test_spot_nodepool_exists(self):
        assert (KARPENTER_DIR / "nodepool-gpu-spot.yaml").exists()

    def test_spot_capacity_type(self):
        obj = _load(KARPENTER_DIR / "nodepool-gpu-spot.yaml")
        reqs = obj["spec"]["template"]["spec"]["requirements"]
        spot = [r for r in reqs if r["key"] == "karpenter.sh/capacity-type"]
        assert len(spot) == 1
        assert "spot" in spot[0]["values"]

    def test_spot_instance_categories_gpu(self):
        obj = _load(KARPENTER_DIR / "nodepool-gpu-spot.yaml")
        reqs = obj["spec"]["template"]["spec"]["requirements"]
        cats = [r for r in reqs if r["key"] == "karpenter.k8s.aws/instance-category"]
        assert len(cats) == 1
        assert "g" in cats[0]["values"]
        assert "p" in cats[0]["values"]

    def test_spot_gpu_count_exists(self):
        obj = _load(KARPENTER_DIR / "nodepool-gpu-spot.yaml")
        reqs = obj["spec"]["template"]["spec"]["requirements"]
        gpu = [r for r in reqs if r["key"] == "karpenter.k8s.aws/instance-gpu-count"]
        assert len(gpu) == 1
        assert gpu[0]["operator"] == "Exists"

    def test_spot_component_label_worker(self):
        obj = _load(KARPENTER_DIR / "nodepool-gpu-spot.yaml")
        labels = obj["metadata"].get("labels", {})
        assert labels.get("app.kubernetes.io/component") == "worker"

    def test_spot_nodeclass_ref(self):
        obj = _load(KARPENTER_DIR / "nodepool-gpu-spot.yaml")
        ref = obj["spec"]["template"]["spec"]["nodeClassRef"]
        assert ref["name"] == "gpu-ec2-class"
        assert ref["kind"] == "EC2NodeClass"

    def test_spot_limits_defined(self):
        obj = _load(KARPENTER_DIR / "nodepool-gpu-spot.yaml")
        assert "limits" in obj["spec"]
        assert obj["spec"]["limits"]["cpu"] >= 100


class TestKarpenterNodePoolOnDemand:
    def test_ondemand_nodepool_exists(self):
        assert (KARPENTER_DIR / "nodepool-gpu-ondemand.yaml").exists()

    def test_ondemand_capacity_type(self):
        obj = _load(KARPENTER_DIR / "nodepool-gpu-ondemand.yaml")
        reqs = obj["spec"]["template"]["spec"]["requirements"]
        od = [r for r in reqs if r["key"] == "karpenter.sh/capacity-type"]
        assert len(od) == 1
        assert "on-demand" in od[0]["values"]

    def test_ondemand_instance_categories_general(self):
        obj = _load(KARPENTER_DIR / "nodepool-gpu-ondemand.yaml")
        reqs = obj["spec"]["template"]["spec"]["requirements"]
        cats = [r for r in reqs if r["key"] == "karpenter.k8s.aws/instance-category"]
        assert "m" in cats[0]["values"]
        assert "c" in cats[0]["values"]

    def test_ondemand_component_label_coordinator(self):
        obj = _load(KARPENTER_DIR / "nodepool-gpu-ondemand.yaml")
        labels = obj["metadata"].get("labels", {})
        assert labels.get("app.kubernetes.io/component") == "coordinator"

    def test_ondemand_consolidation_policy(self):
        obj = _load(KARPENTER_DIR / "nodepool-gpu-ondemand.yaml")
        disruption = obj["spec"].get("disruption", {})
        assert disruption.get("consolidationPolicy") == "WhenUnderutilized"


class TestKarpenterKustomization:
    def test_kustomization_exists(self):
        assert (KARPENTER_DIR / "kustomization.yaml").exists()

    def test_kustomization_includes_all_resources(self):
        obj = _load(KARPENTER_DIR / "kustomization.yaml")
        resources = obj.get("resources", [])
        assert "ec2nodeclass-gpu.yaml" in resources
        assert "nodepool-gpu-spot.yaml" in resources
        assert "nodepool-gpu-ondemand.yaml" in resources

    def test_kustomization_common_labels(self):
        obj = _load(KARPENTER_DIR / "kustomization.yaml")
        labels = obj.get("commonLabels", {})
        assert labels.get("app.kubernetes.io/name") == "distributed-llm"


# ===========================================================================
# ArgoCD — Application
# ===========================================================================


class TestArgoCDApplication:
    def test_application_exists(self):
        assert (ARGOCD_DIR / "application.yaml").exists()

    def test_application_kind(self):
        obj = _load(ARGOCD_DIR / "application.yaml")
        assert obj["kind"] == "Application"

    def test_application_source_path(self):
        obj = _load(ARGOCD_DIR / "application.yaml")
        assert obj["spec"]["source"]["path"] == "deploy/kustomize/dev"

    def test_application_target_revision(self):
        obj = _load(ARGOCD_DIR / "application.yaml")
        assert obj["spec"]["source"]["targetRevision"] == "main"

    def test_application_dest_namespace(self):
        obj = _load(ARGOCD_DIR / "application.yaml")
        assert obj["spec"]["destination"]["namespace"] == "distllm-dev"

    def test_application_sync_policy_automated(self):
        obj = _load(ARGOCD_DIR / "application.yaml")
        assert obj["spec"]["syncPolicy"]["automated"]["prune"] is True
        assert obj["spec"]["syncPolicy"]["automated"]["selfHeal"] is True

    def test_application_retry_config(self):
        obj = _load(ARGOCD_DIR / "application.yaml")
        retry = obj["spec"]["syncPolicy"]["retry"]
        assert retry["limit"] == 5
        assert retry["backoff"]["maxDuration"] == "3m0s"

    def test_application_sync_wave_annotation(self):
        obj = _load(ARGOCD_DIR / "application.yaml")
        ann = obj["metadata"]["annotations"]
        assert ann["argocd.argoproj.io/sync-wave"] == "0"

    def test_application_finalizer(self):
        obj = _load(ARGOCD_DIR / "application.yaml")
        assert "resources-finalizer.argocd.argoproj.io" in obj["metadata"]["finalizers"]


# ===========================================================================
# ArgoCD — ApplicationSet
# ===========================================================================


class TestArgoCDApplicationSet:
    def test_applicationset_exists(self):
        assert (ARGOCD_DIR / "applicationset.yaml").exists()

    def test_applicationset_kind(self):
        obj = _load(ARGOCD_DIR / "applicationset.yaml")
        assert obj["kind"] == "ApplicationSet"

    def test_applicationset_three_environments(self):
        obj = _load(ARGOCD_DIR / "applicationset.yaml")
        elements = obj["spec"]["generators"][0]["list"]["elements"]
        assert len(elements) == 3
        envs = [e["environment"] for e in elements]
        assert "dev" in envs
        assert "staging" in envs
        assert "production" in envs

    def test_applicationset_dev_autosync(self):
        obj = _load(ARGOCD_DIR / "applicationset.yaml")
        elements = obj["spec"]["generators"][0]["list"]["elements"]
        dev = [e for e in elements if e["environment"] == "dev"][0]
        assert dev["autoSync"] is True

    def test_applicationset_production_no_autosync(self):
        obj = _load(ARGOCD_DIR / "applicationset.yaml")
        elements = obj["spec"]["generators"][0]["list"]["elements"]
        prod = [e for e in elements if e["environment"] == "production"][0]
        assert prod["autoSync"] is False

    def test_applicationset_production_revision_pinned(self):
        obj = _load(ARGOCD_DIR / "applicationset.yaml")
        elements = obj["spec"]["generators"][0]["list"]["elements"]
        prod = [e for e in elements if e["environment"] == "production"][0]
        assert prod["revision"] == "v0.4.0"

    def test_applicationset_staging_revision(self):
        obj = _load(ARGOCD_DIR / "applicationset.yaml")
        elements = obj["spec"]["generators"][0]["list"]["elements"]
        staging = [e for e in elements if e["environment"] == "staging"][0]
        assert staging["revision"] == "release"

    def test_applicationset_template_uses_go_template(self):
        obj = _load(ARGOCD_DIR / "applicationset.yaml")
        tmpl = obj["spec"]["template"]
        assert "{{environment}}" in tmpl["metadata"]["name"]
        assert "{{revision}}" in tmpl["spec"]["source"]["targetRevision"]

    def test_applicationset_namespace_per_environment(self):
        obj = _load(ARGOCD_DIR / "applicationset.yaml")
        tmpl = obj["spec"]["template"]
        assert "{{namespace}}" in tmpl["spec"]["destination"]["namespace"]


# ===========================================================================
# Flux — GitRepository
# ===========================================================================


class TestFluxGitRepository:
    def test_gitrepository_exists(self):
        assert (FLUX_DIR / "gitrepository.yaml").exists()

    def test_gitrepository_kind(self):
        obj = _load(FLUX_DIR / "gitrepository.yaml")
        assert obj["kind"] == "GitRepository"

    def test_gitrepository_interval(self):
        obj = _load(FLUX_DIR / "gitrepository.yaml")
        assert obj["spec"]["interval"] == "1m"

    def test_gitrepository_branch(self):
        obj = _load(FLUX_DIR / "gitrepository.yaml")
        assert obj["spec"]["ref"]["branch"] == "main"

    def test_gitrepository_secret_ref(self):
        obj = _load(FLUX_DIR / "gitrepository.yaml")
        assert obj["spec"]["secretRef"]["name"] == "github-credentials"

    def test_gitrepository_url(self):
        obj = _load(FLUX_DIR / "gitrepository.yaml")
        assert "github.com" in obj["spec"]["url"]


# ===========================================================================
# Flux — Kustomization (postBuild variables)
# ===========================================================================


class TestFluxKustomization:
    def test_kustomization_exists(self):
        assert (FLUX_DIR / "kustomization.yaml").exists()

    def test_kustomization_kind(self):
        obj = _load(FLUX_DIR / "kustomization.yaml")
        assert obj["kind"] == "Kustomization"

    def test_kustomization_interval(self):
        obj = _load(FLUX_DIR / "kustomization.yaml")
        assert obj["spec"]["interval"] == "5m"

    def test_kustomization_path_uses_environment_var(self):
        obj = _load(FLUX_DIR / "kustomization.yaml")
        assert "${environment}" in obj["spec"]["path"]

    def test_kustomization_prune_enabled(self):
        obj = _load(FLUX_DIR / "kustomization.yaml")
        assert obj["spec"]["prune"] is True

    def test_kustomization_source_ref(self):
        obj = _load(FLUX_DIR / "kustomization.yaml")
        ref = obj["spec"]["sourceRef"]
        assert ref["kind"] == "GitRepository"
        assert ref["name"] == "distllm-repo"

    def test_kustomization_health_checks(self):
        obj = _load(FLUX_DIR / "kustomization.yaml")
        health = obj["spec"].get("healthChecks", [])
        assert len(health) >= 1
        check = health[0]
        assert check["kind"] == "Deployment"
        assert "${environment}" in check["namespace"]

    def test_kustomization_postbuild_substitute(self):
        obj = _load(FLUX_DIR / "kustomization.yaml")
        sub = obj["spec"]["postBuild"]["substitute"]
        assert "${replicas}" in sub["replica_count"]
        assert "${model_name}" in sub["model_name"]

    def test_kustomization_depends_on(self):
        obj = _load(FLUX_DIR / "kustomization.yaml")
        deps = obj["spec"].get("dependsOn", [])
        assert any(d["name"] == "distllm-crds" for d in deps)
