"""Tests for distllm.dist.provisioning."""
from __future__ import annotations

import time

import pytest

from distllm.dist.cloud_selector import RegionOffer
from distllm.dist.provisioning import Deployment, DeploymentStatus, ProvisioningEngine


class TestDeploymentStatus:
    """Tests for the DeploymentStatus enum."""

    def test_member_values(self) -> None:
        """Each member has the expected string value."""
        assert DeploymentStatus.PENDING.value == "pending"
        assert DeploymentStatus.PROVISIONING.value == "provisioning"
        assert DeploymentStatus.RUNNING.value == "running"
        assert DeploymentStatus.FAILED.value == "failed"
        assert DeploymentStatus.TERMINATED.value == "terminated"

    def test_member_identity(self) -> None:
        """Enum members are comparable by identity."""
        assert DeploymentStatus.PENDING is DeploymentStatus("pending")
        assert DeploymentStatus.RUNNING is not DeploymentStatus.PENDING

    def test_all_members_have_unique_values(self) -> None:
        """All enum values are unique (no duplicates)."""
        values = [m.value for m in DeploymentStatus]
        assert len(values) == len(set(values))


class TestDeployment:
    """Tests for the Deployment dataclass."""

    def test_default_values(self) -> None:
        """Deployment fields have sensible defaults."""
        dep = Deployment(
            deployment_id="dep-abc123",
            tenant_id="tenant-1",
            model_name="llama-70b",
        )
        assert dep.deployment_id == "dep-abc123"
        assert dep.tenant_id == "tenant-1"
        assert dep.model_name == "llama-70b"
        assert dep.gpu_count == 1
        assert dep.min_gpu_memory_gb == 80.0
        assert dep.max_budget_per_hour == 50.0
        assert dep.preferred_regions == []
        assert dep.preferred_providers == []
        assert dep.region_offer is None
        assert dep.assigned_cluster_id == ""
        assert dep.endpoint_url == ""
        assert dep.status == DeploymentStatus.PENDING
        assert dep.created_at > 0
        assert dep.provisioned_at == 0.0
        assert dep.terminated_at == 0.0
        assert dep.error_message == ""
        assert dep.tokens_served == 0
        assert dep.cost_incurred == 0.0

    def test_created_at_is_recent(self) -> None:
        """created_at is set to the current time on construction."""
        before = time.time()
        dep = Deployment(
            deployment_id="dep-abc123",
            tenant_id="tenant-1",
            model_name="llama-70b",
        )
        after = time.time()
        assert before <= dep.created_at <= after

    def test_age_hours_for_running_deployment(self) -> None:
        """age_hours returns elapsed time when deployment is not terminated."""
        created = time.time() - 7200  # 2 hours ago
        dep = Deployment(
            deployment_id="dep-abc123",
            tenant_id="tenant-1",
            model_name="llama-70b",
            created_at=created,
            status=DeploymentStatus.RUNNING,
        )
        age = dep.age_hours
        assert 1.99 <= age <= 2.01  # Allow small clock drift

    def test_age_hours_for_terminated_deployment(self) -> None:
        """age_hours uses terminated_at when deployment is terminated."""
        created = time.time() - 7200  # 2 hours ago
        terminated = time.time() - 1800  # 30 min ago
        dep = Deployment(
            deployment_id="dep-abc123",
            tenant_id="tenant-1",
            model_name="llama-70b",
            created_at=created,
            terminated_at=terminated,
            status=DeploymentStatus.TERMINATED,
        )
        age = dep.age_hours
        assert 1.49 <= age <= 1.51  # Terminated 30 min after creation

    def test_age_hours_for_terminated_with_zero_terminated_at(self) -> None:
        """When terminated but terminated_at is 0, age_hours uses time.time()."""
        created = time.time() - 3600  # 1 hour ago
        dep = Deployment(
            deployment_id="dep-abc123",
            tenant_id="tenant-1",
            model_name="llama-70b",
            created_at=created,
            terminated_at=0.0,
            status=DeploymentStatus.TERMINATED,
        )
        age = dep.age_hours
        assert 0.99 <= age <= 1.01

    def test_to_dict_no_region_offer(self) -> None:
        """to_dict works when region_offer is None."""
        dep = Deployment(
            deployment_id="dep-abc123",
            tenant_id="tenant-1",
            model_name="llama-70b",
        )
        d = dep.to_dict()
        assert d["deployment_id"] == "dep-abc123"
        assert d["tenant_id"] == "tenant-1"
        assert d["model_name"] == "llama-70b"
        assert d["gpu_count"] == 1
        assert d["status"] == "pending"
        assert d["endpoint_url"] == ""
        assert d["assigned_cluster_id"] == ""
        assert d["region"] == ""
        assert d["provider"] == ""
        assert d["price_per_hour"] == 0.0
        assert d["created_at"] == dep.created_at
        assert isinstance(d["age_hours"], float)
        assert d["tokens_served"] == 0
        assert d["cost_incurred"] == 0.0

    def test_to_dict_with_region_offer(self) -> None:
        """to_dict includes region offer details when set."""
        offer = RegionOffer(
            provider="gcp",
            region="us-central1",
            gpu_type="H100",
            gpu_count=8,
            price_per_hour=37.84,
            spot_price_per_hour=11.35,
        )
        dep = Deployment(
            deployment_id="dep-xyz789",
            tenant_id="tenant-2",
            model_name="llama-405b",
            gpu_count=8,
            region_offer=offer,
            status=DeploymentStatus.PROVISIONING,
            endpoint_url="https://example.com/v1/completions",
        )
        d = dep.to_dict()
        assert d["region"] == "us-central1"
        assert d["provider"] == "gcp"
        assert d["price_per_hour"] == 37.84
        assert d["gpu_count"] == 8

    def test_to_dict_contains_all_keys(self) -> None:
        """to_dict returns all expected keys."""
        dep = Deployment(
            deployment_id="dep-abc123",
            tenant_id="tenant-1",
            model_name="llama-70b",
        )
        d = dep.to_dict()
        expected_keys = {
            "deployment_id", "tenant_id", "model_name", "gpu_count",
            "status", "endpoint_url", "assigned_cluster_id",
            "region", "provider", "price_per_hour",
            "created_at", "age_hours", "tokens_served", "cost_incurred",
        }
        assert set(d.keys()) == expected_keys


class TestProvisioningEngine:
    """Tests for the ProvisioningEngine class."""

    def test_init_defaults(self) -> None:
        """Engine can be created with no arguments."""
        engine = ProvisioningEngine()
        assert engine._marketplace is None
        assert engine._cloud_selector is not None
        assert engine._auto_scaler is None
        assert engine._federation is None
        assert engine._deployments == {}

    def test_request_deployment_basic(self) -> None:
        """A basic deployment request returns a PENDING then PROVISIONING deployment."""
        engine = ProvisioningEngine()
        dep = engine.request_deployment(
            tenant_id="acme-corp",
            model_name="llama-70b",
        )
        assert dep.tenant_id == "acme-corp"
        assert dep.model_name == "llama-70b"
        assert dep.deployment_id.startswith("dep-")
        assert len(dep.deployment_id) == 16  # "dep-" + 12 hex chars
        # The deployment is either PROVISIONING (region found) or FAILED.
        # With default params and static fallback data a region should be found.
        assert dep.status in (DeploymentStatus.PROVISIONING, DeploymentStatus.RUNNING)
        # The engine stores the deployment.
        assert engine.get_deployment(dep.deployment_id) is not None

    def test_request_deployment_zero_budget_fails(self) -> None:
        """A deployment with zero max budget cannot find a region."""
        engine = ProvisioningEngine()
        dep = engine.request_deployment(
            tenant_id="acme-corp",
            model_name="llama-70b",
            max_budget_per_hour=0.001,
        )
        assert dep.status == DeploymentStatus.FAILED
        assert dep.error_message == "No qualifying region found"

    def test_request_deployment_with_preferred_regions(self) -> None:
        """Preferred regions filter the candidates."""
        engine = ProvisioningEngine()
        dep = engine.request_deployment(
            tenant_id="acme-corp",
            model_name="llama-70b",
            preferred_regions=["nonexistent-region"],
        )
        # If no region matches the preference, it fails.
        assert dep.status == DeploymentStatus.FAILED
        assert dep.error_message == "No qualifying region found"

    def test_request_deployment_recorded_in_engine(self) -> None:
        """request_deployment stores the deployment in the engine."""
        engine = ProvisioningEngine()
        dep = engine.request_deployment(
            tenant_id="acme-corp",
            model_name="llama-70b",
            preferred_regions=["us-east-1"],
        )
        retrieved = engine.get_deployment(dep.deployment_id)
        assert retrieved is not None
        assert retrieved["tenant_id"] == "acme-corp"
        assert retrieved["model_name"] == "llama-70b"

    def test_terminate_deployment(self) -> None:
        """terminate_deployment sets status to TERMINATED and returns True."""
        engine = ProvisioningEngine()
        dep = engine.request_deployment(
            tenant_id="acme-corp",
            model_name="llama-70b",
        )
        result = engine.terminate_deployment(dep.deployment_id)
        assert result is True
        retrieved = engine.get_deployment(dep.deployment_id)
        assert retrieved is not None
        assert retrieved["status"] == "terminated"

    def test_terminate_deployment_unknown(self) -> None:
        """terminate_deployment returns False for unknown deployment."""
        engine = ProvisioningEngine()
        result = engine.terminate_deployment("nonexistent")
        assert result is False

    def test_record_usage_updates_tokens_and_cost(self) -> None:
        """record_usage increments tokens and computes cost."""
        engine = ProvisioningEngine()
        dep = engine.request_deployment(
            tenant_id="acme-corp",
            model_name="llama-70b",
        )
        engine.record_usage(dep.deployment_id, tokens=5000)
        retrieved = engine.get_deployment(dep.deployment_id)
        assert retrieved is not None
        assert retrieved["tokens_served"] == 5000
        assert retrieved["cost_incurred"] > 0

    def test_record_usage_accumulates(self) -> None:
        """Multiple record_usage calls accumulate tokens and cost."""
        engine = ProvisioningEngine()
        dep = engine.request_deployment(
            tenant_id="acme-corp",
            model_name="llama-70b",
        )
        engine.record_usage(dep.deployment_id, tokens=1000)
        engine.record_usage(dep.deployment_id, tokens=2000)
        retrieved = engine.get_deployment(dep.deployment_id)
        assert retrieved is not None
        assert retrieved["tokens_served"] == 3000

    def test_record_usage_unknown_noop(self) -> None:
        """record_usage does nothing for unknown deployment."""
        engine = ProvisioningEngine()
        # Should not raise.
        engine.record_usage("nonexistent", tokens=1000)

    def test_get_deployment_unknown(self) -> None:
        """get_deployment returns None for unknown deployment."""
        engine = ProvisioningEngine()
        assert engine.get_deployment("nonexistent") is None

    def test_get_deployment_returns_dict(self) -> None:
        """get_deployment returns a dict with the expected keys."""
        engine = ProvisioningEngine()
        dep = engine.request_deployment(
            tenant_id="acme-corp",
            model_name="llama-70b",
        )
        result = engine.get_deployment(dep.deployment_id)
        assert result is not None
        assert isinstance(result, dict)
        assert result["deployment_id"] == dep.deployment_id

    def test_list_deployments_all(self) -> None:
        """list_deployments returns all deployments when no filters."""
        engine = ProvisioningEngine()
        d1 = engine.request_deployment(tenant_id="t1", model_name="llama-70b")
        d2 = engine.request_deployment(tenant_id="t2", model_name="llama-13b")
        all_deps = engine.list_deployments()
        dep_ids = {d["deployment_id"] for d in all_deps}
        assert d1.deployment_id in dep_ids
        assert d2.deployment_id in dep_ids

    def test_list_deployments_filter_by_tenant(self) -> None:
        """list_deployments filters by tenant_id."""
        engine = ProvisioningEngine()
        d1 = engine.request_deployment(tenant_id="t1", model_name="llama-70b")
        engine.request_deployment(tenant_id="t2", model_name="llama-13b")
        t1_deps = engine.list_deployments(tenant_id="t1")
        assert len(t1_deps) == 1
        assert t1_deps[0]["deployment_id"] == d1.deployment_id

    def test_list_deployments_filter_by_status(self) -> None:
        """list_deployments filters by status."""
        engine = ProvisioningEngine()
        engine.request_deployment(tenant_id="t1", model_name="llama-70b")
        # Terminate one to create a varied status.
        d2 = engine.request_deployment(tenant_id="t1", model_name="llama-13b")
        engine.terminate_deployment(d2.deployment_id)

        terminated = engine.list_deployments(status=DeploymentStatus.TERMINATED)
        for d in terminated:
            assert d["status"] == "terminated"

    def test_list_deployments_single_tenant_no_match(self) -> None:
        """list_deployments returns empty list for non-existent tenant."""
        engine = ProvisioningEngine()
        engine.request_deployment(tenant_id="t1", model_name="llama-70b")
        result = engine.list_deployments(tenant_id="nonexistent")
        assert result == []

    def test_get_usage_report_empty(self) -> None:
        """get_usage_report returns zeros when no deployments exist."""
        engine = ProvisioningEngine()
        report = engine.get_usage_report()
        assert report["total_deployments"] == 0
        assert report["active_deployments"] == 0
        assert report["total_tokens_served"] == 0
        assert report["total_cost_incurred"] == 0.0
        assert report["active_by_tenant"] == {}

    def test_get_usage_report_with_deployments(self) -> None:
        """get_usage_report reflects engine state."""
        engine = ProvisioningEngine()
        d1 = engine.request_deployment(tenant_id="t1", model_name="llama-70b")
        engine.request_deployment(tenant_id="t2", model_name="llama-13b")
        engine.record_usage(d1.deployment_id, tokens=1000)

        report = engine.get_usage_report()
        assert report["total_deployments"] == 2
        assert report["total_tokens_served"] >= 1000
        assert report["total_cost_incurred"] > 0
        assert "t1" in report["active_by_tenant"]
        assert "t2" in report["active_by_tenant"]

    def test_get_usage_report_active_by_tenant(self) -> None:
        """active_by_tenant lists each tenant with a count."""
        engine = ProvisioningEngine()
        engine.request_deployment(tenant_id="t1", model_name="llama-70b")
        engine.request_deployment(tenant_id="t1", model_name="llama-13b")
        engine.request_deployment(tenant_id="t2", model_name="llama-70b")

        report = engine.get_usage_report()
        assert report["total_deployments"] == 3
        # All deployments are still PROVISIONING (bg provisioning thread
        # sleeps for 5s before setting RUNNING), so active counts are 0.
        assert set(report["active_by_tenant"].keys()) == {"t1", "t2"}
        assert report["active_by_tenant"]["t1"] == 0
        assert report["active_by_tenant"]["t2"] == 0

    def test_request_deployment_without_cloud_selector_fails(self) -> None:
        """When cloud_selector is None but needed, it fails gracefully."""
        engine = ProvisioningEngine(
            cloud_selector=None,
        )
        # The constructor creates a default CloudRegionSelector when None is passed.
        dep = engine.request_deployment(
            tenant_id="acme-corp",
            model_name="llama-70b",
        )
        # With the default selector, deployment should proceed.
        assert dep.status in (
            DeploymentStatus.PROVISIONING,
            DeploymentStatus.RUNNING,
            DeploymentStatus.FAILED,
        )

    def test_multiple_independent_deployments(self) -> None:
        """Multiple deployments from different tenants are independent."""
        engine = ProvisioningEngine()
        d1 = engine.request_deployment(
            tenant_id="t1", model_name="llama-70b",
        )
        d2 = engine.request_deployment(
            tenant_id="t2", model_name="llama-13b",
        )
        assert d1.deployment_id != d2.deployment_id
        assert engine.get_deployment(d1.deployment_id) is not None
        assert engine.get_deployment(d2.deployment_id) is not None

    def test_record_usage_with_zero_tokens(self) -> None:
        """record_usage with zero tokens does not change cost."""
        engine = ProvisioningEngine()
        dep = engine.request_deployment(
            tenant_id="acme-corp",
            model_name="llama-70b",
        )
        engine.record_usage(dep.deployment_id, tokens=0)
        retrieved = engine.get_deployment(dep.deployment_id)
        assert retrieved is not None
        assert retrieved["tokens_served"] == 0
        assert retrieved["cost_incurred"] == 0.0

    def test_deployment_id_uniqueness(self) -> None:
        """Each deployment gets a unique ID."""
        engine = ProvisioningEngine()
        ids = set()
        for _ in range(10):
            dep = engine.request_deployment(
                tenant_id="t1",
                model_name="llama-70b",
            )
            ids.add(dep.deployment_id)
        assert len(ids) == 10
