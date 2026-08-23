"""Tests for geo-aware cross-cluster routing (GeoRouter, LoadReporter, ClusterLoad).

Covers:
- ClusterLoad dataclass and computed properties (overloaded, capacity)
- LoadReporter: report, get, remove, stale detection
- TrafficSplit dataclass
- GeoRouter: traffic splits, version selection, version pinning
- GeoRouter: target cluster selection (local, spillover, fallback)
- GeoRouter: edge node selection
- GeoRouter: A/B metrics recording and retrieval
"""

from __future__ import annotations

import random
import time
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Test doubles for federation and latency monitor interfaces (no mocks)
# ---------------------------------------------------------------------------

class _FakeFederation:
    """Minimal federation double exposing the methods GeoRouter calls."""

    def __init__(self) -> None:
        self._clusters: list[str] = ["us-east", "us-west", "eu-west"]
        self._edge_nodes: dict[str, list[str]] = {
            "us-east": ["node-1", "node-2"],
            "us-west": ["node-3"],
            "eu-west": [],
        }
        self._nodes_in_cluster: dict[str, list[str]] = {
            "us-east": ["node-1", "node-2"],
            "us-west": ["node-3", "node-4"],
            "eu-west": ["node-5"],
            "empty-cluster": [],
        }

    def list_clusters(self) -> list[str]:
        return list(self._clusters)

    def get_edge_nodes(self, cluster_id: str) -> list[str]:
        return list(self._edge_nodes.get(cluster_id, []))

    def get_nodes_in_cluster(self, cluster_id: str) -> list[str]:
        return list(self._nodes_in_cluster.get(cluster_id, []))

    def stats(self) -> dict[str, Any]:
        return {"clusters": len(self._clusters)}


class _FakeLatencyMonitor:
    """Minimal latency monitor double."""

    def __init__(self) -> None:
        self._latencies: dict[tuple[str, str], float] = {
            ("default", "us-east"): 5.0,
            ("default", "us-west"): 30.0,
            ("default", "eu-west"): 100.0,
            ("us-east", "us-west"): 10.0,
            ("us-east", "eu-west"): 80.0,
            ("us-west", "eu-west"): 60.0,
        }

    def get_latency(self, source: str, target: str) -> float:
        return self._latencies.get((source, target), 999.0)

    def stats(self) -> dict[str, Any]:
        return {"entries": len(self._latencies)}


# ===================================================================
# ClusterLoad
# ===================================================================

class TestClusterLoad:
    """ClusterLoad dataclass and its computed properties."""

    def test_default_values(self) -> None:
        from distllm.dist.geo import ClusterLoad

        load = ClusterLoad(cluster_id="test-cluster")
        assert load.cluster_id == "test-cluster"
        assert load.active_requests == 0
        assert load.pending_requests == 0
        assert load.gpu_utilization == 0.0
        assert load.queue_depth == 0
        assert isinstance(load.last_updated, float)

    def test_is_overloaded_false_by_default(self) -> None:
        from distllm.dist.geo import ClusterLoad

        load = ClusterLoad(cluster_id="c")
        assert not load.is_overloaded

    def test_is_overloaded_true_high_gpu(self) -> None:
        from distllm.dist.geo import ClusterLoad

        load = ClusterLoad(cluster_id="c", gpu_utilization=0.86)
        assert load.is_overloaded

    def test_is_overloaded_true_high_queue(self) -> None:
        from distllm.dist.geo import ClusterLoad

        load = ClusterLoad(cluster_id="c", queue_depth=51)
        assert load.is_overloaded

    def test_is_overloaded_at_threshold(self) -> None:
        from distllm.dist.geo import ClusterLoad

        load = ClusterLoad(cluster_id="c", gpu_utilization=0.75, queue_depth=10)
        assert load.is_overloaded_at(0.7)
        assert not load.is_overloaded_at(0.8)

    def test_is_overloaded_at_queue_over_threshold(self) -> None:
        from distllm.dist.geo import ClusterLoad

        # queue_threshold = int(0.5 * 100) = 50; queue_depth=60 > 50
        load = ClusterLoad(cluster_id="c", gpu_utilization=0.1, queue_depth=60)
        assert load.is_overloaded_at(0.5)

    def test_available_capacity_full(self) -> None:
        from distllm.dist.geo import ClusterLoad

        load = ClusterLoad(cluster_id="c", gpu_utilization=0.0, queue_depth=0)
        assert load.available_capacity == 1.0

    def test_available_capacity_partial(self) -> None:
        from distllm.dist.geo import ClusterLoad

        load = ClusterLoad(cluster_id="c", gpu_utilization=0.3, queue_depth=0)
        assert load.available_capacity == 0.7

    def test_available_capacity_zero(self) -> None:
        from distllm.dist.geo import ClusterLoad

        load = ClusterLoad(cluster_id="c", gpu_utilization=0.95, queue_depth=100)
        assert load.available_capacity == 0.0

    def test_available_capacity_clamps_to_zero(self) -> None:
        from distllm.dist.geo import ClusterLoad

        # queue_depth/100 = 2.0, min(2.0, 1.0) = 1.0, capacity = 0.0
        load = ClusterLoad(cluster_id="c", gpu_utilization=0.5, queue_depth=200)
        assert load.available_capacity == 0.0

    def test_available_capacity_overflow_gpu_clamps(self) -> None:
        from distllm.dist.geo import ClusterLoad

        # gpu_utilization > 1.0 should still result in 0.0 capacity
        load = ClusterLoad(cluster_id="c", gpu_utilization=1.5, queue_depth=0)
        assert load.available_capacity == 0.0

    def test_is_overloaded_boundary_gpu(self) -> None:
        """Boundary: gpu_utilization == 0.85 is not overloaded (> not >=)."""
        from distllm.dist.geo import ClusterLoad

        load = ClusterLoad(cluster_id="c", gpu_utilization=0.85, queue_depth=0)
        assert not load.is_overloaded

    def test_is_overloaded_boundary_queue(self) -> None:
        """Boundary: queue_depth == 50 is not overloaded."""
        from distllm.dist.geo import ClusterLoad

        load = ClusterLoad(cluster_id="c", gpu_utilization=0.0, queue_depth=50)
        assert not load.is_overloaded


# ===================================================================
# LoadReporter
# ===================================================================

class TestLoadReporter:
    """LoadReporter — report, get, remove, and stale-data detection."""

    def test_report_and_get_load(self) -> None:
        from distllm.dist.geo import LoadReporter

        reporter = LoadReporter(stale_threshold_s=300)
        reporter.report("us-east", active=10, pending=2, gpu_util=0.5, queue_depth=5)
        load = reporter.get_load("us-east")
        assert load is not None
        assert load.cluster_id == "us-east"
        assert load.active_requests == 10
        assert load.pending_requests == 2
        assert load.gpu_utilization == 0.5
        assert load.queue_depth == 5

    def test_get_load_missing(self) -> None:
        from distllm.dist.geo import LoadReporter

        reporter = LoadReporter()
        assert reporter.get_load("nonexistent") is None

    def test_get_all_loads(self) -> None:
        from distllm.dist.geo import LoadReporter

        reporter = LoadReporter()
        reporter.report("a", active=1)
        reporter.report("b", active=2)
        loads = reporter.get_all_loads()
        assert set(loads) == {"a", "b"}
        assert loads["a"].active_requests == 1
        assert loads["b"].active_requests == 2

    def test_remove_cluster(self) -> None:
        from distllm.dist.geo import LoadReporter

        reporter = LoadReporter()
        reporter.report("us-east")
        assert reporter.get_load("us-east") is not None
        reporter.remove_cluster("us-east")
        assert reporter.get_load("us-east") is None

    def test_get_all_loads_returns_copy(self) -> None:
        """get_all_loads should return a dict that is independent of internal state."""
        from distllm.dist.geo import LoadReporter

        reporter = LoadReporter()
        reporter.report("a")
        loads = reporter.get_all_loads()
        reporter.remove_cluster("a")
        # The previously returned dict should still have "a"
        assert "a" in loads

    def test_stale_load_returns_none(self) -> None:
        from distllm.dist.geo import LoadReporter

        reporter = LoadReporter(stale_threshold_s=0.0)
        reporter.report("us-east", active=5)
        # With a zero threshold, any elapsed time makes the data stale.
        time.sleep(0.01)
        assert reporter.get_load("us-east") is None

    def test_report_updates_existing(self) -> None:
        from distllm.dist.geo import LoadReporter

        reporter = LoadReporter(stale_threshold_s=300)
        reporter.report("c", active=1)
        reporter.report("c", active=99)
        load = reporter.get_load("c")
        assert load is not None
        assert load.active_requests == 99

    def test_remove_nonexistent(self) -> None:
        """Removing a cluster that was never added should not raise."""
        from distllm.dist.geo import LoadReporter

        reporter = LoadReporter()
        reporter.remove_cluster("ghost")
        # Implicit pass — no exception means success.


# ===================================================================
# TrafficSplit
# ===================================================================

class TestTrafficSplit:
    """TrafficSplit dataclass."""

    def test_default_values(self) -> None:
        from distllm.dist.geo import TrafficSplit

        split = TrafficSplit(version="v2", weight=0.1)
        assert split.version == "v2"
        assert split.weight == 0.1
        assert split.deployment_id == ""
        assert isinstance(split.created_at, float)

    def test_with_deployment_id(self) -> None:
        from distllm.dist.geo import TrafficSplit

        split = TrafficSplit(version="v2", weight=0.25, deployment_id="canary-042")
        assert split.deployment_id == "canary-042"

    def test_zero_weight(self) -> None:
        from distllm.dist.geo import TrafficSplit

        split = TrafficSplit(version="v2", weight=0.0)
        assert split.weight == 0.0

    def test_full_weight(self) -> None:
        from distllm.dist.geo import TrafficSplit

        split = TrafficSplit(version="v2", weight=1.0)
        assert split.weight == 1.0


# ===================================================================
# GeoRouter
# ===================================================================

class TestGeoRouter:
    """GeoRouter — traffic splits, version routing, target/edge selection."""

    # -- constructor ---------------------------------------------------

    def test_creates_default_load_reporter(self) -> None:
        from distllm.dist.geo import GeoRouter, LoadReporter

        router = GeoRouter(_FakeFederation(), _FakeLatencyMonitor())
        assert isinstance(router._load_reporter, LoadReporter)

    # -- traffic split API --------------------------------------------

    def test_set_traffic_split(self) -> None:
        from distllm.dist.geo import GeoRouter

        router = GeoRouter(_FakeFederation(), _FakeLatencyMonitor())
        router.set_traffic_split("model-v2", 0.1, deployment_id="canary-042")
        splits = router.get_traffic_splits()
        assert "model-v2" in splits
        assert splits["model-v2"]["weight"] == 0.1
        assert splits["model-v2"]["deployment_id"] == "canary-042"

    def test_set_traffic_split_zero_removes(self) -> None:
        from distllm.dist.geo import GeoRouter

        router = GeoRouter(_FakeFederation(), _FakeLatencyMonitor())
        router.set_traffic_split("model-v2", 0.1)
        router.set_traffic_split("model-v2", 0.0)  # removes it
        assert "model-v2" not in router.get_traffic_splits()

    def test_set_traffic_split_invalid_weight_negative(self) -> None:
        from distllm.dist.geo import GeoRouter

        router = GeoRouter(_FakeFederation(), _FakeLatencyMonitor())
        with pytest.raises(ValueError, match="weight"):
            router.set_traffic_split("model-v2", -0.1)

    def test_set_traffic_split_invalid_weight_over_one(self) -> None:
        from distllm.dist.geo import GeoRouter

        router = GeoRouter(_FakeFederation(), _FakeLatencyMonitor())
        with pytest.raises(ValueError, match="weight"):
            router.set_traffic_split("model-v2", 1.5)

    def test_get_traffic_splits_empty(self) -> None:
        from distllm.dist.geo import GeoRouter

        router = GeoRouter(_FakeFederation(), _FakeLatencyMonitor())
        assert router.get_traffic_splits() == {}

    # -- version pinning (blue/green) ---------------------------------

    def test_set_version_route_reflected_in_splits(self) -> None:
        from distllm.dist.geo import GeoRouter

        router = GeoRouter(_FakeFederation(), _FakeLatencyMonitor())
        router.set_version_route("model-v2", "eu-west")
        router.set_traffic_split("model-v2", 0.5)
        splits = router.get_traffic_splits()
        assert splits["model-v2"]["pinned_cluster"] == "eu-west"

    # -- version selection (A/B routing) ------------------------------

    def test_select_version_no_split(self) -> None:
        """When no traffic split exists for the version, it is used as-is."""
        from distllm.dist.geo import GeoRouter

        router = GeoRouter(_FakeFederation(), _FakeLatencyMonitor())
        _, _, meta = router.select_target_cluster(model_version="model-v1")
        assert meta["model_version"] == "model-v1"
        assert meta["deployment"] is None

    def test_select_version_none(self) -> None:
        """model_version=None should stay None (no A/B involved)."""
        from distllm.dist.geo import GeoRouter

        router = GeoRouter(_FakeFederation(), _FakeLatencyMonitor())
        _, _, meta = router.select_target_cluster(model_version=None)
        assert meta["model_version"] is None
        assert meta["deployment"] is None

    def test_select_version_traffic_split_full_migration(self) -> None:
        """weight=1.0 always selects the new version (deployment-only when
        the selected version differs from the requested version)."""
        from distllm.dist.geo import GeoRouter

        router = GeoRouter(_FakeFederation(), _FakeLatencyMonitor())
        router.set_traffic_split("model-v2", 1.0, deployment_id="full-rollout")
        _, _, meta = router.select_target_cluster(model_version="model-v2")
        # The version is selected as-is; deployment metadata is only attached
        # when the selected version differs from the requested version.
        assert meta["model_version"] == "model-v2"
        # When model_version equals the selected version, no A/B tag is set.
        assert meta["deployment"] is None

    def test_select_version_traffic_split_zero_is_removed(self) -> None:
        """weight=0.0 removes the split, so the version is used as-is."""
        from distllm.dist.geo import GeoRouter

        router = GeoRouter(_FakeFederation(), _FakeLatencyMonitor())
        router.set_traffic_split("model-v2", 0.0)
        _, _, meta = router.select_target_cluster(model_version="model-v2")
        assert meta["model_version"] == "model-v2"
        assert meta["deployment"] is None

    def test_version_pinned_routing_bypasses_geo_scoring(self) -> None:
        """When a version is pinned, select_target_cluster routes directly."""
        from distllm.dist.geo import GeoRouter

        router = GeoRouter(_FakeFederation(), _FakeLatencyMonitor())
        router.set_version_route("model-v2", "eu-west")
        router.set_traffic_split("model-v2", 1.0)
        target, reason, meta = router.select_target_cluster(
            model_version="model-v2"
        )
        assert target == "eu-west"
        assert reason == "version_pinned:model-v2"

    # -- target cluster selection --------------------------------------

    def test_local_capacity_routing(self) -> None:
        """When the source cluster has capacity, route to self."""
        from distllm.dist.geo import GeoRouter

        router = GeoRouter(_FakeFederation(), _FakeLatencyMonitor())
        router.report_cluster_load("default", active=1, gpu_util=0.3, queue_depth=5)
        target, reason, _ = router.select_target_cluster(source_cluster="default")
        assert target == "default"
        assert reason == "local_capacity"

    def test_spillover_to_nearest_with_capacity(self) -> None:
        """When local is overloaded, spill to the nearest non-overloaded cluster."""
        from distllm.dist.geo import GeoRouter

        router = GeoRouter(_FakeFederation(), _FakeLatencyMonitor())
        # Overload local cluster
        router.report_cluster_load("default", active=50, gpu_util=0.95, queue_depth=100)
        # Remote clusters have capacity
        router.report_cluster_load("us-east", active=5, gpu_util=0.3, queue_depth=5)
        router.report_cluster_load("us-west", active=5, gpu_util=0.3, queue_depth=5)
        router.report_cluster_load("eu-west", active=5, gpu_util=0.3, queue_depth=5)

        target, reason, _ = router.select_target_cluster(source_cluster="default")
        # us-east has the lowest latency from default (5ms)
        assert target == "us-east"
        assert "nearest_with_capacity" in reason

    def test_all_remote_overloaded_fallback(self) -> None:
        """When every remote cluster is overloaded, fall back to source."""
        from distllm.dist.geo import GeoRouter

        router = GeoRouter(_FakeFederation(), _FakeLatencyMonitor())
        router.report_cluster_load("default", active=50, gpu_util=0.95, queue_depth=100)
        router.report_cluster_load("us-east", gpu_util=0.9, queue_depth=60)
        router.report_cluster_load("us-west", gpu_util=0.9, queue_depth=60)
        router.report_cluster_load("eu-west", gpu_util=0.9, queue_depth=60)

        target, reason, _ = router.select_target_cluster(source_cluster="default")
        assert target == "default"
        assert reason == "all_remote_overloaded"

    def test_no_alternative_cluster(self) -> None:
        """When only the local cluster exists, fall back with no_alternative."""
        from distllm.dist.geo import GeoRouter

        fed = _FakeFederation()
        fed._clusters = ["default"]
        router = GeoRouter(fed, _FakeLatencyMonitor())
        router.report_cluster_load("default", active=50, gpu_util=0.95, queue_depth=100)

        target, reason, _ = router.select_target_cluster(source_cluster="default")
        assert target == "default"
        assert reason == "no_alternative"

    def test_remote_overloaded_skipped_in_scoring(self) -> None:
        """Overloaded remote clusters are skipped during candidate scoring."""
        from distllm.dist.geo import GeoRouter

        router = GeoRouter(_FakeFederation(), _FakeLatencyMonitor())
        router.report_cluster_load("default", active=50, gpu_util=0.95, queue_depth=100)
        # Only us-west has capacity
        router.report_cluster_load("us-east", gpu_util=0.9, queue_depth=60)
        router.report_cluster_load("us-west", active=2, gpu_util=0.2, queue_depth=3)
        router.report_cluster_load("eu-west", gpu_util=0.9, queue_depth=60)

        target, reason, _ = router.select_target_cluster(source_cluster="default")
        assert target == "us-west"
        assert "nearest_with_capacity" in reason

    # -- edge node selection -------------------------------------------

    def test_select_edge_node_returns_node_from_edge_list(self) -> None:
        from distllm.dist.geo import GeoRouter

        router = GeoRouter(_FakeFederation(), _FakeLatencyMonitor())
        node = router.select_edge_node("us-east")
        assert node in ("node-1", "node-2")

    def test_select_edge_node_fallback_to_nodes_in_cluster(self) -> None:
        """When edge_nodes is empty, fall back to get_nodes_in_cluster."""
        from distllm.dist.geo import GeoRouter

        router = GeoRouter(_FakeFederation(), _FakeLatencyMonitor())
        # eu-west has no edge nodes but has node-5 via get_nodes_in_cluster
        node = router.select_edge_node("eu-west")
        assert node == "node-5"

    def test_select_edge_node_no_nodes_returns_none(self) -> None:
        """A cluster with no edge nodes and no in-cluster nodes returns None."""
        from distllm.dist.geo import GeoRouter

        router = GeoRouter(_FakeFederation(), _FakeLatencyMonitor())
        # empty-cluster has no edge nodes and an empty nodes_in_cluster list
        node = router.select_edge_node("empty-cluster")
        assert node is None

    def test_select_edge_node_single_node(self) -> None:
        """A cluster with exactly one edge node returns that node directly."""
        from distllm.dist.geo import GeoRouter

        router = GeoRouter(_FakeFederation(), _FakeLatencyMonitor())
        # us-west has exactly one edge node
        node = router.select_edge_node("us-west")
        assert node == "node-3"

    # -- A/B metrics ---------------------------------------------------

    def test_record_and_get_ab_metrics(self) -> None:
        from distllm.dist.geo import GeoRouter

        router = GeoRouter(_FakeFederation(), _FakeLatencyMonitor())
        router.record_ab_metric("model-v2", "latency_p50", 42.0)
        router.record_ab_metric("model-v2", "latency_p50", 58.0)
        metrics = router.get_ab_metrics()
        assert "model-v2" in metrics
        # EMA: 0.9*0 + 0.1*42 = 4.2, then 0.9*4.2 + 0.1*58 = 9.58
        assert metrics["model-v2"]["latency_p50"] == pytest.approx(9.58, rel=1e-9)

    def test_get_ab_metrics_returns_copy(self) -> None:
        """get_ab_metrics should return a dict independent from internal state."""
        from distllm.dist.geo import GeoRouter

        router = GeoRouter(_FakeFederation(), _FakeLatencyMonitor())
        router.record_ab_metric("v1", "err", 1.0)
        metrics = router.get_ab_metrics()
        router.record_ab_metric("v1", "err", 2.0)
        # The previously returned dict should still have the old value
        assert metrics["v1"]["err"] != router.get_ab_metrics()["v1"]["err"]

    def test_record_ab_metric_multiple_versions(self) -> None:
        from distllm.dist.geo import GeoRouter

        router = GeoRouter(_FakeFederation(), _FakeLatencyMonitor())
        router.record_ab_metric("v1", "latency", 10.0)
        router.record_ab_metric("v2", "latency", 20.0)
        metrics = router.get_ab_metrics()
        assert metrics["v1"]["latency"] == pytest.approx(1.0, rel=1e-9)
        assert metrics["v2"]["latency"] == pytest.approx(2.0, rel=1e-9)

    # -- load passthrough helpers --------------------------------------

    def test_get_cluster_load(self) -> None:
        from distllm.dist.geo import GeoRouter

        router = GeoRouter(_FakeFederation(), _FakeLatencyMonitor())
        router.report_cluster_load("us-east", active=5)
        load = router.get_cluster_load("us-east")
        assert load is not None
        assert load.active_requests == 5

    def test_get_cluster_load_none(self) -> None:
        from distllm.dist.geo import GeoRouter

        router = GeoRouter(_FakeFederation(), _FakeLatencyMonitor())
        assert router.get_cluster_load("nonexistent") is None

    # -- stats ---------------------------------------------------------

    def test_stats_structure(self) -> None:
        from distllm.dist.geo import GeoRouter

        router = GeoRouter(_FakeFederation(), _FakeLatencyMonitor())
        router.report_cluster_load("us-east", active=3, gpu_util=0.5)
        s = router.stats()
        assert "cluster_loads" in s
        assert "latency_matrix" in s
        assert "federation" in s
        # Check load entry
        assert s["cluster_loads"]["us-east"]["active"] == 3
        assert s["cluster_loads"]["us-east"]["overloaded"] is False
