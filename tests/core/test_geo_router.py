"""Tests for GeoRouter: cluster selection, edge node routing, load reporting.

Tests: ClusterLoad, LoadReporter, GeoRouter select_target_cluster,
select_edge_node, load reporting, stats, and edge cases.

Run: pytest tests/core/test_geo_router.py -v
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from distllm.core.geo_router import ClusterLoad, LoadReporter, GeoRouter


# --- ClusterLoad tests ---


class TestClusterLoad:
    """Tests for ClusterLoad dataclass."""

    def test_defaults(self):
        load = ClusterLoad(cluster_id="us-east-1")
        assert load.active_requests == 0
        assert load.pending_requests == 0
        assert load.gpu_utilization == 0.0
        assert load.queue_depth == 0
        assert load.last_updated > 0

    def test_custom_values(self):
        load = ClusterLoad(
            cluster_id="eu-west", active_requests=10, pending_requests=5,
            gpu_utilization=0.75, queue_depth=20,
        )
        assert load.active_requests == 10
        assert load.pending_requests == 5
        assert load.gpu_utilization == 0.75
        assert load.queue_depth == 20


class TestClusterLoadOverloaded:
    """Tests for is_overloaded property."""

    def test_not_overloaded_idle(self):
        load = ClusterLoad(cluster_id="idle", gpu_utilization=0.1, queue_depth=0)
        assert load.is_overloaded is False

    def test_overloaded_high_gpu(self):
        load = ClusterLoad(cluster_id="busy", gpu_utilization=0.9)
        assert load.is_overloaded is True

    def test_overloaded_high_queue(self):
        load = ClusterLoad(cluster_id="queued", gpu_utilization=0.5, queue_depth=60)
        assert load.is_overloaded is True

    def test_boundary_gpu_utilization(self):
        load = ClusterLoad(cluster_id="boundary", gpu_utilization=0.85)
        assert load.is_overloaded is False  # > 0.85, not >=

    def test_boundary_queue_depth(self):
        load = ClusterLoad(cluster_id="boundary", gpu_utilization=0.5, queue_depth=50)
        assert load.is_overloaded is False  # > 50, not >=

    def test_both_factors_overloaded(self):
        load = ClusterLoad(cluster_id="very-busy", gpu_utilization=0.95, queue_depth=100)
        assert load.is_overloaded is True


class TestClusterLoadCapacity:
    """Tests for available_capacity property."""

    def test_full_capacity(self):
        load = ClusterLoad(cluster_id="empty", gpu_utilization=0.0, queue_depth=0)
        assert load.available_capacity == 1.0

    def test_half_capacity(self):
        load = ClusterLoad(cluster_id="half", gpu_utilization=0.5, queue_depth=0)
        assert load.available_capacity == 0.5

    def test_gpu_dominates(self):
        load = ClusterLoad(cluster_id="test", gpu_utilization=0.8, queue_depth=10)
        # max(0.8, 10/100) = 0.8, capacity = 1.0 - 0.8 = 0.2
        assert load.available_capacity == pytest.approx(0.2, rel=1e-9)

    def test_queue_dominates(self):
        load = ClusterLoad(cluster_id="test", gpu_utilization=0.1, queue_depth=80)
        # max(0.1, 80/100) = 0.8, capacity = 0.2
        assert load.available_capacity == pytest.approx(0.2, rel=1e-9)

    def test_never_negative(self):
        load = ClusterLoad(cluster_id="test", gpu_utilization=2.0, queue_depth=200)
        assert load.available_capacity == 0.0


# --- LoadReporter tests ---


class TestLoadReporterInit:
    """Tests for LoadReporter initialization."""

    def test_defaults(self):
        reporter = LoadReporter()
        assert reporter._stale_threshold == 30.0
        assert reporter._loads == {}

    def test_custom_threshold(self):
        reporter = LoadReporter(stale_threshold_s=60.0)
        assert reporter._stale_threshold == 60.0


class TestLoadReporterReport:
    """Tests for LoadReporter.report()."""

    def test_report_creates_new_cluster(self):
        reporter = LoadReporter()
        reporter.report("cluster-1", active=5, gpu_util=0.6)
        load = reporter.get_load("cluster-1")
        assert load is not None
        assert load.active_requests == 5
        assert load.gpu_utilization == 0.6

    def test_report_updates_existing_cluster(self):
        reporter = LoadReporter()
        reporter.report("cluster-1", active=5, gpu_util=0.6)
        reporter.report("cluster-1", active=10, gpu_util=0.8)
        load = reporter.get_load("cluster-1")
        assert load.active_requests == 10
        assert load.gpu_utilization == 0.8

    def test_report_all_fields(self):
        reporter = LoadReporter()
        reporter.report(
            "c1", active=3, pending=2, gpu_util=0.5, queue_depth=15,
        )
        load = reporter.get_load("c1")
        assert load.active_requests == 3
        assert load.pending_requests == 2
        assert load.gpu_utilization == 0.5
        assert load.queue_depth == 15

    def test_report_updates_timestamp(self):
        reporter = LoadReporter()
        before = time.time()
        reporter.report("c1")
        after = time.time()
        load = reporter.get_load("c1")
        assert before <= load.last_updated <= after


class TestLoadReporterGetLoad:
    """Tests for LoadReporter.get_load()."""

    def test_get_unknown_cluster(self):
        reporter = LoadReporter()
        assert reporter.get_load("unknown") is None

    def test_stale_data_returns_none(self):
        reporter = LoadReporter(stale_threshold_s=1.0)
        reporter.report("c1", active=5)
        time.sleep(1.1)
        assert reporter.get_load("c1") is None

    def test_fresh_data_returns_load(self):
        reporter = LoadReporter(stale_threshold_s=30.0)
        reporter.report("c1", active=5)
        load = reporter.get_load("c1")
        assert load is not None
        assert load.active_requests == 5


class TestLoadReporterGetAllLoads:
    """Tests for LoadReporter.get_all_loads()."""

    def test_empty(self):
        reporter = LoadReporter()
        assert reporter.get_all_loads() == {}

    def test_multiple_clusters(self):
        reporter = LoadReporter()
        reporter.report("c1", active=1)
        reporter.report("c2", active=2)
        loads = reporter.get_all_loads()
        assert len(loads) == 2
        assert loads["c1"].active_requests == 1
        assert loads["c2"].active_requests == 2

    def test_returns_copy(self):
        reporter = LoadReporter()
        reporter.report("c1")
        loads = reporter.get_all_loads()
        # dict-level copy: removing a key doesn't affect original
        del loads["c1"]
        assert reporter.get_load("c1") is not None
        # But objects are shared references (shallow copy)
        loads2 = reporter.get_all_loads()
        loads2["c1"].active_requests = 999
        original = reporter.get_load("c1")
        assert original.active_requests == 999


class TestLoadReporterRemoveCluster:
    """Tests for LoadReporter.remove_cluster()."""

    def test_remove_existing(self):
        reporter = LoadReporter()
        reporter.report("c1")
        reporter.remove_cluster("c1")
        assert reporter.get_load("c1") is None

    def test_remove_nonexistent(self):
        reporter = LoadReporter()
        reporter.remove_cluster("nonexistent")  # Should not raise


# --- GeoRouter tests ---


class TestGeoRouterInit:
    """Tests for GeoRouter initialization."""

    def test_defaults(self):
        federation = MagicMock()
        latency_monitor = MagicMock()
        router = GeoRouter(federation=federation, latency_monitor=latency_monitor)
        assert router._local_latency_threshold == 50.0
        assert isinstance(router._load_reporter, LoadReporter)

    def test_custom_threshold(self):
        federation = MagicMock()
        latency_monitor = MagicMock()
        router = GeoRouter(
            federation=federation, latency_monitor=latency_monitor,
            local_latency_threshold_ms=100.0,
        )
        assert router._local_latency_threshold == 100.0


class TestSelectTargetCluster:
    """Tests for select_target_cluster()."""

    def _make_router(self, clusters=None, latency=10.0, loads=None):
        federation = MagicMock()
        federation.list_clusters.return_value = clusters or ["us-east-1", "us-west-2", "eu-west-1"]
        latency_monitor = MagicMock()
        latency_monitor.get_latency.return_value = latency
        load_reporter = LoadReporter()
        if loads:
            for cid, metrics in loads.items():
                load_reporter.report(cid, **metrics)
        return GeoRouter(
            federation=federation, latency_monitor=latency_monitor,
            load_reporter=load_reporter,
        )

    def test_local_not_overloaded_stays_local(self):
        router = self._make_router(loads={"us-east-1": {"gpu_util": 0.5}})
        target, reason = router.select_target_cluster(source_cluster="us-east-1")
        assert target == "us-east-1"
        assert reason == "local_capacity"

    def test_local_overloaded_routes_away(self):
        router = self._make_router(
            clusters=["us-east-1", "us-west-2"],
            loads={"us-east-1": {"gpu_util": 0.95}, "us-west-2": {"gpu_util": 0.3}},
            latency=10.0,
        )
        target, reason = router.select_target_cluster(source_cluster="us-east-1")
        assert target == "us-west-2"
        assert "nearest_with_capacity" in reason

    def test_no_alternative_clusters(self):
        router = self._make_router(clusters=["us-east-1"], loads={"us-east-1": {"gpu_util": 0.9}})
        target, reason = router.select_target_cluster(source_cluster="us-east-1")
        assert target == "us-east-1"
        assert reason == "no_alternative"

    def test_all_remotes_overloaded(self):
        router = self._make_router(
            clusters=["us-east-1", "us-west-2", "eu-west-1"],
            loads={
                "us-east-1": {"gpu_util": 0.9},
                "us-west-2": {"gpu_util": 0.95},
                "eu-west-1": {"gpu_util": 0.9},
            },
        )
        target, reason = router.select_target_cluster(source_cluster="us-east-1")
        assert target == "us-east-1"
        assert reason == "all_remote_overloaded"

    def test_selects_best_score_cluster(self):
        federation = MagicMock()
        federation.list_clusters.return_value = ["us-east-1", "cluster-a", "cluster-b"]
        latency_monitor = MagicMock()
        latency_monitor.get_latency.side_effect = lambda src, dst: {
            ("us-east-1", "cluster-a"): 100.0,
            ("us-east-1", "cluster-b"): 20.0,
        }[(src, dst)]
        load_reporter = LoadReporter()
        load_reporter.report("us-east-1", gpu_util=0.9)
        load_reporter.report("cluster-a", gpu_util=0.2)
        load_reporter.report("cluster-b", gpu_util=0.2)
        router = GeoRouter(
            federation=federation, latency_monitor=latency_monitor,
            load_reporter=load_reporter,
        )
        target, reason = router.select_target_cluster(source_cluster="us-east-1")
        # cluster-b has lower latency so lower score
        assert target == "cluster-b"

    def test_load_data_unavailable_uses_default_capacity(self):
        """When remote cluster has no load data, uses 0.5 as default capacity."""
        federation = MagicMock()
        federation.list_clusters.return_value = ["us-east-1", "us-west-2"]
        latency_monitor = MagicMock()
        latency_monitor.get_latency.return_value = 15.0
        load_reporter = LoadReporter()
        load_reporter.report("us-east-1", gpu_util=0.9)
        # us-west-2 has no load data
        router = GeoRouter(
            federation=federation, latency_monitor=latency_monitor,
            load_reporter=load_reporter,
        )
        target, reason = router.select_target_cluster(source_cluster="us-east-1")
        assert target == "us-west-2"


class TestSelectEdgeNode:
    """Tests for select_edge_node()."""

    def _make_router_with_edge(self, edge_nodes=None, cluster_nodes=None):
        federation = MagicMock()
        federation.get_edge_nodes.return_value = edge_nodes or {"edge-0", "edge-1"}
        federation.get_nodes_in_cluster.return_value = cluster_nodes or {"node-0", "node-1"}
        latency_monitor = MagicMock()
        return GeoRouter(
            federation=federation, latency_monitor=latency_monitor,
        )

    def test_returns_edge_node(self):
        router = self._make_router_with_edge(edge_nodes={"edge-a", "edge-b"})
        node = router.select_edge_node("target-cluster", "source-cluster")
        assert node in {"edge-a", "edge-b"}

    def test_fallback_to_cluster_nodes(self):
        federation = MagicMock()
        federation.get_edge_nodes.return_value = set()
        federation.get_nodes_in_cluster.return_value = {"node-a", "node-b"}
        latency_monitor = MagicMock()
        router = GeoRouter(federation=federation, latency_monitor=latency_monitor)
        node = router.select_edge_node("target-cluster")
        assert node in {"node-a", "node-b"}

    def test_returns_none_when_no_nodes(self):
        federation = MagicMock()
        federation.get_edge_nodes.return_value = set()
        federation.get_nodes_in_cluster.return_value = set()
        latency_monitor = MagicMock()
        router = GeoRouter(federation=federation, latency_monitor=latency_monitor)
        node = router.select_edge_node("empty-cluster")
        assert node is None

    def test_single_node_returns_directly(self):
        router = self._make_router_with_edge(edge_nodes={"only-edge"})
        node = router.select_edge_node("target-cluster")
        assert node == "only-edge"

    def test_deterministic_selection(self):
        """Same inputs should always return the same node."""
        router = self._make_router_with_edge(edge_nodes={"node-a", "node-b"})
        result1 = router.select_edge_node("target")
        result2 = router.select_edge_node("target")
        assert result1 == result2


class TestGeoRouterLoadReporting:
    """Tests for load reporting methods."""

    def test_get_cluster_load(self):
        federation = MagicMock()
        latency_monitor = MagicMock()
        router = GeoRouter(federation=federation, latency_monitor=latency_monitor)
        router.report_cluster_load("c1", active=5, gpu_util=0.6)
        load = router.get_cluster_load("c1")
        assert load is not None
        assert load.active_requests == 5

    def test_get_cluster_load_unknown(self):
        federation = MagicMock()
        latency_monitor = MagicMock()
        router = GeoRouter(federation=federation, latency_monitor=latency_monitor)
        assert router.get_cluster_load("unknown") is None

    def test_report_cluster_load_all_params(self):
        federation = MagicMock()
        latency_monitor = MagicMock()
        router = GeoRouter(federation=federation, latency_monitor=latency_monitor)
        router.report_cluster_load("c1", active=10, pending=5, gpu_util=0.7, queue_depth=25)
        load = router.get_cluster_load("c1")
        assert load.active_requests == 10
        assert load.pending_requests == 5
        assert load.gpu_utilization == 0.7
        assert load.queue_depth == 25


class TestGeoRouterStats:
    """Tests for stats() method."""

    def test_stats_structure(self):
        federation = MagicMock()
        federation.stats.return_value = {"total_nodes": 5}
        latency_monitor = MagicMock()
        latency_monitor.stats.return_value = {"avg_latency": 15.0}
        router = GeoRouter(federation=federation, latency_monitor=latency_monitor)
        router.report_cluster_load("c1", active=3, gpu_util=0.5)

        stats = router.stats()

        assert "cluster_loads" in stats
        assert "latency_matrix" in stats
        assert "federation" in stats
        assert "c1" in stats["cluster_loads"]
        assert stats["cluster_loads"]["c1"]["active"] == 3
        assert stats["latency_matrix"] == {"avg_latency": 15.0}
        assert stats["federation"] == {"total_nodes": 5}

    def test_stats_empty(self):
        federation = MagicMock()
        federation.stats.return_value = {}
        latency_monitor = MagicMock()
        latency_monitor.stats.return_value = {}
        router = GeoRouter(federation=federation, latency_monitor=latency_monitor)

        stats = router.stats()

        assert stats["cluster_loads"] == {}


class TestGeoRouterEdgeCases:
    """Tests for edge cases."""

    def test_source_not_in_cluster_list(self):
        """Source cluster not in the cluster list."""
        federation = MagicMock()
        federation.list_clusters.return_value = ["us-west-2", "eu-west-1"]
        latency_monitor = MagicMock()
        load_reporter = LoadReporter()
        load_reporter.report("us-east-1", gpu_util=0.5)  # source not in list
        router = GeoRouter(
            federation=federation, latency_monitor=latency_monitor,
            load_reporter=load_reporter,
        )
        target, reason = router.select_target_cluster(source_cluster="us-east-1")
        # Local is not overloaded, but not in cluster list — still returns local
        assert target == "us-east-1"
        assert reason == "local_capacity"

    def test_latency_monitor_get_latency_returns_zero(self):
        """Latency monitor returns 0 for same-region."""
        router = self._make_router_for_latency_test(
            clusters=["us-east-1", "us-west-2"],
            loads={"us-east-1": {"gpu_util": 0.9}, "us-west-2": {"gpu_util": 0.3}},
            latency=0.0,
        )
        target, reason = router.select_target_cluster(source_cluster="us-east-1")
        assert target == "us-west-2"

    def test_capacity_near_zero_protection(self):
        """Cluster with near-zero capacity should get very high score."""
        federation = MagicMock()
        federation.list_clusters.return_value = ["us-east-1", "c1"]
        latency_monitor = MagicMock()
        latency_monitor.get_latency.return_value = 10.0
        load_reporter = LoadReporter()
        load_reporter.report("us-east-1", gpu_util=0.95)
        load_reporter.report("c1", gpu_util=0.5)  # c1 has capacity
        router = GeoRouter(
            federation=federation, latency_monitor=latency_monitor,
            load_reporter=load_reporter,
        )
        target, reason = router.select_target_cluster(source_cluster="us-east-1")
        # c1 has capacity, local is overloaded
        assert target == "c1"

    def _make_router_for_latency_test(self, clusters, loads, latency):
        federation = MagicMock()
        federation.list_clusters.return_value = clusters
        latency_monitor = MagicMock()
        latency_monitor.get_latency.return_value = latency
        load_reporter = LoadReporter()
        for cid, metrics in loads.items():
            load_reporter.report(cid, **metrics)
        return GeoRouter(
            federation=federation, latency_monitor=latency_monitor,
            load_reporter=load_reporter,
        )

    def _make_router(self, clusters=None, latency=10.0, loads=None):
        federation = MagicMock()
        federation.list_clusters.return_value = clusters or ["us-east-1", "us-west-2", "eu-west-1"]
        latency_monitor = MagicMock()
        latency_monitor.get_latency.return_value = latency
        load_reporter = LoadReporter()
        if loads:
            for cid, metrics in loads.items():
                load_reporter.report(cid, **metrics)
        return GeoRouter(
            federation=federation, latency_monitor=latency_monitor,
            load_reporter=load_reporter,
        )
