"""Tests for distllm.dist.topology -- FederationManager, ClusterInfo, CrossClusterLatencyMonitor."""

from __future__ import annotations

import pytest

from distllm.dist.topology import (
    ClusterInfo,
    CrossClusterLatencyMonitor,
    FederationManager,
)


# ---------------------------------------------------------------------------
# ClusterInfo
# ---------------------------------------------------------------------------


class TestClusterInfo:
    def test_default_construction(self) -> None:
        info = ClusterInfo(cluster_id="us-east-1")
        assert info.cluster_id == "us-east-1"
        assert info.region == "unknown"
        assert info.nodes == set()
        assert info.base_latency_ms == 1.0
        assert info.edge_nodes == set()
        assert info.size == 0

    def test_full_construction(self) -> None:
        info = ClusterInfo(
            cluster_id="eu-west-1",
            region="frankfurt",
            nodes={"node-a", "node-b"},
            base_latency_ms=12.5,
            edge_nodes={"node-a"},
        )
        assert info.cluster_id == "eu-west-1"
        assert info.region == "frankfurt"
        assert info.nodes == {"node-a", "node-b"}
        assert info.base_latency_ms == 12.5
        assert info.edge_nodes == {"node-a"}
        assert info.size == 2

    def test_size_property_empty(self) -> None:
        info = ClusterInfo(cluster_id="empty")
        assert info.size == 0

    def test_size_property(self) -> None:
        info = ClusterInfo(cluster_id="c", nodes={"n1", "n2", "n3"})
        assert info.size == 3

    def test_size_reflects_mutation(self) -> None:
        info = ClusterInfo(cluster_id="c")
        info.nodes.add("x")
        assert info.size == 1

    def test_edge_nodes_independent_from_nodes(self) -> None:
        info = ClusterInfo(cluster_id="c", nodes={"a", "b"}, edge_nodes={"a"})
        assert info.edge_nodes == {"a"}
        assert "b" not in info.edge_nodes

    def test_cluster_id_empty_string(self) -> None:
        info = ClusterInfo(cluster_id="")
        assert info.cluster_id == ""

    def test_region_custom(self) -> None:
        info = ClusterInfo(cluster_id="x", region="us-west-2")
        assert info.region == "us-west-2"


# ---------------------------------------------------------------------------
# FederationManager
# ---------------------------------------------------------------------------


class TestFederationManager:
    def test_default_construction_registers_local_cluster(self) -> None:
        mgr = FederationManager()
        assert mgr.local_cluster_id == "default"
        assert mgr.list_clusters() == ["default"]
        assert mgr.get_nodes_in_cluster("default") == set()

    def test_custom_local_cluster_id(self) -> None:
        mgr = FederationManager(local_cluster_id="dc-1")
        assert mgr.local_cluster_id == "dc-1"
        assert mgr.list_clusters() == ["dc-1"]

    def test_register_cluster_and_list(self) -> None:
        mgr = FederationManager()
        cluster = ClusterInfo(cluster_id="eu-west", region="dublin")
        mgr.register_cluster(cluster)
        assert "eu-west" in mgr.list_clusters()

    def test_register_cluster_replaces_existing(self) -> None:
        mgr = FederationManager()
        mgr.register_cluster(ClusterInfo(cluster_id="c1", region="old"))
        mgr.register_cluster(ClusterInfo(cluster_id="c1", region="new", nodes={"n1"}))
        assert mgr.clusters["c1"].region == "new"
        assert mgr.clusters["c1"].nodes == {"n1"}

    def test_register_node_into_registered_cluster(self) -> None:
        mgr = FederationManager()
        mgr.register_cluster(ClusterInfo(cluster_id="dc-west"))
        mgr.register_node("node-1", "dc-west")
        assert mgr.get_cluster("node-1") == "dc-west"
        assert mgr.get_nodes_in_cluster("dc-west") == {"node-1"}

    def test_register_node_does_not_auto_create_cluster(self) -> None:
        mgr = FederationManager()
        mgr.register_node("n1", "unreg")
        assert mgr.get_cluster("n1") == "unreg"
        assert "unreg" not in mgr.list_clusters()
        assert mgr.get_nodes_in_cluster("unreg") == set()

    def test_register_node_existing_cluster(self) -> None:
        mgr = FederationManager()
        mgr.register_cluster(ClusterInfo(cluster_id="dc"))
        mgr.register_node("n1", "dc")
        mgr.register_node("n2", "dc")
        assert mgr.get_nodes_in_cluster("dc") == {"n1", "n2"}

    def test_register_node_edge_flag(self) -> None:
        mgr = FederationManager()
        mgr.register_cluster(ClusterInfo(cluster_id="c"))
        mgr.register_node("n1", "c", is_edge=True)
        mgr.register_node("n2", "c", is_edge=False)
        assert mgr.get_edge_nodes("c") == {"n1"}
        assert "n2" not in mgr.get_edge_nodes("c")

    def test_register_node_skips_edge_if_cluster_unregistered(self) -> None:
        mgr = FederationManager()
        mgr.register_node("n1", "nowhere", is_edge=True)
        assert mgr.get_edge_nodes("nowhere") == set()

    def test_unregister_node_removes_mapping(self) -> None:
        mgr = FederationManager()
        mgr.register_node("n1", "c")
        mgr.unregister_node("n1")
        assert mgr.get_cluster("n1") is None

    def test_unregister_node_removes_from_cluster_set(self) -> None:
        mgr = FederationManager()
        mgr.register_cluster(ClusterInfo(cluster_id="c"))
        mgr.register_node("n1", "c")
        mgr.unregister_node("n1")
        assert mgr.get_nodes_in_cluster("c") == set()

    def test_unregister_node_unknown_node(self) -> None:
        mgr = FederationManager()
        mgr.unregister_node("nonexistent")
        # Should not raise

    def test_unregister_node_cleans_edge_nodes(self) -> None:
        mgr = FederationManager()
        mgr.register_cluster(ClusterInfo(cluster_id="c"))
        mgr.register_node("n1", "c", is_edge=True)
        mgr.unregister_node("n1")
        assert mgr.get_edge_nodes("c") == set()

    def test_unregister_node_partial_cleanup(self) -> None:
        mgr = FederationManager()
        mgr.register_cluster(ClusterInfo(cluster_id="c"))
        mgr.register_node("n1", "c", is_edge=True)
        mgr.register_node("n2", "c")
        mgr.unregister_node("n1")
        assert mgr.get_nodes_in_cluster("c") == {"n2"}
        assert mgr.get_edge_nodes("c") == set()

    def test_get_cluster_nonexistent_node(self) -> None:
        mgr = FederationManager()
        assert mgr.get_cluster("ghost") is None

    def test_get_cluster_after_unregister(self) -> None:
        mgr = FederationManager()
        mgr.register_node("n1", "c")
        mgr.unregister_node("n1")
        assert mgr.get_cluster("n1") is None

    def test_get_nodes_in_cluster_nonexistent_cluster(self) -> None:
        mgr = FederationManager()
        assert mgr.get_nodes_in_cluster("nowhere") == set()

    def test_get_nodes_in_cluster_empty_cluster(self) -> None:
        mgr = FederationManager()
        mgr.register_cluster(ClusterInfo(cluster_id="empty"))
        assert mgr.get_nodes_in_cluster("empty") == set()

    def test_get_edge_nodes_nonexistent_cluster(self) -> None:
        mgr = FederationManager()
        assert mgr.get_edge_nodes("nowhere") == set()

    def test_get_edge_nodes_registered_no_edges(self) -> None:
        mgr = FederationManager()
        mgr.register_cluster(ClusterInfo(cluster_id="c"))
        mgr.register_node("n1", "c")
        assert mgr.get_edge_nodes("c") == set()

    def test_is_local_true(self) -> None:
        mgr = FederationManager(local_cluster_id="home")
        mgr.register_node("my-node", "home")
        assert mgr.is_local("my-node") is True

    def test_is_local_false(self) -> None:
        mgr = FederationManager(local_cluster_id="home")
        mgr.register_node("remote-node", "away")
        assert mgr.is_local("remote-node") is False

    def test_is_local_unknown_node(self) -> None:
        mgr = FederationManager()
        assert mgr.is_local("unknown-node") is False

    def test_stats_returns_correct_counts(self) -> None:
        mgr = FederationManager(local_cluster_id="dc1")
        mgr.register_cluster(ClusterInfo(cluster_id="dc2"))
        mgr.register_node("a", "dc1")
        mgr.register_node("b", "dc1")
        mgr.register_node("c", "dc2", is_edge=True)
        s = mgr.stats()
        assert s["cluster_count"] == 2
        assert s["node_count"] == 3
        assert s["clusters"]["dc1"]["nodes"] == 2
        assert s["clusters"]["dc1"]["edge_nodes"] == 0
        assert s["clusters"]["dc2"]["nodes"] == 1
        assert s["clusters"]["dc2"]["edge_nodes"] == 1

    def test_stats_empty(self) -> None:
        mgr = FederationManager(local_cluster_id="only")
        s = mgr.stats()
        assert s["cluster_count"] == 1
        assert s["node_count"] == 0
        assert len(s["clusters"]) == 1

    def test_registered_nodes_reflect_after_removal(self) -> None:
        mgr = FederationManager()
        mgr.register_cluster(ClusterInfo(cluster_id="c"))
        mgr.register_node("n1", "c")
        mgr.register_node("n2", "c")
        mgr.unregister_node("n1")
        assert mgr.get_nodes_in_cluster("c") == {"n2"}
        assert mgr.get_cluster("n1") is None

    def test_list_clusters_order_independent(self) -> None:
        mgr = FederationManager(local_cluster_id="a")
        mgr.register_cluster(ClusterInfo(cluster_id="b"))
        mgr.register_cluster(ClusterInfo(cluster_id="c"))
        result = mgr.list_clusters()
        assert sorted(result) == ["a", "b", "c"]

    def test_node_to_cluster_mapping_isolated(self) -> None:
        mgr = FederationManager()
        mgr.register_node("n1", "c1")
        mgr.register_node("n2", "c2")
        assert mgr.get_cluster("n1") == "c1"
        assert mgr.get_cluster("n2") == "c2"


# ---------------------------------------------------------------------------
# CrossClusterLatencyMonitor
# ---------------------------------------------------------------------------


class TestCrossClusterLatencyMonitor:
    def test_empty_latency_returns_default(self) -> None:
        fm = FederationManager()
        mon = CrossClusterLatencyMonitor(fm)
        assert mon.get_latency("us", "eu") == 1000.0

    def test_zero_latency_same_cluster(self) -> None:
        fm = FederationManager()
        mon = CrossClusterLatencyMonitor(fm)
        assert mon.get_latency("c", "c") == 0.0

    def test_record_and_retrieve_latency(self) -> None:
        fm = FederationManager()
        mon = CrossClusterLatencyMonitor(fm)
        mon.record_latency("us", "eu", 80.0)
        assert mon.get_latency("us", "eu") == 80.0

    def test_latency_is_symmetric(self) -> None:
        fm = FederationManager()
        mon = CrossClusterLatencyMonitor(fm)
        mon.record_latency("us", "eu", 75.0)
        assert mon.get_latency("eu", "us") == 75.0

    def test_latency_exponential_moving_average(self) -> None:
        fm = FederationManager()
        mon = CrossClusterLatencyMonitor(fm)
        mon.record_latency("a", "b", 100.0)
        mon.record_latency("a", "b", 200.0)
        # EMA: 100 * 0.7 + 200 * 0.3 = 70 + 60 = 130
        assert mon.get_latency("a", "b") == 130.0

    def test_latency_ema_multiple_updates(self) -> None:
        fm = FederationManager()
        mon = CrossClusterLatencyMonitor(fm)
        mon.record_latency("a", "b", 10.0)
        mon.record_latency("a", "b", 20.0)
        mon.record_latency("a", "b", 30.0)
        # Step 1: 10
        # Step 2: 10 * 0.7 + 20 * 0.3 = 13
        # Step 3: 13 * 0.7 + 30 * 0.3 = 9.1 + 9.0 = 18.1
        assert mon.get_latency("a", "b") == 18.1

    def test_latency_to_node_local(self) -> None:
        fm = FederationManager(local_cluster_id="home")
        mon = CrossClusterLatencyMonitor(fm)
        fm.register_node("local-node", "home")
        # local -> local = 0
        assert mon.get_latency_to_node("local-node") == 0.0

    def test_latency_to_node_remote(self) -> None:
        fm = FederationManager(local_cluster_id="home")
        mon = CrossClusterLatencyMonitor(fm)
        fm.register_node("remote-node", "away")
        mon.record_latency("home", "away", 55.0)
        assert mon.get_latency_to_node("remote-node") == 55.0

    def test_latency_to_node_unknown(self) -> None:
        fm = FederationManager()
        mon = CrossClusterLatencyMonitor(fm)
        assert mon.get_latency_to_node("ghost") == 1000.0

    def test_latency_to_node_orphan_node(self) -> None:
        """Node is registered to a cluster that was later removed."""
        fm = FederationManager(local_cluster_id="home")
        mon = CrossClusterLatencyMonitor(fm)
        fm.register_cluster(ClusterInfo(cluster_id="gone"))
        fm.register_node("orphan", "gone")
        # Simulate the cluster being removed from the manager
        del fm.clusters["gone"]
        # node_to_cluster still maps "orphan" -> "gone",
        # but the cluster is not in the matrix => default 1000.0
        assert mon.get_latency_to_node("orphan") == 1000.0

    def test_get_closest_cluster_empty_list(self) -> None:
        fm = FederationManager()
        mon = CrossClusterLatencyMonitor(fm)
        assert mon.get_closest_cluster([]) is None

    def test_get_closest_cluster_single(self) -> None:
        fm = FederationManager(local_cluster_id="home")
        mon = CrossClusterLatencyMonitor(fm)
        assert mon.get_closest_cluster(["far"]) == "far"

    def test_get_closest_cluster_picks_lowest_latency(self) -> None:
        fm = FederationManager(local_cluster_id="home")
        mon = CrossClusterLatencyMonitor(fm)
        mon.record_latency("home", "close", 5.0)
        mon.record_latency("home", "far", 50.0)
        mon.record_latency("home", "mid", 20.0)
        assert mon.get_closest_cluster(["far", "mid", "close"]) == "close"

    def test_get_closest_cluster_tie_breaks_by_first_in_list(self) -> None:
        """When two have equal latency, min() returns the first in the list."""
        fm = FederationManager(local_cluster_id="home")
        mon = CrossClusterLatencyMonitor(fm)
        mon.record_latency("home", "alpha", 10.0)
        mon.record_latency("home", "beta", 10.0)
        assert mon.get_closest_cluster(["beta", "alpha"]) == "beta"

    def test_get_closest_cluster_all_default_latency(self) -> None:
        fm = FederationManager(local_cluster_id="home")
        mon = CrossClusterLatencyMonitor(fm)
        # All default to 1000.0, min() returns the first element
        result = mon.get_closest_cluster(["z", "m", "a"])
        assert result == "z"

    def test_get_closest_cluster_unknown_candidates(self) -> None:
        """Candidates not in the matrix all default to 1000.0."""
        fm = FederationManager(local_cluster_id="home")
        mon = CrossClusterLatencyMonitor(fm)
        assert mon.get_closest_cluster(["x", "y"]) == "x"

    def test_stats_empty(self) -> None:
        fm = FederationManager()
        mon = CrossClusterLatencyMonitor(fm)
        assert mon.stats() == {}

    def test_stats_after_recordings(self) -> None:
        fm = FederationManager()
        mon = CrossClusterLatencyMonitor(fm)
        mon.record_latency("a", "b", 10.0)
        mon.record_latency("c", "d", 20.0)
        s = mon.stats()
        assert s["a->b"] == 10.0
        assert s["b->a"] == 10.0
        assert s["c->d"] == 20.0
        assert s["d->c"] == 20.0
        assert len(s) == 4

    def test_stats_sorted_keys(self) -> None:
        fm = FederationManager()
        mon = CrossClusterLatencyMonitor(fm)
        mon.record_latency("z", "a", 1.0)
        mon.record_latency("m", "n", 2.0)
        keys = list(mon.stats().keys())
        assert keys == sorted(keys)

    def test_latency_update_overwrite_ema(self) -> None:
        fm = FederationManager()
        mon = CrossClusterLatencyMonitor(fm)
        mon.record_latency("x", "y", 50.0)
        assert mon.get_latency("x", "y") == 50.0
        mon.record_latency("x", "y", 150.0)
        # 50 * 0.7 + 150 * 0.3 = 35 + 45 = 80
        assert mon.get_latency("x", "y") == 80.0
        mon.record_latency("x", "y", 0.0)
        # 80 * 0.7 + 0 * 0.3 = 56
        assert mon.get_latency("x", "y") == 56.0


# ---------------------------------------------------------------------------
# Integration: FederationManager + CrossClusterLatencyMonitor
# ---------------------------------------------------------------------------


class TestTopologyIntegration:
    def test_full_workflow(self) -> None:
        fm = FederationManager(local_cluster_id="dc1")
        mon = CrossClusterLatencyMonitor(fm)

        # Register remote cluster
        fm.register_cluster(ClusterInfo(cluster_id="dc2"))
        fm.register_node("worker-1", "dc2", is_edge=True)

        # Record inter-cluster latency
        mon.record_latency("dc1", "dc2", 42.0)

        # Check latency to remote node
        assert mon.get_latency_to_node("worker-1") == 42.0

        # Local node has zero latency
        fm.register_node("local-w", "dc1")
        assert mon.get_latency_to_node("local-w") == 0.0

        # Closest cluster routing
        fm.register_cluster(ClusterInfo(cluster_id="dc3"))
        mon.record_latency("dc1", "dc3", 100.0)

        assert mon.get_closest_cluster(["dc2", "dc3"]) == "dc2"
        assert mon.get_closest_cluster(["dc3"]) == "dc3"
        assert mon.get_closest_cluster([]) is None

        # Unregister and verify cleanup
        fm.unregister_node("worker-1")
        assert mon.get_latency_to_node("worker-1") == 1000.0
        assert fm.get_nodes_in_cluster("dc2") == set()

    def test_multiple_nodes_same_cluster(self) -> None:
        fm = FederationManager(local_cluster_id="local")
        mon = CrossClusterLatencyMonitor(fm)

        fm.register_cluster(ClusterInfo(cluster_id="remote"))
        for i in range(10):
            fm.register_node(f"r-node-{i}", "remote")

        mon.record_latency("local", "remote", 30.0)

        for i in range(10):
            assert mon.get_latency_to_node(f"r-node-{i}") == 30.0

        assert fm.stats()["clusters"]["remote"]["nodes"] == 10
