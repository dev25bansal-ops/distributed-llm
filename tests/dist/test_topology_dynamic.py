"""Tests for distllm.dist.topology_dynamic — DynamicClusterTopology, NodeInfo."""

from __future__ import annotations

import time

import pytest

from distllm.dist.topology_dynamic import (
    DynamicClusterTopology,
    NodeInfo,
)
from distllm.dist.partition.topology import LinkProfile, TopologyGraph, TopologyProber


# ---------------------------------------------------------------------------
# NodeInfo
# ---------------------------------------------------------------------------


class TestNodeInfo:
    def test_default_construction(self) -> None:
        info = NodeInfo(node_id="node-0", host="192.168.1.10")
        assert info.node_id == "node-0"
        assert info.host == "192.168.1.10"
        assert info.port == 50051
        assert info.gpu_count == 1
        assert info.healthy is True
        assert info.joined_at == 0.0
        assert info.tags == {}

    def test_full_construction(self) -> None:
        info = NodeInfo(
            node_id="n1",
            host="10.0.0.1",
            port=50052,
            gpu_count=8,
            healthy=False,
            joined_at=1234.5,
            tags={"rack": "a1", "region": "us-east"},
        )
        assert info.node_id == "n1"
        assert info.host == "10.0.0.1"
        assert info.port == 50052
        assert info.gpu_count == 8
        assert info.healthy is False
        assert info.joined_at == 1234.5
        assert info.tags == {"rack": "a1", "region": "us-east"}

    def test_tags_default_is_empty_dict(self) -> None:
        info = NodeInfo(node_id="x", host="h")
        assert info.tags == {}
        # Ensure field is isolated across instances
        info.tags["k"] = "v"
        info2 = NodeInfo(node_id="y", host="h")
        assert info2.tags == {}

    def test_immutable_fields_after_construction(self) -> None:
        info = NodeInfo(node_id="n", host="h", port=50053, gpu_count=4)
        assert info.node_id == "n"
        assert info.host == "h"
        assert info.port == 50053
        assert info.gpu_count == 4

    def test_host_defaults_to_node_id_when_empty(self) -> None:
        """Host defaults to node_id in on_node_join, not in NodeInfo itself."""
        info = NodeInfo(node_id="node-x", host="")
        assert info.host == ""  # NodeInfo stores as-is


# ---------------------------------------------------------------------------
# DynamicClusterTopology — Construction
# ---------------------------------------------------------------------------


class TestDynamicClusterTopologyConstruction:
    def test_default_construction(self) -> None:
        topo = DynamicClusterTopology()
        assert topo.node_count() == 0
        assert topo.total_gpus() == 0
        assert topo.get_healthy_nodes() == []
        graph = topo.get_graph()
        assert graph.node_ids == []
        assert graph.links == []
        assert graph.gpu_counts == {}

    def test_custom_defaults(self) -> None:
        topo = DynamicClusterTopology(
            default_bandwidth=100.0,
            default_latency_us=200.0,
        )
        topo.on_node_join("a", host="h1")
        topo.on_node_join("b", host="h2")
        bw = topo.get_bandwidth("a", "b")
        lat = topo.get_latency("a", "b")
        assert bw == 100.0
        assert lat == 200.0

    def test_custom_prober(self) -> None:
        prober = TopologyProber()
        topo = DynamicClusterTopology(prober=prober)
        assert topo._prober is prober

    def test_probe_on_join_false(self) -> None:
        """_probe_active_links is a no-op, but verify probe_on_join=False works."""
        topo = DynamicClusterTopology(probe_on_join=False)
        topo.on_node_join("a", host="h1")
        topo.on_node_join("b", host="h2")
        assert topo.node_count() == 2


# ---------------------------------------------------------------------------
# DynamicClusterTopology — Node Join / Leave
# ---------------------------------------------------------------------------


class TestNodeJoin:
    def test_join_first_node(self) -> None:
        topo = DynamicClusterTopology()
        graph = topo.on_node_join("node-0", host="10.0.0.1", gpu_count=4)
        assert topo.node_count() == 1
        assert topo.total_gpus() == 4
        assert graph.node_ids == ["node-0"]
        assert graph.gpu_counts == {"node-0": 4}
        assert graph.links == []

    def test_join_second_node_cross_host_creates_link(self) -> None:
        topo = DynamicClusterTopology()
        topo.on_node_join("node-0", host="10.0.0.1")
        graph = topo.on_node_join("node-1", host="10.0.0.2")
        assert topo.node_count() == 2
        assert len(graph.links) == 1
        link = graph.links[0]
        assert {link.source, link.target} == {"node-0", "node-1"}
        # cross-host defaults
        assert link.bandwidth_gbps == 12.5
        assert link.latency_us == 500.0
        assert link.is_nvlink is False
        assert link.is_infiniband is False

    def test_join_second_node_same_host_creates_nvlink(self) -> None:
        topo = DynamicClusterTopology()
        topo.on_node_join("node-0", host="10.0.0.1")
        graph = topo.on_node_join("node-1", host="10.0.0.1")
        assert len(graph.links) == 1
        link = graph.links[0]
        assert link.bandwidth_gbps == 600.0
        assert link.latency_us == 5.0
        assert link.is_nvlink is True
        assert link.is_infiniband is False

    def test_join_third_node_creates_links_to_all_existing(self) -> None:
        topo = DynamicClusterTopology()
        topo.on_node_join("a", host="h1")
        topo.on_node_join("b", host="h2")
        graph = topo.on_node_join("c", host="h3")
        assert len(graph.links) == 3  # a-b, a-c, b-c
        sources_targets = {(l.source, l.target) for l in graph.links}
        assert ("a", "b") in sources_targets or ("b", "a") in sources_targets
        assert ("a", "c") in sources_targets or ("c", "a") in sources_targets
        assert ("b", "c") in sources_targets or ("c", "b") in sources_targets

    def test_join_duplicate_node(self) -> None:
        topo = DynamicClusterTopology()
        topo.on_node_join("node-0", host="10.0.0.1", gpu_count=2)
        graph = topo.on_node_join("node-0", host="10.0.0.1", gpu_count=8)
        assert topo.node_count() == 1
        # gpu_count should be updated
        node = topo.get_node("node-0")
        assert node is not None
        assert node.gpu_count == 8

    def test_join_duplicate_does_not_add_extra_links(self) -> None:
        topo = DynamicClusterTopology()
        topo.on_node_join("a", host="h1")
        topo.on_node_join("b", host="h2")
        topo.on_node_join("a", host="h1")  # re-join
        assert len(topo.get_graph().links) == 1  # still just a-b

    def test_join_updates_host_update(self) -> None:
        topo = DynamicClusterTopology()
        topo.on_node_join("n", host="old-host")
        topo.on_node_join("n", host="new-host")
        assert topo.get_node("n") is not None
        assert topo.get_node("n").host == "new-host"  # type: ignore[union-attr]

    def test_join_with_tags(self) -> None:
        topo = DynamicClusterTopology()
        topo.on_node_join("n", host="h", tags={"rack": "42"})
        node = topo.get_node("n")
        assert node is not None
        assert node.tags == {"rack": "42"}

    def test_join_host_defaults_to_node_id(self) -> None:
        topo = DynamicClusterTopology()
        topo.on_node_join("my-node")
        node = topo.get_node("my-node")
        assert node is not None
        assert node.host == "my-node"

    def test_join_custom_port(self) -> None:
        topo = DynamicClusterTopology()
        topo.on_node_join("n", host="h", port=9999)
        node = topo.get_node("n")
        assert node is not None
        assert node.port == 9999

    def test_join_high_infiniband_bandwidth(self) -> None:
        """Cross-host links with bandwidth > 25.0 are marked as infiniband."""
        topo = DynamicClusterTopology(default_bandwidth=100.0)
        topo.on_node_join("a", host="h1")
        graph = topo.on_node_join("b", host="h2")
        link = graph.links[0]
        assert link.is_infiniband is True


class TestNodeLeave:
    def test_leave_removes_node(self) -> None:
        topo = DynamicClusterTopology()
        topo.on_node_join("n", host="h")
        topo.on_node_leave("n")
        assert topo.node_count() == 0
        assert topo.get_node("n") is None

    def test_leave_clears_links(self) -> None:
        topo = DynamicClusterTopology()
        topo.on_node_join("a", host="h1")
        topo.on_node_join("b", host="h2")
        topo.on_node_join("c", host="h3")
        topo.on_node_leave("b")
        graph = topo.get_graph()
        assert len(graph.links) == 1  # only a-c remains
        for link in graph.links:
            assert "b" not in (link.source, link.target)

    def test_leave_clears_graph_metadata(self) -> None:
        topo = DynamicClusterTopology()
        topo.on_node_join("n", host="h", gpu_count=4)
        topo.on_node_leave("n")
        graph = topo.get_graph()
        assert "n" not in graph.node_ids
        assert "n" not in graph.gpu_counts
        assert "n" not in graph.node_hostnames

    def test_leave_unknown_node_returns_current_graph(self) -> None:
        topo = DynamicClusterTopology()
        graph = topo.on_node_leave("nonexistent")
        assert graph is not None
        assert graph.node_ids == []

    def test_leave_unknown_node_does_not_raise(self) -> None:
        topo = DynamicClusterTopology()
        topo.on_node_leave("ghost")  # must not raise

    def test_leave_last_node_cleans_up(self) -> None:
        topo = DynamicClusterTopology()
        topo.on_node_join("n", host="h")
        topo.on_node_leave("n")
        assert topo.node_count() == 0
        assert topo.total_gpus() == 0
        assert topo.get_healthy_nodes() == []

    def test_leave_then_rejoin(self) -> None:
        topo = DynamicClusterTopology()
        topo.on_node_join("a", host="h1")
        topo.on_node_join("b", host="h2")
        topo.on_node_leave("b")
        graph = topo.on_node_join("b", host="h2")
        assert topo.node_count() == 2
        # link should be recreated
        assert len(graph.links) == 1

    def test_leave_node_id_not_in_graph_node_ids(self) -> None:
        topo = DynamicClusterTopology()
        topo.on_node_join("a", host="h")
        topo.on_node_leave("a")
        graph = topo.get_graph()
        assert "a" not in graph.node_ids


# ---------------------------------------------------------------------------
# DynamicClusterTopology — Health State
# ---------------------------------------------------------------------------


class TestHealthManagement:
    def test_mark_unhealthy(self) -> None:
        topo = DynamicClusterTopology()
        topo.on_node_join("n", host="h")
        topo.mark_unhealthy("n")
        assert topo.get_healthy_nodes() == []
        node = topo.get_node("n")
        assert node is not None
        assert node.healthy is False

    def test_mark_healthy(self) -> None:
        topo = DynamicClusterTopology()
        topo.on_node_join("n", host="h")
        topo.mark_unhealthy("n")
        topo.mark_healthy("n")
        assert topo.get_healthy_nodes() == ["n"]
        node = topo.get_node("n")
        assert node is not None
        assert node.healthy is True

    def test_mark_unhealthy_unknown_node(self) -> None:
        topo = DynamicClusterTopology()
        topo.mark_unhealthy("ghost")  # must not raise

    def test_mark_healthy_unknown_node(self) -> None:
        topo = DynamicClusterTopology()
        topo.mark_healthy("ghost")  # must not raise

    def test_healthy_nodes_multiple(self) -> None:
        topo = DynamicClusterTopology()
        topo.on_node_join("a", host="h1")
        topo.on_node_join("b", host="h2")
        topo.on_node_join("c", host="h3")
        topo.mark_unhealthy("b")
        healthy = topo.get_healthy_nodes()
        assert sorted(healthy) == ["a", "c"]

    def test_mark_unhealthy_then_leave(self) -> None:
        topo = DynamicClusterTopology()
        topo.on_node_join("n", host="h")
        topo.mark_unhealthy("n")
        topo.on_node_leave("n")
        assert topo.node_count() == 0

    def test_new_node_is_healthy(self) -> None:
        topo = DynamicClusterTopology()
        topo.on_node_join("n", host="h")
        assert topo.get_node("n") is not None
        assert topo.get_node("n").healthy is True  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# DynamicClusterTopology — Graph Queries
# ---------------------------------------------------------------------------


class TestGraphQueries:
    def test_get_graph_returns_copy(self) -> None:
        topo = DynamicClusterTopology()
        topo.on_node_join("a", host="h1")
        topo.on_node_join("b", host="h2")
        graph1 = topo.get_graph()
        graph2 = topo.get_graph()
        # Mutating the returned graph should not affect the internal state
        graph1.links.clear()
        assert len(topo.get_graph().links) == 1

    def test_get_graph_deep_copies_lists_and_dicts(self) -> None:
        topo = DynamicClusterTopology()
        topo.on_node_join("a", host="h1", gpu_count=2)
        graph = topo.get_graph()
        graph.node_ids.append("bogus")
        graph.gpu_counts["bogus"] = 99
        assert topo.get_graph().node_ids == ["a"]
        assert topo.get_graph().gpu_counts == {"a": 2}

    def test_get_node_returns_node_info(self) -> None:
        topo = DynamicClusterTopology()
        topo.on_node_join("n", host="h", gpu_count=8)
        node = topo.get_node("n")
        assert node is not None
        assert node.node_id == "n"
        assert node.host == "h"
        assert node.gpu_count == 8

    def test_get_node_nonexistent_returns_none(self) -> None:
        topo = DynamicClusterTopology()
        assert topo.get_node("nobody") is None

    def test_get_nodes_returns_copy(self) -> None:
        topo = DynamicClusterTopology()
        topo.on_node_join("a", host="h1")
        topo.on_node_join("b", host="h2")
        nodes = topo.get_nodes()
        nodes.pop("a")
        assert "a" in topo.get_nodes()

    def test_get_bandwidth_no_link_returns_default(self) -> None:
        topo = DynamicClusterTopology()
        assert topo.get_bandwidth("x", "y") == 1.0

    def test_get_bandwidth_existing_link(self) -> None:
        topo = DynamicClusterTopology()
        topo.on_node_join("a", host="h1")
        topo.on_node_join("b", host="h2")
        bw = topo.get_bandwidth("a", "b")
        assert bw == 12.5

    def test_get_latency_no_link_returns_default(self) -> None:
        topo = DynamicClusterTopology()
        assert topo.get_latency("x", "y") == 1000.0

    def test_get_latency_existing_link(self) -> None:
        topo = DynamicClusterTopology()
        topo.on_node_join("a", host="h1")
        topo.on_node_join("b", host="h2")
        lat = topo.get_latency("a", "b")
        assert lat == 500.0

    def test_node_count_zero_on_empty(self) -> None:
        topo = DynamicClusterTopology()
        assert topo.node_count() == 0

    def test_total_gpus_multiple_nodes(self) -> None:
        topo = DynamicClusterTopology()
        topo.on_node_join("a", host="h1", gpu_count=4)
        topo.on_node_join("b", host="h2", gpu_count=8)
        assert topo.total_gpus() == 12


# ---------------------------------------------------------------------------
# DynamicClusterTopology — Callbacks
# ---------------------------------------------------------------------------


class TestCallbacks:
    def test_on_change_fires_on_join(self) -> None:
        events: list[tuple[str, str]] = []
        topo = DynamicClusterTopology()

        def cb(node_id: str, event: str, graph: object) -> None:
            events.append((node_id, event))

        topo.on_change(cb)
        topo.on_node_join("n", host="h")
        assert events == [("n", "join")]

    def test_on_change_fires_on_leave(self) -> None:
        events: list[tuple[str, str]] = []
        topo = DynamicClusterTopology()
        topo.on_node_join("n", host="h")
        topo.on_change(lambda nid, ev, g: events.append((nid, ev)))
        topo.on_node_leave("n")
        assert events == [("n", "leave")]

    def test_on_change_multiple_callbacks(self) -> None:
        call_count = 0
        topo = DynamicClusterTopology()

        def cb1(*_: object) -> None:
            nonlocal call_count
            call_count += 1

        def cb2(*_: object) -> None:
            nonlocal call_count
            call_count += 1

        topo.on_change(cb1)
        topo.on_change(cb2)
        topo.on_node_join("n", host="h")
        assert call_count == 2

    def test_callback_receives_graph_on_join(self) -> None:
        received_graph: list[object] = []
        topo = DynamicClusterTopology()
        topo.on_change(lambda nid, ev, g: received_graph.append(g))
        graph = topo.on_node_join("a", host="h")
        assert received_graph[0] is graph  # same object

    def test_callback_error_does_not_block(self) -> None:
        """A failing callback should not prevent other callbacks or the operation."""
        events: list[str] = []
        topo = DynamicClusterTopology()
        topo.on_change(lambda *_: (_ for _ in ()).throw(RuntimeError("boom")))
        topo.on_change(lambda nid, ev, g: events.append(nid))
        topo.on_node_join("n", host="h")
        assert events == ["n"]

    def test_callback_not_called_after_node_join_for_leave(self) -> None:
        """Verify join callback doesn't get called on leave."""
        events: list[str] = []
        topo = DynamicClusterTopology()
        topo.on_node_join("n", host="h")
        topo.on_change(lambda nid, ev, g: events.append(ev))
        topo.on_node_leave("n")
        assert events == ["leave"]


# ---------------------------------------------------------------------------
# DynamicClusterTopology — Stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_stats_empty(self) -> None:
        topo = DynamicClusterTopology()
        s = topo.stats()
        assert s["node_count"] == 0
        assert s["total_gpus"] == 0
        assert s["link_count"] == 0
        assert s["healthy_nodes"] == 0
        assert s["unhealthy_nodes"] == 0
        assert s["nodes"] == {}

    def test_stats_with_nodes(self) -> None:
        topo = DynamicClusterTopology()
        topo.on_node_join("a", host="h1", gpu_count=4)
        topo.on_node_join("b", host="h2", gpu_count=2)
        topo.mark_unhealthy("b")
        s = topo.stats()
        assert s["node_count"] == 2
        assert s["total_gpus"] == 6
        assert s["link_count"] == 1
        assert s["healthy_nodes"] == 1
        assert s["unhealthy_nodes"] == 1

    def test_stats_node_details(self) -> None:
        topo = DynamicClusterTopology()
        topo.on_node_join("a", host="10.0.0.1", gpu_count=8)
        s = topo.stats()
        assert "a" in s["nodes"]
        na = s["nodes"]["a"]
        assert na["host"] == "10.0.0.1"
        assert na["gpus"] == 8
        assert na["healthy"] is True
        assert isinstance(na["uptime_s"], float)

    def test_stats_uptime_is_positive(self) -> None:
        topo = DynamicClusterTopology()
        topo.on_node_join("a", host="h")
        s = topo.stats()
        assert s["nodes"]["a"]["uptime_s"] >= 0.0

    def test_stats_after_leave(self) -> None:
        topo = DynamicClusterTopology()
        topo.on_node_join("a", host="h1")
        topo.on_node_join("b", host="h2")
        topo.on_node_leave("a")
        s = topo.stats()
        assert s["node_count"] == 1
        assert "a" not in s["nodes"]


# ---------------------------------------------------------------------------
# DynamicClusterTopology — Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_everything(self) -> None:
        topo = DynamicClusterTopology()
        topo.on_node_join("a", host="h1", gpu_count=4)
        topo.on_node_join("b", host="h2", gpu_count=8)
        topo.on_change(lambda *_: None)
        topo.reset()
        assert topo.node_count() == 0
        assert topo.total_gpus() == 0
        assert topo.get_healthy_nodes() == []
        assert topo.get_graph().node_ids == []
        assert topo.get_graph().links == []

    def test_reset_after_reset_is_idempotent(self) -> None:
        topo = DynamicClusterTopology()
        topo.reset()
        topo.reset()
        assert topo.node_count() == 0

    def test_reset_allows_reuse(self) -> None:
        topo = DynamicClusterTopology()
        topo.on_node_join("a", host="h1")
        topo.reset()
        topo.on_node_join("b", host="h2")
        assert topo.node_count() == 1
        assert topo.get_node("b") is not None


# ---------------------------------------------------------------------------
# DynamicClusterTopology — Link Building Details
# ---------------------------------------------------------------------------


class TestLinkBuilding:
    def test_same_host_nvlink_high_bandwidth(self) -> None:
        topo = DynamicClusterTopology()
        topo.on_node_join("a", host="10.0.0.1")
        graph = topo.on_node_join("b", host="10.0.0.1")
        link = graph.links[0]
        assert link.bandwidth_gbps == 600.0
        assert link.latency_us == 5.0
        assert link.is_nvlink is True
        assert link.is_infiniband is False

    def test_cross_host_default_link(self) -> None:
        topo = DynamicClusterTopology()
        topo.on_node_join("a", host="10.0.0.1")
        graph = topo.on_node_join("b", host="10.0.0.2")
        link = graph.links[0]
        assert link.bandwidth_gbps == 12.5
        assert link.latency_us == 500.0
        assert link.is_nvlink is False
        assert link.is_infiniband is False

    def test_cross_host_link_with_high_bw_is_infiniband(self) -> None:
        topo = DynamicClusterTopology(default_bandwidth=40.0)
        topo.on_node_join("a", host="h1")
        graph = topo.on_node_join("b", host="h2")
        link = graph.links[0]
        assert link.is_infiniband is True

    def test_cross_host_link_boundary_infiniband(self) -> None:
        """Exactly 25.0 should NOT be infiniband (bw > 25.0)."""
        topo = DynamicClusterTopology(default_bandwidth=25.0)
        topo.on_node_join("a", host="h1")
        graph = topo.on_node_join("b", host="h2")
        link = graph.links[0]
        assert link.is_infiniband is False

    def test_links_are_bidirectional_in_graph(self) -> None:
        topo = DynamicClusterTopology()
        topo.on_node_join("a", host="h1")
        topo.on_node_join("b", host="h2")
        bw_ab = topo.get_bandwidth("a", "b")
        bw_ba = topo.get_bandwidth("b", "a")
        assert bw_ab == bw_ba
        assert bw_ab > 0


# ---------------------------------------------------------------------------
# DynamicClusterTopology — Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_node_id_on_join(self) -> None:
        topo = DynamicClusterTopology()
        graph = topo.on_node_join("", host="h")
        assert topo.node_count() == 1
        assert topo.get_node("") is not None

    def test_empty_node_id_on_leave(self) -> None:
        topo = DynamicClusterTopology()
        topo.on_node_join("", host="h")
        topo.on_node_leave("")
        assert topo.node_count() == 0

    def test_join_leave_join_same_node(self) -> None:
        topo = DynamicClusterTopology()
        topo.on_node_join("n", host="h")
        topo.on_node_leave("n")
        topo.on_node_join("n", host="h")
        assert topo.node_count() == 1
        assert topo.get_node("n") is not None

    def test_many_nodes_no_crash(self) -> None:
        topo = DynamicClusterTopology()
        for i in range(100):
            topo.on_node_join(f"node-{i}", host=f"10.0.0.{i}")
        assert topo.node_count() == 100
        assert len(topo.get_graph().links) == 4950  # 100*99/2

    def test_many_nodes_leave_all(self) -> None:
        topo = DynamicClusterTopology()
        for i in range(50):
            topo.on_node_join(f"n-{i}", host=f"h{i}")
        for i in range(50):
            topo.on_node_leave(f"n-{i}")
        assert topo.node_count() == 0
        assert topo.get_graph().links == []

    def test_get_node_returns_none_after_leave(self) -> None:
        topo = DynamicClusterTopology()
        topo.on_node_join("n", host="h")
        topo.on_node_leave("n")
        assert topo.get_node("n") is None

    def test_join_with_zero_gpus(self) -> None:
        topo = DynamicClusterTopology()
        topo.on_node_join("n", host="h", gpu_count=0)
        assert topo.total_gpus() == 0
        assert topo.get_graph().gpu_counts == {"n": 0}


# ---------------------------------------------------------------------------
# DynamicClusterTopology — Concurrency Safety (basic)
# ---------------------------------------------------------------------------


class TestConcurrencySafety:
    def test_concurrent_join_from_multiple_threads(self) -> None:
        """Basic sanity: multiple threads joining nodes should not corrupt state."""
        import threading

        topo = DynamicClusterTopology()
        n_threads = 10
        barrier = threading.Barrier(n_threads)
        errors: list[Exception] = []

        def worker(idx: int) -> None:
            try:
                barrier.wait()
                topo.on_node_join(f"n-{idx}", host=f"h{idx}", gpu_count=idx)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(i,)) for i in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors
        assert topo.node_count() == n_threads

    def test_concurrent_leave_does_not_crash(self) -> None:
        topo = DynamicClusterTopology()
        for i in range(10):
            topo.on_node_join(f"n-{i}", host=f"h{i}")

        import threading

        errors: list[Exception] = []

        def leaver(idx: int) -> None:
            try:
                topo.on_node_leave(f"n-{idx}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=leaver, args=(i,)) for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors
        assert topo.node_count() == 0


# ---------------------------------------------------------------------------
# DynamicClusterTopology — Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_full_lifecycle(self) -> None:
        topo = DynamicClusterTopology()

        # Join nodes
        topo.on_node_join("server-1", host="10.0.0.1", gpu_count=4)
        topo.on_node_join("server-2", host="10.0.0.2", gpu_count=8)
        topo.on_node_join("server-3", host="10.0.0.1", gpu_count=4)

        # Verify state
        assert topo.node_count() == 3
        assert topo.total_gpus() == 16
        assert len(topo.get_healthy_nodes()) == 3
        assert len(topo.get_graph().links) == 3

        # server-1 and server-3 share a host -> NVLink
        link_13 = next(
            l for l in topo.get_graph().links
            if {l.source, l.target} == {"server-1", "server-3"}
        )
        assert link_13.is_nvlink
        assert link_13.bandwidth_gbps == 600.0

        # Cross-host links are default
        link_12 = next(
            l for l in topo.get_graph().links
            if {l.source, l.target} == {"server-1", "server-2"}
        )
        assert not link_12.is_nvlink
        assert link_12.bandwidth_gbps == 12.5

        # Mark one unhealthy
        topo.mark_unhealthy("server-2")
        assert topo.get_healthy_nodes() == ["server-1", "server-3"]

        # Remove a node
        topo.on_node_leave("server-2")
        assert topo.node_count() == 2
        assert topo.total_gpus() == 8

        # Reset
        topo.reset()
        assert topo.node_count() == 0
        assert topo.get_graph().links == []

    def test_callback_receives_correct_graph_on_join(self) -> None:
        events: list[TopologyGraph] = []
        topo = DynamicClusterTopology()

        def cb(node_id: str, event: str, graph: TopologyGraph) -> None:
            events.append(graph)

        topo.on_change(cb)
        graph = topo.on_node_join("a", host="h1")
        assert len(events) == 1
        assert events[0] is graph

    def test_callback_receives_correct_graph_on_leave(self) -> None:
        graphs: list[TopologyGraph] = []
        topo = DynamicClusterTopology()
        topo.on_node_join("a", host="h1")
        topo.on_node_join("b", host="h2")

        def cb(node_id: str, event: str, graph: TopologyGraph) -> None:
            graphs.append(graph)

        topo.on_change(cb)
        result = topo.on_node_leave("a")
        assert len(graphs) == 1
        assert graphs[0] is result

    def test_large_bandwidth_infiniband_through_stats(self) -> None:
        topo = DynamicClusterTopology(default_bandwidth=200.0)
        topo.on_node_join("a", host="h1")
        topo.on_node_join("b", host="h2")
        s = topo.stats()
        assert s["link_count"] == 1
        assert topo.get_bandwidth("a", "b") == 200.0
