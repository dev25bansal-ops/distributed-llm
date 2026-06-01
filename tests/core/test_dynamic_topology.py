"""Tests for dynamic cluster topology discovery and updates."""

import pytest
import threading
from distllm.core.dynamic_topology import DynamicClusterTopology, NodeInfo
from distllm.core.auto_partition.topology import TopologyGraph


class TestNodeInfo:
    def test_defaults(self):
        info = NodeInfo(node_id="node-0", host="192.168.1.1")
        assert info.node_id == "node-0"
        assert info.host == "192.168.1.1"
        assert info.port == 50051
        assert info.gpu_count == 1
        assert info.healthy is True
        assert info.joined_at == 0.0
        assert info.tags == {}

    def test_custom(self):
        info = NodeInfo(
            node_id="gpu-node",
            host="10.0.0.1",
            port=50052,
            gpu_count=8,
            healthy=False,
            tags={"region": "us-east"},
        )
        assert info.port == 50052
        assert info.gpu_count == 8
        assert info.healthy is False
        assert info.tags["region"] == "us-east"


class TestDynamicClusterTopology:
    @pytest.fixture
    def topo(self):
        return DynamicClusterTopology(probe_on_join=False)

    def test_init(self, topo):
        graph = topo.get_graph()
        assert graph.node_ids == []
        assert graph.links == []
        assert topo.node_count() == 0
        assert topo.total_gpus() == 0

    def test_node_join_first_node(self, topo):
        """First node join creates entry but no links."""
        graph = topo.on_node_join("node-0", host="10.0.0.1", gpu_count=4)
        assert topo.node_count() == 1
        assert topo.total_gpus() == 4
        assert graph.node_ids == ["node-0"]
        assert len(graph.links) == 0

    def test_node_join_creates_links(self, topo):
        """Second node join creates a link between both nodes."""
        topo.on_node_join("node-0", host="10.0.0.1", gpu_count=4)
        graph = topo.on_node_join("node-1", host="10.0.0.2", gpu_count=8)
        assert topo.node_count() == 2
        assert len(graph.links) == 1
        link = graph.links[0]
        assert link.source == "node-1"
        assert link.target == "node-0"

    def test_node_join_same_host_detects_nvlink(self, topo):
        """Nodes on the same host get NVLink bandwidth."""
        topo.on_node_join("node-0", host="10.0.0.1")
        graph = topo.on_node_join("node-1", host="10.0.0.1")
        link = graph.links[0]
        assert link.is_nvlink is True
        assert link.bandwidth_gbps == 600.0
        assert link.latency_us == 5.0

    def test_node_join_different_host_default_bandwidth(self, topo):
        """Nodes on different hosts get default inter-node bandwidth."""
        topo.on_node_join("node-0", host="10.0.0.1")
        graph = topo.on_node_join("node-1", host="10.0.0.2")
        link = graph.links[0]
        assert link.is_nvlink is False
        assert link.bandwidth_gbps == 12.5

    def test_node_join_idempotent(self, topo):
        """Re-joining same node updates metadata but doesn't duplicate."""
        topo.on_node_join("node-0", host="10.0.0.1", gpu_count=4)
        topo.on_node_join("node-1", host="10.0.0.2", gpu_count=8)
        assert topo.node_count() == 2

        topo.on_node_join("node-0", host="10.0.0.1", gpu_count=16)
        assert topo.node_count() == 2
        assert topo._nodes["node-0"].gpu_count == 16

    def test_node_leave_removes_node(self, topo):
        """Node departure removes node and its links."""
        topo.on_node_join("node-0", host="10.0.0.1")
        topo.on_node_join("node-1", host="10.0.0.2")
        topo.on_node_join("node-2", host="10.0.0.3")
        assert topo.node_count() == 3

        graph = topo.on_node_leave("node-1")
        assert topo.node_count() == 2
        assert "node-1" not in graph.node_ids
        # No links involving node-1
        assert not any(
            l.source == "node-1" or l.target == "node-1"
            for l in graph.links
        )

    def test_node_leave_unknown(self, topo):
        """Leaving an unknown node is a no-op."""
        topo.on_node_join("node-0", host="10.0.0.1")
        graph = topo.on_node_leave("nonexistent")
        assert topo.node_count() == 1
        assert graph.node_ids == ["node-0"]

    def test_node_leave_last_node(self, topo):
        """Removing the last node leaves empty topology."""
        topo.on_node_join("node-0", host="10.0.0.1")
        graph = topo.on_node_leave("node-0")
        assert topo.node_count() == 0
        assert graph.node_ids == []

    def test_mark_unhealthy(self, topo):
        topo.on_node_join("node-0", host="10.0.0.1")
        assert topo.get_node("node-0").healthy is True
        topo.mark_unhealthy("node-0")
        assert topo.get_node("node-0").healthy is False

    def test_mark_healthy(self, topo):
        topo.on_node_join("node-0", host="10.0.0.1")
        topo.mark_unhealthy("node-0")
        topo.mark_healthy("node-0")
        assert topo.get_node("node-0").healthy is True

    def test_get_healthy_nodes(self, topo):
        topo.on_node_join("node-0", host="10.0.0.1")
        topo.on_node_join("node-1", host="10.0.0.2")
        topo.mark_unhealthy("node-0")
        healthy = topo.get_healthy_nodes()
        assert "node-1" in healthy
        assert "node-0" not in healthy

    def test_get_node(self, topo):
        topo.on_node_join("node-0", host="10.0.0.1", gpu_count=4)
        info = topo.get_node("node-0")
        assert info is not None
        assert info.host == "10.0.0.1"
        assert info.gpu_count == 4

        assert topo.get_node("nonexistent") is None

    def test_get_nodes(self, topo):
        topo.on_node_join("node-0", host="10.0.0.1")
        topo.on_node_join("node-1", host="10.0.0.2")
        nodes = topo.get_nodes()
        assert len(nodes) == 2
        assert "node-0" in nodes
        assert "node-1" in nodes

    def test_bandwidth_and_latency_queries(self, topo):
        topo.on_node_join("node-0", host="10.0.0.1")
        topo.on_node_join("node-1", host="10.0.0.2")
        bw = topo.get_bandwidth("node-0", "node-1")
        lat = topo.get_latency("node-0", "node-1")
        assert bw == 12.5
        assert lat == 500.0

    def test_bandwidth_same_host(self, topo):
        topo.on_node_join("node-0", host="10.0.0.1")
        topo.on_node_join("node-1", host="10.0.0.1")
        bw = topo.get_bandwidth("node-0", "node-1")
        assert bw == 600.0

    def test_stats(self, topo):
        topo.on_node_join("node-0", host="10.0.0.1", gpu_count=4)
        topo.on_node_join("node-1", host="10.0.0.2", gpu_count=8)
        topo.mark_unhealthy("node-0")
        stats = topo.stats()
        assert stats["node_count"] == 2
        assert stats["total_gpus"] == 12
        assert stats["link_count"] == 1
        assert stats["healthy_nodes"] == 1
        assert stats["unhealthy_nodes"] == 1
        assert "node-0" in stats["nodes"]
        assert "node-1" in stats["nodes"]

    def test_reset(self, topo):
        topo.on_node_join("node-0", host="10.0.0.1")
        topo.on_node_join("node-1", host="10.0.0.2")
        topo.reset()
        assert topo.node_count() == 0
        assert topo.get_graph().node_ids == []
        assert topo.get_graph().links == []

    def test_callbacks_on_join(self, topo):
        events = []
        topo.on_change(lambda nid, evt, g: events.append((nid, evt)))
        topo.on_node_join("node-0", host="10.0.0.1")
        assert len(events) == 1
        assert events[0] == ("node-0", "join")

    def test_callbacks_on_leave(self, topo):
        events = []
        topo.on_node_join("node-0", host="10.0.0.1")
        topo.on_node_join("node-1", host="10.0.0.2")
        topo.on_change(lambda nid, evt, g: events.append((nid, evt)))
        topo.on_node_leave("node-0")
        assert ("node-0", "leave") in events

    def test_multiple_callbacks(self, topo):
        results = []
        topo.on_change(lambda nid, evt, g: results.append(f"cb1:{nid}"))
        topo.on_change(lambda nid, evt, g: results.append(f"cb2:{nid}"))
        topo.on_node_join("node-0", host="10.0.0.1")
        assert len(results) == 2
        assert "cb1:node-0" in results
        assert "cb2:node-0" in results

    def test_callback_exception_does_not_block(self, topo):
        results = []
        def failing_cb(nid, evt, g):
            raise ValueError("oops")
        def working_cb(nid, evt, g):
            results.append(nid)
        topo.on_change(failing_cb)
        topo.on_change(working_cb)
        topo.on_node_join("node-0", host="10.0.0.1")
        assert results == ["node-0"]

    def test_get_graph_returns_copy(self, topo):
        topo.on_node_join("node-0", host="10.0.0.1")
        graph1 = topo.get_graph()
        graph2 = topo.get_graph()
        graph1.node_ids.append("injected")
        assert "injected" not in graph2.node_ids

    def test_multiple_nodes_multiple_links(self, topo):
        for i in range(4):
            topo.on_node_join(f"node-{i}", host=f"10.0.0.{i+1}")
        assert topo.node_count() == 4
        # n*(n-1)/2 links = 6
        assert len(topo.get_graph().links) == 6

    def test_probe_on_join_default(self):
        """With probe_on_join, links are created at join time."""
        topo = DynamicClusterTopology(probe_on_join=True)
        topo.on_node_join("node-0", host="10.0.0.1")
        assert topo.get_graph().links == []
        topo.on_node_join("node-1", host="10.0.0.2")
        assert len(topo.get_graph().links) == 1

    def test_get_graph_after_leave_maintains_links(self, topo):
        topo.on_node_join("node-0", host="10.0.0.1")
        topo.on_node_join("node-1", host="10.0.0.2")
        topo.on_node_join("node-2", host="10.0.0.3")
        links_before = len(topo.get_graph().links)
        topo.on_node_leave("node-1")
        graph = topo.get_graph()
        assert len(graph.links) == links_before - 2
        # Remaining links: node-0<->node-2 only


class TestPipelineOrchestratorIntegration:
    @pytest.fixture
    def orchestrator(self):
        from distllm.dist.pipeline import PipelineOrchestrator
        from distllm.core.dynamic_topology import DynamicClusterTopology
        topo = DynamicClusterTopology(probe_on_join=False)
        return PipelineOrchestrator(total_layers=24, topology=topo)

    def test_register_updates_topology(self, orchestrator):
        orchestrator.register_node(
            node_id="node-0", host="10.0.0.1", port=50051,
            start_layer=0, end_layer=11,
        )
        topo = orchestrator.get_dynamic_topology()
        assert topo.node_count() == 1
        assert topo.get_node("node-0").host == "10.0.0.1"

    def test_register_multiple_nodes_creates_links(self, orchestrator):
        orchestrator.register_node(
            node_id="node-0", host="10.0.0.1", port=50051,
            start_layer=0, end_layer=11,
        )
        orchestrator.register_node(
            node_id="node-1", host="10.0.0.2", port=50052,
            start_layer=12, end_layer=23,
        )
        topo = orchestrator.get_dynamic_topology()
        assert topo.node_count() == 2
        assert len(topo.get_graph().links) == 1

    def test_register_same_host_nvlink(self, orchestrator):
        orchestrator.register_node(
            node_id="node-0", host="10.0.0.1", port=50051,
            start_layer=0, end_layer=11,
        )
        orchestrator.register_node(
            node_id="node-1", host="10.0.0.1", port=50052,
            start_layer=12, end_layer=23,
        )
        bw = orchestrator.get_topology_graph().get_bandwidth("node-0", "node-1")
        assert bw == 600.0

    def test_unregister_updates_topology(self, orchestrator):
        orchestrator.register_node(
            node_id="node-0", host="10.0.0.1", port=50051,
            start_layer=0, end_layer=11,
        )
        orchestrator.register_node(
            node_id="node-1", host="10.0.0.2", port=50052,
            start_layer=12, end_layer=23,
        )
        orchestrator.unregister_node("node-0")
        topo = orchestrator.get_dynamic_topology()
        assert topo.node_count() == 1
        assert topo.get_node("node-1") is not None
        assert topo.get_node("node-0") is None

    def test_unregister_last_node(self, orchestrator):
        orchestrator.register_node(
            node_id="node-0", host="10.0.0.1", port=50051,
            start_layer=0, end_layer=23,
        )
        orchestrator.unregister_node("node-0")
        topo = orchestrator.get_dynamic_topology()
        assert topo.node_count() == 0

    def test_get_topology_graph_returns_graph(self, orchestrator):
        orchestrator.register_node(
            node_id="node-0", host="10.0.0.1", port=50051,
            start_layer=0, end_layer=23,
        )
        graph = orchestrator.get_topology_graph()
        assert "node-0" in graph.node_ids
        assert graph.gpu_counts["node-0"] == 1

    def test_topology_pipeline_registration_order(self, orchestrator):
        for i in range(3):
            orchestrator.register_node(
                node_id=f"node-{i}", host=f"10.0.0.{i+1}", port=50051,
                start_layer=i * 8, end_layer=(i + 1) * 8 - 1,
            )
        topo = orchestrator.get_dynamic_topology()
        assert topo.node_count() == 3
        assert len(topo.get_graph().links) == 3
