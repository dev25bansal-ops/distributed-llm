"""Gap tests: Multi-Node & Distributed (cluster topology, NCCL, federation, gossip, zero-copy)."""

import pytest

from distllm.core.cluster_topology import FederationManager, ClusterInfo, CrossClusterLatencyMonitor
from distllm.core.federation_load_balancer import FederationLoadBalancer, RemoteClusterLoad
from distllm.core.gossip_protocol import GossipProtocol, VectorClock, LWWRegister
from distllm.core.zero_copy_transfer import ZeroCopyTransferEngine, TransferBackend, TransferStats
from distllm.core.dynamic_topology import DynamicClusterTopology, NodeInfo


class TestFederationManager:
    def test_register_cluster(self):
        fm = FederationManager()
        fm.register_cluster(ClusterInfo("cluster-1"))
        assert "cluster-1" in fm.list_clusters()

    def test_register_node(self):
        fm = FederationManager()
        fm.register_cluster(ClusterInfo("c1"))
        fm.register_node("node-1", "c1")
        assert "node-1" in fm.get_nodes_in_cluster("c1")

    def test_unregister_node(self):
        fm = FederationManager()
        fm.register_cluster(ClusterInfo("c1"))
        fm.register_node("node-1", "c1")
        fm.unregister_node("node-1")
        assert "node-1" not in fm.get_nodes_in_cluster("c1")

    def test_get_cluster_for_node(self):
        fm = FederationManager()
        fm.register_cluster(ClusterInfo("c1"))
        fm.register_node("node-1", "c1")
        assert fm.get_cluster("node-1") == "c1"

    def test_is_local(self):
        fm = FederationManager("local-cluster")
        fm.register_cluster(ClusterInfo("local-cluster"))
        assert fm.is_local("node-1") is False

    def test_stats(self):
        fm = FederationManager()
        fm.register_cluster(ClusterInfo("c1"))
        s = fm.stats()
        assert "clusters" in s


class TestCrossClusterLatencyMonitor:
    def test_record_and_get_latency(self):
        fm = FederationManager("local")
        fm.register_cluster(ClusterInfo("local"))
        fm.register_cluster(ClusterInfo("remote"))
        monitor = CrossClusterLatencyMonitor(fm)
        monitor.record_latency("local", "remote", 50.0)
        lat = monitor.get_latency("local", "remote")
        assert lat == pytest.approx(50.0, rel=0.5)

    def test_same_cluster_latency_zero(self):
        fm = FederationManager("local")
        fm.register_cluster(ClusterInfo("local"))
        monitor = CrossClusterLatencyMonitor(fm)
        assert monitor.get_latency("local", "local") == 0.0

    def test_unknown_cluster_latency(self):
        fm = FederationManager("local")
        monitor = CrossClusterLatencyMonitor(fm)
        lat = monitor.get_latency("local", "unknown")
        assert lat == 1000.0

    def test_stats(self):
        fm = FederationManager("local")
        fm.register_cluster(ClusterInfo("local"))
        fm.register_cluster(ClusterInfo("remote"))
        monitor = CrossClusterLatencyMonitor(fm)
        monitor.record_latency("local", "remote", 30.0)
        s = monitor.stats()
        assert isinstance(s, dict)


class TestFederationLoadBalancer:
    def test_report_and_get(self):
        lb = FederationLoadBalancer()
        lb.report_load("cluster-1", active_requests=10, pending_requests=2, gpu_utilization=0.5, queue_depth=5)
        load = lb.get_remote_load("cluster-1")
        assert load is not None

    def test_get_all_loads(self):
        lb = FederationLoadBalancer()
        lb.report_load("c1", active_requests=5, pending_requests=1, gpu_utilization=0.3, queue_depth=2)
        lb.report_load("c2", active_requests=50, pending_requests=20, gpu_utilization=0.9, queue_depth=100)
        loads = lb.get_all_loads()
        assert "c1" in loads

    def test_remove_cluster(self):
        lb = FederationLoadBalancer()
        lb.report_load("c1", active_requests=0, pending_requests=0, gpu_utilization=0, queue_depth=0)
        lb.remove_cluster("c1")
        assert lb.get_remote_load("c1") is None

    def test_to_dict(self):
        lb = FederationLoadBalancer()
        lb.report_load("c1", active_requests=5, pending_requests=1, gpu_utilization=0.5, queue_depth=3)
        d = lb.to_dict()
        assert isinstance(d, dict)


class TestGossipProtocol:
    def test_store_and_lookup(self):
        gp = GossipProtocol("node-1")
        gp.store_local("hash123", "ref456")
        result = gp.lookup("hash123")
        assert result == "node-1" or result == "ref456"

    def test_add_remove_peer(self):
        gp = GossipProtocol("node-1")
        gp.add_peer("node-2")
        assert "node-2" in gp.get_peers()
        gp.remove_peer("node-2")
        assert "node-2" not in gp.get_peers()

    def test_advertise_builds_dict(self):
        gp = GossipProtocol("node-1")
        ad = gp.advertise()
        assert isinstance(ad, dict)

    def test_select_peer_returns_none_when_empty(self):
        gp = GossipProtocol("node-1")
        assert gp.select_peer() is None

    def test_tombstone_entry(self):
        gp = GossipProtocol("node-1")
        gp.store_local("hash1", "ref1")
        gp.tombstone_entry("hash1")
        assert gp.lookup("hash1") is not None  # tombstone prevents re-gossip


class TestVectorClock:
    def test_increment(self):
        vc = VectorClock()
        vc.increment("node-1")
        assert vc.clocks["node-1"] == 1

    def test_merge(self):
        vc1 = VectorClock()
        vc1.increment("node-1")
        vc2 = VectorClock()
        vc2.increment("node-2")
        vc1.merge(vc2)
        assert vc1.clocks.get("node-2") == 1

    def test_concurrent(self):
        vc1 = VectorClock()
        vc1.increment("a")
        vc2 = VectorClock()
        vc2.increment("b")
        assert vc1.is_concurrent(vc2)


class TestZeroCopyTransfer:
    def test_engine_init(self):
        engine = ZeroCopyTransferEngine()
        assert engine is not None

    def test_get_stats_returns_list(self):
        engine = ZeroCopyTransferEngine()
        s = engine.get_stats()
        assert isinstance(s, list)

    def test_get_aggregate_stats(self):
        engine = ZeroCopyTransferEngine()
        s = engine.get_aggregate_stats()
        assert isinstance(s, dict)


class TestDynamicClusterTopology:
    def test_on_node_join(self):
        topo = DynamicClusterTopology()
        graph = topo.on_node_join("node-1", "host1", 50051, gpu_count=4)
        assert topo.node_count() == 1
        assert topo.total_gpus() == 4

    def test_on_node_leave(self):
        topo = DynamicClusterTopology()
        topo.on_node_join("node-1", "host1", 50051)
        topo.on_node_leave("node-1")
        assert topo.node_count() == 0

    def test_mark_unhealthy_healthy(self):
        topo = DynamicClusterTopology()
        topo.on_node_join("node-1", "host1", 50051)
        topo.mark_unhealthy("node-1")
        assert "node-1" not in topo.get_healthy_nodes()
        topo.mark_healthy("node-1")
        assert "node-1" in topo.get_healthy_nodes()

    def test_get_graph(self):
        topo = DynamicClusterTopology()
        topo.on_node_join("node-1", "host1", 50051)
        graph = topo.get_graph()
        assert graph is not None
        assert graph.total_gpus() >= 1

    def test_stats(self):
        topo = DynamicClusterTopology()
        topo.on_node_join("node-1", "host1", 50051)
        s = topo.stats()
        assert isinstance(s, dict)

    def test_reset(self):
        topo = DynamicClusterTopology()
        topo.on_node_join("node-1", "host1", 50051)
        topo.reset()
        assert topo.node_count() == 0
