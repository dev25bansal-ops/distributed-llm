"""Tests for federated inference fabric components."""

import asyncio
import time

from distllm.core.latency_prober import LatencyProber, ProbeResult
from distllm.core.federation_discovery import FederationPeerDiscovery, PeerInfo
from distllm.core.cross_cluster_forwarder import CrossClusterForwarder
from distllm.core.federation_load_balancer import FederationLoadBalancer, RemoteClusterLoad
from distllm.core.cache_migration import CacheMigrator


class TestLatencyProber:
    def test_add_remove_target(self):
        prober = LatencyProber()
        prober.add_target("cluster-b", "node-1", "10.0.0.1", 50051)
        assert "cluster-b" in prober._targets
        assert len(prober._targets["cluster-b"]) == 1

        prober.remove_target("cluster-b", "node-1")
        assert len(prober._targets["cluster-b"]) == 0

    def test_custom_ping_function(self):
        prober = LatencyProber()
        prober.set_ping_function(lambda host, port, timeout: 5.0)  # Always 5ms

        result = asyncio.get_event_loop().run_until_complete(
            prober.probe_once("cluster-b", "node-1", "10.0.0.1", 50051)
        )
        assert result.success
        assert result.rtt_ms >= 5.0  # Includes measurement overhead

    def test_result_callback(self):
        prober = LatencyProber()
        results = []
        prober.set_result_callback(lambda r: results.append(r))
        prober.set_ping_function(lambda host, port, timeout: 3.0)

        asyncio.get_event_loop().run_until_complete(
            prober.probe_once("cluster-b", "node-1", "10.0.0.1", 50051)
        )
        assert len(results) == 1
        assert results[0].success

    def test_history(self):
        prober = LatencyProber()
        prober.set_ping_function(lambda host, port, timeout: 1.0)

        for i in range(5):
            asyncio.get_event_loop().run_until_complete(
                prober.probe_once("cluster-b", f"node-{i}", "10.0.0.1", 50051)
            )

        history = prober.get_history(limit=3)
        assert len(history) == 3

    def test_latest_latency(self):
        prober = LatencyProber()
        prober.set_ping_function(lambda host, port, timeout: 10.0)

        asyncio.get_event_loop().run_until_complete(
            prober.probe_once("cluster-b", "node-1", "10.0.0.1", 50051)
        )
        latency = prober.get_latest_latency("cluster-b")
        assert latency is not None
        assert latency >= 10.0


class TestFederationPeerDiscovery:
    def test_add_seed_nodes(self):
        discovery = FederationPeerDiscovery("cluster-a", "localhost", 8000)
        discovery.add_seed_nodes(["http://peer1:8000", "http://peer2:8000"])
        assert len(discovery._seed_nodes) == 2

    def test_register_peer(self):
        discovery = FederationPeerDiscovery("cluster-a", "localhost", 8000)
        peer = PeerInfo(cluster_id="cluster-b", host="10.0.0.1", port=8000)
        discovery._register_peer(peer)

        assert discovery.get_peer("cluster-b") is not None
        assert discovery.get_peer("cluster-b").last_seen > 0

    def test_get_peers(self):
        discovery = FederationPeerDiscovery("cluster-a", "localhost", 8000)
        discovery._register_peer(PeerInfo(cluster_id="b", host="10.0.0.1", port=8000))
        discovery._register_peer(PeerInfo(cluster_id="c", host="10.0.0.2", port=8000))

        peers = discovery.get_peers()
        assert len(peers) == 2

    def test_own_url(self):
        discovery = FederationPeerDiscovery("cluster-a", "coord.example.com", 9000)
        # PeerInfo.url property test
        peer = PeerInfo(cluster_id="b", host="coord.example.com", port=9000)
        assert peer.url == "http://coord.example.com:9000"


class TestFederationLoadBalancer:
    def test_report_load(self):
        lb = FederationLoadBalancer()
        lb.report_load("cluster-b", active_requests=10, pending_requests=5,
                        gpu_utilization=70.0, queue_depth=20)

        load = lb.get_remote_load("cluster-b")
        assert load is not None
        assert load.gpu_utilization == 70.0

    def test_overloaded_detection(self):
        lb = FederationLoadBalancer()
        lb.report_load("cluster-b", active_requests=100, pending_requests=100,
                        gpu_utilization=95.0, queue_depth=60)

        load = lb.get_remote_load("cluster-b")
        assert load.is_overloaded
        assert load.available_capacity == 0.0

    def test_best_cluster_selection(self):
        lb = FederationLoadBalancer()
        lb.report_load("b", 10, 5, 50.0, 10)
        lb.report_load("c", 20, 15, 80.0, 30)
        lb.report_load("d", 5, 2, 30.0, 5)

        best = lb.get_best_cluster(["b", "c", "d"])
        assert best == "d"  # Lowest score

    def test_skips_overloaded(self):
        lb = FederationLoadBalancer()
        lb.report_load("b", 100, 100, 95.0, 60)
        lb.report_load("c", 5, 2, 30.0, 5)

        best = lb.get_best_cluster(["b", "c"])
        assert best == "c"  # b is overloaded

    def test_stale_detection(self):
        lb = FederationLoadBalancer(stale_threshold_s=1.0)
        lb.report_load("b", 10, 5, 50.0, 10)

        # Immediately not stale
        load = lb.get_remote_load("b")
        assert not load.stale

    def test_remove_cluster(self):
        lb = FederationLoadBalancer()
        lb.report_load("b", 10, 5, 50.0, 10)
        lb.remove_cluster("b")
        assert lb.get_remote_load("b") is None


class TestCacheMigrator:
    def test_no_transport_returns_false(self):
        migrator = CacheMigrator()
        result = migrator.migrate_cache("http://src:8000", "http://dst:8000", ["hash1"])
        assert result is False
