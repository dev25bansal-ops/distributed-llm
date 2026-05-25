"""Tests for FederationRouter, DNSGeoResolver, KVReplicationQueue."""

from unittest.mock import MagicMock, patch

import pytest

from distllm.core.federation_router import (
    DNSGeoResolver,
    FederationRouter,
    KVReplicationQueue,
)


class TestDNSGeoResolver:
    def test_default_region(self):
        resolver = DNSGeoResolver(default_region="us-east-1")
        assert resolver._default_region == "us-east-1"
        assert resolver._region_map == {}
        assert resolver._ip_prefix_map == {}

    def test_map_region(self):
        resolver = DNSGeoResolver()
        resolver.map_region("us-east-1", "cluster-a")
        assert resolver.get_cluster_for_region("us-east-1") == "cluster-a"

    def test_map_region_unknown(self):
        resolver = DNSGeoResolver()
        assert resolver.get_cluster_for_region("nonexistent") is None

    def test_map_ip_prefix_cidr(self):
        resolver = DNSGeoResolver()
        resolver.map_ip_prefix("10.0.0.0/8", "cluster-a")
        assert resolver.resolve(client_ip="10.1.2.3") == "cluster-a"
        assert resolver.resolve(client_ip="192.168.1.1") is None

    def test_map_ip_prefix_exact(self):
        resolver = DNSGeoResolver()
        resolver.map_ip_prefix("10.0.0.", "cluster-a")
        assert resolver.resolve(client_ip="10.0.0.5") == "cluster-a"
        assert resolver.resolve(client_ip="10.0.1.5") is None

    def test_map_ip_prefix_invalid(self):
        resolver = DNSGeoResolver()
        resolver.map_ip_prefix("not-a-prefix", "cluster-a")
        assert resolver.resolve(client_ip="not-a-prefix") == "cluster-a"

    def test_region_before_ip(self):
        resolver = DNSGeoResolver()
        resolver.map_region("us-east-1", "region-cluster")
        resolver.map_ip_prefix("10.0.0.0/8", "ip-cluster")
        result = resolver.resolve(client_ip="10.0.0.1")
        assert result == "ip-cluster"

    @patch("distllm.core.federation_router.socket.getaddrinfo")
    def test_dns_name_resolution(self, mock_getaddrinfo):
        mock_getaddrinfo.return_value = [(None, None, None, None, ("10.0.0.1", 0))]
        resolver = DNSGeoResolver()
        resolver.map_ip_prefix("10.0.0.0/8", "dns-cluster")
        result = resolver.resolve(dns_name="my-cluster.example.com")
        assert result == "dns-cluster"

    @patch("distllm.core.federation_router.socket.getaddrinfo")
    def test_dns_name_resolution_failure(self, mock_getaddrinfo):
        mock_getaddrinfo.side_effect = OSError("DNS resolution failed")
        resolver = DNSGeoResolver()
        resolver.map_ip_prefix("10.0.0.0/8", "dns-cluster")
        result = resolver.resolve(
            client_ip="10.0.0.1", dns_name="my-cluster.example.com"
        )
        assert result == "dns-cluster"
        assert resolver.resolve(client_ip="192.168.1.1") is None

    def test_resolve_none(self):
        resolver = DNSGeoResolver()
        assert resolver.resolve() is None
        assert resolver.resolve(client_ip="10.0.0.1") is None


class TestKVReplicationQueue:
    def test_init(self):
        queue = KVReplicationQueue()
        assert queue.size() == 0

    def test_enqueue(self):
        queue = KVReplicationQueue()
        queue.enqueue("h1", {"data": "v1"}, ["cluster-a", "cluster-b"])
        assert queue.size() == 1

    def test_flush_empty(self):
        queue = KVReplicationQueue()
        assert queue.flush() == 0

    def test_flush_with_forwarder(self):
        forwarder = MagicMock()
        forwarder.forward_kv_cache.return_value = True
        queue = KVReplicationQueue(forwarder=forwarder)
        queue.enqueue("h1", {"data": "v1"}, ["http://node1:8000"])
        result = queue.flush()
        assert result == 1
        assert queue.size() == 0
        forwarder.forward_kv_cache.assert_called_once_with(
            remote_node_url="http://node1:8000",
            prefix_hash="h1",
            kv_data={"data": "v1"},
        )

    def test_flush_batch_size(self):
        forwarder = MagicMock()
        forwarder.forward_kv_cache.return_value = True
        queue = KVReplicationQueue(forwarder=forwarder)
        for i in range(10):
            queue.enqueue(f"h{i}", {}, ["http://node1:8000"])
        result = queue.flush(batch_size=3)
        assert result == 3
        assert queue.size() == 7

    def test_flush_forwarder_failure(self):
        forwarder = MagicMock()
        forwarder.forward_kv_cache.return_value = False
        queue = KVReplicationQueue(forwarder=forwarder)
        queue.enqueue("h1", {}, ["http://node1:8000"])
        result = queue.flush()
        assert result == 0

    def test_size(self):
        queue = KVReplicationQueue()
        queue.enqueue("h1", {}, [])
        queue.enqueue("h2", {}, [])
        assert queue.size() == 2
        queue.flush()
        assert queue.size() == 0


class TestFederationRouterLoadBalancing:
    def test_forward_to_best_cluster(self):
        router = FederationRouter(local_cluster_id="local")
        router.federation_manager.register_cluster(
            type("ClusterInfo", (), {"cluster_id": "remote", "nodes": {"http://remote:8000"}, "edge_nodes": set()})()
        )
        router.federation_manager.clusters["remote"].nodes = {"http://remote:8000"}
        router.load_balancer.report_load("remote", 5, 2, 30.0, 10)

        with patch.object(router._forwarder, "forward_request", return_value={"ok": True}) as mock_fwd:
            resp = router.forward_to_best_cluster(
                {"model": "test"}, candidate_clusters=["remote"]
            )
            assert resp == {"ok": True}
            mock_fwd.assert_called_once()

    def test_forward_to_best_cluster_no_candidates(self):
        router = FederationRouter(local_cluster_id="local")
        with pytest.raises(RuntimeError, match="No available cluster"):
            router.forward_to_best_cluster({"model": "test"}, candidate_clusters=[])

    def test_forward_to_best_cluster_all_overloaded(self):
        router = FederationRouter(local_cluster_id="local")
        router.federation_manager.clusters["remote"] = type("CI", (), {"cluster_id": "remote", "nodes": {"http://r:8000"}})()
        router.federation_manager.clusters["remote"].nodes = {"http://r:8000"}
        router.load_balancer.report_load("remote", 100, 100, 95.0, 60)
        with pytest.raises(RuntimeError, match="No available cluster"):
            router.forward_to_best_cluster({"model": "test"}, candidate_clusters=["remote"])

    def test_forward_to_best_cluster_all_candidates(self):
        router = FederationRouter(local_cluster_id="local")
        router.federation_manager.register_cluster(
            type("ClusterInfo", (), {"cluster_id": "remote", "nodes": {"http://r:8000"}, "edge_nodes": set()})()
        )
        router.federation_manager.clusters["remote"].nodes = {"http://r:8000"}
        router.load_balancer.report_load("remote", 5, 2, 30.0, 10)

        with patch.object(router._forwarder, "forward_request", return_value={"ok": True}) as mock_fwd:
            resp = router.forward_to_best_cluster({"model": "test"})
            assert resp == {"ok": True}

    def test_forward_to_cluster(self):
        router = FederationRouter(local_cluster_id="local")
        router.federation_manager.register_cluster(
            type("ClusterInfo", (), {"cluster_id": "eu-west", "nodes": {"http://eu:8000"}, "edge_nodes": set()})()
        )
        router.federation_manager.clusters["eu-west"].nodes = {"http://eu:8000"}

        with patch.object(router._forwarder, "forward_request", return_value={"ok": True}) as mock_fwd:
            resp = router.forward_to_cluster("eu-west", {"model": "test"})
            assert resp == {"ok": True}

    def test_forward_to_cluster_no_nodes(self):
        router = FederationRouter(local_cluster_id="local")
        router.federation_manager.register_cluster(
            type("ClusterInfo", (), {"cluster_id": "empty", "nodes": set(), "edge_nodes": set()})()
        )
        router.federation_manager.clusters["empty"].nodes = set()
        with pytest.raises(RuntimeError, match="no registered nodes"):
            router.forward_to_cluster("empty", {"model": "test"})


class TestFederationRouterDNS:
    def test_route_with_dns_region(self):
        router = FederationRouter(local_cluster_id="local")
        router.dns_resolver.map_region("us-east-1", "east-cluster")
        target, reason = router.route_with_dns(client_region="us-east-1")
        assert target == "east-cluster"
        assert "dns_region" in reason

    def test_route_with_dns_ip(self):
        router = FederationRouter(local_cluster_id="local")
        router.dns_resolver.map_ip_prefix("10.0.0.0/8", "ip-cluster")
        target, reason = router.route_with_dns(client_ip="10.0.0.5")
        assert target == "ip-cluster"
        assert "dns_ip" in reason

    def test_route_with_dns_fallback_to_capacity(self):
        router = FederationRouter(local_cluster_id="local")
        with patch.object(router.geo_router, "select_target_cluster", return_value=("fallback", "local_capacity")):
            target, reason = router.route_with_dns()
            assert target == "fallback"
            assert reason == "local_capacity"

    def test_route_with_spillover_local_ok(self):
        router = FederationRouter(local_cluster_id="local")
        router.load_reporter.report("local", active=5, gpu_util=0.5)
        target, reason = router.route_with_spillover()
        assert target == "local"
        assert reason == "local_capacity"

    def test_route_with_spillover_triggers(self):
        router = FederationRouter(local_cluster_id="local")
        router.load_reporter.report("local", active=50, gpu_util=0.95)
        with patch.object(router.geo_router, "select_target_cluster", return_value=("remote", "nearest_with_capacity")):
            target, reason = router.route_with_spillover(spill_threshold=0.5)
            assert target == "remote"

    def test_report_load_delegates(self):
        router = FederationRouter(local_cluster_id="local")
        with patch.object(router.geo_router, "report_cluster_load") as mock_report:
            router.report_load("c1", active=5, pending=2, gpu_util=0.6, queue_depth=10)
            mock_report.assert_called_once_with(
                cluster_id="c1", active=5, pending=2, gpu_util=0.6, queue_depth=10
            )

    def test_attach_registrar(self):
        router = FederationRouter(local_cluster_id="local")
        registrar = MagicMock()
        router.attach_registrar(registrar)
        assert registrar.federation_manager is router.federation_manager
        assert registrar.expert_registry is None


class TestFederationRouterStats:
    def test_stats_structure(self):
        router = FederationRouter(local_cluster_id="local")
        stats = router.stats()
        assert "cluster_loads" in stats
        assert "latency_matrix" in stats
        assert "federation" in stats
        assert "geo_regions" in stats
        assert "kv_replication_queue" in stats

    def test_stats_geo_regions(self):
        router = FederationRouter(local_cluster_id="local")
        router.dns_resolver.map_region("us-east-1", "east")
        stats = router.stats()
        assert stats["geo_regions"] == {"us-east-1": "east"}
