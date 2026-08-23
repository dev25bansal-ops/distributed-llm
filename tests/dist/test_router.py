"""Tests for distllm.dist.p2p.router — zero mocks, real objects only."""

from __future__ import annotations

import threading

import pytest

from distllm.dist.p2p.router import DNSGeoResolver, FederationRouter, KVReplicationQueue
from distllm.dist.topology import ClusterInfo
from distllm.dist.geo import LoadReporter
from distllm.dist.cross_cluster import CrossClusterForwarder


# ═══════════════════════════════════════════════════════════════════════════════
# DNSGeoResolver
# ═══════════════════════════════════════════════════════════════════════════════


class TestDNSGeoResolver:
    """DNSGeoResolver: IP-prefix / region mapping and resolution."""

    def test_default_region_default(self) -> None:
        resolver = DNSGeoResolver()
        assert resolver._default_region == "default"

    def test_default_region_custom(self) -> None:
        resolver = DNSGeoResolver(default_region="us-east")
        assert resolver._default_region == "us-east"

    def test_map_region_and_get(self) -> None:
        resolver = DNSGeoResolver()
        resolver.map_region("us-east", "cluster-a")
        assert resolver.get_cluster_for_region("us-east") == "cluster-a"

    def test_map_region_twice_overwrites(self) -> None:
        resolver = DNSGeoResolver()
        resolver.map_region("us-east", "cluster-a")
        resolver.map_region("us-east", "cluster-b")
        assert resolver.get_cluster_for_region("us-east") == "cluster-b"

    def test_get_region_unknown(self) -> None:
        resolver = DNSGeoResolver()
        assert resolver.get_cluster_for_region("nonexistent") is None

    def test_map_ip_prefix_cidr(self) -> None:
        resolver = DNSGeoResolver()
        resolver.map_ip_prefix("10.0.0.0/24", "cluster-a")
        assert resolver._resolve_ip("10.0.0.1") == "cluster-a"
        assert resolver._resolve_ip("10.0.1.1") is None

    def test_map_ip_prefix_dot_prefix(self) -> None:
        resolver = DNSGeoResolver()
        resolver.map_ip_prefix("192.168.", "cluster-b")
        assert resolver._resolve_ip("192.168.1.1") == "cluster-b"
        assert resolver._resolve_ip("10.0.0.1") is None

    def test_resolve_client_ip_matches(self) -> None:
        resolver = DNSGeoResolver()
        resolver.map_ip_prefix("10.0.0.0/8", "cluster-c")
        assert resolver.resolve(client_ip="10.1.2.3") == "cluster-c"

    def test_resolve_client_ip_no_match(self) -> None:
        resolver = DNSGeoResolver()
        resolver.map_ip_prefix("10.0.0.0/8", "cluster-c")
        assert resolver.resolve(client_ip="192.168.1.1") is None

    def test_resolve_none(self) -> None:
        resolver = DNSGeoResolver()
        assert resolver.resolve() is None

    def test_multiple_prefixes(self) -> None:
        resolver = DNSGeoResolver()
        resolver.map_ip_prefix("10.0.0.0/8", "cluster-x")
        resolver.map_ip_prefix("192.168.0.0/16", "cluster-y")
        assert resolver._resolve_ip("10.10.10.10") == "cluster-x"
        assert resolver._resolve_ip("192.168.1.1") == "cluster-y"

    def test_ip_with_slash_triggers_cidr(self) -> None:
        resolver = DNSGeoResolver()
        resolver.map_ip_prefix("172.16.0.0/12", "cidr-cluster")
        assert resolver._resolve_ip("172.20.30.40") == "cidr-cluster"

    def test_invalid_prefix_falls_back_to_startswith(self) -> None:
        resolver = DNSGeoResolver()
        resolver.map_ip_prefix("not-a-prefix", "fallback-cluster")
        assert resolver._resolve_ip("not-a-prefix-something") == "fallback-cluster"

    def test_invalid_prefix_non_matching(self) -> None:
        resolver = DNSGeoResolver()
        resolver.map_ip_prefix("not-a-prefix", "fallback-cluster")
        assert resolver._resolve_ip("other-value") is None

    def test_concurrent_map_and_get(self) -> None:
        resolver = DNSGeoResolver()
        errors: list[Exception] = []

        def mapper() -> None:
            for i in range(100):
                resolver.map_region(f"region-{i}", f"cluster-{i}")

        def reader() -> None:
            for i in range(100):
                try:
                    resolver.get_cluster_for_region(f"region-{i}")
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=mapper) for _ in range(3)]
        threads += [threading.Thread(target=reader) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors

    def test_concurrent_map_ip_and_resolve(self) -> None:
        resolver = DNSGeoResolver()
        errors: list[Exception] = []

        def mapper() -> None:
            for i in range(100):
                resolver.map_ip_prefix(f"10.0.{i}.0/24", f"c-{i}")

        def resolver_func() -> None:
            for _ in range(100):
                try:
                    resolver._resolve_ip("10.0.50.1")
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=mapper) for _ in range(3)]
        threads += [threading.Thread(target=resolver_func) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors


# ═══════════════════════════════════════════════════════════════════════════════
# KVReplicationQueue
# ═══════════════════════════════════════════════════════════════════════════════


class TestKVReplicationQueue:
    """KVReplicationQueue: enqueue, flush, size."""

    def test_default_forwarder(self) -> None:
        queue = KVReplicationQueue()
        assert isinstance(queue._forwarder, CrossClusterForwarder)

    def test_custom_forwarder(self) -> None:
        fwd = CrossClusterForwarder()
        queue = KVReplicationQueue(forwarder=fwd)
        assert queue._forwarder is fwd

    def test_size_empty(self) -> None:
        queue = KVReplicationQueue()
        assert queue.size() == 0

    def test_enqueue_single(self) -> None:
        queue = KVReplicationQueue()
        queue.enqueue("hash1", {"key": [1.0]}, ["cluster-a"])
        assert queue.size() == 1

    def test_enqueue_multiple(self) -> None:
        queue = KVReplicationQueue()
        queue.enqueue("h1", {}, ["c1"])
        queue.enqueue("h2", {}, ["c2"])
        queue.enqueue("h3", {}, ["c3"])
        assert queue.size() == 3

    def test_flush_empty_returns_zero(self) -> None:
        queue = KVReplicationQueue()
        assert queue.flush() == 0

    def test_flush_bad_target_returns_zero(self) -> None:
        """flush returns 0 when forward_kv_cache fails (bad URL)."""
        queue = KVReplicationQueue()
        queue.enqueue("hash1", {"key": [1.0]}, ["http://127.0.0.1:1"])
        result = queue.flush(batch_size=8)
        assert result == 0

    def test_flush_empty_targets_returns_zero(self) -> None:
        """flush on entry with empty target list returns 0."""
        queue = KVReplicationQueue()
        queue.enqueue("h1", {}, [])
        assert queue.flush() == 0

    def test_flush_batch_size_limits_dequeue(self) -> None:
        queue = KVReplicationQueue()
        for i in range(10):
            queue.enqueue(f"h{i}", {}, ["target"])
        assert queue.size() == 10
        queue.flush(batch_size=3)
        assert queue.size() == 7

    def test_flush_all_when_batch_exceeds_queue(self) -> None:
        queue = KVReplicationQueue()
        for i in range(3):
            queue.enqueue(f"h{i}", {}, ["target"])
        queue.flush(batch_size=10)
        assert queue.size() == 0

    def test_enqueue_round_trip_updates_size(self) -> None:
        queue = KVReplicationQueue()
        assert queue.size() == 0
        queue.enqueue("h1", {"a": 1}, ["c1"])
        assert queue.size() == 1
        queue.flush()
        assert queue.size() == 0

    def test_enqueue_empty_kv_data(self) -> None:
        queue = KVReplicationQueue()
        queue.enqueue("h1", {}, ["http://127.0.0.1:1"])
        result = queue.flush()
        assert result == 0


# ═══════════════════════════════════════════════════════════════════════════════
# FederationRouter
# ═══════════════════════════════════════════════════════════════════════════════


class TestFederationRouterConstruction:
    """FederationRouter: construction and basic attribute access."""

    def test_default_construction(self) -> None:
        router = FederationRouter(local_cluster_id="test-cluster")
        assert router.federation_manager.local_cluster_id == "test-cluster"
        assert router.load_reporter is not None
        assert router.dns_resolver is not None
        assert router.kv_replication is not None
        assert router.latency_monitor is not None
        assert router.load_balancer is not None
        assert router.geo_router is not None

    def test_default_local_cluster_id(self) -> None:
        router = FederationRouter()
        assert router.federation_manager.local_cluster_id == "default"

    def test_custom_load_reporter(self) -> None:
        reporter = LoadReporter()
        router = FederationRouter(load_reporter=reporter)
        assert router.load_reporter is reporter
        assert router.geo_router._load_reporter is reporter


class TestFederationRouterReporting:
    """FederationRouter: load reporting."""

    def test_report_load(self) -> None:
        router = FederationRouter()
        router.report_load("cluster-x", active=5, pending=2, gpu_util=0.6, queue_depth=3)
        load = router.load_reporter.get_load("cluster-x")
        assert load is not None
        assert load.active_requests == 5
        assert load.pending_requests == 2
        assert load.gpu_utilization == 0.6
        assert load.queue_depth == 3

    def test_report_load_updates(self) -> None:
        router = FederationRouter()
        router.report_load("cluster-x", active=5)
        router.report_load("cluster-x", active=10)
        load = router.load_reporter.get_load("cluster-x")
        assert load is not None
        assert load.active_requests == 10

    def test_report_load_minimal_args(self) -> None:
        router = FederationRouter()
        router.report_load("cluster-x")
        load = router.load_reporter.get_load("cluster-x")
        assert load is not None
        assert load.active_requests == 0
        assert load.gpu_utilization == 0.0


class _RegistrarStub:
    """Minimal stub with writable attributes for attach_registrar tests."""

    def __init__(self) -> None:
        self.federation_manager = None
        self.expert_registry = None


class TestFederationRouterAttach:
    """FederationRouter: attach_registrar."""

    def test_attach_sets_federation_manager(self) -> None:
        router = FederationRouter()
        registrar = _RegistrarStub()
        router.attach_registrar(registrar)
        assert registrar.federation_manager is router.federation_manager

    def test_attach_with_expert_registry(self) -> None:
        router = FederationRouter()
        registrar = _RegistrarStub()
        expert_reg = object()
        router.attach_registrar(registrar, expert_registry=expert_reg)
        assert registrar.expert_registry is expert_reg

    def test_attach_no_expert_registry(self) -> None:
        router = FederationRouter()
        registrar = _RegistrarStub()
        router.attach_registrar(registrar)
        assert registrar.federation_manager is router.federation_manager
        assert registrar.expert_registry is None


class TestFederationRouterDNSRouting:
    """FederationRouter: route_with_dns."""

    def test_dns_region_match(self) -> None:
        router = FederationRouter()
        router.dns_resolver.map_region("eu-west", "cluster-eu")
        target, reason, *_ = router.route_with_dns(client_region="eu-west")
        assert target == "cluster-eu"
        assert "dns_region" in reason

    def test_dns_region_priority_over_ip(self) -> None:
        router = FederationRouter()
        router.dns_resolver.map_region("us-east", "region-cluster")
        router.dns_resolver.map_ip_prefix("10.0.0.0/8", "ip-cluster")
        target, reason, *_ = router.route_with_dns(
            client_region="us-east", client_ip="10.0.0.1"
        )
        assert target == "region-cluster"
        assert "dns_region" in reason

    def test_dns_ip_match(self) -> None:
        router = FederationRouter()
        router.dns_resolver.map_ip_prefix("192.168.0.0/16", "ip-cluster")
        target, reason, *_ = router.route_with_dns(client_ip="192.168.1.100")
        assert target == "ip-cluster"
        assert "dns_ip" in reason

    def test_dns_no_match_returns_local(self) -> None:
        router = FederationRouter(local_cluster_id="my-cluster")
        target, reason, *_ = router.route_with_dns()
        assert target == "my-cluster"
        assert reason == "no_alternative"

    def test_dns_no_region_or_ip_uses_source_cluster(self) -> None:
        """Without client hints, route_with_dns selects based on source."""
        router = FederationRouter(local_cluster_id="local-cluster")
        target, reason, *_ = router.route_with_dns(source_cluster="local-cluster")
        assert target == "local-cluster"
        assert reason == "no_alternative"

    def test_dns_region_no_match_falls_back_to_ip(self) -> None:
        router = FederationRouter()
        router.dns_resolver.map_ip_prefix("10.0.0.0/8", "ip-cluster")
        target, reason, *_ = router.route_with_dns(
            client_region="unknown-region", client_ip="10.0.0.1"
        )
        assert target == "ip-cluster"
        assert "dns_ip" in reason


class TestFederationRouterSpillover:
    """FederationRouter: route_with_spillover."""

    def test_spillover_local_capacity(self) -> None:
        router = FederationRouter(local_cluster_id="local-cluster")
        router.report_load("local-cluster", gpu_util=0.5)
        target, reason, *_ = router.route_with_spillover(spill_threshold=0.85)
        assert target == "local-cluster"
        assert reason == "local_capacity"

    def test_spillover_triggers_when_overloaded(self) -> None:
        router = FederationRouter(local_cluster_id="local-cluster")
        router.report_load("local-cluster", gpu_util=0.95)
        target, reason, *_ = router.route_with_spillover(spill_threshold=0.85)
        assert target == "local-cluster"
        assert reason == "no_alternative"

    def test_spillover_no_load_data_skips_capacity_check(self) -> None:
        router = FederationRouter(local_cluster_id="local-cluster")
        target, reason, *_ = router.route_with_spillover(spill_threshold=0.85)
        assert target == "local-cluster"
        assert "no_alternative" in reason

    def test_spillover_uses_custom_threshold(self) -> None:
        router = FederationRouter(local_cluster_id="local-cluster")
        router.report_load("local-cluster", gpu_util=0.90)
        target, reason, *_ = router.route_with_spillover(spill_threshold=0.95)
        assert target == "local-cluster"
        assert reason == "local_capacity"

    def test_spillover_routes_to_remote_with_capacity(self) -> None:
        router = FederationRouter(local_cluster_id="local-cluster")
        router.federation_manager.register_cluster(ClusterInfo(cluster_id="remote-cluster"))
        router.report_load("local-cluster", gpu_util=0.95)
        router.report_load("remote-cluster", gpu_util=0.3)
        target, reason, *_ = router.route_with_spillover(spill_threshold=0.85)
        assert target == "remote-cluster"
        assert "nearest_with_capacity" in reason

    def test_spillover_all_remote_overloaded(self) -> None:
        router = FederationRouter(local_cluster_id="local-cluster")
        router.federation_manager.register_cluster(ClusterInfo(cluster_id="remote-cluster"))
        router.report_load("local-cluster", gpu_util=0.95)
        router.report_load("remote-cluster", gpu_util=0.95, queue_depth=100)
        target, reason, *_ = router.route_with_spillover(spill_threshold=0.85)
        assert target == "local-cluster"
        assert "all_remote_overloaded" in reason

    def test_spillover_explicit_source_cluster(self) -> None:
        router = FederationRouter(local_cluster_id="local-cluster")
        router.federation_manager.register_cluster(ClusterInfo(cluster_id="other-cluster"))
        router.report_load("other-cluster", gpu_util=0.5)
        target, reason, *_ = router.route_with_spillover(
            source_cluster="other-cluster", spill_threshold=0.85
        )
        assert target == "other-cluster"
        assert reason == "local_capacity"


class TestFederationRouterRoute:
    """FederationRouter: route() delegates to route_with_spillover."""

    def test_route_returns_strings(self) -> None:
        router = FederationRouter(local_cluster_id="local-cluster")
        target, reason, *_ = router.route()
        assert isinstance(target, str)
        assert isinstance(reason, str)

    def test_route_no_other_clusters(self) -> None:
        router = FederationRouter(local_cluster_id="solo")
        target, reason, *_ = router.route()
        assert target == "solo"
        assert reason == "no_alternative"

    def test_route_does_not_consult_dns(self) -> None:
        """route() uses spillover, not dns routing."""
        router = FederationRouter(local_cluster_id="local-cluster")
        router.dns_resolver.map_region("us-east", "dns-cluster")
        router.report_load("local-cluster", gpu_util=0.5)
        target, *_ = router.route()
        assert target == "local-cluster"


class TestFederationRouterForward:
    """FederationRouter: forward_to_cluster / forward_to_best_cluster.

    Only tests error paths that do not reach HTTP calls.
    """

    def test_forward_to_cluster_no_nodes(self) -> None:
        router = FederationRouter(local_cluster_id="local-cluster")
        router.federation_manager.register_cluster(ClusterInfo(cluster_id="remote-cluster"))
        with pytest.raises(RuntimeError, match="no registered nodes"):
            router.forward_to_cluster("remote-cluster", {"request": "test"})

    def test_forward_to_cluster_unknown_cluster(self) -> None:
        router = FederationRouter()
        with pytest.raises(RuntimeError):
            router.forward_to_cluster("does-not-exist", {"request": "test"})

    def test_forward_to_best_cluster_no_candidates(self) -> None:
        router = FederationRouter()
        with pytest.raises(RuntimeError, match="No available cluster"):
            router.forward_to_best_cluster({"request": "test"}, candidate_clusters=[])

    def test_forward_to_best_cluster_remote_no_nodes(self) -> None:
        router = FederationRouter(local_cluster_id="local-cluster")
        router.federation_manager.register_cluster(ClusterInfo(cluster_id="remote-cluster"))
        with pytest.raises(RuntimeError, match="no registered nodes"):
            router.forward_to_best_cluster({"request": "test"})

    def test_forward_to_best_cluster_overloaded(self) -> None:
        router = FederationRouter(local_cluster_id="local-cluster")
        router.federation_manager.register_cluster(ClusterInfo(cluster_id="remote-cluster"))
        router.federation_manager.register_node("http://node-1:8000", "remote-cluster")
        router.load_balancer.report_load(
            "remote-cluster",
            active_requests=100,
            pending_requests=0,
            gpu_utilization=95.0,
            queue_depth=60,
        )
        with pytest.raises(RuntimeError, match="No available cluster"):
            router.forward_to_best_cluster({"request": "test"})


class TestFederationRouterKVReplication:
    """FederationRouter: replicate_prefix_cache."""

    def test_replicate_no_peers(self) -> None:
        router = FederationRouter(local_cluster_id="solo-cluster")
        router.replicate_prefix_cache("hash-a", {"layers": []})
        assert router.kv_replication.size() == 0

    def test_replicate_with_peer_flushed(self) -> None:
        router = FederationRouter(local_cluster_id="local-cluster")
        router.federation_manager.register_cluster(ClusterInfo(cluster_id="peer-cluster"))
        router.replicate_prefix_cache("hash-a", {"layers": []})
        assert router.kv_replication.size() == 0

    def test_replicate_partial_kv_data(self) -> None:
        router = FederationRouter(local_cluster_id="local-cluster")
        router.replicate_prefix_cache("hash-b", {"key": [0.5], "value": [0.5]})
        assert router.kv_replication.size() == 0

    def test_replicate_preserves_queue_on_flush_failure(self) -> None:
        """When flush fails (bad targets), queue is still drained."""
        router = FederationRouter(local_cluster_id="local-cluster")
        router.federation_manager.register_cluster(ClusterInfo(cluster_id="bad-peer"))
        # flush returns 0 but still dequeues
        router.replicate_prefix_cache("hash-c", {"layers": []})
        assert router.kv_replication.size() == 0


class TestFederationRouterStats:
    """FederationRouter: stats()."""

    def test_stats_structure(self) -> None:
        router = FederationRouter(local_cluster_id="stats-cluster")
        router.report_load("stats-cluster", active=3, gpu_util=0.5)
        s = router.stats()
        assert isinstance(s, dict)
        assert "cluster_loads" in s
        assert "latency_matrix" in s
        assert "federation" in s
        assert "geo_regions" in s
        assert "kv_replication_queue" in s

    def test_stats_kv_queue_size(self) -> None:
        router = FederationRouter()
        router.kv_replication.enqueue("h1", {}, ["t"])
        s = router.stats()
        assert s["kv_replication_queue"] == 1

    def test_stats_geo_regions(self) -> None:
        router = FederationRouter()
        router.dns_resolver.map_region("us-east", "cluster-a")
        router.dns_resolver.map_region("eu-west", "cluster-b")
        s = router.stats()
        assert s["geo_regions"] == {"us-east": "cluster-a", "eu-west": "cluster-b"}

    def test_stats_empty(self) -> None:
        router = FederationRouter()
        s = router.stats()
        assert s["cluster_loads"] == {}
        assert s["kv_replication_queue"] == 0
        assert s["geo_regions"] == {}

    def test_stats_federation_info(self) -> None:
        router = FederationRouter(local_cluster_id="my-cluster")
        router.federation_manager.register_cluster(ClusterInfo(
            cluster_id="peer", region="us-east"
        ))
        s = router.stats()
        fed = s["federation"]
        assert fed["cluster_count"] == 2
        assert "peer" in fed["clusters"]


class TestFederationRouterRoundRobin:
    """FederationRouter internal _next_node_url."""

    def test_single_node(self) -> None:
        router = FederationRouter()
        url = router._next_node_url("c1", ["node-1"])
        assert url == "node-1"

    def test_multiple_nodes(self) -> None:
        router = FederationRouter()
        url = router._next_node_url("c1", ["n1", "n2", "n3"])
        assert url in ["n1", "n2", "n3"]

    def test_advances_on_subsequent_calls(self) -> None:
        router = FederationRouter()
        url1 = router._next_node_url("c1", ["a", "b"])
        url2 = router._next_node_url("c1", ["a", "b"])
        assert url1 in ("a", "b")
        assert url2 in ("a", "b")

    def test_rebuilds_on_node_list_change(self) -> None:
        router = FederationRouter()
        _ = router._next_node_url("c1", ["n1", "n2"])
        url2 = router._next_node_url("c1", ["n1"])
        assert url2 == "n1"

    def test_new_cluster_gets_fresh_cycle(self) -> None:
        router = FederationRouter()
        url1 = router._next_node_url("c1", ["n1"])
        url2 = router._next_node_url("c2", ["n2"])
        assert url1 == "n1"
        assert url2 == "n2"

    def test_accepts_duplicate_nodes_in_list(self) -> None:
        router = FederationRouter()
        url = router._next_node_url("c1", ["n1", "n1"])
        assert url == "n1"
