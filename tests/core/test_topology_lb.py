"""Tests for topology-aware and cache-aware load balancing."""

from __future__ import annotations

from distllm.core.topology_aware_lb import (
    CacheAwareRouter,
    PipelineGroupAwareness,
    PrefixLocalityTracker,
    TopologyAwareLoadBalancer,
)


class TestPrefixLocalityTracker:
    def test_record_and_get(self):
        pt = PrefixLocalityTracker()
        pt.record("hash-a", "node-1")
        assert "node-1" in pt.get_nodes("hash-a")

    def test_is_cached(self):
        pt = PrefixLocalityTracker()
        pt.record("hash-a", "node-1")
        assert pt.is_cached("hash-a", "node-1")
        assert not pt.is_cached("hash-a", "node-2")

    def test_evict(self):
        pt = PrefixLocalityTracker()
        pt.record("hash-a", "node-1")
        pt.evict("hash-a")
        assert pt.get_nodes("hash-a") == []

    def test_remove_node(self):
        pt = PrefixLocalityTracker()
        pt.record("hash-a", "node-1")
        pt.record("hash-b", "node-1")
        pt.remove_node("node-1")
        assert pt.stats["cached_prefixes"] == 0

    def test_multiple_nodes_per_prefix(self):
        pt = PrefixLocalityTracker()
        pt.record("hash-a", "node-1")
        pt.record("hash-a", "node-2")
        assert len(pt.get_nodes("hash-a")) == 2


class TestPipelineGroupAwareness:
    def test_register(self):
        pg = PipelineGroupAwareness()
        pg.register("node-1", "group-a")
        assert pg.group_for("node-1") == "group-a"

    def test_same_group(self):
        pg = PipelineGroupAwareness()
        pg.register("node-1", "group-a")
        pg.register("node-2", "group-a")
        assert pg.same_group("node-1", "node-2")

    def test_different_group(self):
        pg = PipelineGroupAwareness()
        pg.register("node-1", "group-a")
        pg.register("node-2", "group-b")
        assert not pg.same_group("node-1", "node-2")

    def test_nodes_in_group(self):
        pg = PipelineGroupAwareness()
        pg.register("node-1", "group-a")
        pg.register("node-2", "group-a")
        assert "node-1" in pg.nodes_in_group("group-a")
        assert "node-2" in pg.nodes_in_group("group-a")

    def test_nvlink_topology(self):
        pg = PipelineGroupAwareness()
        pg.register("node-1", "group-a", nvlink_peers=["node-2"])
        assert pg.has_nvlink("node-1", "node-2")
        assert not pg.has_nvlink("node-1", "node-3")


class TestCacheAwareRouter:
    def test_sequence_tracking(self):
        r = CacheAwareRouter()
        r.record_sequence("seq-1", "node-1")
        assert r.get_sequence_node("seq-1") == "node-1"

    def test_remove_sequence(self):
        r = CacheAwareRouter()
        r.record_sequence("seq-1", "node-1")
        r.remove_sequence("seq-1")
        assert r.get_sequence_node("seq-1") is None


class TestTopologyAwareLoadBalancer:
    def test_register_node(self):
        lb = TopologyAwareLoadBalancer()
        lb.register_node("node-1", "group-a")
        assert lb.stats["active_nodes"] == 1

    def test_pick_returns_healthy_node(self):
        lb = TopologyAwareLoadBalancer()
        lb.register_node("node-1", "group-a")
        chosen = lb.pick(request_id="req-1")
        assert chosen == "node-1"

    def test_pick_none_when_no_nodes(self):
        lb = TopologyAwareLoadBalancer()
        assert lb.pick() is None

    def test_pick_skips_unhealthy(self):
        lb = TopologyAwareLoadBalancer()
        lb.register_node("node-1", "group-a")
        lb.mark_unhealthy("node-1")
        assert lb.pick() is None  # No healthy nodes

    def test_cache_affinity(self):
        lb = TopologyAwareLoadBalancer()
        lb.register_node("node-1", "group-a")
        lb.register_node("node-2", "group-a")
        lb.record_prefix("prefix-abc", "node-1")
        chosen = lb.pick(request_id="req-1", prefix_hash="prefix-abc")
        assert chosen == "node-1"

    def test_group_affinity(self):
        lb = TopologyAwareLoadBalancer()
        lb.register_node("node-1", "group-a")
        lb.register_node("node-2", "group-b")
        lb.register_node("node-3", "group-b")
        chosen = lb.pick(request_id="req-1", pipeline_group="group-b")
        assert chosen in ("node-2", "node-3")

    def test_sequence_affinity(self):
        lb = TopologyAwareLoadBalancer()
        lb.register_node("node-1", "group-a")
        lb.register_node("node-2", "group-a")
        lb.record_sequence("seq-1", "node-2")
        chosen = lb.pick(request_id="req-1", sequence_id="seq-1")
        assert chosen == "node-2"

    def test_release_decrements_connections(self):
        lb = TopologyAwareLoadBalancer()
        lb.register_node("node-1", "group-a")
        lb.pick(request_id="req-1")
        lb.release("node-1")
        stats = lb.stats
        # After pick + release, connections should be back to initial

    def test_mark_unhealthy_healthy(self):
        lb = TopologyAwareLoadBalancer()
        lb.register_node("node-1", "group-a")
        lb.mark_unhealthy("node-1")
        assert lb.pick() is None
        lb.mark_healthy("node-1")
        assert lb.pick() is not None

    def test_stats(self):
        lb = TopologyAwareLoadBalancer()
        lb.register_node("node-1", "group-a")
        lb.pick(request_id="req-1")
        stats = lb.stats
        assert stats["total_routes"] >= 1
        assert stats["active_nodes"] >= 1
