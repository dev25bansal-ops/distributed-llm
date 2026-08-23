"""Tests for distllm.dist.scheduling.batcher module.

Covers the full public API: BatchGroup and LatencyAwareBatcher.
Deterministic -- no time.sleep, no threading, no network, no GPU.
"""

from __future__ import annotations

import time

import pytest

from distllm.dist.scheduling.batcher import BatchGroup, LatencyAwareBatcher


# ── BatchGroup Dataclass ────────────────────────────────────────────────────


class TestBatchGroup:
    """BatchGroup dataclass construction, fields, and edge cases."""

    def test_minimal_constructor(self) -> None:
        group = BatchGroup(group_id="g1", cluster_id="c1")
        assert group.group_id == "g1"
        assert group.cluster_id == "c1"
        assert group.request_ids == []
        assert isinstance(group.created_at, float)
        assert group.created_at > 0

    def test_full_constructor(self) -> None:
        group = BatchGroup(
            group_id="g2",
            cluster_id="c2",
            request_ids=["r1", "r2"],
        )
        assert group.group_id == "g2"
        assert group.cluster_id == "c2"
        assert group.request_ids == ["r1", "r2"]

    def test_created_at_is_timestamp(self) -> None:
        before = time.time()
        group = BatchGroup(group_id="g3", cluster_id="c3")
        after = time.time()
        assert before <= group.created_at <= after

    def test_mutable_request_ids(self) -> None:
        """request_ids is a mutable list that supports append."""
        group = BatchGroup(group_id="g4", cluster_id="c4")
        group.request_ids.append("r1")
        assert group.request_ids == ["r1"]


# ── LatencyAwareBatcher ─────────────────────────────────────────────────────


class TestLatencyAwareBatcherInit:
    """Constructor and cluster map setup."""

    def test_default_init(self) -> None:
        batcher = LatencyAwareBatcher()
        assert batcher.node_to_cluster == {}
        assert batcher._groups == {}
        assert batcher._group_counter == 0

    def test_init_with_cluster_map(self) -> None:
        batcher = LatencyAwareBatcher(node_to_cluster={"n1": "c1"})
        assert batcher.node_to_cluster == {"n1": "c1"}

    def test_set_cluster_map(self) -> None:
        batcher = LatencyAwareBatcher()
        batcher.set_cluster_map({"n1": "c1", "n2": "c2"})
        assert batcher.node_to_cluster == {"n1": "c1", "n2": "c2"}

    def test_set_cluster_map_overwrites(self) -> None:
        batcher = LatencyAwareBatcher(node_to_cluster={"n1": "old"})
        batcher.set_cluster_map({"n1": "new"})
        assert batcher.node_to_cluster == {"n1": "new"}


class TestLatencyAwareBatcherGroupRequests:
    """Core grouping logic."""

    def test_empty_requests(self) -> None:
        batcher = LatencyAwareBatcher()
        groups = batcher.group_requests([])
        assert groups == []
        assert batcher.stats()["pending_groups"] == 0

    def test_single_request(self) -> None:
        batcher = LatencyAwareBatcher()
        groups = batcher.group_requests([{"request_id": "r1", "cluster_id": "c1"}])
        assert len(groups) == 1
        assert len(groups[0].request_ids) == 1
        assert groups[0].cluster_id == "c1"

    def test_requests_grouped_by_cluster(self) -> None:
        batcher = LatencyAwareBatcher()
        reqs = [
            {"request_id": "r1", "cluster_id": "c1"},
            {"request_id": "r2", "cluster_id": "c1"},
            {"request_id": "r3", "cluster_id": "c2"},
        ]
        groups = batcher.group_requests(reqs)
        c1_groups = [g for g in groups if g.cluster_id == "c1"]
        c2_groups = [g for g in groups if g.cluster_id == "c2"]
        assert len(c1_groups) == 1
        assert c1_groups[0].request_ids == ["r1", "r2"]
        assert len(c2_groups) == 1
        assert c2_groups[0].request_ids == ["r3"]

    def test_max_group_size_splitting(self) -> None:
        batcher = LatencyAwareBatcher()
        reqs = [{"request_id": f"r{i}", "cluster_id": "c1"} for i in range(10)]
        groups = batcher.group_requests(reqs, max_group_size=3)
        assert len(groups) == 4  # 10 / 3 = 4 chunks
        assert [len(g.request_ids) for g in groups] == [3, 3, 3, 1]

    def test_max_group_size_exact(self) -> None:
        batcher = LatencyAwareBatcher()
        reqs = [{"request_id": f"r{i}", "cluster_id": "c1"} for i in range(8)]
        groups = batcher.group_requests(reqs, max_group_size=8)
        assert len(groups) == 1
        assert len(groups[0].request_ids) == 8

    def test_max_group_size_one(self) -> None:
        batcher = LatencyAwareBatcher()
        reqs = [{"request_id": f"r{i}", "cluster_id": "c1"} for i in range(3)]
        groups = batcher.group_requests(reqs, max_group_size=1)
        assert len(groups) == 3
        for g in groups:
            assert len(g.request_ids) == 1

    def test_node_id_lookup_when_cluster_missing(self) -> None:
        batcher = LatencyAwareBatcher(node_to_cluster={"n42": "c99"})
        groups = batcher.group_requests([{"request_id": "r1", "node_id": "n42"}])
        assert len(groups) == 1
        assert groups[0].cluster_id == "c99"

    def test_default_cluster_when_no_lookup(self) -> None:
        batcher = LatencyAwareBatcher()
        groups = batcher.group_requests([{"request_id": "r1"}])
        assert len(groups) == 1
        assert groups[0].cluster_id == "default"

    def test_empty_request_id(self) -> None:
        batcher = LatencyAwareBatcher()
        groups = batcher.group_requests([{"cluster_id": "c1"}])
        assert len(groups) == 1
        # request_id defaults to ""
        assert groups[0].request_ids == [""]

    def test_group_ids_are_unique_and_monotonic(self) -> None:
        batcher = LatencyAwareBatcher()
        g1 = batcher.group_requests([{"request_id": "r1", "cluster_id": "c1"}])
        g2 = batcher.group_requests([{"request_id": "r2", "cluster_id": "c1"}])
        assert len(g1) == 1
        assert len(g2) == 1
        assert g1[0].group_id != g2[0].group_id
        # group_counter increments: first call yields batch-1, second batch-2
        assert g1[0].group_id == "batch-1"
        assert g2[0].group_id == "batch-2"


class TestLatencyAwareBatcherGroupLifecycle:
    """Getting, completing, and listing pending groups."""

    def test_get_group_found(self) -> None:
        batcher = LatencyAwareBatcher()
        groups = batcher.group_requests([{"request_id": "r1", "cluster_id": "c1"}])
        gid = groups[0].group_id
        retrieved = batcher.get_group(gid)
        assert retrieved is not None
        assert retrieved.group_id == gid
        assert retrieved.cluster_id == "c1"

    def test_get_group_not_found(self) -> None:
        batcher = LatencyAwareBatcher()
        assert batcher.get_group("nonexistent") is None

    def test_get_group_after_complete(self) -> None:
        batcher = LatencyAwareBatcher()
        groups = batcher.group_requests([{"request_id": "r1", "cluster_id": "c1"}])
        gid = groups[0].group_id
        batcher.complete_group(gid)
        assert batcher.get_group(gid) is None

    def test_complete_group_nonexistent(self) -> None:
        """Completing a nonexistent group should not raise."""
        batcher = LatencyAwareBatcher()
        batcher.complete_group("nonexistent")  # no error

    def test_pending_groups_empty(self) -> None:
        batcher = LatencyAwareBatcher()
        assert batcher.pending_groups() == []

    def test_pending_groups_returns_snapshot(self) -> None:
        batcher = LatencyAwareBatcher()
        batcher.group_requests([{"request_id": "r1", "cluster_id": "c1"}])
        pending = batcher.pending_groups()
        assert len(pending) == 1
        assert pending[0].cluster_id == "c1"

    def test_pending_groups_after_complete(self) -> None:
        batcher = LatencyAwareBatcher()
        groups = batcher.group_requests([{"request_id": "r1", "cluster_id": "c1"}])
        gid = groups[0].group_id
        batcher.complete_group(gid)
        assert batcher.pending_groups() == []


class TestLatencyAwareBatcherPrioritize:
    """Execution-order prioritization logic."""

    def make_group(self, group_id: str, cluster_id: str) -> BatchGroup:
        return BatchGroup(
            group_id=group_id,
            cluster_id=cluster_id,
            request_ids=["r1"],
        )

    def test_all_local_returns_same_order(self) -> None:
        batcher = LatencyAwareBatcher()
        groups = [self.make_group("g1", "default"), self.make_group("g2", "default")]
        result = batcher.prioritize_execution_order(groups)
        assert result == groups

    def test_local_first(self) -> None:
        batcher = LatencyAwareBatcher()
        groups = [
            self.make_group("g1", "remote"),
            self.make_group("g2", "default"),
            self.make_group("g3", "remote2"),
        ]
        result = batcher.prioritize_execution_order(groups)
        assert result[0].cluster_id == "default"
        # remaining order among remotes is stable (sorted by cluster_id tie-break not defined,
        # but they maintain their relative order from the group due to stable sort)
        assert [g.cluster_id for g in result[1:]] == ["remote", "remote2"]

    def test_empty_list(self) -> None:
        batcher = LatencyAwareBatcher()
        assert batcher.prioritize_execution_order([]) == []


class TestLatencyAwareBatcherStats:
    """Stats reporting."""

    def test_stats_empty(self) -> None:
        batcher = LatencyAwareBatcher()
        stats = batcher.stats()
        assert stats == {
            "pending_groups": 0,
            "total_pending_requests": 0,
            "by_cluster": {},
        }

    def test_stats_with_pending(self) -> None:
        batcher = LatencyAwareBatcher()
        batcher.group_requests([
            {"request_id": "r1", "cluster_id": "c1"},
            {"request_id": "r2", "cluster_id": "c1"},
            {"request_id": "r3", "cluster_id": "c2"},
        ])
        stats = batcher.stats()
        assert stats["pending_groups"] == 2
        assert stats["total_pending_requests"] == 3
        assert stats["by_cluster"] == {"c1": 2, "c2": 1}

    def test_stats_after_complete(self) -> None:
        batcher = LatencyAwareBatcher()
        groups = batcher.group_requests([{"request_id": "r1", "cluster_id": "c1"}])
        batcher.complete_group(groups[0].group_id)
        stats = batcher.stats()
        assert stats["pending_groups"] == 0
        assert stats["total_pending_requests"] == 0


class TestLatencyAwareBatcherEdgeCases:
    """Edge cases and boundary conditions."""

    def test_multiple_clusters_same_batch(self) -> None:
        """Requests from different clusters produce separate groups."""
        batcher = LatencyAwareBatcher()
        reqs = [
            {"request_id": "r1", "cluster_id": "c1"},
            {"request_id": "r2", "cluster_id": "c2"},
            {"request_id": "r3", "cluster_id": "c1"},
        ]
        groups = batcher.group_requests(reqs)
        assert len(groups) == 2
        c1 = [g for g in groups if g.cluster_id == "c1"]
        c2 = [g for g in groups if g.cluster_id == "c2"]
        assert c1[0].request_ids == ["r1", "r3"]
        assert c2[0].request_ids == ["r2"]

    def test_prioritize_does_not_mutate_input(self) -> None:
        """prioritize_execution_order returns a new list, does not modify input."""
        batcher = LatencyAwareBatcher()
        groups = [
            BatchGroup(group_id="g1", cluster_id="remote"),
            BatchGroup(group_id="g2", cluster_id="default"),
        ]
        original_order = [g.group_id for g in groups]
        _ = batcher.prioritize_execution_order(groups)
        assert [g.group_id for g in groups] == original_order

    def test_group_requests_respects_counter_outside_lock(self) -> None:
        """Calling group_requests multiple times increments counter."""
        batcher = LatencyAwareBatcher()
        assert batcher._group_counter == 0
        batcher.group_requests([{"request_id": "r1", "cluster_id": "c"}])
        assert batcher._group_counter == 1
        batcher.group_requests([{"request_id": "r2", "cluster_id": "c"}])
        assert batcher._group_counter == 2
