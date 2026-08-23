"""Tests for the disaggregated prefill/decode node pools.

Covers ``PrefillPool``, ``DecodePool``, ``NodeRegistration``,
and ``PoolRole`` from ``distllm.dist.disagg.pool``.
"""

from __future__ import annotations

import pytest

from distllm.dist.disagg.pool import DecodePool, NodeRegistration, PoolRole, PrefillPool


class TestPoolRole:
    def test_enum_values(self):
        assert PoolRole.PREFILL.value == "prefill"
        assert PoolRole.DECODE.value == "decode"

    def test_enum_members(self):
        assert set(PoolRole.__members__) == {"PREFILL", "DECODE"}


class TestNodeRegistration:
    def test_defaults(self):
        reg = NodeRegistration(node_id="n1", address="10.0.0.1:50051", role=PoolRole.PREFILL)
        assert reg.node_id == "n1"
        assert reg.address == "10.0.0.1:50051"
        assert reg.role == PoolRole.PREFILL
        assert reg.max_num_seqs == 8
        assert reg.gpu_memory_gb == 0.0
        assert reg.healthy is True
        assert reg.active_requests == 0
        assert reg.last_seen > 0

    def test_custom_values(self):
        reg = NodeRegistration(
            node_id="n2", address="10.0.0.2:50051",
            role=PoolRole.DECODE, max_num_seqs=64,
            gpu_memory_gb=80.0, healthy=False, active_requests=3,
        )
        assert reg.max_num_seqs == 64
        assert reg.gpu_memory_gb == 80.0
        assert reg.healthy is False
        assert reg.active_requests == 3


class TestPrefillPool:
    def test_init(self):
        pool = PrefillPool(max_concurrent_prefills=8)
        assert pool.total_nodes == 0
        assert pool.available_nodes == 0
        assert pool.active_count == 0

    def test_register_node(self):
        pool = PrefillPool()
        pool.register_node("n1", "10.0.0.1:50051", max_num_seqs=32, gpu_memory_gb=40.0)
        assert pool.total_nodes == 1
        assert pool.available_nodes == 1
        assert pool.active_count == 0

    def test_register_multiple_nodes(self):
        pool = PrefillPool()
        pool.register_node("n1", "10.0.0.1:50051")
        pool.register_node("n2", "10.0.0.2:50051")
        pool.register_node("n3", "10.0.0.3:50051")
        assert pool.total_nodes == 3

    def test_register_duplicate_overwrites(self):
        pool = PrefillPool()
        pool.register_node("n1", "10.0.0.1:50051", max_num_seqs=32)
        pool.register_node("n1", "10.0.0.1:50051", max_num_seqs=64)
        assert pool.total_nodes == 1
        node = pool._nodes["n1"]
        assert node.max_num_seqs == 64

    @pytest.mark.asyncio
    async def test_select_node_returns_least_loaded(self):
        pool = PrefillPool()
        pool.register_node("n1", "10.0.0.1:50051", max_num_seqs=32)
        pool.register_node("n2", "10.0.0.2:50051", max_num_seqs=32)
        pool._nodes["n1"].active_requests = 5
        pool._nodes["n2"].active_requests = 2
        node = await pool.select_node()
        assert node is not None
        assert node.node_id == "n2"

    @pytest.mark.asyncio
    async def test_select_node_returns_none_when_no_nodes(self):
        pool = PrefillPool()
        node = await pool.select_node()
        assert node is None

    @pytest.mark.asyncio
    async def test_select_node_skips_unhealthy(self):
        pool = PrefillPool()
        pool.register_node("n1", "10.0.0.1:50051")
        pool._nodes["n1"].healthy = False
        node = await pool.select_node()
        assert node is None

    @pytest.mark.asyncio
    async def test_select_node_skips_full_nodes(self):
        pool = PrefillPool()
        pool.register_node("n1", "10.0.0.1:50051", max_num_seqs=2)
        pool._nodes["n1"].active_requests = 2
        node = await pool.select_node()
        assert node is None

    @pytest.mark.asyncio
    async def test_acquire_success(self):
        pool = PrefillPool(max_concurrent_prefills=2)
        pool.register_node("n1", "10.0.0.1:50051")
        result = await pool.acquire("n1")
        assert result is True
        assert pool._nodes["n1"].active_requests == 1

    @pytest.mark.asyncio
    async def test_acquire_unknown_node(self):
        pool = PrefillPool()
        result = await pool.acquire("nonexistent")
        assert result is False

    def test_release(self):
        pool = PrefillPool()
        pool.register_node("n1", "10.0.0.1:50051")
        pool._nodes["n1"].active_requests = 3
        pool.release("n1")
        assert pool._nodes["n1"].active_requests == 2

    def test_release_bottom_clamp(self):
        pool = PrefillPool()
        pool.register_node("n1", "10.0.0.1:50051")
        pool.release("n1")
        assert pool._nodes["n1"].active_requests == 0

    def test_mark_unhealthy(self):
        pool = PrefillPool()
        pool.register_node("n1", "10.0.0.1:50051")
        pool.mark_unhealthy("n1")
        assert pool._nodes["n1"].healthy is False
        assert pool.available_nodes == 0

    def test_mark_healthy(self):
        pool = PrefillPool()
        pool.register_node("n1", "10.0.0.1:50051")
        pool.mark_unhealthy("n1")
        pool.mark_healthy("n1")
        assert pool._nodes["n1"].healthy is True

    def test_remove_node(self):
        pool = PrefillPool()
        pool.register_node("n1", "10.0.0.1:50051")
        pool.register_node("n2", "10.0.0.2:50051")
        pool.remove_node("n1")
        assert pool.total_nodes == 1
        assert "n1" not in pool._nodes

    def test_available_nodes_counts_only_healthy(self):
        pool = PrefillPool()
        pool.register_node("n1", "10.0.0.1:50051")
        pool.register_node("n2", "10.0.0.2:50051")
        pool._nodes["n1"].healthy = False
        assert pool.available_nodes == 1

    def test_active_count(self):
        pool = PrefillPool()
        pool.register_node("n1", "10.0.0.1:50051")
        pool.register_node("n2", "10.0.0.2:50051")
        pool._nodes["n1"].active_requests = 3
        pool._nodes["n2"].active_requests = 5
        assert pool.active_count == 8


class TestDecodePool:
    def test_init(self):
        pool = DecodePool()
        assert pool.total_nodes == 0
        assert pool.active_count == 0

    def test_register_node(self):
        pool = DecodePool()
        pool.register_node("n1", "10.0.0.1:50051", max_num_seqs=8, gpu_memory_gb=80.0)
        assert pool.total_nodes == 1
        assert pool._nodes["n1"].role == PoolRole.DECODE

    @pytest.mark.asyncio
    async def test_select_node_round_robin(self):
        pool = DecodePool()
        pool.register_node("n1", "10.0.0.1:50051", max_num_seqs=8)
        pool.register_node("n2", "10.0.0.2:50051", max_num_seqs=8)
        first = await pool.select_node()
        second = await pool.select_node()
        # Round-robin should alternate
        assert first is not None
        assert second is not None
        assert first.node_id != second.node_id
        # Third call wraps around
        third = await pool.select_node()
        assert third is not None
        assert third.node_id == first.node_id

    @pytest.mark.asyncio
    async def test_select_node_returns_none_when_empty(self):
        pool = DecodePool()
        node = await pool.select_node()
        assert node is None

    def test_get_node_by_handle_success(self):
        pool = DecodePool()
        pool.register_node("n1", "10.0.0.1:50051")
        node = pool.get_node_by_handle("n1")
        assert node is not None
        assert node.node_id == "n1"

    def test_get_node_by_handle_missing(self):
        pool = DecodePool()
        node = pool.get_node_by_handle("nonexistent")
        assert node is None

    def test_store_and_lookup_kv_cache(self):
        pool = DecodePool()
        pool._kv_store["req1"] = {"kv_data": {}, "node_id": "n1", "stored_at": 100.0}
        result = pool.lookup_kv_cache("req1")
        assert result is not None
        assert result["node_id"] == "n1"

    def test_evict_kv_cache(self):
        pool = DecodePool()
        pool._kv_store["req1"] = {"kv_data": {}, "node_id": "n1", "stored_at": 100.0}
        pool.evict_kv_cache("req1")
        assert pool.lookup_kv_cache("req1") is None

    def test_release_node(self):
        pool = DecodePool()
        pool.register_node("n1", "10.0.0.1:50051")
        pool._nodes["n1"].active_requests = 5
        pool.release_node("n1")
        assert pool._nodes["n1"].active_requests == 4

    def test_release_node_bottom_clamp(self):
        pool = DecodePool()
        pool.register_node("n1", "10.0.0.1:50051")
        pool.release_node("n1")
        assert pool._nodes["n1"].active_requests == 0

    def test_mark_unhealthy_and_healthy(self):
        pool = DecodePool()
        pool.register_node("n1", "10.0.0.1:50051")
        pool.mark_unhealthy("n1")
        assert pool._nodes["n1"].healthy is False
        pool.mark_healthy("n1")
        assert pool._nodes["n1"].healthy is True

    def test_remove_node(self):
        pool = DecodePool()
        pool.register_node("n1", "10.0.0.1:50051")
        pool.remove_node("n1")
        assert pool.total_nodes == 0
