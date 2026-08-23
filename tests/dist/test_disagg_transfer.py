"""Tests for KV cache transfer between prefill and decode nodes.

Covers ``KVCacheHandle`` and ``KVCacheTransferScheduler``
from ``distllm.dist.disagg.transfer``.
"""

from __future__ import annotations

import pytest

from distllm.dist.disagg.transfer import KVCacheHandle, KVCacheTransferScheduler


class TestKVCacheHandle:
    def test_create_handle(self):
        handle = KVCacheHandle(
            request_id="req1",
            decode_node_id="decode-1",
            prefill_node_id="prefill-1",
            num_prefill_tokens=128,
        )
        assert handle.request_id == "req1"
        assert handle.decode_node_id == "decode-1"
        assert handle.prefill_node_id == "prefill-1"
        assert handle.num_prefill_tokens == 128
        assert handle.created_at > 0

    def test_handle_with_kv_cache_key(self):
        handle = KVCacheHandle(
            request_id="req1", decode_node_id="d1",
            prefill_node_id="p1", num_prefill_tokens=64,
            kv_cache_key="key_abc123",
        )
        assert handle.kv_cache_key == "key_abc123"

    def test_handle_defaults(self):
        handle = KVCacheHandle(
            request_id="req1", decode_node_id="d1",
            prefill_node_id="p1", num_prefill_tokens=256,
        )
        assert handle.kv_cache_key == ""
        assert handle.created_at > 0


import asyncio

class TestKVCacheTransferScheduler:
    @pytest.mark.asyncio
    async def test_transfer_with_fn(self):
        async def mock_transfer(req_id, src, dst):
            return {"status": "ok", "bytes": 1024}

        scheduler = KVCacheTransferScheduler(transfer_fn=mock_transfer)
        handle = KVCacheHandle("req1", "d1", "p1", 128)
        result = await scheduler.transfer(handle, "node1:50051", "node2:50051")
        assert result is True

    @pytest.mark.asyncio
    async def test_transfer_without_fn(self):
        scheduler = KVCacheTransferScheduler()
        handle = KVCacheHandle("req1", "d1", "p1", 128)
        result = await scheduler.transfer(handle, "node1:50051", "node2:50051")
        assert result is False

    @pytest.mark.asyncio
    async def test_transfer_fn_failure_propagates(self):
        async def failing_transfer(req_id, src, dst):
            raise RuntimeError("Network error")

        scheduler = KVCacheTransferScheduler(transfer_fn=failing_transfer)
        handle = KVCacheHandle("req1", "d1", "p1", 128)
        with pytest.raises(RuntimeError, match="Network error"):
            await scheduler.transfer(handle, "node1", "node2")

    @pytest.mark.asyncio
    async def test_concurrent_transfers_tracked(self):
        call_count = 0

        async def slow_transfer(req_id, src, dst):
            nonlocal call_count
            call_count += 1
            return {"status": "ok"}

        scheduler = KVCacheTransferScheduler(transfer_fn=slow_transfer)
        h1 = KVCacheHandle("req1", "d1", "p1", 128)
        h2 = KVCacheHandle("req2", "d2", "p2", 256)

        t1 = scheduler.transfer(h1, "n1", "n2")
        t2 = scheduler.transfer(h2, "n3", "n4")

        results = await asyncio.gather(t1, t2)
        assert all(results)
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_transfer_status_tracking(self):
        async def mock_transfer(req_id, src, dst):
            return {"status": "ok"}

        import asyncio

        scheduler = KVCacheTransferScheduler(transfer_fn=mock_transfer)
        handle = KVCacheHandle("req1", "d1", "p1", 128)
        await scheduler.transfer(handle, "n1", "n2")
        await asyncio.sleep(0.01)
        # In-flight dict should be empty after completion
        assert len(scheduler._transfers_in_flight) == 0
