"""Disaggregated Prefill + Decode (split architecture).

Separates prefill (compute-bound, large batches) from decode (memory-bound,
small batches) onto different node pools, connected via KV cache transfer.

Architecture::

    ┌──────────────────┐      KV cache       ┌──────────────────┐
    │  Prefill Pool    │ ◄─── transfer ────► │   Decode Pool    │
    │  (large batches) │     (NCCL/RDMA)     │ (small batches)  │
    │  max_num_seqs=32 │                     │ max_num_seqs=4   │
    └──────────────────┘                     └──────────────────┘

Usage::

    from distllm.dist.disagg import DisaggManager

    mgr = DisaggManager()
    mgr.prefill_pool.register_node("node-a", "10.0.0.1:50051")
    mgr.decode_pool.register_node("node-b", "10.0.0.2:50051")

    handle = await mgr.prefill(input_ids, request_id="req-1")
    output = await mgr.decode(input_ids, handle)
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Callable, Optional

from loguru import logger

from distllm.dist.disagg.pool import (
    PoolRole,
    NodeRegistration,
    PrefillPool,
    DecodePool,
)
from distllm.dist.disagg.transfer import (
    KVCacheHandle,
    KVCacheTransferScheduler,
)


class DisaggManager:
    """Top-level orchestrator for disaggregated prefill/decode.

    Usage::

        mgr = DisaggManager()
        mgr.prefill_pool.register_node("node-a", "10.0.0.1:50051")
        mgr.decode_pool.register_node("node-b", "10.0.0.2:50051")

        async def handle_request(input_ids):
            handle = await mgr.prefill(input_ids, request_id=str(uuid.uuid4()))
            output = await mgr.decode(input_ids, handle)
            return output
    """

    def __init__(self, transfer_fn: Optional[Callable] = None):
        self.prefill_pool = PrefillPool()
        self.decode_pool = DecodePool()
        self.transfer_scheduler = KVCacheTransferScheduler(transfer_fn=transfer_fn)

    async def prefill(
        self,
        input_ids: list[int],
        request_id: str,
    ) -> Optional[KVCacheHandle]:
        prefill_node = await self.prefill_pool.select_node()
        if prefill_node is None:
            logger.warning("No prefill nodes available")
            return None

        decode_node = await self.decode_pool.select_node()
        if decode_node is None:
            self.prefill_pool.release_node(prefill_node.node_id)
            logger.warning("No decode nodes available")
            return None

        handle = KVCacheHandle(
            request_id=request_id,
            decode_node_id=decode_node.node_id,
            prefill_node_id=prefill_node.node_id,
            num_prefill_tokens=len(input_ids),
            kv_cache_key=f"kv:{request_id}",
        )

        transfer_task = asyncio.create_task(
            self.transfer_scheduler.transfer(
                handle, prefill_node.address, decode_node.address,
            )
        )
        self.transfer_scheduler._transfers_in_flight[request_id] = transfer_task

        logger.info(
            f"Disagg prefill: {request_id} ({len(input_ids)} tokens) "
            f"on {prefill_node.node_id} → {decode_node.node_id}"
        )
        return handle

    async def decode(
        self,
        input_ids: list[int],
        handle: KVCacheHandle,
    ) -> Optional[list[int]]:
        transfer_task = self.transfer_scheduler._transfers_in_flight.pop(
            handle.request_id, None
        )
        if transfer_task is not None:
            try:
                transferred = await transfer_task
            except Exception as e:
                logger.error(
                    f"KV transfer for {handle.request_id} raised: {e}"
                )
                transferred = False
            if not transferred:
                # Never continue decode from a KV cache that never arrived.
                logger.error(
                    f"KV transfer for {handle.request_id} did not complete; "
                    "aborting decode"
                )
                self.decode_pool.release_node(handle.decode_node_id)
                self.prefill_pool.release_node(handle.prefill_node_id)
                return None

        decode_node = self.decode_pool._nodes.get(handle.decode_node_id)
        if decode_node is None:
            logger.error(f"Decode node {handle.decode_node_id} not found")
            return None

        await asyncio.sleep(0.01)  # placeholder for actual gRPC call
        output_tokens = input_ids[-1:] + [42] * 10  # placeholder generation

        self.decode_pool.release_node(handle.decode_node_id)
        self.prefill_pool.release_node(handle.prefill_node_id)

        return output_tokens

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "prefill_capacity": self.prefill_pool.available_capacity,
            "decode_capacity": sum(
                max(0, n.max_num_seqs - n.active_requests)
                for n in self.decode_pool._nodes.values()
            ),
            "transfers_in_flight": self.transfer_scheduler.in_flight_count(),
        }


__all__ = [
    "DisaggManager",
    "PrefillPool",
    "DecodePool",
    "KVCacheTransferScheduler",
    "KVCacheHandle",
    "NodeRegistration",
    "PoolRole",
]
