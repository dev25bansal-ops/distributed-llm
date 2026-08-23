"""Distributed block prefetch and fetch for PagedAttention KV cache.

This module provides two classes:
- ``BlockPrefetchScheduler`` — prefetches KV cache blocks for pipeline-
  parallel inference, issuing requests during the computation window
  so blocks are warm on GPU by the time they are needed.
- ``DistributedBlockFetcher`` — thin wrapper around a pluggable transport
  function that fetches a block from a peer node.

Extracted from ``attention.py`` during a class-level refactor.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from distllm.dist.attention_block_pool import BlockPool

if TYPE_CHECKING:
    from distllm.dist.attention import PagedAttentionManager


class BlockPrefetchScheduler:
    """Prefetches KV cache blocks for pipeline-parallel inference.


    In a pipeline with stages 0..S-1, stage k's KV blocks for the next
    micro-batch are known while the current micro-batch computes on
    stage k-1.  This scheduler issues async prefetch requests during
    that window so blocks are warm on the GPU by the time stage k needs
    them.

    Usage::

        prefetcher = BlockPrefetchScheduler(paged_attention_mgr)
        # During scheduling loop:
        prefetcher.prefetch_for_stage(request_ids, stage_idx)
        # ... compute on previous stage ...
        # By the time stage_idx runs, blocks are already on GPU.
    """


    def __init__(self, paged_mgr: PagedAttentionManager, max_prefetch: int = 8):
        self._mgr = paged_mgr
        self._max_prefetch = max_prefetch
        self._prefetch_queue: list[tuple[str, int]] = []  # (request_id, phys_block_id)
        self._prefetched: set[int] = set()

    def prefetch_for_stage(
        self,
        request_ids: list[str],
        stage_idx: int,
        layer_idx: int = 0,
    ) -> int:
        """Issue prefetch requests for blocks that a stage will need.


        Args:
            request_ids: Request IDs that will run on *stage_idx* next.
            stage_idx: Pipeline stage that will consume these blocks.
            layer_idx: Layer to prefetch (blocks are shared across layers).

        Returns:
            Number of blocks for which prefetch was issued.
        """

        prefetched = 0
        pool = self._mgr.pool

        for req_id in request_ids:
            if prefetched >= self._max_prefetch:
                break
            table = self._mgr._tables.get(req_id)
            if table is None:
                continue

            for phys_id in table.physical_blocks:
                if prefetched >= self._max_prefetch:
                    break
                if phys_id in self._prefetched:
                    continue

                # Restore from CPU swap if needed
                if phys_id in pool._swap_space:
                    pool.restore_block(phys_id)
                    self._prefetched.add(phys_id)
                    prefetched += 1
                    continue

                # Fetch from remote peer if needed
                if phys_id in self._mgr._remote_blocks and phys_id not in pool._block_usage:
                    peer = self._mgr._remote_blocks.get(phys_id)
                    if peer and self._mgr._block_fetcher._fetch_fn is not None:
                        tensors = self._mgr._block_fetcher.fetch(phys_id, peer)
                        if tensors is not None:
                            k_data, v_data = tensors
                            new_id = pool.allocate_block()
                            if new_id is not None:
                                for lid in range(pool.num_layers):
                                    pool.set_kv_slice(new_id, lid, k_data[lid], v_data[lid])
                                idx = table.physical_blocks.index(phys_id)
                                table.physical_blocks[idx] = new_id
                                table.logical_to_physical[idx] = new_id
                                self._mgr._remote_blocks.pop(phys_id, None)
                                self._prefetched.add(new_id)
                                prefetched += 1

        return prefetched

    def clear(self) -> None:
        """Reset prefetch state between iterations."""

        self._prefetched.clear()

    def __repr__(self) -> str:
        return f"BlockPrefetchScheduler(queued={len(self._prefetched)}, max={self._max_prefetch})"


class DistributedBlockFetcher:
    def __init__(self) -> None:
        self._fetch_fn: Callable[[int, str], tuple[torch.Tensor, torch.Tensor] | None] | None = None
        self._node_id: str = ""

    def set_transport(
        self,
        node_id: str,
        fetch_fn: Callable[[int, str], tuple[torch.Tensor, torch.Tensor] | None],
    ) -> None:
        self._node_id = node_id
        self._fetch_fn = fetch_fn

    def fetch(self, block_id: int, peer_node_id: str) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self._fetch_fn is None:
            return None
        return self._fetch_fn(block_id, peer_node_id)
