"""Paged KV cache backend for block-based allocation.

Extracted from :mod:`distllm.core.kv_cache`.
"""

from __future__ import annotations

import torch
from loguru import logger


class PagedKVCacheBackend:
    """Paged KV cache backend using block-based allocation.

    Wraps PagedAttentionManager to provide a KVCache-compatible interface
    while using paged memory for O(1) allocation and automatic defragmentation.

    Args:
        paged_mgr: PagedAttentionManager instance (backends or dist version).
        max_blocks_per_request: Per-request block limit (0 = use manager default).
    """

    def __init__(self, paged_mgr: object | None = None, max_blocks_per_request: int = 0):
        self._paged_mgr = paged_mgr
        self._request_id: str | None = None
        self._max_blocks_per_request = max_blocks_per_request
        self._request_blocks: dict[str, int] = {}  # request_id -> block count

    def attach(self, request_id: str) -> None:
        self._request_id = request_id
        self._request_blocks[request_id] = 0
        if self._paged_mgr is not None:
            self._paged_mgr.create_sequence(request_id)

    def append_kv(self, layer_idx: int, new_key: torch.Tensor, new_value: torch.Tensor) -> None:
        if self._paged_mgr is not None and self._request_id is not None:
            if self._max_blocks_per_request > 0:
                current = self._request_blocks.get(self._request_id, 0)
                if current >= self._max_blocks_per_request:
                    raise RuntimeError(
                        f"Request {self._request_id} exceeded block budget "
                        f"({self._max_blocks_per_request} blocks)"
                    )
            allocations, _ = self._paged_mgr.append_layer_kv(
                self._request_id, layer_idx, new_key, new_value,
            )
            if allocations:
                blocks_used = sum(1 for _ in allocations)
                self._request_blocks[self._request_id] = (
                    self._request_blocks.get(self._request_id, 0) + blocks_used
                )

    def get_kv(self, request_id: str, layer_idx: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._paged_mgr is not None:
            return self._paged_mgr.gather_kv_for_attention(request_id, layer_idx, seq_len)
        raise RuntimeError("Paged backend not available")

    def free(self, request_id: str) -> None:
        if self._paged_mgr is not None:
            self._paged_mgr.free_sequence(request_id)
        self._request_blocks.pop(request_id, None)

    @property
    def available(self) -> bool:
        return self._paged_mgr is not None

    def memory_usage(self) -> int:
        if self._paged_mgr is not None:
            pool = self._paged_mgr.pool
            used = pool.used_count * pool.num_layers * 2 * pool.num_heads * pool.block_size * pool.head_dim * pool.dtype.itemsize
            return used
        return 0

    @property
    def pool_utilization(self) -> float:
        if self._paged_mgr is not None:
            return self._paged_mgr.pool.utilization
        return 0.0
