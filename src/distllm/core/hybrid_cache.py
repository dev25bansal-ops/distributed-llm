"""Hybrid paged+contiguous KV cache.

Short sequences (<1 block) use contiguous allocation for lower
overhead.  Long sequences automatically switch to paged allocation
for O(1) growth and memory efficiency.

Usage::

    from distllm.core.hybrid_cache import HybridKVCache

    cache = HybridKVCache(
        paged_threshold_tokens=16,
        num_layers=32, num_heads=32, head_dim=128,
        block_size=16, num_blocks=1024,
    )
    cache.allocate("req-1", initial_tokens=8)   # contiguous
    cache.append_tokens("req-1", 100)            # switches to paged
"""

from __future__ import annotations

import threading
from typing import Any

import torch
from loguru import logger


class ContiguousKVBuffer:
    """Simple pre-allocated contiguous KV cache for one sequence."""

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_tokens: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        self.max_tokens = max_tokens
        self.num_tokens = 0
        self.keys = torch.zeros(
            (num_layers, num_heads, max_tokens, head_dim),
            dtype=dtype, device=device,
        )
        self.values = torch.zeros(
            (num_layers, num_heads, max_tokens, head_dim),
            dtype=dtype, device=device,
        )

    def append(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor) -> None:
        n = key.shape[-2]
        self.keys[layer_idx, :, self.num_tokens:self.num_tokens + n, :] = key
        self.values[layer_idx, :, self.num_tokens:self.num_tokens + n, :] = value

    def get(self, layer_idx: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.keys[layer_idx, :, :seq_len, :],
            self.values[layer_idx, :, :seq_len, :],
        )

    def free(self) -> None:
        self.keys = None
        self.values = None


class HybridKVCache:
    """Hybrid KV cache that auto-switches between contiguous and paged.

    Args:
        paged_threshold_tokens: Sequences longer than this use paged allocation.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        head_dim: Dimension per head.
        block_size: Block size for paged allocation.
        num_blocks: Total paged blocks in the pool.
        device: Target device.
    """

    def __init__(
        self,
        paged_threshold_tokens: int = 16,
        num_layers: int = 32,
        num_heads: int = 32,
        head_dim: int = 128,
        block_size: int = 16,
        num_blocks: int = 1024,
        device: str = "cuda",
    ):
        self._threshold = paged_threshold_tokens
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._block_size = block_size
        self._device = device

        # Import here to avoid circular dependency
        from distllm.dist.attention import PagedAttentionManager
        self._paged_mgr = PagedAttentionManager(
            num_blocks=num_blocks,
            block_size=block_size,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            device=device,
        )

        self._contiguous: dict[str, ContiguousKVBuffer] = {}
        self._mode: dict[str, str] = {}  # request_id -> "contiguous" | "paged"
        self._seq_lens: dict[str, int] = {}
        self._lock = threading.Lock()

    @property
    def paged_manager(self) -> Any:
        """Access the underlying PagedAttentionManager."""
        return self._paged_mgr

    def allocate(
        self,
        request_id: str,
        initial_tokens: int = 0,
    ) -> str:
        """Allocate KV cache for a request.

        Returns the allocation mode: "contiguous" or "paged".
        """
        with self._lock:
            if initial_tokens <= self._threshold:
                buf = ContiguousKVBuffer(
                    num_layers=self._num_layers,
                    num_heads=self._num_heads,
                    head_dim=self._head_dim,
                    max_tokens=self._threshold + self._block_size,
                    device=self._device,
                )
                self._contiguous[request_id] = buf
                self._mode[request_id] = "contiguous"
                self._seq_lens[request_id] = initial_tokens
                return "contiguous"
            else:
                self._paged_mgr.create_sequence(request_id)
                self._mode[request_id] = "paged"
                self._seq_lens[request_id] = initial_tokens
                return "paged"

    def append_tokens(
        self,
        request_id: str,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Append KV tokens, switching to paged if threshold is exceeded."""
        with self._lock:
            mode = self._mode.get(request_id)
            if mode is None:
                raise KeyError(f"Sequence {request_id} not found")

            n = key.shape[-2]
            new_len = self._seq_lens.get(request_id, 0) + n

            if mode == "contiguous" and new_len > self._threshold:
                # Migrate from contiguous to paged
                self._migrate_to_paged(request_id)

            if self._mode[request_id] == "paged":
                self._paged_mgr.append_layer_kv(request_id, layer_idx, key, value)
            else:
                buf = self._contiguous[request_id]
                buf.append(layer_idx, key, value)

            self._seq_lens[request_id] = new_len

    def get_kv(
        self,
        request_id: str,
        layer_idx: int,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve KV tensors for attention.

        Thread-safe: acquires ``self._lock`` to prevent TOCTOU races
        with concurrent ``_migrate_to_paged`` or ``free`` operations.
        """
        with self._lock:
            mode = self._mode.get(request_id)
            if mode is None:
                raise KeyError(f"Sequence {request_id} not found")

            if mode == "paged":
                return self._paged_mgr.gather_kv_for_attention(request_id, layer_idx, seq_len)
            else:
                buf = self._contiguous[request_id]
                return buf.get(layer_idx, seq_len)

    def free(self, request_id: str) -> None:
        """Free KV cache for a request."""
        with self._lock:
            mode = self._mode.pop(request_id, None)
            if mode == "contiguous":
                buf = self._contiguous.pop(request_id, None)
                if buf is not None:
                    buf.free()
            elif mode == "paged":
                self._paged_mgr.free_sequence(request_id)
            self._seq_lens.pop(request_id, None)

    def _migrate_to_paged(self, request_id: str) -> None:
        """Migrate a contiguous buffer to paged allocation."""
        buf = self._contiguous.pop(request_id)
        self._paged_mgr.create_sequence(request_id)

        # Copy existing KV data from contiguous buffer to paged blocks
        for layer_idx in range(self._num_layers):
            k, v = buf.get(layer_idx, buf.num_tokens)
            if k.numel() > 0:
                self._paged_mgr.append_layer_kv(
                    request_id, layer_idx,
                    k[:, :, :buf.num_tokens, :],
                    v[:, :, :buf.num_tokens, :],
                )

        buf.free()
        self._mode[request_id] = "paged"
        logger.debug(f"Migrated {request_id} from contiguous to paged ({buf.num_tokens} tokens)")

    def stats(self) -> dict[str, Any]:
        contiguous_count = sum(1 for m in self._mode.values() if m == "contiguous")
        paged_count = sum(1 for m in self._mode.values() if m == "paged")
        return {
            "contiguous_sequences": contiguous_count,
            "paged_sequences": paged_count,
            "total_sequences": len(self._mode),
            "threshold_tokens": self._threshold,
            "paged_pool": self._paged_mgr.pool.stats(),
        }

    def __repr__(self) -> str:
        c = sum(1 for m in self._mode.values() if m == "contiguous")
        p = sum(1 for m in self._mode.values() if m == "paged")
        return (
            f"HybridKVCache(contiguous={c}, paged={p}, "
            f"threshold={self._threshold})"
        )
