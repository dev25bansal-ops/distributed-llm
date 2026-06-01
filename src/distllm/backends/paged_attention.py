"""PagedAttention manager for continuous batching with paged KV cache.

Single-node implementation with per-block KVCacheBlock tensors.
For distributed inference with BlockPool, Merkle sync, and FP8 support,
use ``distllm.dist.attention.PagedAttentionManager`` instead.

This module provides:
- ``KVCacheBlock`` / ``SequenceBlocks`` — per-block metadata
- ``PagedAttentionManager`` — block manager for single-node inference

The distributed variant (``dist/attention.py``) adds:
- Contiguous 6D tensor pool (``BlockPool``) with auto-expand
- LRU heap-based eviction with watermark triggers
- FP8 quantized storage
- MerkleTree-based distributed page-table sync
- CUDA stream-accelerated swap I/O
"""

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Defragmentable(Protocol):
    """Protocol for components that support memory defragmentation."""

    def get_blocks(self) -> list[Any]:
        ...

    def get_seq_blocks(self) -> dict[str, Any]:
        ...

    def set_free_blocks(self, free_ids: list[int]) -> None:
        ...

    def acquire_lock(self) -> Any:
        ...

    def release_lock(self) -> None:
        ...

import torch
from loguru import logger


@dataclass
class KVCacheBlock:
    """A single block of KV cache for one layer."""
    block_id: int
    num_tokens: int = 0
    max_tokens: int = 16  # block size (tokens per block)
    # Shape: (2, num_heads, max_tokens, head_dim) for K and V
    key_cache: torch.Tensor | None = None
    value_cache: torch.Tensor | None = None
    is_allocated: bool = False
    ref_count: int = 0

    def allocate(self, num_heads: int, head_dim: int, device: str = "cuda") -> None:
        """Allocate GPU memory for this block."""
        dtype = torch.float16
        self.key_cache = torch.zeros(
            (2, num_heads, self.max_tokens, head_dim),
            dtype=dtype, device=device,
        )
        self.value_cache = torch.zeros(
            (2, num_heads, self.max_tokens, head_dim),
            dtype=dtype, device=device,
        )
        self.is_allocated = True
        self.num_tokens = 0

    def free(self) -> None:
        """Release GPU memory for this block."""
        self.key_cache = None
        self.value_cache = None
        self.is_allocated = False
        self.num_tokens = 0
        self.ref_count = 0

    def __repr__(self) -> str:
        status = "free" if not self.is_allocated else f"{self.num_tokens}/{self.max_tokens}tok,ref={self.ref_count}"
        return f"KVCacheBlock(id={self.block_id}, {status})"


@dataclass
class SequenceBlocks:
    """Block allocation for a single sequence."""
    sequence_id: str
    block_ids: list[int] = field(default_factory=list)
    num_tokens: int = 0


class PagedAttentionManager:
    """Manages paged KV cache blocks for continuous batching.

    Allocates fixed-size blocks on demand, supports copy-on-write
    for beam search, and provides block swap for CPU offloading.

    Args:
        num_blocks: Total number of blocks in the pool.
        block_size: Tokens per block (must be power of 2).
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads per layer.
        head_dim: Dimension per head.
        device: Target device for block allocation.
    """

    def __init__(
        self,
        num_blocks: int = 1024,
        block_size: int = 16,
        num_layers: int = 32,
        num_heads: int = 32,
        head_dim: int = 128,
        device: str = "cuda",
    ):
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        if block_size <= 0 or (block_size & (block_size - 1)) != 0:
            raise ValueError(f"block_size must be a positive power of 2, got {block_size}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")

        self._num_blocks = num_blocks
        self._block_size = block_size
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._device = device
        self._lock = threading.Lock()
        self._numa_node = self._detect_numa_node()

        # Block pool: pre-allocate all blocks
        self._blocks: list[KVCacheBlock] = [
            KVCacheBlock(block_id=i, max_tokens=block_size)
            for i in range(num_blocks)
        ]
        self._free_blocks: list[int] = list(range(num_blocks))
        self._seq_blocks: dict[str, SequenceBlocks] = {}

        # Stats
        self._stats = {
            "allocations": 0,
            "frees": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "peak_blocks_used": 0,
        }

    # ── Defragmentable protocol conformance ──

    @staticmethod
    def _detect_numa_node() -> int:
        """Detect the NUMA node for the current GPU.

        Returns the NUMA node ID or 0 if detection fails.
        Used to hint block allocation locality.
        """
        try:
            import torch
            if torch.cuda.is_available():
                # CUDA_VISIBLE_DEVICES maps GPU to NUMA node
                gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
                # Try reading from sysfs (Linux)
                numa_path = f"/sys/bus/pci/devices/*/numa_node"
                import glob
                for path in glob.glob(numa_path):
                    try:
                        with open(path) as f:
                            node = int(f.read().strip())
                            if node >= 0:
                                return node
                    except (ValueError, OSError):
                        continue
        except Exception:
            pass
        return 0

    def get_blocks(self) -> list[KVCacheBlock]:
        return self._blocks

    def get_seq_blocks(self) -> dict[str, SequenceBlocks]:
        return self._seq_blocks

    def set_free_blocks(self, free_ids: list[int]) -> None:
        self._free_blocks = free_ids

    def acquire_lock(self) -> threading.RLock:
        self._lock.acquire()
        return self._lock

    def release_lock(self) -> None:
        self._lock.release()

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_blocks)

    @property
    def num_used_blocks(self) -> int:
        return self._num_blocks - len(self._free_blocks)

    @property
    def memory_usage_pct(self) -> float:
        return (self.num_used_blocks / max(self._num_blocks, 1)) * 100

    def _find_fragmented_blocks(self, count: int) -> list[int]:
        """Find free blocks that sit between allocated blocks (fragmented).

        Returns up to *count* block IDs from fragmented free regions,
        preferring blocks with allocated neighbors on both sides.
        Falls back to any free blocks if not enough fragmented ones exist.
        """
        allocated_set = set(
            b.block_id for b in self._blocks if b.is_allocated
        )
        free_set = set(self._free_blocks)

        fragmented = []
        other = []
        for bid in self._free_blocks:
            has_left = (bid - 1) in allocated_set
            has_right = (bid + 1) in allocated_set
            if has_left and has_right:
                fragmented.append(bid)
            else:
                other.append(bid)
            if len(fragmented) >= count:
                break

        result = fragmented[:count]
        if len(result) < count:
            result += [b for b in other if b not in result][: count - len(result)]
        return result

    def allocate_sequence(self, sequence_id: str, num_tokens: int) -> list[int]:
        """Allocate blocks for a new sequence.

        Uses fragmentation-aware allocation: prefers blocks from
        fragmented regions to naturally reduce fragmentation.

        Args:
            sequence_id: Unique sequence identifier.
            num_tokens: Number of tokens in the sequence.

        Returns:
            List of allocated block IDs.

        Raises:
            RuntimeError: If not enough free blocks available.
        """
        with self._lock:
            num_blocks_needed = math.ceil(num_tokens / self._block_size)

            if len(self._free_blocks) < num_blocks_needed:
                raise RuntimeError(
                    f"Not enough KV cache blocks: need {num_blocks_needed}, "
                    f"have {len(self._free_blocks)} free"
                )

            # Fragmentation-aware: prefer blocks from fragmented regions
            candidates = self._find_fragmented_blocks(num_blocks_needed)
            remaining_free = [b for b in self._free_blocks if b not in candidates]
            self._free_blocks = remaining_free

            block_ids = []
            for bid in candidates:
                block = self._blocks[bid]
                if not block.is_allocated:
                    block.allocate(self._num_heads, self._head_dim, self._device)
                block.ref_count = 1
                block_ids.append(bid)

            seq_blocks = SequenceBlocks(
                sequence_id=sequence_id,
                block_ids=block_ids,
                num_tokens=num_tokens,
            )
            self._seq_blocks[sequence_id] = seq_blocks

            self._stats["allocations"] += num_blocks_needed
            self._stats["peak_blocks_used"] = max(
                self._stats["peak_blocks_used"], self.num_used_blocks,
            )

            return block_ids

    def free_sequence(self, sequence_id: str) -> None:
        """Free all blocks belonging to a sequence."""
        with self._lock:
            seq = self._seq_blocks.pop(sequence_id, None)
            if seq is None:
                return

            for bid in seq.block_ids:
                block = self._blocks[bid]
                block.ref_count -= 1
                if block.ref_count <= 0:
                    block.free()
                    self._free_blocks.append(bid)
                    self._stats["frees"] += 1

    def _unshare_block(self, bid: int) -> int:
        """If block has ref_count > 1, allocate a fresh copy and return its ID.

        Decrements the original block's ref_count.  The caller must replace
        *bid* in the sequence's block_ids with the returned ID.
        """
        block = self._blocks[bid]
        if block.ref_count <= 1:
            return bid

        if not self._free_blocks:
            raise RuntimeError("No free KV cache blocks available for COW copy")

        new_bid = self._free_blocks.pop()
        new_block = self._blocks[new_bid]
        if not new_block.is_allocated:
            new_block.allocate(self._num_heads, self._head_dim, self._device)

        # Copy existing KV data from shared block
        if block.key_cache is not None:
            new_block.key_cache.copy_(block.key_cache)
            new_block.value_cache.copy_(block.value_cache)
        new_block.num_tokens = block.num_tokens
        new_block.ref_count = 1

        # Release reference on the old shared block
        block.ref_count -= 1

        self._stats["allocations"] += 1
        self._stats["peak_blocks_used"] = max(
            self._stats["peak_blocks_used"], self.num_used_blocks,
        )
        return new_bid

    def append_token(self, sequence_id: str) -> int:
        """Allocate a new block if the current one is full.

        Returns:
            The block ID where the new token should be written.
        """
        with self._lock:
            seq = self._seq_blocks.get(sequence_id)
            if seq is None:
                raise ValueError(f"Unknown sequence: {sequence_id}")

            seq.num_tokens += 1

            # Check if current block has space
            last_idx = len(seq.block_ids) - 1
            last_bid = seq.block_ids[last_idx]
            last_block = self._blocks[last_bid]

            # Unshare if this block is shared with another sequence (COW)
            if last_block.ref_count > 1:
                new_bid = self._unshare_block(last_bid)
                seq.block_ids[last_idx] = new_bid
                last_bid = new_bid
                last_block = self._blocks[new_bid]

            if last_block.num_tokens < last_block.max_tokens:
                last_block.num_tokens += 1
                return last_bid

            # Need a new block
            if not self._free_blocks:
                raise RuntimeError("No free KV cache blocks available")

            new_bid = self._free_blocks.pop()
            new_block = self._blocks[new_bid]
            if not new_block.is_allocated:
                new_block.allocate(self._num_heads, self._head_dim, self._device)
            new_block.ref_count = 1
            new_block.num_tokens = 1
            seq.block_ids.append(new_bid)

            self._stats["allocations"] += 1
            self._stats["peak_blocks_used"] = max(
                self._stats["peak_blocks_used"], self.num_used_blocks,
            )
            return new_bid

    def get_block_table(self, sequence_id: str) -> list[int]:
        """Get the block table (list of block IDs) for a sequence."""
        seq = self._seq_blocks.get(sequence_id)
        if seq is None:
            return []
        return list(seq.block_ids)

    def get_kv_cache(
        self, sequence_id: str, layer_idx: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Get concatenated KV cache tensors for a sequence at a layer.

        Returns:
            (key_cache, value_cache) tensors, or (None, None) if not found.
        """
        seq = self._seq_blocks.get(sequence_id)
        if seq is None:
            return None, None

        keys = []
        values = []
        for bid in seq.block_ids:
            block = self._blocks[bid]
            if block.key_cache is not None:
                keys.append(block.key_cache)
                values.append(block.value_cache)

        if not keys:
            return None, None

        return torch.cat(keys, dim=2), torch.cat(values, dim=2)

    def copy_on_write(self, source_id: str, dest_id: str) -> None:
        """Copy block references for beam search (copy-on-write).

        Increments ref_count on shared blocks instead of copying data.
        """
        with self._lock:
            source = self._seq_blocks.get(source_id)
            if source is None:
                return

            dest = SequenceBlocks(
                sequence_id=dest_id,
                block_ids=list(source.block_ids),
                num_tokens=source.num_tokens,
            )
            self._seq_blocks[dest_id] = dest

            for bid in source.block_ids:
                self._blocks[bid].ref_count += 1

    def swap_blocks_to_cpu(self, sequence_id: str) -> int:
        """Offload a sequence's KV cache blocks to CPU memory.

        Returns:
            Number of blocks swapped.
        """
        with self._lock:
            seq = self._seq_blocks.get(sequence_id)
            if seq is None:
                return 0

            swapped = 0
            for bid in seq.block_ids:
                block = self._blocks[bid]
                if block.is_allocated and block.key_cache is not None:
                    gpu_key = block.key_cache
                    gpu_val = block.value_cache
                    block.key_cache = gpu_key.cpu()
                    block.value_cache = gpu_val.cpu()
                    del gpu_key, gpu_val
                    swapped += 1

        # Note: torch.cuda.empty_cache() removed — it causes 10-100ms stalls
        # in the hot path. GPU memory is reclaimed automatically by PyTorch's
        # caching allocator. Only call empty_cache() during shutdown.

        return swapped

    def swap_blocks_to_gpu(self, sequence_id: str) -> int:
        """Move a sequence's KV cache blocks back to GPU.

        Returns:
            Number of blocks swapped.
        """
        with self._lock:
            seq = self._seq_blocks.get(sequence_id)
            if seq is None:
                return 0

            swapped = 0
            for bid in seq.block_ids:
                block = self._blocks[bid]
                if block.is_allocated and block.key_cache is not None:
                    block.key_cache = block.key_cache.to(self._device)
                    block.value_cache = block.value_cache.to(self._device)
                    swapped += 1

            return swapped

    def get_stats(self) -> dict[str, Any]:
        """Get block manager statistics."""
        return {
            **self._stats,
            "num_blocks": self._num_blocks,
            "block_size": self._block_size,
            "free_blocks": self.num_free_blocks,
            "used_blocks": self.num_used_blocks,
            "memory_usage_pct": round(self.memory_usage_pct, 1),
            "active_sequences": len(self._seq_blocks),
        }

    def reset(self) -> None:
        """Reset all blocks to free state."""
        with self._lock:
            for block in self._blocks:
                block.free()
            self._free_blocks = list(range(self._num_blocks))
            self._seq_blocks.clear()
            self._stats = {
                "allocations": 0,
                "frees": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "peak_blocks_used": 0,
            }

    def __repr__(self) -> str:
        return (
            f"PagedAttentionManager(blocks={self._num_blocks}, "
            f"size={self._block_size}, used={self.num_used_blocks}, "
            f"seqs={len(self._seq_blocks)})"
        )

    # ── Async API for non-blocking serving ─────────────────────────

    async def async_allocate_sequence(self, sequence_id: str, num_tokens: int) -> list[int]:
        """Async wrapper for allocate_sequence — yields to event loop."""
        import asyncio
        return await asyncio.to_thread(self.allocate_sequence, sequence_id, num_tokens)

    async def async_free_sequence(self, sequence_id: str) -> None:
        """Async wrapper for free_sequence — yields to event loop."""
        import asyncio
        return await asyncio.to_thread(self.free_sequence, sequence_id)

    async def async_append_token(self, sequence_id: str) -> int:
        """Async wrapper for append_token — yields to event loop."""
        import asyncio
        return await asyncio.to_thread(self.append_token, sequence_id)
