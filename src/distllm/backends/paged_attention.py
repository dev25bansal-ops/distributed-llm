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

import contextlib
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

    def acquire_lock(self, timeout: float | None = None) -> Any:
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
        # Enhancement #2: sharded block pool. A single global lock serialized
        # every sequence's allocations against every other sequence. 16
        # sequence-keyed RLock shards let different sequences allocate,
        # append, and free concurrently. The free-list itself remains
        # guarded by its own dedicated lock; shards are acquired before
        # free-list ops to keep lock ordering strict:
        #   shard(seqA) -> shard(seqB) (sorted) -> free-list -> block state
        # preventing deadlock in the cross-sequence COW paths.
        # B-02: RLock (not Lock) — leaf readers (num_free_blocks,
        # get_stats) re-acquire it while callers already hold it
        # (e.g. allocate_sequence updates stats after taking the
        # shard+global pair). Lock ordering (global, deadlock-free):
        #   sequence shard(s), sorted by index -> self._lock
        # Every mutator/reader of seq metadata, ref_counts, block
        # tensors, or the free-list must follow that order.
        self._lock = threading.RLock()         # legacy attribute: free-list + global ops
        self._lock_shards: list[threading.RLock] = [threading.RLock() for _ in range(16)]
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
                numa_path = "/sys/bus/pci/devices/*/numa_node"
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
            # CQ-08: best-effort: failure is non-fatal (optional capability / telemetry / cleanup).
            pass
        return 0

    def get_blocks(self) -> list[KVCacheBlock]:
        return self._blocks

    def get_seq_blocks(self) -> dict[str, SequenceBlocks]:
        return self._seq_blocks

    def set_free_blocks(self, free_ids: list[int]) -> None:
        self._free_blocks = free_ids

    def acquire_lock(self, timeout: float | None = None) -> threading.RLock:
        """Take the allocator lock and return it.

        ``timeout`` bounds the wait, in seconds. It is optional so existing
        ``acquire_lock()`` callers keep their blocking behaviour, but a caller
        that must not hang — the concurrent defrag stress test drives alloc and
        free from several threads — can pass one instead of deadlocking.
        Raises ``TimeoutError`` when the lock is not acquired in time.
        """
        if not self._lock.acquire(timeout=-1 if timeout is None else timeout):
            raise TimeoutError(
                f"could not acquire the paged-attention lock within {timeout}s"
            )
        return self._lock

    def release_lock(self) -> None:
        self._lock.release()

    @property
    def block_size(self) -> int:
        return self._block_size

    def _shard_for(self, sequence_id: str) -> threading.RLock:
        """Per-sequence shard lock (16-way sharded pool)."""
        return self._lock_shards[hash(sequence_id) % len(self._lock_shards)]

    def _shards_for(self, *sequence_ids: str) -> list[threading.RLock]:
        """Distinct shard locks for the given ids, in acquisition-safe order.

        Deduplicates (a cross-sequence op touching two sequences that hash
        to the same shard acquires it once — RLock makes re-entry safe
        anyway) and returns them sorted by index so every caller acquires
        in the same global order: no deadlock.
        """
        idxs = {hash(sid) % len(self._lock_shards) for sid in sequence_ids}
        return [self._lock_shards[i] for i in sorted(idxs)]

    # B-01/BK-01: unified acquisition context. Every public method funnels
    # through this helper so shard acquisition order is defined in exactly
    # one place: shards sorted by index → the global free-list lock.
    # The previous code violated this in three places: copy_on_write
    # acquired two shards in caller order, swap_blocks_to_cpu took a shard
    # but no free-list lock, and swap_blocks_to_gpu took the free-list lock
    # but no shard — allowing deadlock and races with concurrent
    # alloc/free/append.
    @contextlib.contextmanager
    def _locked(self, *sequence_ids: str):
        shards = self._shards_for(*sequence_ids)
        with contextlib.ExitStack() as stack:
            for shard in shards:
                stack.enter_context(shard)
            stack.enter_context(self._lock)
            yield

    @contextlib.contextmanager
    def _locked_all_shards(self):
        """Acquire every shard (sorted) then the global lock.

        Used by operations that touch all sequences (reset): without
        the shards they cannot detect/mutate state held by an
        in-flight per-sequence op and could corrupt the free-list.
        """
        with contextlib.ExitStack() as stack:
            for shard in self._lock_shards:
                stack.enter_context(shard)
            stack.enter_context(self._lock)
            yield

    @property
    def num_free_blocks(self) -> int:
        # B-02: leaf reader — takes only the global lock (no shard),
        # matching the tail of the ordering above.
        with self._lock:
            return len(self._free_blocks)

    @property
    def num_used_blocks(self) -> int:
        with self._lock:
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
        if num_tokens < 0:
            raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
        # An empty sequence still needs one block so the first
        # append_token() call has a block to write into (B-03).
        num_blocks_needed = max(1, math.ceil(num_tokens / self._block_size))
        with self._locked(sequence_id):

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
            # Seed per-block num_tokens from the prefill length: full
            # block_size for every block except the last, which takes
            # the remainder (B-03). Without this, block.num_tokens is
            # always 0 after allocation, so append_token() never
            # advances to a new block on the boundary (388-390).
            remaining = num_tokens
            for bid in candidates:
                block = self._blocks[bid]
                if not block.is_allocated:
                    block.allocate(self._num_heads, self._head_dim, self._device)
                block.ref_count = 1
                block.num_tokens = min(self._block_size, remaining)
                remaining -= block.num_tokens
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
        with self._locked(sequence_id):
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
        with self._locked(sequence_id):
            seq = self._seq_blocks.get(sequence_id)
            if seq is None:
                raise ValueError(f"Unknown sequence: {sequence_id}")

            seq.num_tokens += 1

            if not seq.block_ids:
                # Defensive: a sequence with no blocks (e.g. created
                # before the allocate_sequence(0) guard, or via
                # copy_on_write of an empty sequence) must not hit
                # IndexError on the next append (B-03).
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
        with self._locked(sequence_id):
            seq = self._seq_blocks.get(sequence_id)
            if seq is None:
                return []
            # Snapshot copy under the lock — returning live state let
            # readers race with append/free mutations (B-01/BK-01).
            return list(seq.block_ids)

    def get_kv_cache(
        self, sequence_id: str, layer_idx: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Get concatenated KV cache tensors for a sequence at a layer.

        Block storage is single-layer: each ``KVCacheBlock`` holds data
        for exactly one layer (the allocation has no layer dimension),
        so only ``layer_idx=0`` is supported. Any other layer index
        raises ``NotImplementedError`` rather than silently returning
        the wrong layer's data.

        Returns:
            (key_cache, value_cache) tensors, or (None, None) if not found.
        """
        if layer_idx != 0:
            raise NotImplementedError(
                f"Per-layer KV retrieval not supported: blocks store a "
                f"single layer (layer_idx=0 only), got layer_idx={layer_idx}"
            )
        with self._locked(sequence_id):
            seq = self._seq_blocks.get(sequence_id)
            if seq is None:
                return None, None

            # Snapshot block refs under the lock; the tensors themselves
            # are read-only here. Reading seq.block_ids without the lock
            # raced concurrent append/free (B-01/BK-01).
            block_ids = list(seq.block_ids)
            blocks = [self._blocks[bid] for bid in block_ids]

        keys = []
        values = []
        for block in blocks:
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
        # B-01/BK-01: shards acquired in sorted order via _locked — the
        # previous caller-order acquisition deadlocked when two COW calls
        # crossed two sequences hashed to different shards.
        with self._locked(source_id, dest_id):
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
        # B-01/BK-01: also take the global free-list lock (previously
        # shard-only), so swap is ordered vs. concurrent alloc/free/append.
        with self._locked(sequence_id):
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
        # B-01/BK-01: take the sequence shard in addition to the global
        # lock (previously lock-only), matching every other mutator.
        with self._locked(sequence_id):
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
        # B-02: snapshot stats + seq count under the global lock so a
        # concurrent alloc/free cannot tear the accounting.
        with self._lock:
            return {
                **self._stats,
                "num_blocks": self._num_blocks,
                "block_size": self._block_size,
                "free_blocks": len(self._free_blocks),
                "used_blocks": self._num_blocks - len(self._free_blocks),
                "memory_usage_pct": round(
                    (self._num_blocks - len(self._free_blocks))
                    / max(self._num_blocks, 1) * 100, 1,
                ),
                "active_sequences": len(self._seq_blocks),
            }

    def reset(self) -> None:
        """Reset all blocks to free state."""
        # B-02: reset() holds only the global lock while every
        # per-sequence mutator also takes a shard first — a shard
        # holder joins the allocation AFTER the reset and its
        # block then collides with post-reset state (double-used
        # block / lost free-list entry). Acquire all shards in
        # sorted order via _locked_all_shards().
        with self._locked_all_shards():
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
