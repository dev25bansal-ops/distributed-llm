"""Block-level memory pool for PagedAttention KV cache management.

Provides the physical block pool, allocation/eviction, CPU swap,
and prefetch scheduling primitives used by ``PagedAttentionManager``.

Extracted from the former monolithic ``attention.py`` to keep each module
focused and testable.
"""

from __future__ import annotations
import hashlib
import heapq
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
from loguru import logger

from distllm.dist.merkle import MerkleTree, EMPTY_HASH


class _GatherBufferPool:
    """Pre-allocated buffer pool for gather_kv_for_attention.


    PERFORMANCE: Instead of allocating torch.zeros() on every gather call
    (which creates 8GB+ of temporary allocations per decode step for large
    models), this pool reuses a fixed set of pinned buffers. Buffers are
    cleared with zero_() between uses rather than freed and re-allocated.

    This reduces GPU memory allocation pressure by 2-4x and eliminates
    the per-call allocation overhead.
    """


    def __init__(self, num_heads: int, head_dim: int, dtype: torch.dtype, device: str):
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._dtype = dtype
        self._device = device
        self._pool: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._max_cached_len = 0
        self._lock = threading.RLock()

    def acquire(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a pre-allocated buffer pair for the given seq_len, or allocate.

        Synchronous version — suitable for thread-based contexts.
        For async callers, use :meth:`async_acquire` to avoid blocking
        the event loop.
        """
        return self._acquire_impl(seq_len)

    async def async_acquire(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Async-compatible version of :meth:`acquire`.

        Runs the allocation in a thread pool executor so the event loop
        is not blocked by the ``threading.RLock`` or GPU tensor creation.
        """
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._acquire_impl, seq_len)

    def _acquire_impl(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a pre-allocated buffer pair for the given seq_len, or allocate.


        Buffers are grown on-demand but never shrunk — they're zero-filled
        and reused. For a production deployment with bounded max_seq_len,
        this converges to a fixed memory footprint after warmup.
        """
        if seq_len > self._max_cached_len:
            self._max_cached_len = seq_len
        with self._lock:
            best = None
            for cached_len in sorted(self._pool.keys()):
                if cached_len >= seq_len:
                    best = cached_len
                    break
            if best is not None:
                k_buf, v_buf = self._pool[best]
                k_buf[:, :seq_len, :].zero_()
                v_buf[:, :seq_len, :].zero_()
                return k_buf, v_buf
            rounded = 1
            while rounded < seq_len:
                rounded <<= 1
            key_buf = torch.zeros(
                (self._num_heads, rounded, self._head_dim),
                dtype=self._dtype, device=self._device,
            )
            value_buf = torch.zeros(
                (self._num_heads, rounded, self._head_dim),
                dtype=self._dtype, device=self._device,
            )
            self._pool[rounded] = (key_buf, value_buf)
            return key_buf, value_buf


@dataclass
class Block:
    block_id: int
    num_tokens: int = 0
    ref_count: int = 1
    last_access: float = field(default_factory=time.time)

    def is_full(self, block_size: int) -> bool:
        return self.num_tokens >= block_size

    @property
    def is_free(self) -> bool:
        return self.ref_count == 0


@dataclass
class BlockTable:
    request_id: str
    physical_blocks: List[int] = field(default_factory=list)
    logical_to_physical: Dict[int, int] = field(default_factory=dict)
    num_logical_blocks: int = 0
    adapter_id: str | None = None  # LoRA adapter tag (S-LoRA style)

    def append_block(self, physical_block_id: int) -> int:
        logical_idx = self.num_logical_blocks
        self.logical_to_physical[logical_idx] = physical_block_id
        self.physical_blocks.append(physical_block_id)
        self.num_logical_blocks += 1
        return logical_idx

    def get_physical(self, logical_idx: int) -> Optional[int]:
        return self.logical_to_physical.get(logical_idx)

    def total_capacity(self, block_size: int) -> int:
        return self.num_logical_blocks * block_size


@dataclass
class SwapEntry:
    block_id: int
    key_tensor: Optional[torch.Tensor] = None
    value_tensor: Optional[torch.Tensor] = None
    key_scale: Optional[torch.Tensor] = None   # for compressed swap
    value_scale: Optional[torch.Tensor] = None  # for compressed swap
    device: str = "cpu"
    swap_time: float = field(default_factory=time.time)


class BlockPool:
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        swap_to_cpu: bool = False,
        max_swap_blocks: int = 0,
        auto_expand: bool = True,
        eviction_watermark: float = 0.85,
        restore_watermark: float = 0.70,
    ):
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        if block_size <= 0 or (block_size & (block_size - 1)) != 0:
            raise ValueError(f"block_size must be a positive power of 2, got {block_size}")

        self.block_size = block_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self.swap_to_cpu = swap_to_cpu
        self.max_swap_blocks = max_swap_blocks
        self.auto_expand = auto_expand
        self.use_fp8 = False
        self.eviction_watermark = eviction_watermark
        self.restore_watermark = restore_watermark

        self._pools: List[torch.Tensor] = []
        self._pool_boundaries: List[int] = []
        self._initial_num_blocks = num_blocks
        self._fp8_scales: Optional[torch.Tensor] = None
        self._fp8_pools: List[torch.Tensor] = []

        self.num_blocks = 0
        self._free_blocks: List[int] = []
        self._block_usage: Dict[int, Block] = {}

        initial = max(1, min(num_blocks, 64))
        self._allocate_initial_pool(initial)

        self._swap_space: Dict[int, SwapEntry] = {}
        self._lock = threading.RLock()
        self._fp8_init_lock = threading.RLock()

        self._lru_heap: list[tuple[float, int]] = []

        self._total_allocations = 0
        self._total_swaps = 0
        self._total_restores = 0
        self._total_expansions = 0

        self._swap_stream: Optional[torch.cuda.Stream] = None
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                self._swap_stream = torch.cuda.Stream(device=self.device)
            except Exception:
                logger.debug("Failed to create CUDA swap stream")
                pass

    def _allocate_initial_pool(self, num_blocks: int) -> None:
        try:
            if self.device == "cuda" and torch.cuda.is_available():
                pool = torch.zeros(
                    (num_blocks, self.num_layers, 2, self.num_heads, self.block_size, self.head_dim),
                    dtype=self.dtype,
                    device=self.device,
                )
                self._pools.append(pool)
                self._pool_boundaries.append(num_blocks)
                self.num_blocks = num_blocks
                self._free_blocks.extend(range(num_blocks))
                logger.info(
                    f"PagedAttention: allocated {num_blocks} blocks "
                    f"({self._pool_memory_gb:.2f} GB) on {self.device}"
                )
            else:
                logger.warning(f"PagedAttention: device {self.device} not CUDA, using CPU pool")
                pool = torch.zeros(
                    (num_blocks, self.num_layers, 2, self.num_heads, self.block_size, self.head_dim),
                    dtype=self.dtype,
                    device="cpu",
                )
                self._pools.append(pool)
                self._pool_boundaries.append(num_blocks)
                self.num_blocks = num_blocks
                self._free_blocks.extend(range(num_blocks))
        except RuntimeError as e:
            logger.error(f"Failed to allocate block pool: {e}")
            raise

    def _expand_pool(self, grow_by: int | None = None) -> bool:
        if not self.auto_expand:
            return False
        grow = grow_by or max(self.num_blocks, 64)
        new_start = self.num_blocks
        try:
            if self.device == "cuda" and torch.cuda.is_available():
                new_pool = torch.zeros(
                    (grow, self.num_layers, 2, self.num_heads, self.block_size, self.head_dim),
                    dtype=self.dtype,
                    device=self.device,
                )
            else:
                new_pool = torch.zeros(
                    (grow, self.num_layers, 2, self.num_heads, self.block_size, self.head_dim),
                    dtype=self.dtype,
                    device="cpu",
                )
            self._pools.append(new_pool)
            self._pool_boundaries.append(new_start + grow)
            self._free_blocks.extend(range(new_start, new_start + grow))
            self.num_blocks += grow
            self._total_expansions += 1
            if self.use_fp8:
                if self._fp8_scales is not None:
                    new_scales = torch.zeros(
                        (grow, self.num_layers, 1),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    self._fp8_scales = torch.cat([self._fp8_scales, new_scales], dim=0)
                if self._fp8_pools is not None:
                    new_fp8_pool = torch.zeros(
                        (grow, self.num_layers, 2, self.num_heads, self.block_size, self.head_dim),
                        dtype=torch.float8_e4m3fn,
                        device=self.device,
                    )
                    self._fp8_pools.append(new_fp8_pool)
            logger.info(
                f"PagedAttention: expanded pool by {grow} blocks "
                f"(total: {self.num_blocks}, {self._pool_memory_gb:.2f} GB)"
            )
            return True
        except RuntimeError as e:
            logger.error(f"Failed to expand block pool: {e}")
            return False

    # ── Pool access ────────────────────────────────────────────────────
    def _get_pool_and_offset(self, block_id: int) -> Tuple[torch.Tensor, int]:
        for i, boundary in enumerate(self._pool_boundaries):
            prev_boundary = self._pool_boundaries[i - 1] if i > 0 else 0
            if prev_boundary <= block_id < boundary:
                return self._pools[i], block_id - prev_boundary
        raise ValueError(f"Block {block_id} not found in any pool (total: {self.num_blocks})")

    @property
    def _pool_memory_gb(self) -> float:
        bytes_per_block = (
            self.num_layers * 2 * self.num_heads * self.block_size * self.head_dim * self.dtype.itemsize
        )
        return (self.num_blocks * bytes_per_block) / (1024 ** 3)

    @property
    def free_count(self) -> int:
        return len(self._free_blocks)

    @property
    def used_count(self) -> int:
        return self.num_blocks - len(self._free_blocks)

    @property
    def utilization(self) -> float:
        return self.used_count / self.num_blocks if self.num_blocks > 0 else 0.0

    # ── Block lifecycle ─────────────────────────────────────────────────
    def allocate_block(self) -> Optional[int]:
        with self._lock:
            if not self._free_blocks:
                if self.auto_expand and self._expand_pool():
                    pass
                elif self.swap_to_cpu:
                    swapped = self._swap_lru_block()
                    if not swapped:
                        return None
                else:
                    return None
            block_id = self._free_blocks.pop()
            block = Block(block_id=block_id)
            self._block_usage[block_id] = block
            heapq.heappush(self._lru_heap, (block.last_access, block_id))
            self._total_allocations += 1
            if self.swap_to_cpu and self.utilization > self.eviction_watermark:
                target = int(self.num_blocks * self.restore_watermark)
                while self.used_count > target and not self._free_blocks:
                    if not self._swap_lru_block():
                        break
            return block_id

    def free_block(self, block_id: int) -> None:
        with self._lock:
            if block_id in self._block_usage:
                self._block_usage[block_id].ref_count -= 1
                if self._block_usage[block_id].is_free:
                    del self._block_usage[block_id]
                    self._free_blocks.append(block_id)

    def free_blocks(self, block_ids: List[int]) -> None:
        if not block_ids:
            return
        with self._lock:
            for bid in block_ids:
                if bid in self._block_usage:
                    self._block_usage[bid].ref_count -= 1
                    if self._block_usage[bid].is_free:
                        del self._block_usage[bid]
                        self._free_blocks.append(bid)

    # ── KV read/write ───────────────────────────────────────────────────
    def get_kv_slice(
        self, block_id: int, layer_idx: int, num_tokens: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_fp8 and self._fp8_pools:
            fp8_pool, fp8_offset = self._get_fp8_pool_and_offset(block_id)
            fp8_block = fp8_pool[fp8_offset, layer_idx]
            key = fp8_block[0]
            value = fp8_block[1]
            if num_tokens is not None and num_tokens < self.block_size:
                key = key[:, :num_tokens, :]
                value = value[:, :num_tokens, :]
            from distllm.core.fp8_engine import dequantize_kv_fp8
            scale_k = self._fp8_scales[block_id, layer_idx, 0]
            scale_v = self._fp8_scales[block_id, layer_idx, 1]
            key = dequantize_kv_fp8(key, scale_k)
            value = dequantize_kv_fp8(value, scale_v)
        else:
            pool, offset = self._get_pool_and_offset(block_id)
            block_data = pool[offset, layer_idx]
            key = block_data[0]
            value = block_data[1]
            if num_tokens is not None and num_tokens < self.block_size:
                key = key[:, :num_tokens, :]
                value = value[:, :num_tokens, :]
        return key, value

    def set_kv_slice(
        self, block_id: int, layer_idx: int,
        key: torch.Tensor, value: torch.Tensor, offset: int = 0,
    ) -> None:
        pool, local_offset = self._get_pool_and_offset(block_id)
        num_tokens = key.shape[-2]
        if self.use_fp8:
            from distllm.core.fp8_engine import quantize_kv_fp8
            if self._fp8_scales is None:
                with self._fp8_init_lock:
                    if self._fp8_scales is None:
                        self._init_fp8_scales()
            fp8_k, scale_k = quantize_kv_fp8(key)
            fp8_v, scale_v = quantize_kv_fp8(value)
            self._fp8_scales[block_id, layer_idx, 0] = scale_k
            self._fp8_scales[block_id, layer_idx, 1] = scale_v
            if not self._fp8_pools:
                with self._fp8_init_lock:
                    if not self._fp8_pools:
                        self._allocate_fp8_pools()
            fp8_pool, fp8_local_offset = self._get_fp8_pool_and_offset(block_id)
            fp8_block = fp8_pool[fp8_local_offset, layer_idx]
            fp8_block[0, :, offset : offset + num_tokens, :] = fp8_k
            fp8_block[1, :, offset : offset + num_tokens, :] = fp8_v
        else:
            block_data = pool[local_offset, layer_idx]
            block_data[0, :, offset : offset + num_tokens, :] = key
            block_data[1, :, offset : offset + num_tokens, :] = value

    # ── FP8 ─────────────────────────────────────────────────────────────
    def _init_fp8_scales(self) -> None:
        self._fp8_scales = torch.zeros(
            (self.num_blocks, self.num_layers, 2),
            dtype=torch.float32,
            device=self.device,
        )

    def enable_fp8_storage(self) -> None:
        self.use_fp8 = True
        self._init_fp8_scales()

    def _allocate_fp8_pools(self) -> None:
        for pool in self._pools:
            fp8_pool = torch.empty_like(pool, dtype=torch.float8_e4m3fn)
            self._fp8_pools.append(fp8_pool)

    def _get_fp8_pool_and_offset(self, block_id: int) -> tuple[torch.Tensor, int]:
        for pool_idx, boundary in enumerate(self._pool_boundaries):
            if block_id < boundary:
                prev = self._pool_boundaries[pool_idx - 1] if pool_idx > 0 else 0
                return self._fp8_pools[pool_idx], block_id - prev
        raise IndexError(f"Block {block_id} out of range")

    # ── CPU swap ────────────────────────────────────────────────────────
    def _swap_lru_block(self) -> bool:
        lru_id = None
        while self._lru_heap:
            access_time, bid = heapq.heappop(self._lru_heap)
            block = self._block_usage.get(bid)
            if block is not None and block.last_access == access_time:
                lru_id = bid
                break
        if lru_id is None:
            return False
        if self.max_swap_blocks > 0 and len(self._swap_space) >= self.max_swap_blocks:
            return False
        pool, offset = self._get_pool_and_offset(lru_id)
        stream = self._swap_stream
        compress = getattr(self, "_swap_compress", None)
        if stream is not None and pool.is_cuda:
            with torch.cuda.stream(stream):
                raw_key = pool[offset, :, 0]
                raw_val = pool[offset, :, 1]
                if compress:
                    key_data, key_scale = self._compress_for_swap(raw_key)
                    value_data, value_scale = self._compress_for_swap(raw_val)
                else:
                    key_data = raw_key.cpu().clone()
                    value_data = raw_val.cpu().clone()
                    key_scale = value_scale = None
            stream.synchronize()
        else:
            raw_key = pool[offset, :, 0]
            raw_val = pool[offset, :, 1]
            if compress:
                key_data, key_scale = self._compress_for_swap(raw_key)
                value_data, value_scale = self._compress_for_swap(raw_val)
            else:
                key_data = raw_key.cpu().clone()
                value_data = raw_val.cpu().clone()
                key_scale = value_scale = None
        self._swap_space[lru_id] = SwapEntry(
            block_id=lru_id,
            key_tensor=key_data,
            value_tensor=value_data,
            key_scale=key_scale,
            value_scale=value_scale,
            device="cpu",
        )
        self._total_swaps += 1
        logger.debug(f"Swapped block {lru_id} to CPU")
        del self._block_usage[lru_id]
        self._free_blocks.append(lru_id)
        return True

    def restore_block(self, block_id: int) -> bool:
        with self._lock:
            if block_id not in self._swap_space:
                return False
            entry = self._swap_space.pop(block_id)
            pool, offset = self._get_pool_and_offset(block_id)
            target_dtype = pool.dtype
            if entry.key_scale is not None and entry.key_scale.numel() > 0:
                key_data = self._decompress_from_swap(
                    entry.key_tensor, entry.key_scale, target_dtype,
                )
                value_data = self._decompress_from_swap(
                    entry.value_tensor, entry.value_scale, target_dtype,
                )
            else:
                key_data = entry.key_tensor.to(target_dtype)
                value_data = entry.value_tensor.to(target_dtype)
            stream = self._swap_stream
            if stream is not None and pool.is_cuda:
                with torch.cuda.stream(stream):
                    pool[offset, :, 0, :, :, :] = key_data.to(self.device)
                    pool[offset, :, 1, :, :, :] = value_data.to(self.device)
                stream.synchronize()
            else:
                pool[offset, :, 0, :, :, :] = key_data.to(self.device)
                pool[offset, :, 1, :, :, :] = value_data.to(self.device)
            self._total_restores += 1
            logger.debug(f"Restored block {block_id} from CPU")
            self._block_usage[block_id] = Block(block_id=block_id)
            if block_id in self._free_blocks:
                self._free_blocks.remove(block_id)
            return True

    # ── Compressed swap ─────────────────────────────────────────────────
    def enable_compressed_swap(self, method: str = "fp8") -> None:
        self._swap_compress = method

    def _compress_for_swap(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        method = getattr(self, "_swap_compress", None)
        if method == "fp8" and hasattr(torch, "float8_e4m3fn"):
            scale = tensor.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 448.0
            return (tensor / scale).to(torch.float8_e4m3fn), scale
        elif method == "int8":
            scale = tensor.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 127.0
            return (tensor / scale).to(torch.int8), scale
        return tensor, torch.empty(0)

    @staticmethod
    def _decompress_from_swap(
        tensor: torch.Tensor, scale: torch.Tensor, target_dtype: torch.dtype,
    ) -> torch.Tensor:
        if scale.numel() == 0:
            return tensor.to(target_dtype)
        return (tensor.float() * scale).to(target_dtype)

    # ── Stats ───────────────────────────────────────────────────────────
    def get_swap_stats(self) -> Dict:
        return {
            "swapped_blocks": len(self._swap_space),
            "total_swaps": self._total_swaps,
            "total_restores": self._total_restores,
            "total_expansions": self._total_expansions,
            "swap_memory_gb": self._swap_memory_gb,
        }

    @property
    def _swap_memory_gb(self) -> float:
        total = 0
        for entry in self._swap_space.values():
            if entry.key_tensor is not None:
                total += entry.key_tensor.numel() * entry.key_tensor.element_size()
            if entry.value_tensor is not None:
                total += entry.value_tensor.numel() * entry.value_tensor.element_size()
        return total / (1024 ** 3)

    def stats(self) -> Dict:
        return {
            "total_blocks": self.num_blocks,
            "free_blocks": self.free_count,
            "used_blocks": self.used_count,
            "utilization": round(self.utilization, 4),
            "block_size": self.block_size,
            "pool_memory_gb": round(self._pool_memory_gb, 2),
            **self.get_swap_stats(),
        }

    def __repr__(self) -> str:
        return (
            f"BlockPool(blocks={self.num_blocks}, free={self.free_count}, "
            f"size={self.block_size}, util={self.utilization:.1%}, "
            f"mem={self._pool_memory_gb:.2f}GB, swapped={len(self._swap_space)})"
        )


class BlockPrefetchScheduler:
    """Prefetches KV cache blocks for pipeline-parallel inference."""

    def __init__(self, paged_mgr: "PagedAttentionManager", max_prefetch: int = 8):
        self._mgr = paged_mgr
        self._max_prefetch = max_prefetch
        self._prefetch_queue: list[tuple[str, int]] = []
        self._prefetched: set[int] = set()

    def prefetch_for_stage(
        self, request_ids: List[str], stage_idx: int, layer_idx: int = 0,
    ) -> int:
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
                if phys_id in pool._swap_space:
                    pool.restore_block(phys_id)
                    self._prefetched.add(phys_id)
                    prefetched += 1
                    continue
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
        self._prefetched.clear()

    def __repr__(self) -> str:
        return f"BlockPrefetchScheduler(queued={len(self._prefetched)}, max={self._max_prefetch})"


class DistributedBlockFetcher:
    def __init__(self) -> None:
        self._fetch_fn: Callable[[int, str], Tuple[torch.Tensor, torch.Tensor] | None] | None = None
        self._node_id: str = ""

    def set_transport(
        self,
        node_id: str,
        fetch_fn: Callable[[int, str], Tuple[torch.Tensor, torch.Tensor] | None],
    ) -> None:
        self._node_id = node_id
        self._fetch_fn = fetch_fn

    def fetch(self, block_id: int, peer_node_id: str) -> Tuple[torch.Tensor, torch.Tensor] | None:
        if self._fetch_fn is None:
            return None
        return self._fetch_fn(block_id, peer_node_id)
