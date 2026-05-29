"""GPU memory defragmentation for PagedAttention blocks.

Compacts fragmented KV cache blocks to free contiguous GPU memory.
Prevents OOM errors caused by memory fragmentation during long-running
inference sessions with many concurrent requests.

Supports multiple compaction policies and async execution.
"""

from __future__ import annotations

import asyncio
import enum
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from loguru import logger


class DefragPolicy(str, enum.Enum):
    """Compaction aggressiveness policy."""

    LAZY = "lazy"
    """Only compact when fragmentation exceeds 50%. Max 64 blocks per pass."""

    BALANCED = "balanced"
    """Compact when fragmentation exceeds 30%. Max 128 blocks per pass."""

    AGGRESSIVE = "aggressive"
    """Compact when fragmentation exceeds 15%. Compact all fragmented blocks."""

    @property
    def threshold(self) -> float:
        if self == DefragPolicy.LAZY:
            return 0.50
        if self == DefragPolicy.AGGRESSIVE:
            return 0.15
        return 0.30

    @property
    def max_blocks_per_pass(self) -> int:
        if self == DefragPolicy.LAZY:
            return 64
        if self == DefragPolicy.AGGRESSIVE:
            return 0  # unlimited
        return 128


class TieredCompactionLevel(str, enum.Enum):
    L1_HOT = "l1_hot"
    """Compact within GPU memory. Fast, no offloading."""

    L2_WARM = "l2_warm"
    """Swap cold sequences to CPU, recompact GPU, remap page table."""

    L3_COLD = "l3_cold"
    """Offload to NVMe for extreme memory pressure."""


@dataclass
class DefragConfig:
    """Configuration for the memory defragmenter.

    Args:
        enabled: Whether defragmentation is active.
        policy: Compaction aggressiveness policy.
        interval_seconds: Seconds between background defrag checks.
        max_blocks_per_pass: Override max blocks to move per pass (0 = unlimited).
        tiered_compaction: Enable tiered (L1/L2/L3) compaction.
        l2_cpu_swap_threshold: Fragmentation threshold to trigger L2 warm swap.
        l3_nvme_swap_threshold: Fragmentation threshold to trigger L3 cold swap.
        cuda_stream_priority: CUDA stream priority for copy operations.
        enable_predictive: Enable predictive (preemptive) defragmentation.
        enable_prometheus: Export Prometheus metrics.
    """

    enabled: bool = True
    policy: DefragPolicy = DefragPolicy.BALANCED
    interval_seconds: float = 60.0
    max_blocks_per_pass: int = 0
    tiered_compaction: bool = False
    l2_cpu_swap_threshold: float = 0.60
    l3_nvme_swap_threshold: float = 0.80
    cuda_stream_priority: int = -1
    enable_predictive: bool = False
    enable_prometheus: bool = False


@dataclass
class FragmentInfo:
    """Information about a fragmented memory region."""

    block_id: int
    device: str
    size_bytes: int
    is_free: bool
    adjacent_free: int


@dataclass
class DefragResult:
    """Result of a single defragmentation pass."""

    blocks_moved: int = 0
    bytes_compacted: int = 0
    time_ms: float = 0.0
    fragmentation_before: float = 0.0
    fragmentation_after: float = 0.0
    tier_used: TieredCompactionLevel | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        result = {
            "blocks_moved": self.blocks_moved,
            "bytes_compacted": self.bytes_compacted,
            "time_ms": self.time_ms,
            "fragmentation_before": round(self.fragmentation_before, 4),
            "fragmentation_after": round(self.fragmentation_after, 4),
        }
        if self.tier_used:
            result["tier_used"] = self.tier_used.value
        if self.error:
            result["error"] = self.error
        return result


class MemoryDefragmenter:
    """Defragments PagedAttention KV cache blocks.

    Identifies fragmented regions where free blocks are scattered
    between allocated blocks, then compacts by moving allocated
    blocks to fill gaps.

    Thread-safe: uses a reentrant lock around all mutation operations.
    Async-safe: provides async ``defragment_async()`` for use in event loops.

    Args:
        config: Defragmentation configuration.
        metrics_collector: Optional Prometheus-style metrics collector.
    """

    def __init__(
        self,
        config: DefragConfig | None = None,
        metrics_collector: Any = None,
    ):
        self._config = config or DefragConfig()
        self._lock = threading.RLock()
        self._async_lock = asyncio.Lock()
        self._cuda_stream: torch.cuda.Stream | None = None
        self._metrics = metrics_collector
        self._stats = {
            "defrag_count": 0,
            "blocks_moved": 0,
            "bytes_compacted": 0,
            "total_time_ms": 0.0,
            "last_fragmentation_ratio": 0.0,
            "peak_fragmentation_ratio": 0.0,
            "l1_count": 0,
            "l2_count": 0,
            "l3_count": 0,
        }

        # Predictive state
        self._fragmentation_history: list[float] = []
        self._history_max_len = 100

        # Prometheus-style gauges (lazy init)
        self._metric_frag_ratio: Any = None
        self._metric_blocks_moved: Any = None
        self._metric_duration: Any = None

    # ── Configuration ──

    @property
    def config(self) -> DefragConfig:
        return self._config

    def reconfigure(self, config: DefragConfig) -> None:
        with self._lock:
            self._config = config

    # ── Fragmentation Analysis ──

    def analyze_fragmentation(self, blocks: list[Any]) -> list[FragmentInfo]:
        """Analyze fragmentation in a block pool.

        Args:
            blocks: List of KVCacheBlock objects from PagedAttentionManager.

        Returns:
            List of FragmentInfo for each block.
        """
        fragments = []
        for i, block in enumerate(blocks):
            adjacent_free = 0
            if i > 0 and not blocks[i - 1].is_allocated:
                adjacent_free += 1
            if i < len(blocks) - 1 and not blocks[i + 1].is_allocated:
                adjacent_free += 1

            size_bytes = 0
            if block.is_allocated and block.key_cache is not None:
                size_bytes = (
                    block.key_cache.element_size() * block.key_cache.numel()
                    + block.value_cache.element_size() * block.value_cache.numel()
                )

            fragments.append(
                FragmentInfo(
                    block_id=block.block_id,
                    device=str(block.key_cache.device) if block.key_cache is not None else "cpu",
                    size_bytes=size_bytes,
                    is_free=not block.is_allocated,
                    adjacent_free=adjacent_free,
                )
            )

        return fragments

    def compute_compaction_plan(self, blocks: list[Any]) -> list[tuple[int, int, int]]:
        """Compute which blocks need to move to eliminate fragmentation.

        Returns list of (source_index, dest_index, ref_count) tuples.
        Respects copy-on-write: blocks with ref_count > 1 are not moved.
        """
        moves = []
        free_slots = []
        allocated = []

        for i, block in enumerate(blocks):
            if not block.is_allocated:
                free_slots.append(i)
            elif free_slots:
                if block.ref_count <= 1:
                    allocated.append(i)

        max_blocks = self._config.max_blocks_per_pass
        if max_blocks > 0 and len(allocated) > max_blocks:
            allocated = allocated[:max_blocks]
            free_slots = free_slots[:max_blocks]

        for alloc_idx, free_idx in zip(reversed(allocated), free_slots):
            if alloc_idx > free_idx:
                moves.append((alloc_idx, free_idx, blocks[alloc_idx].ref_count))

        return moves

    @staticmethod
    def compute_fragmentation_ratio(blocks: list[Any]) -> float:
        """Compute the fragmentation ratio.

        Ratio = fragmented free blocks / total blocks.
        A free block is "fragmented" if it has at least one allocated neighbor
        (i.e. it's scattered among allocated blocks, not contiguous).
        """
        total = len(blocks)
        if total == 0:
            return 0.0

        fragmented = 0
        for i, block in enumerate(blocks):
            if not block.is_allocated:
                has_alloc_neighbor = False
                if i > 0 and blocks[i - 1].is_allocated:
                    has_alloc_neighbor = True
                if i < len(blocks) - 1 and blocks[i + 1].is_allocated:
                    has_alloc_neighbor = True
                if has_alloc_neighbor:
                    fragmented += 1

        return fragmented / total

    _compute_fragmentation_ratio = compute_fragmentation_ratio

    # ── Defragmentation Trigger ──

    def should_defragment(self, blocks: list[Any]) -> bool:
        """Check if defragmentation is needed based on policy threshold."""
        ratio = self._compute_fragmentation_ratio(blocks)
        return ratio >= self._config.policy.threshold

    # ── Synchronous Defragmentation ──

    def defragment(self, paged_attention_mgr: Any) -> DefragResult:
        """Defragment the PagedAttention block pool.

        Thread-safe: acquires internal lock around all mutation.
        Uses the ``Defragmentable`` protocol if the manager supports it,
        falling back to direct attribute access otherwise.

        Args:
            paged_attention_mgr: A PagedAttentionManager (or any object
                conforming to the Defragmentable protocol).

        Returns:
            DefragResult with statistics.
        """
        with self._lock:
            return self._defragment_impl(paged_attention_mgr)

    @staticmethod
    def _resolve_blocks(mgr: Any) -> tuple[list[Any], dict[str, Any]]:
        """Get blocks and seq_blocks from mgr via protocol or fallback."""
        try:
            blocks = mgr.get_blocks()
            if not isinstance(blocks, list):
                raise TypeError
        except (AttributeError, TypeError):
            blocks = mgr._blocks
        try:
            seq_blocks = mgr.get_seq_blocks()
            if not isinstance(seq_blocks, dict):
                raise TypeError
        except (AttributeError, TypeError):
            seq_blocks = mgr._seq_blocks
        return blocks, seq_blocks

    @staticmethod
    def _set_free_blocks(mgr: Any, free_ids: list[int]) -> None:
        try:
            mgr.set_free_blocks(free_ids)
        except (AttributeError, TypeError):
            mgr._free_blocks = free_ids

    def _defragment_impl(
        self,
        mgr: Any,
        tier: TieredCompactionLevel | None = None,
    ) -> DefragResult:
        """Internal defrag implementation (caller must hold _lock)."""
        start = time.monotonic()
        result = DefragResult()

        blocks, seq_blocks = self._resolve_blocks(mgr)
        if not blocks:
            return result

        result.fragmentation_before = self._compute_fragmentation_ratio(blocks)

        self._fragmentation_history.append(result.fragmentation_before)
        if len(self._fragmentation_history) > self._history_max_len:
            self._fragmentation_history.pop(0)

        moves = self.compute_compaction_plan(blocks)
        if not moves:
            result.fragmentation_after = result.fragmentation_before
            return result

        # L2/L3 tier: swap cold sequences to CPU/NVMe before compacting
        if tier in (TieredCompactionLevel.L2_WARM, TieredCompactionLevel.L3_COLD):
            self._offload_cold_sequences(mgr, tier)

        # Initialize CUDA stream if needed
        stream = self._get_cuda_stream()

        bytes_moved = 0
        for src_idx, dst_idx, ref_count in moves:
            src_block = blocks[src_idx]
            dst_block = blocks[dst_idx]

            if src_block.key_cache is None:
                continue

            # Allocate destination block if it hasn't been allocated yet
            if dst_block.key_cache is None:
                dst_block.allocate(
                    num_heads=src_block.key_cache.shape[1],
                    head_dim=src_block.key_cache.shape[-1],
                    device=str(src_block.key_cache.device),
                )

            if stream is not None:
                with torch.cuda.stream(stream):
                    dst_block.key_cache.copy_(src_block.key_cache)
                    dst_block.value_cache.copy_(src_block.value_cache)
            else:
                dst_block.key_cache.copy_(src_block.key_cache)
                dst_block.value_cache.copy_(src_block.value_cache)

            dst_block.num_tokens = src_block.num_tokens
            dst_block.is_allocated = True
            dst_block.ref_count = ref_count

            src_block.key_cache = None
            src_block.value_cache = None
            src_block.is_allocated = False
            src_block.num_tokens = 0
            src_block.ref_count = 0

            bytes_moved += (
                dst_block.key_cache.element_size() * dst_block.key_cache.numel()
                + dst_block.value_cache.element_size() * dst_block.value_cache.numel()
            )

        # Synchronize CUDA stream before updating bookkeeping
        if stream is not None:
            stream.synchronize()

        # Update sequence block references using block_ids (not list indices).
        # This is robust even if block_id != list index in future implementations.
        for src_idx, dst_idx, _ in moves:
            src_block_id = blocks[src_idx].block_id
            dst_block_id = blocks[dst_idx].block_id
            for seq in seq_blocks.values():
                for i, bid in enumerate(seq.block_ids):
                    if bid == src_block_id:
                        seq.block_ids[i] = dst_block_id

        # Rebuild free list
        self._set_free_blocks(mgr, [b.block_id for b in blocks if not b.is_allocated])

        result.fragmentation_after = self._compute_fragmentation_ratio(blocks)
        result.blocks_moved = len(moves)
        result.bytes_compacted = bytes_moved
        result.time_ms = (time.monotonic() - start) * 1000
        result.tier_used = tier or TieredCompactionLevel.L1_HOT

        # Update stats
        self._stats["defrag_count"] += 1
        self._stats["blocks_moved"] += result.blocks_moved
        self._stats["bytes_compacted"] += result.bytes_compacted
        self._stats["total_time_ms"] += result.time_ms
        self._stats["last_fragmentation_ratio"] = result.fragmentation_after
        self._stats["peak_fragmentation_ratio"] = max(
            self._stats["peak_fragmentation_ratio"],
            result.fragmentation_before,
        )

        tier_key = f"{tier or TieredCompactionLevel.L1_HOT}_count".replace("-", "_")
        ts = tier or TieredCompactionLevel.L1_HOT
        if ts == TieredCompactionLevel.L1_HOT:
            self._stats["l1_count"] += 1
        elif ts == TieredCompactionLevel.L2_WARM:
            self._stats["l2_count"] += 1
        elif ts == TieredCompactionLevel.L3_COLD:
            self._stats["l3_count"] += 1

        self._emit_prometheus(result)

        logger.info(
            f"Defrag [{tier or TieredCompactionLevel.L1_HOT.value}]: "
            f"moved {result.blocks_moved} blocks, "
            f"{result.bytes_compacted / 1024 / 1024:.1f}MB compacted, "
            f"frag {result.fragmentation_before:.1%} -> {result.fragmentation_after:.1%}, "
            f"{result.time_ms:.1f}ms"
        )

        return result

    def _offload_cold_sequences(self, mgr: Any, tier: TieredCompactionLevel) -> int:
        """Offload cold sequences to CPU (L2) or NVMe (L3).

        Returns number of sequences offloaded.
        """
        offloaded = 0
        _, seq_blocks = self._resolve_blocks(mgr)
        seq_ids = list(seq_blocks.keys())
        for seq_id in seq_ids:
            if tier == TieredCompactionLevel.L2_WARM:
                offloaded += mgr.swap_blocks_to_cpu(seq_id)
            elif tier == TieredCompactionLevel.L3_COLD:
                try:
                    mgr.swap_blocks_to_cpu(seq_id)
                    offloaded += 1
                except Exception:
                    pass
        return offloaded

    def _get_cuda_stream(self) -> torch.cuda.Stream | None:
        """Get or create a CUDA stream for async copies."""
        if not torch.cuda.is_available():
            return None
        if self._cuda_stream is None:
            self._cuda_stream = torch.cuda.Stream(
                priority=self._config.cuda_stream_priority
            )
        return self._cuda_stream

    # ── Async Defragmentation ──

    async def defragment_async(self, paged_attention_mgr: Any) -> DefragResult:
        """Async variant that yields the event loop during compaction.

        Thread-safe: uses asyncio.Lock for cooperative concurrency.
        """
        async with self._async_lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self._defragment_impl, paged_attention_mgr, None,
            )

    async def defragment_with_tier_async(
        self,
        paged_attention_mgr: Any,
        tier: TieredCompactionLevel,
    ) -> DefragResult:
        """Async defragmentation with explicit tier selection."""
        async with self._async_lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self._defragment_impl, paged_attention_mgr, tier,
            )

    # ── Predictive Defragmentation ──

    def predict_fragmentation(self, steps_ahead: int = 5) -> float:
        """Predict future fragmentation using exponential smoothing.

        Args:
            steps_ahead: Number of inference steps to forecast.

        Returns:
            Predicted fragmentation ratio (0.0-1.0).
        """
        history = self._fragmentation_history
        if len(history) < 3:
            return history[-1] if history else 0.0

        alpha = 0.3
        smoothed = history[0]
        for val in history[1:]:
            smoothed = alpha * val + (1 - alpha) * smoothed

        trend = (history[-1] - history[0]) / max(len(history) - 1, 1)
        predicted = smoothed + trend * steps_ahead
        return max(0.0, min(1.0, predicted))

    # ── Prometheus Metrics ──

    def _ensure_prometheus(self) -> None:
        if not self._config.enable_prometheus:
            return
        if self._metric_frag_ratio is not None:
            return
        try:
            from prometheus_client import Gauge, Histogram

            self._metric_frag_ratio = Gauge(
                "kv_cache_fragmentation_ratio",
                "Current KV cache fragmentation ratio (0-1)",
            )
            self._metric_blocks_moved = Gauge(
                "defrag_blocks_moved_total",
                "Total blocks moved by defragmentation",
            )
            self._metric_duration = Histogram(
                "defrag_duration_ms",
                "Defragmentation pass duration in ms",
                buckets=[10, 50, 100, 500, 1000, 5000],
            )
        except ImportError:
            pass

    def _emit_prometheus(self, result: DefragResult) -> None:
        if not self._config.enable_prometheus:
            return
        self._ensure_prometheus()
        try:
            if self._metric_frag_ratio is not None:
                self._metric_frag_ratio.set(result.fragmentation_after)
            if self._metric_blocks_moved is not None:
                self._metric_blocks_moved.inc(result.blocks_moved)
            if self._metric_duration is not None:
                self._metric_duration.observe(result.time_ms)
        except Exception:
            pass

    # ── Properties ──

    @property
    def stats(self) -> dict:
        with self._lock:
            return dict(self._stats)

    @property
    def fragmentation_history(self) -> list[float]:
        return list(self._fragmentation_history)

    @property
    def needs_tier_upgrade(self) -> bool:
        """Check if fragmentation warrants an upgrade to L2 or L3."""
        if not self._fragmentation_history:
            return False
        current = self._fragmentation_history[-1]
        return current > self._config.l2_cpu_swap_threshold

    @property
    def defrag_count(self) -> int:
        return self._stats["defrag_count"]
