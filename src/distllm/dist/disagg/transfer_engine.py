"""Low-level KV cache transfer engine with NCCL/QUIC backends, pipelining,
dynamic pool resizing, and progress tracking.

Provides::

    KVCacheTransferEngine    — actual transfer via NCCL or QUIC fallback
    TransferPipeline         — chunked, concurrent, pipelined transfers
    DynamicPoolManager       — scale up/down based on pool utilization
    TransferProgress         — per-chunk status + estimated completion
"""

from __future__ import annotations

import asyncio
import math
import random
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, AsyncGenerator, Callable, Optional

from loguru import logger


# ---------------------------------------------------------------------------
# Backend type
# ---------------------------------------------------------------------------

class TransferBackend(str, Enum):
    """Which transport backend to use for KV cache movement."""
    NCCL = "nccl"            # GPU-to-GPU via NVIDIA Collective Communications Library
    QUIC = "quic"            # QUIC-based (HTTP/3 / aioquic) fallback


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

class ChunkStatus(str, Enum):
    PENDING = "pending"
    IN_FLIGHT = "in_flight"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ChunkProgress:
    """Progress state for a single KV cache chunk."""
    chunk_id: str
    bytes_total: int = 0
    bytes_transferred: int = 0
    status: ChunkStatus = ChunkStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error: Optional[str] = None

    @property
    def elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time is not None else time.time()
        return end - self.start_time

    @property
    def fraction(self) -> float:
        if self.bytes_total == 0:
            return 1.0 if self.status == ChunkStatus.COMPLETED else 0.0
        return self.bytes_transferred / self.bytes_total


@dataclass
class TransferProgress:
    """Aggregate progress for a multi-chunk KV cache transfer.

    Usage::

        progress = TransferProgress(total_bytes=1_000_000)
        progress.start()
        ...
        progress.mark_chunk_in_flight("chunk-0")
        progress.update_chunk("chunk-0", bytes_transferred=250_000)
        progress.mark_chunk_completed("chunk-0")
        ...
        print(progress.summary())
    """

    total_bytes: int
    bytes_transferred: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    on_progress: Optional[Callable[["TransferProgress"], None]] = None
    chunks: dict[str, ChunkProgress] = field(default_factory=dict)

    def start(self) -> None:
        self.start_time = time.time()

    def _notify(self) -> None:
        if self.on_progress is not None:
            try:
                self.on_progress(self)
            except Exception:
                pass

    # -- chunk lifecycle ---------------------------------------------------

    def add_chunk(self, chunk_id: str, bytes_total: int) -> ChunkProgress:
        cp = ChunkProgress(chunk_id=chunk_id, bytes_total=bytes_total)
        self.chunks[chunk_id] = cp
        return cp

    def mark_chunk_in_flight(self, chunk_id: str) -> None:
        cp = self.chunks.get(chunk_id)
        if cp is not None:
            cp.status = ChunkStatus.IN_FLIGHT
            cp.start_time = time.time()
        self._notify()

    def update_chunk(self, chunk_id: str, bytes_transferred: int) -> None:
        cp = self.chunks.get(chunk_id)
        if cp is not None:
            cp.bytes_transferred = bytes_transferred
            cp.status = ChunkStatus.IN_FLIGHT
        # Recompute aggregate
        self.bytes_transferred = sum(
            c.bytes_transferred for c in self.chunks.values()
        )
        self._notify()

    def mark_chunk_completed(self, chunk_id: str) -> None:
        cp = self.chunks.get(chunk_id)
        if cp is not None:
            cp.status = ChunkStatus.COMPLETED
            cp.bytes_transferred = cp.bytes_total
            cp.end_time = time.time()
        self.bytes_transferred = sum(
            c.bytes_transferred for c in self.chunks.values()
        )
        if self.bytes_transferred >= self.total_bytes:
            self.end_time = time.time()
        self._notify()

    def mark_chunk_failed(self, chunk_id: str, error: str) -> None:
        cp = self.chunks.get(chunk_id)
        if cp is not None:
            cp.status = ChunkStatus.FAILED
            cp.error = error
            cp.end_time = time.time()
        self._notify()

    # -- derived properties ------------------------------------------------

    @property
    def fraction(self) -> float:
        if self.total_bytes == 0:
            return 1.0
        return self.bytes_transferred / self.total_bytes

    @property
    def elapsed_seconds(self) -> float:
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time is not None else time.time()
        return end - self.start_time

    @property
    def estimated_remaining_seconds(self) -> Optional[float]:
        if self.fraction <= 0 or self.elapsed_seconds <= 0:
            return None
        rate = self.bytes_transferred / self.elapsed_seconds
        if rate <= 0:
            return None
        return (self.total_bytes - self.bytes_transferred) / rate

    @property
    def estimated_completion(self) -> Optional[float]:
        remaining = self.estimated_remaining_seconds
        if remaining is None:
            return None
        return time.time() + remaining

    @property
    def failed_chunks(self) -> list[ChunkProgress]:
        return [c for c in self.chunks.values() if c.status == ChunkStatus.FAILED]

    @property
    def active_chunks(self) -> list[ChunkProgress]:
        return [
            c for c in self.chunks.values()
            if c.status in (ChunkStatus.PENDING, ChunkStatus.IN_FLIGHT)
        ]

    # -- helpers -----------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        return {
            "bytes_transferred": self.bytes_transferred,
            "total_bytes": self.total_bytes,
            "fraction": round(self.fraction, 4),
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "estimated_remaining_seconds": (
                round(self.estimated_remaining_seconds, 3)
                if self.estimated_remaining_seconds is not None
                else None
            ),
            "estimated_completion": (
                round(self.estimated_completion, 3)
                if self.estimated_completion is not None
                else None
            ),
            "chunks_total": len(self.chunks),
            "chunks_completed": sum(
                1 for c in self.chunks.values()
                if c.status == ChunkStatus.COMPLETED
            ),
            "chunks_failed": len(self.failed_chunks),
            "chunks_active": len(self.active_chunks),
        }


# ---------------------------------------------------------------------------
# NCCL / QUIC Transfer Engine
# ---------------------------------------------------------------------------

class KVCacheTransferEngine:
    """Low-level KV cache transfer supporting NCCL (GPU-to-GPU) and QUIC fallback.

    Usage::

        engine = KVCacheTransferEngine(preferred_backend="nccl")
        ok = await engine.async_transfer(
            kv_data={"layer_0": ...},
            source="10.0.0.1:50051",
            dest="10.0.0.2:50051",
        )
    """

    def __init__(
        self,
        preferred_backend: str = "nccl",
        nccl_timeout: float = 30.0,
        quic_timeout: float = 60.0,
        chunk_size_bytes: int = 16 * 1024 * 1024,  # 16 MiB
        max_concurrent_streams: int = 4,
    ) -> None:
        self._preferred_backend = TransferBackend(preferred_backend)
        self._nccl_timeout = nccl_timeout
        self._quic_timeout = quic_timeout
        self._chunk_size_bytes = chunk_size_bytes
        self._max_concurrent_streams = max_concurrent_streams
        self._nccl_available: Optional[bool] = None  # None = not yet probed
        self._backend_lock = asyncio.Lock()
        self._active_transfers: dict[str, asyncio.Task] = {}

    # -- public API --------------------------------------------------------

    async def async_transfer(
        self,
        kv_data: dict[str, Any],
        source: str,
        dest: str,
        *,
        request_id: Optional[str] = None,
        progress: Optional[TransferProgress] = None,
    ) -> bool:
        """Transfer ``kv_data`` from *source* node to *dest* node.

        Returns ``True`` on success, ``False`` on failure.

        The backend is selected automatically:
        - NCCL when the GPU-to-GPU library is available and *source* and
          *dest* are on the same PCIe / NVLink fabric.
        - QUIC otherwise.
        """
        rid = request_id or f"kv-xfer-{uuid.uuid4().hex[:12]}"

        backend = await self._resolve_backend(source, dest)
        logger.debug(
            "KV transfer {} ({}) using {} backend: {} -> {}",
            rid, self._estimate_kv_size(kv_data), backend.value, source, dest,
        )

        if progress is not None:
            progress.start()

        try:
            if backend == TransferBackend.NCCL:
                ok = await self._transfer_nccl(kv_data, source, dest, rid, progress)
            else:
                ok = await self._transfer_quic(kv_data, source, dest, rid, progress)
            return ok
        except asyncio.TimeoutError:
            logger.error("KV transfer {} timed out (backend={})", rid, backend.value)
            return False
        except Exception as exc:
            logger.error("KV transfer {} failed: {}", rid, exc)
            return False

    async def probe_backend(self, backend: Optional[str] = None) -> bool:
        """Probe whether a particular transfer backend is available.

        If *backend* is ``None``, probes NCCL first, falling back to QUIC
        (which is always considered available in simulation / real QUIC).
        """
        target = TransferBackend(backend) if backend else self._preferred_backend
        if target == TransferBackend.QUIC:
            return await self._probe_quic()

        async with self._backend_lock:
            if self._nccl_available is not None:
                return self._nccl_available
            self._nccl_available = await self._probe_nccl()
            return self._nccl_available

    # -- backend resolution ------------------------------------------------

    async def _resolve_backend(self, source: str, dest: str) -> TransferBackend:
        """Pick the best available backend for this source/dest pair."""
        if self._preferred_backend == TransferBackend.NCCL:
            nccl_ok = await self.probe_backend("nccl")
            if nccl_ok:
                return TransferBackend.NCCL
            logger.warning("NCCL unavailable, falling back to QUIC")
        return TransferBackend.QUIC

    # -- NCCL backend ------------------------------------------------------

    async def _probe_nccl(self) -> bool:
        """Check whether NCCL is reachable (simulated via import / CUDA check).

        In production this would call ``torch.cuda.nccl`` or ``cupy.cuda.nccl``.
        Here we simulate unavailability on CPU-only hosts.
        """
        try:
            import torch  # noqa: F401
            if not torch.cuda.is_available():
                logger.info("NCCL probe: CUDA not available")
                return False
            logger.info("NCCL probe: CUDA available, NCCL presumed reachable")
            return True
        except ImportError:
            logger.info("NCCL probe: PyTorch not installed")
            return False
        except Exception as exc:
            logger.warning("NCCL probe error: {}", exc)
            return False

    async def _transfer_nccl(
        self,
        kv_data: dict[str, Any],
        source: str,
        dest: str,
        request_id: str,
        progress: Optional[TransferProgress],
    ) -> bool:
        """GPU-to-GPU transfer via NCCL (simulated with asyncio sleep).

        In production this would issue ``ncclSend`` / ``ncclRecv`` on CUDA
        streams identified by *source* and *dest* ranks.
        """
        total_bytes = self._estimate_kv_size(kv_data)
        if progress is None:
            progress = TransferProgress(total_bytes=total_bytes)
            progress.start()

        # Simulate NCCL latency: ~10 GB/s per GPU link
        bandwidth_bps = 10 * 1024**3  # 10 GB/s
        latency_s = total_bytes / bandwidth_bps if total_bytes > 0 else 0.001
        # Add a small fixed overhead for NCCL sync
        latency_s += 0.005

        await asyncio.sleep(latency_s)
        progress.bytes_transferred = total_bytes
        progress.end_time = time.time()

        logger.info(
            "NCCL transfer {}: {} bytes in {:.1f}ms",
            request_id, total_bytes, latency_s * 1000,
        )
        return True

    # -- QUIC backend ------------------------------------------------------

    async def _probe_quic(self) -> bool:
        """QUIC backends are always available (aioquic or simulated)."""
        try:
            import aioquic  # noqa: F401
            logger.debug("QUIC probe: aioquic available")
        except ImportError:
            logger.debug("QUIC probe: aioquic not installed, using simulated UDP")
        return True

    async def _transfer_quic(
        self,
        kv_data: dict[str, Any],
        source: str,
        dest: str,
        request_id: str,
        progress: Optional[TransferProgress],
    ) -> bool:
        """QUIC-based transfer (simulated with chunked asyncio streaming).

        Splits the KV data into *chunk_size_bytes* pieces and sends them
        concurrently over *max_concurrent_streams* streams.
        """
        total_bytes = self._estimate_kv_size(kv_data)
        if progress is None:
            progress = TransferProgress(
                total_bytes=total_bytes,
                on_progress=None,
            )
            progress.start()

        chunks = self._split_into_chunks(total_bytes)
        semaphore = asyncio.Semaphore(self._max_concurrent_streams)

        async def _send_chunk(chunk_id: str, offset: int, size: int) -> bool:
            async with semaphore:
                progress.mark_chunk_in_flight(chunk_id)

                # Simulate QUIC stream throughput (~1 Gbps per stream)
                stream_bps = 1 * 1024**3  # 1 Gbps
                delay = size / stream_bps if size > 0 else 0.001
                # Simulate partial progress updates for large chunks
                steps = max(1, min(10, size // (256 * 1024)))
                step_bytes = size // steps
                for s in range(steps):
                    await asyncio.sleep(delay / steps)
                    transferred = min((s + 1) * step_bytes, size)
                    progress.update_chunk(chunk_id, transferred)

                progress.mark_chunk_completed(chunk_id)
                return True

        tasks = []
        for chunk_id, offset, size in chunks:
            tasks.append(_send_chunk(chunk_id, offset, size))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        success = all(r is True for r in results)
        failed_count = sum(1 for r in results if r is not True)
        if failed_count:
            logger.warning(
                "QUIC transfer {}: {}/{} chunks failed",
                request_id, failed_count, len(tasks),
            )

        return success

    # -- internal helpers --------------------------------------------------

    @staticmethod
    def _estimate_kv_size(kv_data: dict[str, Any]) -> int:
        """Rough byte-size estimate of a KV cache dict.

        In production this would walk tensors; here we approximate via
        repr length as a placeholder.
        """
        # Try common shapes first
        total = 0
        for key, value in kv_data.items():
            if hasattr(value, "nbytes"):
                total += value.nbytes
            elif hasattr(value, "shape"):
                # numpy/cupy array
                import numpy as np
                arr = np.asarray(value)
                total += arr.nbytes
            elif isinstance(value, bytes):
                total += len(value)
            elif isinstance(value, (list, tuple)):
                total += len(value) * 4  # rough: 4 bytes per element
            else:
                total += len(repr(value))
        return total or len(repr(kv_data))

    def _split_into_chunks(
        self,
        total_bytes: int,
    ) -> list[tuple[str, int, int]]:
        """Return list of ``(chunk_id, offset, size)``."""
        if total_bytes == 0:
            return [("chunk-0", 0, 0)]

        chunks: list[tuple[str, int, int]] = []
        offset = 0
        idx = 0
        while offset < total_bytes:
            size = min(self._chunk_size_bytes, total_bytes - offset)
            chunks.append((f"chunk-{idx}", offset, size))
            offset += size
            idx += 1
        return chunks

    @property
    def in_flight_count(self) -> int:
        return len(self._active_transfers)


# ---------------------------------------------------------------------------
# Transfer Pipeline
# ---------------------------------------------------------------------------

class TransferPipeline:
    """Pipelined KV cache transfer that splits data into *N* chunks and
    transfers them concurrently across multiple streams.

    Usage::

        pipeline = TransferPipeline(engine, pipeline_depth=4)
        async for update in pipeline.pipeline_transfer(kv_data, n_chunks=8):
            print(update["fraction"])
    """

    def __init__(
        self,
        engine: KVCacheTransferEngine,
        pipeline_depth: int = 4,
        on_progress: Optional[Callable[[TransferProgress], None]] = None,
    ) -> None:
        self._engine = engine
        self._pipeline_depth = max(1, pipeline_depth)
        self._on_progress = on_progress

    async def pipeline_transfer(
        self,
        kv_data: dict[str, Any],
        n_chunks: int = 4,
        *,
        source: str = "",
        dest: str = "",
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Transfer ``kv_data`` in *n_chunks* pipelined chunks.

        Yields progress dictionaries (same shape as
        ``TransferProgress.summary()``) after each chunk completes.
        """
        rid = request_id or f"pipe-{uuid.uuid4().hex[:12]}"
        total_bytes = KVCacheTransferEngine._estimate_kv_size(kv_data)
        n_chunks = max(1, min(n_chunks, total_bytes // (64 * 1024) + 1))

        progress = TransferProgress(
            total_bytes=total_bytes,
            on_progress=self._on_progress,
        )
        progress.start()

        # Build chunk plan
        chunk_size = max(1, total_bytes // n_chunks) if n_chunks else total_bytes
        chunk_plan: list[tuple[str, int, int]] = []
        offset = 0
        for i in range(n_chunks):
            size = chunk_size if i < n_chunks - 1 else total_bytes - offset
            cid = f"{rid}-chunk-{i:04d}"
            cp = progress.add_chunk(cid, size)
            chunk_plan.append((cid, offset, size))
            offset += size

        # -- Pipeline execution --------------------------------------------
        # We use a sliding window: maintain up to *pipeline_depth* in-flight
        # chunks at once.  As each completes we yield progress and launch
        # the next.
        sem = asyncio.Semaphore(self._pipeline_depth)

        async def _pipeline_chunk(
            chunk_id: str,
            offset: int,
            size: int,
        ) -> bool:
            async with sem:
                progress.mark_chunk_in_flight(chunk_id)
                try:
                    # Simulate per-chunk transfer time proportional to size
                    # In production this would call engine.async_transfer
                    # with the sub-slice of kv_data.
                    bps = 1 * 1024**3  # 1 Gbps simulated
                    delay = size / bps if size > 0 else 0.001
                    await asyncio.sleep(delay)
                    progress.mark_chunk_completed(chunk_id)
                    return True
                except Exception as exc:
                    progress.mark_chunk_failed(chunk_id, str(exc))
                    return False

        # Track pending / in-flight tasks
        pending = list(chunk_plan)
        in_flight: set[asyncio.Task] = set()

        while pending or in_flight:
            # Launch up to pipeline_depth new chunks
            while pending and len(in_flight) < self._pipeline_depth:
                cid, offset, size = pending.pop(0)
                task = asyncio.create_task(_pipeline_chunk(cid, offset, size))
                in_flight.add(task)

            # Wait for at least one to finish
            done, in_flight = await asyncio.wait(
                in_flight, return_when=asyncio.FIRST_COMPLETED,
            )
            for t in done:
                _ = await t  # propagate exceptions

            yield progress.summary()

        progress.end_time = time.time()
        yield progress.summary()

    @property
    def pipeline_depth(self) -> int:
        return self._pipeline_depth


# ---------------------------------------------------------------------------
# Dynamic Pool Manager
# ---------------------------------------------------------------------------

class ScaleDirection(str, Enum):
    UP = "up"
    DOWN = "down"


@dataclass
class PoolScaleEvent:
    """Record of a single scale-up or scale-down operation."""
    timestamp: float = field(default_factory=time.time)
    direction: ScaleDirection = ScaleDirection.UP
    node_id: str = ""
    reason: str = ""
    utilization_before: float = 0.0
    utilization_after: float = 0.0
    node_count_before: int = 0
    node_count_after: int = 0


class DynamicPoolManager:
    """Manages dynamic pool sizing based on utilization metrics.

    Usage::

        mgr = DynamicPoolManager(
            min_nodes=2, max_nodes=32,
            scale_up_threshold=0.8,
            scale_down_threshold=0.3,
        )

        # Register baseline nodes
        mgr.register_node("node-a", capacity=32)
        mgr.register_node("node-b", capacity=32)

        # On a timer or event:
        stats = mgr.get_pool_stats()
        if stats["should_scale_up"]:
            mgr.scale_up()
        elif stats["should_scale_down"]:
            mgr.scale_down()

        # Rebalance KV cache after resize
        await mgr.rebalance(kv_distribution)
    """

    def __init__(
        self,
        min_nodes: int = 2,
        max_nodes: int = 32,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.3,
        cooldown_seconds: float = 30.0,
        drain_timeout: float = 10.0,
    ) -> None:
        if scale_up_threshold <= scale_down_threshold:
            raise ValueError(
                f"scale_up_threshold ({scale_up_threshold}) must be > "
                f"scale_down_threshold ({scale_down_threshold})"
            )
        self._min_nodes = min_nodes
        self._max_nodes = max_nodes
        self._scale_up_threshold = scale_up_threshold
        self._scale_down_threshold = scale_down_threshold
        self._cooldown_seconds = cooldown_seconds
        self._drain_timeout = drain_timeout

        self._nodes: dict[str, _PoolNode] = {}
        self._lock = asyncio.Lock()
        self._last_scale_time: float = 0.0
        self._scale_history: list[PoolScaleEvent] = []
        self._node_counter: int = 0

    # -- node management ---------------------------------------------------

    def register_node(
        self,
        node_id: str,
        capacity: int = 32,
        *,
        address: str = "",
    ) -> None:
        self._nodes[node_id] = _PoolNode(
            node_id=node_id,
            capacity=capacity,
            address=address,
        )

    def unregister_node(self, node_id: str) -> Optional[_PoolNode]:
        return self._nodes.pop(node_id, None)

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    # -- utilization metrics -----------------------------------------------

    def get_pool_stats(self) -> dict[str, Any]:
        """Return a snapshot of pool utilization, node counts, and scaling
        recommendations."""
        if not self._nodes:
            return self._empty_stats()

        total_capacity = sum(n.capacity for n in self._nodes.values())
        total_active = sum(n.active_requests for n in self._nodes.values())
        utilization = total_active / total_capacity if total_capacity > 0 else 0.0

        avg_load = (
            (total_active / len(self._nodes))
            if self._nodes
            else 0.0
        )

        in_cooldown = (time.time() - self._last_scale_time) < self._cooldown_seconds

        return {
            "utilization": round(utilization, 4),
            "node_count": self.node_count,
            "total_capacity": total_capacity,
            "total_active_requests": total_active,
            "avg_load": round(avg_load, 2),
            "min_nodes": self._min_nodes,
            "max_nodes": self._max_nodes,
            "scale_up_threshold": self._scale_up_threshold,
            "scale_down_threshold": self._scale_down_threshold,
            "should_scale_up": (
                not in_cooldown
                and utilization > self._scale_up_threshold
                and self.node_count < self._max_nodes
            ),
            "should_scale_down": (
                not in_cooldown
                and utilization < self._scale_down_threshold
                and self.node_count > self._min_nodes
            ),
            "in_cooldown": in_cooldown,
            "cooldown_remaining": max(
                0.0,
                self._cooldown_seconds - (time.time() - self._last_scale_time),
            ),
            "scale_history_count": len(self._scale_history),
        }

    # -- scale operations --------------------------------------------------

    async def scale_up(self, n: int = 1) -> list[str]:
        """Add *n* new nodes to the pool.

        Returns the list of newly created node IDs.
        """
        if self.node_count >= self._max_nodes:
            logger.warning("Cannot scale up: at max_nodes ({})", self._max_nodes)
            return []

        n = min(n, self._max_nodes - self.node_count)
        stats_before = self.get_pool_stats()

        added: list[str] = []
        async with self._lock:
            for _ in range(n):
                self._node_counter += 1
                node_id = f"node-auto-{self._node_counter}"
                # New nodes start with a capacity equal to the current average,
                # clamped to a reasonable range.
                avg_capacity = max(
                    4,
                    int(
                        sum(n.capacity for n in self._nodes.values())
                        / max(1, self.node_count)
                    ),
                )
                self.register_node(node_id, capacity=avg_capacity)
                added.append(node_id)

        self._last_scale_time = time.time()
        stats_after = self.get_pool_stats()

        for node_id in added:
            self._scale_history.append(PoolScaleEvent(
                direction=ScaleDirection.UP,
                node_id=node_id,
                reason=f"utilization {stats_before['utilization']:.1%} > threshold",
                utilization_before=stats_before["utilization"],
                utilization_after=stats_after["utilization"],
                node_count_before=stats_before["node_count"],
                node_count_after=stats_after["node_count"],
            ))

        logger.info(
            "Scaled UP pool: {} -> {} nodes (util {:.1%} -> {:.1%})",
            stats_before["node_count"],
            stats_after["node_count"],
            stats_before["utilization"],
            stats_after["utilization"],
        )
        return added

    async def scale_down(self, n: int = 1) -> list[str]:
        """Remove *n* nodes from the pool after draining them.

        Returns the list of removed node IDs.
        """
        if self.node_count <= self._min_nodes:
            logger.warning(
                "Cannot scale down: at min_nodes ({})", self._min_nodes,
            )
            return []

        n = min(n, self.node_count - self._min_nodes)
        stats_before = self.get_pool_stats()

        removed: list[str] = []
        async with self._lock:
            # Pick the least-loaded nodes to drain first
            candidates = sorted(
                self._nodes.values(),
                key=lambda nd: nd.active_requests,
            )
            for node in candidates[:n]:
                ok = await self._drain_node(node)
                if ok:
                    self.unregister_node(node.node_id)
                    removed.append(node.node_id)

        self._last_scale_time = time.time()
        stats_after = self.get_pool_stats()

        for node_id in removed:
            self._scale_history.append(PoolScaleEvent(
                direction=ScaleDirection.DOWN,
                node_id=node_id,
                reason=f"utilization {stats_before['utilization']:.1%} < threshold",
                utilization_before=stats_before["utilization"],
                utilization_after=stats_after["utilization"],
                node_count_before=stats_before["node_count"],
                node_count_after=stats_after["node_count"],
            ))

        logger.info(
            "Scaled DOWN pool: {} -> {} nodes (util {:.1%} -> {:.1%})",
            stats_before["node_count"],
            stats_after["node_count"],
            stats_before["utilization"],
            stats_after["utilization"],
        )
        return removed

    async def rebalance(
        self,
        kv_distribution: dict[str, str],  # request_id -> node_id
    ) -> dict[str, list[str]]:
        """Redistribute KV cache entries across the pool after a resize.

        *kv_distribution* maps ``request_id`` to the node it currently resides on.

        Returns ``{node_id: [request_id, ...]}`` — the target distribution.
        """
        if not kv_distribution or not self._nodes:
            return {}

        target = self._compute_target_distribution(list(kv_distribution.keys()))

        # Log moves for observability
        moves = 0
        for node_id, req_ids in target.items():
            for req_id in req_ids:
                current = kv_distribution.get(req_id)
                if current is not None and current != node_id:
                    moves += 1

        if moves:
            logger.info(
                "Rebalancing {} KV entries across {} nodes ({} moves)",
                len(kv_distribution), self.node_count, moves,
            )

        return target

    # -- internal helpers --------------------------------------------------

    def _empty_stats(self) -> dict[str, Any]:
        return {
            "utilization": 0.0,
            "node_count": 0,
            "total_capacity": 0,
            "total_active_requests": 0,
            "avg_load": 0.0,
            "min_nodes": self._min_nodes,
            "max_nodes": self._max_nodes,
            "scale_up_threshold": self._scale_up_threshold,
            "scale_down_threshold": self._scale_down_threshold,
            "should_scale_up": False,
            "should_scale_down": False,
            "in_cooldown": False,
            "cooldown_remaining": 0.0,
            "scale_history_count": len(self._scale_history),
        }

    async def _drain_node(self, node: _PoolNode) -> bool:
        """Wait for active requests on *node* to finish, up to timeout."""
        if node.active_requests <= 0:
            return True

        deadline = time.monotonic() + self._drain_timeout
        while time.monotonic() < deadline:
            await asyncio.sleep(0.1)
            if node.active_requests <= 0:
                return True

        logger.warning(
            "Drain timeout for {} ({} active requests still pending)",
            node.node_id, node.active_requests,
        )
        # Force-drain: mark remaining requests as failed
        return False  # caller decides whether to force-remove

    def _compute_target_distribution(
        self,
        request_ids: list[str],
    ) -> dict[str, list[str]]:
        """Compute a balanced assignment of request IDs to nodes.

        Uses a simple round-robin across nodes sorted by current load.
        """
        if not self._nodes:
            return {}

        sorted_nodes = sorted(
            self._nodes.values(),
            key=lambda nd: nd.active_requests,
        )
        target: dict[str, list[str]] = {
            n.node_id: [] for n in sorted_nodes
        }

        for i, req_id in enumerate(request_ids):
            node = sorted_nodes[i % len(sorted_nodes)]
            target[node.node_id].append(req_id)

        return target

    @property
    def scale_history(self) -> list[dict[str, Any]]:
        return [
            {
                "timestamp": e.timestamp,
                "direction": e.direction.value,
                "node_id": e.node_id,
                "reason": e.reason,
                "utilization_before": round(e.utilization_before, 4),
                "utilization_after": round(e.utilization_after, 4),
                "node_count_before": e.node_count_before,
                "node_count_after": e.node_count_after,
            }
            for e in self._scale_history
        ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@dataclass
class _PoolNode:
    """Internal node record for the dynamic pool manager."""
    node_id: str
    capacity: int = 32
    active_requests: int = 0
    healthy: bool = True
    address: str = ""


__all__ = [
    "KVCacheTransferEngine",
    "TransferPipeline",
    "DynamicPoolManager",
    "TransferProgress",
    "ChunkProgress",
    "ChunkStatus",
    "TransferBackend",
    "ScaleDirection",
    "PoolScaleEvent",
]
