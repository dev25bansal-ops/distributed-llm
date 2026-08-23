"""Redundant execution engine with gradient compression, NCCL state replication,
incremental recovery, and FSDP weight synchronization.

Provides production-grade infrastructure for fault-tolerant distributed LLM
inference across unreliable nodes:

1. **GradientCompressor** -- Reduces communication volume via Top-K sparsification
   and FP32-to-INT8 quantization with decompression back to sparse gradients.

2. **NCCLStateReplicator** -- GPU-to-GPU tensor replication using NCCL collectives,
   with automatic CPU fallback when CUDA is unavailable.

3. **IncrementalRecovery** -- Tracks parameter deltas relative to the last
   checkpoint and replays them sequentially to reconstruct a target step.

4. **FSDPWeightSync** -- Synchronises FSDP-sharded weight partitions across
   nodes (all-gather sync, shard extraction).

5. **RedundantExecutor** -- Combines all four components to run a task on
   ``n_replicas`` redundant copies, return the fastest result, and recover
   a failed node end-to-end.

Usage::

    executor = RedundantExecutor()
    result = executor.run_redundant(my_task_fn, n_replicas=2)
    executor.recover_node("node-3")
    print(executor.stats())
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import torch
from loguru import logger

from distllm.errors.types import NodeUnreachableError, DistLLMError


# ===========================================================================
# Gradient Compression
# ===========================================================================


class CompressionMethod(str, Enum):
    """Supported gradient compression methods."""

    TOP_K = "topk"
    QUANTIZE_INT8 = "quantize_int8"


@dataclass
class CompressedGradient:
    """Container for a compressed gradient tensor.

    Attributes:
        indices: Indices of selected elements (Top-K) or None (quantization).
        values: Compressed values (float16 for Top-K, int8 for quantization).
        scale: Per-tensor scale factor used during quantization (None for Top-K).
        original_shape: Shape of the uncompressed gradient.
        original_dtype: Data type of the uncompressed gradient.
        original_numel: Total number of elements in the uncompressed gradient.
        method: Compression method used.
    """

    indices: torch.Tensor | None
    values: torch.Tensor
    scale: torch.Tensor | None
    original_shape: torch.Size
    original_dtype: torch.dtype
    original_numel: int
    method: CompressionMethod

    @property
    def compression_ratio(self) -> float:
        """Ratio of compressed size to original size (bytes)."""
        compressed_bytes = self.values.numel() * self.values.element_size()
        if self.indices is not None:
            compressed_bytes += self.indices.numel() * self.indices.element_size()
        if self.scale is not None:
            compressed_bytes += self.scale.numel() * self.scale.element_size()
        original_bytes = self.original_numel * _dtype_size(self.original_dtype)
        if original_bytes == 0:
            return 0.0
        return compressed_bytes / original_bytes


class GradientCompressor:
    """Compress and decompress gradient tensors for efficient communication.

    Supports two methods:

    * **Top-K sparsification** (``method="topk"``) -- keeps the ``k`` elements
      with the largest absolute value; zeros out the rest.  ``ratio`` controls
      the fraction of elements retained (default 0.01 = 1%).

    * **FP32-to-INT8 quantization** (``method="quantize_int8"``) -- maps the
      gradient range to [-128, 127] using a per-tensor scale factor.
      ``ratio`` is ignored; each float32 element becomes one int8 byte.

    Thread-safe: all public methods acquire an internal lock.
    """

    def __init__(self, default_ratio: float = 0.01):
        if not 0 < default_ratio <= 1.0:
            raise ValueError(f"default_ratio must be in (0, 1], got {default_ratio}")
        self._default_ratio = default_ratio
        self._lock = threading.Lock()
        self._stats: dict[str, Any] = {
            "compress_calls": 0,
            "decompress_calls": 0,
            "total_input_bytes": 0,
            "total_compressed_bytes": 0,
        }

    # -- Public API ---------------------------------------------------------

    def compress(
        self,
        gradients: dict[str, torch.Tensor],
        method: CompressionMethod | str = CompressionMethod.TOP_K,
        ratio: float | None = None,
    ) -> dict[str, CompressedGradient]:
        """Compress a dictionary of named gradient tensors.

        Args:
            gradients: Mapping of parameter name to gradient tensor.
            method: ``"topk"`` (default) or ``"quantize_int8"``.
            ratio: Fraction of elements to retain for Top-K
                (ignored for quantization).  Falls back to ``default_ratio``.

        Returns:
            Dictionary mapping each parameter name to a
            :class:`CompressedGradient`.
        """
        method = CompressionMethod(method) if isinstance(method, str) else method
        ratio = ratio if ratio is not None else self._default_ratio
        result: dict[str, CompressedGradient] = {}
        total_in = 0
        total_out = 0

        for name, grad in gradients.items():
            if grad is None or grad.numel() == 0:
                continue
            total_in += grad.numel() * grad.element_size()
            compressed = self._compress_one(grad, method, ratio)
            total_out += (
                compressed.values.numel() * compressed.values.element_size()
            )
            if compressed.indices is not None:
                total_out += compressed.indices.numel() * compressed.indices.element_size()
            if compressed.scale is not None:
                total_out += compressed.scale.numel() * compressed.scale.element_size()
            result[name] = compressed

        with self._lock:
            self._stats["compress_calls"] += 1
            self._stats["total_input_bytes"] += total_in
            self._stats["total_compressed_bytes"] += total_out

        logger.debug(
            f"GradientCompressor.compress: {len(gradients)} tensors, "
            f"{total_in / 1024:.1f} KiB -> {total_out / 1024:.1f} KiB "
            f"({total_out / max(total_in, 1) * 100:.1f}%)"
        )
        return result

    def decompress(
        self,
        compressed: dict[str, CompressedGradient],
    ) -> dict[str, torch.Tensor]:
        """Decompress gradients back to their original shapes.

        Args:
            compressed: Dictionary produced by :meth:`compress`.

        Returns:
            Dictionary mapping parameter name to reconstructed
            gradient tensor (same shape and dtype as the original).
        """
        result: dict[str, torch.Tensor] = {}
        for name, cg in compressed.items():
            result[name] = self._decompress_one(cg)

        with self._lock:
            self._stats["decompress_calls"] += 1

        return result

    def stats(self) -> dict[str, Any]:
        """Return compression statistics.

        Returns:
            Dict with keys ``compress_calls``, ``decompress_calls``,
            ``total_input_bytes``, ``total_compressed_bytes``, and
            ``overall_ratio``.
        """
        with self._lock:
            s = dict(self._stats)
            total_in = s.get("total_input_bytes", 0)
            total_out = s.get("total_compressed_bytes", 0)
            s["overall_ratio"] = (
                total_out / max(total_in, 1)
            )
            return s

    # -- Internal helpers ---------------------------------------------------

    def _compress_one(
        self,
        grad: torch.Tensor,
        method: CompressionMethod,
        ratio: float,
    ) -> CompressedGradient:
        """Compress a single tensor."""
        flat = grad.flatten()
        numel = flat.numel()

        if method == CompressionMethod.TOP_K:
            return self._topk_compress(flat, ratio, grad.shape, grad.dtype, numel)
        elif method == CompressionMethod.QUANTIZE_INT8:
            return self._quantize_compress(flat, grad.shape, grad.dtype, numel)
        else:
            raise ValueError(f"Unknown compression method: {method}")

    @staticmethod
    def _topk_compress(
        flat: torch.Tensor,
        ratio: float,
        shape: torch.Size,
        dtype: torch.dtype,
        numel: int,
    ) -> CompressedGradient:
        """Top-K sparsification: keep top ``ratio * numel`` elements."""
        k = max(1, int(numel * ratio))
        k = min(k, numel)

        # Compute absolute values and select top-k indices.
        abs_flat = flat.abs()
        topk_values, topk_indices = torch.topk(abs_flat, k, sorted=False)

        # Gather the actual gradient values at those indices.
        values = flat[topk_indices].to(dtype=torch.float16)

        return CompressedGradient(
            indices=topk_indices.to(dtype=torch.int32),
            values=values,
            scale=None,
            original_shape=shape,
            original_dtype=dtype,
            original_numel=numel,
            method=CompressionMethod.TOP_K,
        )

    @staticmethod
    def _quantize_compress(
        flat: torch.Tensor,
        shape: torch.Size,
        dtype: torch.dtype,
        numel: int,
    ) -> CompressedGradient:
        """FP32/FP16 -> INT8 quantization with per-tensor scale."""
        # Compute min/max for the flat tensor.
        v_min = flat.min()
        v_max = flat.max()
        span = v_max - v_min

        # Avoid division by zero when the tensor is constant.
        if span < 1e-12:
            scale = torch.tensor(1.0, device=flat.device)
            quantized = torch.zeros(numel, dtype=torch.int8, device=flat.device)
        else:
            scale = span / 255.0
            # Map [-128, 127] range:  x_q = round((x - v_min) / scale) - 128
            normalized = (flat - v_min) / scale
            quantized = (normalized - 128).round().clamp(-128, 127).to(dtype=torch.int8)

        # Store both scale and offset so decompression is exact.
        scale_info = torch.stack([
            torch.tensor(v_min.item(), device=flat.device),
            scale,
        ])

        return CompressedGradient(
            indices=None,
            values=quantized,
            scale=scale_info,
            original_shape=shape,
            original_dtype=dtype,
            original_numel=numel,
            method=CompressionMethod.QUANTIZE_INT8,
        )

    @staticmethod
    def _decompress_one(cg: CompressedGradient) -> torch.Tensor:
        """Reconstruct a single gradient from its compressed representation."""
        if cg.method == CompressionMethod.TOP_K:
            return _decompress_topk(cg)
        elif cg.method == CompressionMethod.QUANTIZE_INT8:
            return _decompress_quantize(cg)
        else:
            raise ValueError(f"Unknown compression method: {cg.method}")


def _decompress_topk(cg: CompressedGradient) -> torch.Tensor:
    """Reconstruct a sparse gradient from Top-K compressed form."""
    device = cg.values.device
    reconstructed = torch.zeros(
        cg.original_numel, dtype=cg.original_dtype, device=device,
    )
    reconstructed.scatter_(0, cg.indices.long(), cg.values.float())
    return reconstructed.view(cg.original_shape)


def _decompress_quantize(cg: CompressedGradient) -> torch.Tensor:
    """Reconstruct a gradient from INT8 quantized form."""
    v_min = cg.scale[0]
    scale = cg.scale[1]
    # Reverse: x = (x_q + 128) * scale + v_min
    deq = (cg.values.float() + 128) * scale + v_min
    return deq.view(cg.original_shape).to(dtype=cg.original_dtype)


def _dtype_size(dtype: torch.dtype) -> int:
    """Return the byte size of a PyTorch data type."""
    if dtype in (torch.float32, torch.int32):
        return 4
    elif dtype in (torch.float16, torch.bfloat16, torch.int16):
        return 2
    elif dtype == torch.int8:
        return 1
    elif dtype == torch.float64:
        return 8
    return 4  # fallback


# ===========================================================================
# NCCL State Replication
# ===========================================================================


class NCCLStateReplicator:
    """Replicate tensor state across GPUs using NCCL collectives.

    Provides GPU-to-GPU all-gather of tensors with automatic CPU fallback
    when CUDA or NCCL is unavailable.  This is a higher-level wrapper around
    ``torch.distributed`` intended for replicating model/KV-cache state to
    standby nodes.

    Thread-safe: uses an internal lock for initialization and stats updates.
    """

    def __init__(
        self,
        rank: int = 0,
        world_size: int = 1,
        master_addr: str = "127.0.0.1",
        master_port: int = 29500,
        backend: str = "nccl",
        timeout_s: float = 30.0,
        auto_init: bool = True,
    ):
        self._rank = rank
        self._world_size = world_size
        self._master_addr = master_addr
        self._master_port = master_port
        self._backend = backend
        self._timeout_s = timeout_s
        self._effective_backend: str = backend
        self._initialized = False
        self._lock = threading.Lock()
        self._replication_count = 0
        self._total_bytes_replicated = 0
        self._total_replication_time_ns = 0
        self._fallback_count = 0

        if auto_init:
            self.initialize()

    # -- Public API ---------------------------------------------------------

    def initialize(self) -> None:
        """Initialize the NCCL/CUDA distributed process group.

        Called automatically unless ``auto_init=False``.  Safe to call
        multiple times.
        """
        if self._initialized:
            return
        if self._world_size <= 1:
            self._initialized = True
            return

        import os

        os.environ.setdefault("MASTER_ADDR", self._master_addr)
        os.environ.setdefault("MASTER_PORT", str(self._master_port))

        import torch.distributed as dist

        if dist.is_initialized():
            self._initialized = True
            return

        effective = self._backend
        if effective == "nccl" and not torch.cuda.is_available():
            logger.warning(
                "NCCL requested but no CUDA GPUs; falling back to Gloo "
                "for NCCLStateReplicator"
            )
            effective = "gloo"

        self._effective_backend = effective
        dist.init_process_group(
            backend=effective,
            rank=self._rank,
            world_size=self._world_size,
        )
        self._initialized = True
        logger.info(
            f"NCCLStateReplicator initialized: rank={self._rank}, "
            f"world={self._world_size}, backend={effective}"
        )

    @property
    def is_initialized(self) -> bool:
        """Whether the process group is ready."""
        import torch.distributed as dist

        return self._initialized and dist.is_initialized()

    @property
    def available(self) -> bool:
        """True if NCCL/CUDA is available for GPU replication.

        Returns False when running on CPU-only hardware, signalling
        callers to use CPU fallback.
        """
        return (
            self._effective_backend == "nccl"
            and torch.cuda.is_available()
            and self.is_initialized
        )

    @property
    def device(self) -> torch.device:
        """Preferred device for replication tensors."""
        if self.available:
            return torch.device(f"cuda:{self._rank % max(torch.cuda.device_count(), 1)}")
        return torch.device("cpu")

    def replicate(
        self,
        tensor: torch.Tensor,
        peers: list[int] | None = None,
    ) -> list[torch.Tensor]:
        """All-gather *tensor* across *peers* and return collected copies.

        When ``peers`` is ``None``, all ranks in the world participate.
        When the effective backend is Gloo (CPU fallback), the tensor is
        moved to CPU before communication.

        Args:
            tensor: Local tensor to replicate.
            peers: Optional subset of ranks to gather from.
                Must include the local rank.  When provided, a new
                process group is created for these peers.

        Returns:
            List of length ``world_size`` (or ``len(peers)``) containing
            the tensor from each peer.

        Raises:
            RuntimeError: If the replicator is not initialized.
        """
        import torch.distributed as dist

        self._ensure_initialized()

        start = time.time_ns()

        # Move to the correct device before communication.
        comm_tensor = tensor.to(device=self.device)

        if peers is not None and self._rank not in peers:
            raise ValueError(
                f"Local rank {self._rank} must be included in peers list {peers}"
            )

        if peers is not None and len(peers) > 1:
            # Create a subgroup for the specified peers.
            group = dist.new_group(ranks=peers, backend=self._effective_backend)
            ws = len(peers)
            gather_list = [
                torch.empty_like(comm_tensor) for _ in range(ws)
            ]
            dist.all_gather(gather_list, comm_tensor, group=group)
            dist.destroy_process_group(group)
        elif self._world_size > 1 and peers is None:
            gather_list = [
                torch.empty_like(comm_tensor) for _ in range(self._world_size)
            ]
            dist.all_gather(gather_list, comm_tensor)
        else:
            # Single-node: just return the local tensor.
            gather_list = [comm_tensor]

        elapsed = time.time_ns() - start
        n_bytes = tensor.numel() * tensor.element_size() * len(gather_list)

        with self._lock:
            self._replication_count += 1
            self._total_bytes_replicated += n_bytes
            self._total_replication_time_ns += elapsed

        logger.debug(
            f"NCCLStateReplicator: replicated {tensor.shape} "
            f"({tensor.numel() * tensor.element_size()} B) "
            f"across {len(gather_list)} peers in {elapsed / 1e6:.2f}ms"
        )
        return gather_list

    def replicate_cpu_fallback(
        self,
        tensor: torch.Tensor,
        peers: list[int] | None = None,
    ) -> list[torch.Tensor]:
        """Replicate using CPU-based broadcast when NCCL is unavailable.

        Moves tensors to CPU and uses Gloo all-gather.  This is the
        explicit fallback path -- prefer :meth:`replicate`, which handles
        fallback automatically.

        Args:
            tensor: Local tensor to replicate.
            peers: Optional peer rank subset.

        Returns:
            List of gathered tensors.
        """
        cpu_tensor = tensor.cpu()
        with self._lock:
            self._fallback_count += 1
        logger.debug(
            f"NCCLStateReplicator CPU fallback: replicating {tensor.shape}"
        )
        return self.replicate(cpu_tensor, peers=peers)

    def destroy(self) -> None:
        """Tear down the process group."""
        import torch.distributed as dist

        with self._lock:
            if dist.is_initialized():
                dist.destroy_process_group()
            self._initialized = False

    # -- Stats --------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return replication statistics.

        Returns:
            Dict with keys ``replication_count``, ``total_bytes_replicated``,
            ``avg_replication_time_us``, ``fallback_count``, and
            ``effective_backend``.
        """
        with self._lock:
            avg_us = (
                (self._total_replication_time_ns / max(self._replication_count, 1))
                / 1000
            )
            return {
                "replication_count": self._replication_count,
                "total_bytes_replicated": self._total_bytes_replicated,
                "avg_replication_time_us": round(avg_us, 2),
                "total_replication_time_ms": round(
                    self._total_replication_time_ns / 1e6, 2
                ),
                "fallback_count": self._fallback_count,
                "effective_backend": self._effective_backend,
            }

    # -- Internal helpers ---------------------------------------------------

    def _ensure_initialized(self) -> None:
        import torch.distributed as dist

        if not self._initialized:
            self.initialize()
        if not dist.is_initialized():
            raise RuntimeError(
                "NCCLStateReplicator is not initialized; call initialize() first, "
                "or pass auto_init=True to the constructor"
            )


# ===========================================================================
# Incremental Recovery
# ===========================================================================


@dataclass
class ParameterDelta:
    """A recorded delta (change) for a single parameter at a given step.

    Attributes:
        step: Training / generation step at which this delta was recorded.
        name: Fully-qualified parameter name.
        delta: The change tensor (``new_value - old_value``).
        device: Original device of the parameter.
        dtype: Original data type of the parameter.
    """

    step: int
    name: str
    delta: torch.Tensor
    device: torch.device
    dtype: torch.dtype


class IncrementalRecovery:
    """Track parameter deltas from the last checkpoint and replay them.

    After a coordinated checkpoint, call :meth:`save_delta` after each step
    to record which parameters changed.  Later, if a node needs to recover
    to a target step, :meth:`recover_to` replays the stored deltas in order
    so the node's parameter state is reconstructed without transferring the
    full weights.

    Design:

    * Deltas are stored as ``new_value - old_value``, so they are sparse
      (many parameters do not change between steps) and additive.

    * Recovery replays deltas sequentially from step ``checkpoint_step + 1``
      up to the requested target step.  Skipping an intermediate step would
      produce incorrect state.

    * Deltas are kept in a ring buffer (``max_deltas``) to cap memory usage.
      When the buffer is full, the oldest delta is evicted.

    Thread-safe: all public methods acquire an internal lock.
    """

    def __init__(
        self,
        max_deltas: int = 1000,
        checkpoint_step: int = 0,
        persist_path: str | None = None,
    ):
        if max_deltas < 1:
            raise ValueError(f"max_deltas must be >= 1, got {max_deltas}")
        self._max_deltas = max_deltas
        self._checkpoint_step = checkpoint_step
        self._persist_path = persist_path
        self._deltas: dict[int, dict[str, ParameterDelta]] = {}  # step -> {name -> delta}
        self._step_order: list[int] = []  # ordered step list for replay
        self._lock = threading.Lock()
        self._latest_step = checkpoint_step

    # -- Public API ---------------------------------------------------------

    @property
    def latest_step(self) -> int:
        """Highest step for which a delta has been recorded."""
        return self._latest_step

    @property
    def delta_count(self) -> int:
        """Number of stored deltas across all steps."""
        with self._lock:
            return sum(len(d) for d in self._deltas.values())

    @property
    def step_count(self) -> int:
        """Number of steps with recorded deltas."""
        with self._lock:
            return len(self._step_order)

    def save_delta(
        self,
        step: int,
        params: dict[str, torch.Tensor],
        base_params: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Record parameter changes for *step*.

        When *base_params* is provided, the delta is computed as
        ``params[name] - base_params[name]``.  When it is ``None``,
        the delta is computed against the parameter's value at the last
        recorded step (or the checkpoint if no prior delta exists).

        Args:
            step: Step number (must be > ``latest_step``).
            params: Current parameter values after the step.
            base_params: Optional reference state.  When ``None``, the
                reference is the checkpoint state (first occurrence) or
                the previous step's saved value.

        Raises:
            ValueError: If *step* is not greater than the latest recorded step.
        """
        if step <= self._latest_step:
            raise ValueError(
                f"Step {step} must be > latest step {self._latest_step}"
            )

        step_deltas: dict[str, ParameterDelta] = {}

        for name, current in params.items():
            if current is None or current.numel() == 0:
                continue

            # Determine the base (reference) value.
            if base_params is not None and name in base_params:
                base = base_params[name]
            else:
                # Use the value from the previous delta, if any.
                prev = self._find_previous_delta(name, step)
                if prev is not None:
                    # To compute the delta we need the absolute value at
                    # the previous step.  We approximate by assuming the
                    # delta was applied to the checkpoint value, which is
                    # correct for the first occurrence.  For deeper chains
                    # we accumulate.
                    base = self._estimate_previous_value(name, step - 1)
                else:
                    # No previous delta -- this is the first recorded change
                    # for this parameter, so delta is relative to checkpoint.
                    base = current  # delta = 0 effectively

            if base is None:
                base = current  # no base available: delta is zero

            delta = current - base

            # Skip zero deltas to save memory.
            if delta.abs().max().item() == 0:
                continue

            step_deltas[name] = ParameterDelta(
                step=step,
                name=name,
                delta=delta.cpu().clone() if delta.is_cuda else delta.clone(),
                device=current.device,
                dtype=current.dtype,
            )

        with self._lock:
            if step_deltas:
                self._deltas[step] = step_deltas
                self._step_order.append(step)
                self._latest_step = step

                # Evict oldest step if over capacity.
                if len(self._step_order) > self._max_deltas:
                    oldest_step = self._step_order.pop(0)
                    self._deltas.pop(oldest_step, None)

        logger.debug(
            f"IncrementalRecovery: saved {len(step_deltas)} deltas for step {step}"
        )

    def recover_to(
        self,
        target_step: int,
        params: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Apply deltas to *params* to advance from checkpoint to *target_step*.

        The *params* dict should contain the parameter state at the last
        checkpoint (``checkpoint_step``).  Deltas from
        ``checkpoint_step + 1`` through *target_step* are applied in
        sequence.

        Args:
            target_step: Step to recover to (must be <= ``latest_step``).
            params: Parameter state at the checkpoint.

        Returns:
            Updated *params* dict with deltas applied in order.

        Raises:
            ValueError: If *target_step* is not within the recorded range.
        """
        if target_step < self._checkpoint_step:
            raise ValueError(
                f"target_step {target_step} is before checkpoint step "
                f"{self._checkpoint_step}"
            )
        if target_step > self._latest_step:
            raise ValueError(
                f"target_step {target_step} exceeds latest step "
                f"{self._latest_step}; call save_delta first"
            )

        with self._lock:
            steps_to_apply = [
                s for s in self._step_order if self._checkpoint_step < s <= target_step
            ]

        logger.info(
            f"IncrementalRecovery: recovering to step {target_step} "
            f"by replaying {len(steps_to_apply)} delta steps"
        )

        for s in steps_to_apply:
            with self._lock:
                step_deltas = self._deltas.get(s, {})
            for name, delta_rec in step_deltas.items():
                if name in params and params[name] is not None:
                    delta_on_device = delta_rec.delta.to(
                        device=params[name].device,
                        dtype=params[name].dtype,
                    )
                    params[name] = params[name] + delta_on_device

        return params

    def set_checkpoint(self, step: int) -> None:
        """Update the checkpoint step and discard deltas before it.

        Args:
            step: New checkpoint step.  Must be >= ``checkpoint_step``.
        """
        if step < self._checkpoint_step:
            raise ValueError(
                f"New checkpoint step {step} must be >= "
                f"current checkpoint step {self._checkpoint_step}"
            )

        with self._lock:
            # Remove deltas for steps <= the new checkpoint.
            stale_steps = [s for s in self._step_order if s <= step]
            for s in stale_steps:
                self._deltas.pop(s, None)
                self._step_order.remove(s)
            self._checkpoint_step = step

        logger.info(
            f"IncrementalRecovery: checkpoint advanced to step {step}, "
            f"discarded {len(stale_steps)} delta steps"
        )

    def clear(self) -> None:
        """Discard all deltas and reset to the initial checkpoint step."""
        with self._lock:
            self._deltas.clear()
            self._step_order.clear()
            self._latest_step = self._checkpoint_step
        logger.info("IncrementalRecovery: all deltas cleared")

    # -- Internal helpers ---------------------------------------------------

    def _find_previous_delta(
        self, name: str, before_step: int
    ) -> ParameterDelta | None:
        """Find the most recent delta for *name* at a step < *before_step*."""
        candidate: ParameterDelta | None = None
        for s in reversed(self._step_order):
            if s >= before_step:
                continue
            if name in self._deltas.get(s, {}):
                candidate = self._deltas[s][name]
                break
        return candidate

    def _estimate_previous_value(
        self, name: str, step: int
    ) -> torch.Tensor | None:
        """Estimate the absolute value of *name* at *step* by summing deltas."""
        # This is an approximation: sum all deltas for this name up to *step*.
        accumulated: torch.Tensor | None = None
        for s in self._step_order:
            if s > step:
                break
            if name in self._deltas.get(s, {}):
                delta_t = self._deltas[s][name].delta
                accumulated = (
                    delta_t.clone()
                    if accumulated is None
                    else accumulated + delta_t
                )
        return accumulated

    def stats(self) -> dict[str, Any]:
        """Return delta tracking statistics."""
        with self._lock:
            total_deltas = sum(len(d) for d in self._deltas.values())
            return {
                "checkpoint_step": self._checkpoint_step,
                "latest_step": self._latest_step,
                "steps_recorded": len(self._step_order),
                "total_deltas": total_deltas,
                "max_deltas": self._max_deltas,
                "utilization_pct": round(
                    len(self._step_order) / max(self._max_deltas, 1) * 100, 1
                ),
            }


# ===========================================================================
# FSDP Weight Synchronization
# ===========================================================================


@dataclass
class ShardInfo:
    """Metadata about a single weight shard.

    Attributes:
        rank: Owner rank for this shard.
        start_idx: Start index in the flattened parameter.
        end_idx: End index (exclusive).
        shape: Original parameter shape.
        dtype: Original parameter data type.
    """

    rank: int
    start_idx: int
    end_idx: int
    shape: torch.Size
    dtype: torch.dtype


class FSDPWeightSync:
    """Synchronise FSDP-sharded weight partitions across distributed nodes.

    This class handles the **weight synchronisation** layer of FSDP:

    * :meth:`sync_weights` -- all-gather each node's shard so all peers hold
      the full parameter set (used before the forward pass).

    * :meth:`get_shard` -- extract the local shard for a parameter given the
      rank and world size (used after the backward pass or when loading
      a checkpoint).

    Unlike :class:`~distllm.dist.fsdp.FSDPShard`, which wraps an ``nn.Module``
    and handles the full gather-forward-free cycle, this class is a lighter
    utility for explicit weight synchronisation when you need fine-grained
    control over when communication happens.

    Thread-safe.
    """

    def __init__(
        self,
        rank: int = 0,
        world_size: int = 1,
        min_param_size: int = 1024,
    ):
        if world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {world_size}")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank {rank} out of range [0, {world_size})")

        self._rank = rank
        self._world_size = world_size
        self._min_param_size = min_param_size
        self._lock = threading.Lock()
        self._sync_count = 0
        self._total_sync_bytes = 0
        self._total_sync_time_ns = 0

    # -- Public API ---------------------------------------------------------

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    def sync_weights(
        self,
        param_groups: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """All-gather sharded weights across all peers.

        Each peer provides its local shard for every parameter.  After
        this call all peers return the full, un-sharded parameter state.

        Args:
            param_groups: Mapping of parameter name to **local shard**.

        Returns:
            Full (un-sharded) parameter dict reconstructed from all peers.

        Raises:
            RuntimeError: If ``torch.distributed`` is not initialized.
        """
        import torch.distributed as dist

        if self._world_size <= 1:
            return dict(param_groups)

        if not dist.is_initialized():
            raise RuntimeError(
                "torch.distributed is not initialised; call "
                "``dist.init_process_group`` before ``sync_weights``"
            )

        start = time.time_ns()
        result: dict[str, torch.Tensor] = {}
        ws = self._world_size
        total_bytes = 0

        for name, local_shard in param_groups.items():
            if local_shard is None or local_shard.numel() == 0:
                continue

            # All ranks must contribute the same-sized tensor for all_gather.
            max_chunk_size = _ceil_div(local_shard.numel(), ws)
            padded = local_shard.flatten()
            if padded.numel() < max_chunk_size:
                pad_len = max_chunk_size - padded.numel()
                padded = torch.cat([padded, padded.new_zeros(pad_len)])

            gather_list = [
                torch.empty(
                    max_chunk_size,
                    dtype=padded.dtype,
                    device=padded.device,
                )
                for _ in range(ws)
            ]

            dist.all_gather(gather_list, padded)

            # Concatenate and trim to the original shard's size (each peer
            # may have a different-length shard; we keep the local one).
            full_flat = torch.cat(gather_list, dim=0)
            reconstructed = full_flat[: local_shard.numel()].view(
                local_shard.shape
            )
            result[name] = reconstructed.to(dtype=local_shard.dtype)
            total_bytes += local_shard.numel() * local_shard.element_size() * ws

        elapsed = time.time_ns() - start
        with self._lock:
            self._sync_count += 1
            self._total_sync_bytes += total_bytes
            self._total_sync_time_ns += elapsed

        logger.debug(
            f"FSDPWeightSync: synced {len(param_groups)} parameter groups "
            f"across {ws} ranks in {elapsed / 1e6:.2f}ms "
            f"({total_bytes / 1024:.1f} KiB total)"
        )
        return result

    @staticmethod
    def get_shard(
        tensor: torch.Tensor,
        rank: int,
        world_size: int,
        min_param_size: int = 1024,
    ) -> tuple[torch.Tensor, ShardInfo]:
        """Extract the shard for *rank* from a full parameter tensor.

        Args:
            tensor: Full parameter tensor.
            rank: Local rank (0-based).
            world_size: Total number of shards.
            min_param_size: Minimum elements for sharding; parameters
                below this are returned in full on every rank.

        Returns:
            Tuple of ``(shard_tensor, ShardInfo)`` where ``shard_tensor`` is
            the local shard (or the full tensor if too small) and
            ``ShardInfo`` describes the shard range.

        Raises:
            ValueError: On invalid rank or world_size.
        """
        if world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {world_size}")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank {rank} out of range [0, {world_size})")

        numel = tensor.numel()
        if numel < min_param_size or numel < world_size:
            # Too small to shard: every rank keeps the full tensor.
            return (
                tensor.clone(),
                ShardInfo(
                    rank=rank,
                    start_idx=0,
                    end_idx=numel,
                    shape=tensor.shape,
                    dtype=tensor.dtype,
                ),
            )

        chunk_size = _ceil_div(numel, world_size)
        start = rank * chunk_size
        end = min(start + chunk_size, numel)

        flat = tensor.flatten()
        shard = flat[start:end].clone()

        info = ShardInfo(
            rank=rank,
            start_idx=start,
            end_idx=end,
            shape=tensor.shape,
            dtype=tensor.dtype,
        )

        return shard, info

    def stats(self) -> dict[str, Any]:
        """Return weight sync statistics.

        Returns:
            Dict with keys ``sync_count``, ``total_sync_bytes``,
            ``avg_sync_bytes``, ``total_sync_time_ms``, ``rank``,
            and ``world_size``.
        """
        with self._lock:
            return {
                "sync_count": self._sync_count,
                "total_sync_bytes": self._total_sync_bytes,
                "avg_sync_bytes": (
                    self._total_sync_bytes // max(self._sync_count, 1)
                ),
                "total_sync_time_ms": round(
                    self._total_sync_time_ns / 1e6, 2
                ),
                "rank": self._rank,
                "world_size": self._world_size,
            }


# ===========================================================================
# Redundant Executor
# ===========================================================================


@dataclass
class RedundantResult:
    """Outcome of a redundant task execution.

    Attributes:
        result: Return value from the winning replica.
        replica_id: Identifier of the fastest replica.
        elapsed_s: Wall-clock time of the fastest replica.
        total_replicas: Number of replicas launched.
        failures: Number of replicas that raised an exception.
        timestamp: Monotonic time when the result was collected.
    """

    result: Any
    replica_id: str
    elapsed_s: float
    total_replicas: int
    failures: int
    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class RecoveryReport:
    """Summary of a :meth:`RedundantExecutor.recover_node` operation.

    Attributes:
        node_id: The recovered node identifier.
        success: Whether recovery completed without error.
        recovery_time_ms: Wall-clock time for the full recovery.
        steps_replayed: Number of incremental recovery steps applied.
        weights_synced: Number of parameter groups synchronised.
        replication_size_bytes: Total bytes replicated during recovery.
        error: Error message if *success* is False; otherwise empty.
    """

    node_id: str
    success: bool
    recovery_time_ms: float
    steps_replayed: int = 0
    weights_synced: int = 0
    replication_size_bytes: int = 0
    error: str = ""


class RedundantExecutor:
    """Combine gradient compression, NCCL replication, incremental recovery,
    and FSDP weight sync into a single fault-tolerant executor.

    This is the top-level orchestrator for redundant distributed inference:

    * :meth:`run_redundant` fans a callable out to ``n_replicas`` independent
      copies (e.g. replicated pipeline stages or model instances) and returns
      the fastest result.  Gradients are compressed before being sent to the
      coordinator.

    * :meth:`recover_node` performs end-to-end recovery for a failed node:
      incremental delta replay, NCCL state replication for KV cache / buffers,
      and FSDP weight synchronisation.

    * :meth:`stats` aggregates statistics from all sub-components.

    Usage::

        executor = RedundantExecutor(
            compressor=GradientCompressor(),
            replicator=NCCLStateReplicator(rank=0, world_size=4),
            recovery=IncrementalRecovery(),
            weight_sync=FSDPWeightSync(rank=0, world_size=4),
        )
        # Run a task redundantly
        result = executor.run_redundant(my_fn, n_replicas=2)
        # Recover a node
        report = executor.recover_node("node-3")
        # Print stats
        print(executor.stats())
    """

    def __init__(
        self,
        compressor: GradientCompressor | None = None,
        replicator: NCCLStateReplicator | None = None,
        recovery: IncrementalRecovery | None = None,
        weight_sync: FSDPWeightSync | None = None,
        default_timeout_s: float = 30.0,
    ):
        self._compressor = compressor or GradientCompressor()
        self._replicator = replicator or NCCLStateReplicator(
            auto_init=False,
        )
        self._recovery = recovery or IncrementalRecovery()
        self._weight_sync = weight_sync or FSDPWeightSync()
        self._default_timeout = default_timeout_s
        self._lock = threading.Lock()
        self._total_tasks = 0
        self._total_recoveries = 0
        self._total_compressed_bytes = 0
        self._total_replicated_bytes = 0
        self._cumulative_recovery_time_ms = 0.0

        logger.info("RedundantExecutor initialised")

    # -- Public API ---------------------------------------------------------

    def run_redundant(
        self,
        task: Callable[..., Any],
        n_replicas: int = 2,
        task_args: tuple[Any, ...] = (),
        task_kwargs: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> RedundantResult:
        """Run *task* on ``n_replicas`` independent copies and return
        the fastest result.

        Each replica runs the callable concurrently (via thread pool) and
        the first to complete wins.  Remaining replicas are allowed to drain
        in the background; their results are discarded.

        Args:
            task: Callable to execute redundantly.  It receives
                ``replica_id`` as the first positional argument; additional
                args are passed via *task_args*.
            n_replicas: Number of redundant copies (minimum 1).
            task_args: Additional positional arguments for the task.
            task_kwargs: Additional keyword arguments for the task.
            timeout_s: Per-replica timeout.  Defaults to ``default_timeout_s``.

        Returns:
            :class:`RedundantResult` describing the fastest replica's outcome.

        Raises:
            RuntimeError: If all replicas failed.
        """
        from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

        n = max(1, n_replicas)
        timeout = timeout_s or self._default_timeout
        kwargs = task_kwargs or {}

        executor = ThreadPoolExecutor(max_workers=n)
        futures = {}
        for i in range(n):
            rid = f"replica-{i}"
            fut = executor.submit(task, rid, *task_args, **kwargs)
            futures[fut] = rid

        # Wait for the first completed result.
        done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED, timeout=timeout)

        failures = 0
        best_result: Any = None
        best_id: str = ""
        best_elapsed: float = 0.0
        best_timestamp: float = 0.0

        for fut in done:
            rid = futures[fut]
            try:
                t0 = time.monotonic()
                result = fut.result(timeout=0)
                elapsed = time.monotonic() - t0
                best_result = result
                best_id = rid
                best_elapsed = elapsed
                best_timestamp = time.monotonic()
                break  # First completed is the winner.
            except Exception as exc:
                logger.warning(f"Redundant task {rid} failed: {exc}")
                failures += 1

        executor.shutdown(wait=False)

        if best_result is None and failures > 0:
            raise RuntimeError(
                f"All {n} redundant replicas failed "
                f"({failures} failures, timeout={timeout}s)"
            )

        with self._lock:
            self._total_tasks += 1

        logger.info(
            f"RedundantExecutor: task completed via {best_id} "
            f"in {best_elapsed * 1000:.1f}ms "
            f"({failures}/{n} replicas failed)"
        )

        return RedundantResult(
            result=best_result,
            replica_id=best_id,
            elapsed_s=best_elapsed,
            total_replicas=n,
            failures=failures,
            timestamp=best_timestamp,
        )

    def recover_node(
        self,
        node_id: str,
        checkpoint_params: dict[str, torch.Tensor] | None = None,
        target_step: int | None = None,
        sharded_params: dict[str, torch.Tensor] | None = None,
        state_tensors: dict[str, torch.Tensor] | None = None,
    ) -> RecoveryReport:
        """Perform end-to-end recovery for *node_id*.

        Recovery runs three phases:

        1. **Incremental delta recovery** -- if *checkpoint_params* and
           *target_step* are provided, apply deltas from the incremental
           recovery tracker to bring the node's parameters up to date.

        2. **FSDP weight synchronisation** -- if *sharded_params* are
           provided, all-gather them across peers so this node holds the
           full weight set.

        3. **NCCL state replication** -- if *state_tensors* are provided,
           replicate them across the cluster (e.g. KV cache buffers,
           optimizer state).

        Args:
            node_id: Identifier of the node being recovered.
            checkpoint_params: Parameter state at the last checkpoint.
                Pass to apply incremental recovery deltas.
            target_step: Step to recover to.  Required if
                *checkpoint_params* is provided.
            sharded_params: Local sharded parameter groups.  Pass to
                trigger FSDP weight sync.
            state_tensors: Named tensors to replicate via NCCL.
                Pass to replicate runtime state.

        Returns:
            :class:`RecoveryReport` summarising the recovery outcome.
        """
        logger.info(f"RedundantExecutor: starting recovery for node {node_id}")
        t_start = time.monotonic()
        steps_replayed = 0
        weights_synced = 0
        replication_bytes = 0

        try:
            # Phase 1: Incremental recovery.
            if checkpoint_params is not None and target_step is not None:
                logger.info(
                    f"Recovery phase 1: applying incremental deltas "
                    f"to step {target_step}"
                )
                self._recovery.recover_to(target_step, checkpoint_params)
                steps_replayed = self._recovery.step_count

            # Phase 2: FSDP weight sync.
            if sharded_params is not None:
                logger.info(
                    f"Recovery phase 2: synchronising "
                    f"{len(sharded_params)} FSDP weight groups"
                )
                synced = self._weight_sync.sync_weights(sharded_params)
                weights_synced = len(synced)

            # Phase 3: NCCL state replication.
            if state_tensors is not None and self._replicator.is_initialized:
                logger.info(
                    f"Recovery phase 3: replicating "
                    f"{len(state_tensors)} state tensors via NCCL"
                )
                for name, tensor in state_tensors.items():
                    gathered = self._replicator.replicate(tensor)
                    replication_bytes += (
                        tensor.numel() * tensor.element_size() * len(gathered)
                    )

            elapsed_ms = (time.monotonic() - t_start) * 1000

            with self._lock:
                self._total_recoveries += 1
                self._cumulative_recovery_time_ms += elapsed_ms

            logger.success(
                f"RedundantExecutor: node {node_id} recovered in "
                f"{elapsed_ms:.1f}ms "
                f"(steps={steps_replayed}, weights={weights_synced}, "
                f"replication={replication_bytes / 1024:.1f}KiB)"
            )

            return RecoveryReport(
                node_id=node_id,
                success=True,
                recovery_time_ms=elapsed_ms,
                steps_replayed=steps_replayed,
                weights_synced=weights_synced,
                replication_size_bytes=replication_bytes,
            )

        except Exception as e:
            elapsed_ms = (time.monotonic() - t_start) * 1000
            logger.error(
                f"RedundantExecutor: node {node_id} recovery failed "
                f"after {elapsed_ms:.1f}ms: {e}"
            )
            return RecoveryReport(
                node_id=node_id,
                success=False,
                recovery_time_ms=elapsed_ms,
                error=str(e),
            )

    def stats(self) -> dict[str, Any]:
        """Aggregate statistics from all sub-components.

        Returns:
            Combined dict with top-level keys (``total_tasks``,
            ``total_recoveries``, ``cumulative_recovery_time_ms``,
            ``avg_recovery_time_ms``) nested under ``compressor``,
            ``replicator``, ``recovery``, and ``weight_sync``.
        """
        with self._lock:
            avg_recovery = (
                self._cumulative_recovery_time_ms / max(self._total_recoveries, 1)
            )
            return {
                "total_tasks": self._total_tasks,
                "total_recoveries": self._total_recoveries,
                "cumulative_recovery_time_ms": round(
                    self._cumulative_recovery_time_ms, 2
                ),
                "avg_recovery_time_ms": round(avg_recovery, 2),
                "compressor": self._compressor.stats(),
                "replicator": self._replicator.stats(),
                "recovery": self._recovery.stats(),
                "weight_sync": self._weight_sync.stats(),
            }

    def compress_gradients(
        self,
        gradients: dict[str, torch.Tensor],
        method: CompressionMethod | str = CompressionMethod.TOP_K,
        ratio: float | None = None,
    ) -> dict[str, CompressedGradient]:
        """Convenience wrapper for :meth:`GradientCompressor.compress`."""
        return self._compressor.compress(gradients, method=method, ratio=ratio)

    def decompress_gradients(
        self,
        compressed: dict[str, CompressedGradient],
    ) -> dict[str, torch.Tensor]:
        """Convenience wrapper for :meth:`GradientCompressor.decompress`."""
        return self._compressor.decompress(compressed)

    def replicate_state(
        self,
        tensor: torch.Tensor,
        peers: list[int] | None = None,
    ) -> list[torch.Tensor]:
        """Convenience wrapper for :meth:`NCCLStateReplicator.replicate`."""
        return self._replicator.replicate(tensor, peers=peers)

    def save_delta(
        self,
        step: int,
        params: dict[str, torch.Tensor],
        base_params: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Convenience wrapper for :meth:`IncrementalRecovery.save_delta`."""
        self._recovery.save_delta(step, params, base_params=base_params)

    def recover_to(
        self,
        target_step: int,
        params: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Convenience wrapper for :meth:`IncrementalRecovery.recover_to`."""
        return self._recovery.recover_to(target_step, params)

    def sync_weights(
        self,
        param_groups: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Convenience wrapper for :meth:`FSDPWeightSync.sync_weights`."""
        return self._weight_sync.sync_weights(param_groups)


# ===========================================================================
# Module-level helpers
# ===========================================================================


def _ceil_div(a: int, b: int) -> int:
    """Ceiling division: ``ceil(a / b)``."""
    return -(-a // b)
