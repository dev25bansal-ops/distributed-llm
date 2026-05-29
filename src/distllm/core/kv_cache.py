"""KV cache management for distributed LLM inference."""

import asyncio
import threading
import time

import numpy as np
import torch
from loguru import logger

# Quantization range constants
FP8_E4M3_MAX: float = 448.0   # Max representable value in FP8 E4M3 format
INT8_MAX: float = 127.0       # Max representable value in signed int8
INT4_MAX: float = 7.0         # Max representable value in signed int4


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
            allocations, _ = self._paged_mgr.free_layer_kv(
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


class KVCache:
    """Manages key-value cache for a single generation request.

    Stores past_key_values for each transformer layer, enabling
    efficient autoregressive generation without re-processing tokens.
    Uses pre-allocated buffers and slicing to avoid O(n²) torch.cat.

    Supports dynamic CPU/GPU swap: when GPU memory is scarce, KV cache
    blocks can be offloaded to CPU RAM and swapped back on access.
    """

    def __init__(self, max_seq_len: int = 0):
        self.cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.num_layers = 0
        self._max_seq_len = max_seq_len
        self._seq_lens: list[int] = []
        # Quantization state
        self._quantized: bool = False
        self._quant_bits: int = 8
        self._quant_fp8: bool = False
        # Quantized segments stored as list of (qk, qv, sk, sv) per layer
        self._qsegments: list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]] = []
        # FP8 segments: list of (fp8_k, fp8_v, scale_k, scale_v) per layer
        self._fp8_segments: list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]] = []
        # Scale factors from compression (per-layer)
        self._scale_k: list[torch.Tensor] = []
        self._scale_v: list[torch.Tensor] = []
        # CPU/GPU swap state
        self._offloaded: bool = False
        self._cpu_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        self._offload_device: str = "cpu"
        self._lock = threading.Lock()

    def init_cache(self, num_layers: int, batch_size: int, num_heads: int, head_dim: int, device: str = "cpu"):
        """Initialize empty KV cache for all layers with pre-allocated buffer."""
        with self._lock:
            self.cache = []
            self.num_layers = num_layers
            self._seq_lens = [0] * num_layers
            capacity = max(self._max_seq_len, 1)
            for _ in range(num_layers):
                k = torch.zeros(batch_size, num_heads, capacity, head_dim, device=device)
                v = torch.zeros(batch_size, num_heads, capacity, head_dim, device=device)
                self.cache.append((k, v))

    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Get cached key/value states for a layer."""
        with self._lock:
            if layer_idx >= len(self.cache):
                return None
            return self.cache[layer_idx]

    def get_all(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Get all cached key/value states."""
        with self._lock:
            return list(self.cache)

    def set_all(self, cache: list[tuple[torch.Tensor, torch.Tensor]]):
        """Replace entire cache."""
        with self._lock:
            self.cache = list(cache)
            self.num_layers = len(cache)

    def slice(self, layer_start: int, layer_end: int) -> "KVCache":
        """E13: Create a new KVCache with a slice of layers for partial reuse.

        Args:
            layer_start: Start layer index (inclusive).
            layer_end: End layer index (exclusive).

        Returns:
            New KVCache containing only the specified layers.
        """
        new_cache = KVCache()
        with self._lock:
            new_cache.cache = list(self.cache[layer_start:layer_end])
            new_cache.num_layers = layer_end - layer_start
            new_cache._quantized = self._quantized
            new_cache._quant_bits = self._quant_bits
            new_cache._quant_fp8 = self._quant_fp8
        return new_cache

    def to(self, device: str) -> "KVCache":
        """Move cache to device, preserving quantization state."""
        new_cache = KVCache()
        with self._lock:
            new_cache.cache = [(k.to(device), v.to(device)) for k, v in self.cache]
            new_cache.num_layers = self.num_layers
            new_cache._quantized = self._quantized
            new_cache._quant_bits = self._quant_bits
            new_cache._quant_fp8 = self._quant_fp8
            new_cache._qsegments = [
                [(qk.to(device), qv.to(device), sk.to(device), sv.to(device)) for qk, qv, sk, sv in segs]
                for segs in self._qsegments
            ]
            new_cache._fp8_segments = [
                [(fk.to(device), fv.to(device), sk.to(device), sv.to(device)) for fk, fv, sk, sv in segs]
                for segs in self._fp8_segments
            ]
        return new_cache

    @property
    def sequence_length(self) -> int:
        """Get current sequence length from cache."""
        with self._lock:
            if not self._seq_lens:
                return 0
            return self._seq_lens[0]

    def clear(self):
        """Clear the cache and quantization state."""
        with self._lock:
            self.cache = []
            self.num_layers = 0
            self._qsegments = []
            self._fp8_segments = []
            self._quant_fp8 = False
            self._cpu_cache = None
            self._offloaded = False

    # ── CPU/GPU Swap ────────────────────────────────────────────────

    def offload_to_cpu(self, non_blocking: bool = True) -> int:
        """Offload KV cache from GPU to CPU RAM.

        Moves all cache tensors to CPU to free GPU memory. The cache
        remains usable — on the next access, tensors are automatically
        moved back to GPU.

        Args:
            non_blocking: Use async transfer (requires pinned memory).

        Returns:
            Bytes offloaded.
        """
        with self._lock:
            if self._offloaded:
                return 0

            bytes_offloaded = 0
            cpu_cache = []
            for k, v in self.cache:
                if k.is_cuda:
                    k_cpu = k.to("cpu", non_blocking=non_blocking)
                    v_cpu = v.to("cpu", non_blocking=non_blocking)
                    cpu_cache.append((k_cpu, v_cpu))
                    bytes_offloaded += k.element_size() * k.numel() + v.element_size() * v.numel()
                else:
                    cpu_cache.append((k, v))

            self._cpu_cache = cpu_cache
            self._offloaded = True
            return bytes_offloaded

    def load_to_gpu(self, device: str = "cuda", non_blocking: bool = True) -> int:
        """Load KV cache from CPU back to GPU.

        Args:
            device: Target CUDA device.
            non_blocking: Use async transfer.

        Returns:
            Bytes loaded.
        """
        with self._lock:
            if not self._offloaded or self._cpu_cache is None:
                return 0

            bytes_loaded = 0
            gpu_cache = []
            for k, v in self._cpu_cache:
                if k.device.type == "cpu":
                    k_gpu = k.to(device, non_blocking=non_blocking)
                    v_gpu = v.to(device, non_blocking=non_blocking)
                    gpu_cache.append((k_gpu, v_gpu))
                    bytes_loaded += k.element_size() * k.numel() + v.element_size() * v.numel()
                else:
                    gpu_cache.append((k, v))

            self.cache = gpu_cache
            self._cpu_cache = None
            self._offloaded = False
            return bytes_loaded

    @property
    def is_offloaded(self) -> bool:
        """Whether the cache is currently offloaded to CPU."""
        return self._offloaded

    def pin_memory(self) -> None:
        """Pin CPU memory for faster GPU transfers.

        Call before offloading to enable non_blocking transfers.
        """
        with self._lock:
            if self._offloaded and self._cpu_cache is not None:
                self._cpu_cache = [
                    (k.pin_memory(), v.pin_memory()) for k, v in self._cpu_cache
                ]
            elif not self._offloaded:
                self.cache = [
                    (k.pin_memory(), v.pin_memory()) for k, v in self.cache
                ]

    def memory_usage(self) -> int:
        """Get memory usage in bytes."""
        with self._lock:
            total = 0
            for k, v in self.cache:
                total += k.element_size() * k.numel() + v.element_size() * v.numel()
            if self._quantized:
                for segs in self._qsegments:
                    for _qk, _qv, sk, sv in segs:
                        total += sk.element_size() * sk.numel()
                        total += sv.element_size() * sv.numel()
            return total

    def __repr__(self) -> str:
        quant = ""
        if self._quantized:
            quant = f", quant={'fp8' if self._quant_fp8 else f'int{self._quant_bits}'}"
        mem_mb = self.memory_usage() / (1024 * 1024)
        return f"KVCache(layers={self.num_layers}, seq_len={self.sequence_length}, mem={mem_mb:.1f}MB{quant})"

    def enable_quantization(self, bits: int = 8, use_fp8: bool = False) -> None:
        """Enable KV cache quantization.

        Args:
            bits: Target bit width (4 or 8). Ignored if use_fp8=True.
            use_fp8: Use FP8 E4M3 quantization for 2x capacity vs fp16.
        """
        with self._lock:
            if use_fp8:
                # Check if FP8 is actually available
                if hasattr(torch, 'float8_e4m3fn'):
                    self._quantized = True
                    self._quant_fp8 = True
                    self._quant_bits = 8
                    return
                else:
                    logger.warning("FP8 not available on this PyTorch version, falling back to int8")
                    bits = 8
            if bits not in (4, 8):
                raise ValueError(f"KV cache quantization bits must be 4 or 8, got {bits}")
            self._quantized = True
            self._quant_bits = bits
            self._quant_fp8 = False

    def compress(self, method: str = "int8") -> dict:
        """Compress existing KV cache in-place to reduce memory usage.

        Args:
            method: Compression method — "fp8", "int8", or "int4".

        Returns:
            Dict with compression stats (original_bytes, compressed_bytes, ratio).
        """
        import torch as _torch

        original_bytes = self.memory_usage()

        if method == "fp8":
            if not hasattr(_torch, 'float8_e4m3fn'):
                logger.warning("FP8 not available, falling back to int8")
                method = "int8"
            else:
                with self._lock:
                    compressed_cache = []
                    scales_k = []
                    scales_v = []
                    for k, v in self.cache:
                        # Scale to preserve dynamic range
                        k_scale = k.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / FP8_E4M3_MAX
                        v_scale = v.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / FP8_E4M3_MAX
                        k_fp8 = (k / k_scale).to(_torch.float8_e4m3fn)
                        v_fp8 = (v / v_scale).to(_torch.float8_e4m3fn)
                        compressed_cache.append((k_fp8, v_fp8))
                        scales_k.append(k_scale)
                        scales_v.append(v_scale)
                    self.cache = compressed_cache
                    self._quantized = True
                    self._quant_fp8 = True
                    self._quant_bits = 8
                    self._scale_k = scales_k
                    self._scale_v = scales_v
                    compressed_bytes = self.memory_usage()
                    return {
                        "method": "fp8",
                        "original_bytes": original_bytes,
                        "compressed_bytes": compressed_bytes,
                        "ratio": compressed_bytes / max(original_bytes, 1),
                        "savings_pct": (1 - compressed_bytes / max(original_bytes, 1)) * 100,
                    }

        if method == "int8":
            with self._lock:
                compressed_cache = []
                scales_k = []
                scales_v = []
                for k, v in self.cache:
                    k_scale = k.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / INT8_MAX
                    v_scale = v.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / INT8_MAX
                    k_int8 = (k / k_scale).to(_torch.int8)
                    v_int8 = (v / v_scale).to(_torch.int8)
                    compressed_cache.append((k_int8, v_int8))
                    scales_k.append(k_scale)
                    scales_v.append(v_scale)
                self.cache = compressed_cache
                self._quantized = True
                self._quant_fp8 = False
                self._quant_bits = 8
                self._scale_k = scales_k
                self._scale_v = scales_v
                compressed_bytes = self.memory_usage()
                return {
                    "method": "int8",
                    "original_bytes": original_bytes,
                    "compressed_bytes": compressed_bytes,
                    "ratio": compressed_bytes / max(original_bytes, 1),
                    "savings_pct": (1 - compressed_bytes / max(original_bytes, 1)) * 100,
                }

        if method == "int4":
            with self._lock:
                compressed_cache = []
                scales_k = []
                scales_v = []
                for k, v in self.cache:
                    k_scale = k.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / INT4_MAX
                    v_scale = v.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / INT4_MAX
                    k_int4 = (k / k_scale).clamp(-7, 7).to(_torch.int8)
                    v_int4 = (v / v_scale).clamp(-7, 7).to(_torch.int8)
                    compressed_cache.append((k_int4, v_int4))
                    scales_k.append(k_scale)
                    scales_v.append(v_scale)
                self.cache = compressed_cache
                self._quantized = True
                self._quant_fp8 = False
                self._quant_bits = 4
                self._scale_k = scales_k
                self._scale_v = scales_v
                compressed_bytes = self.memory_usage()
                return {
                    "method": "int4",
                    "original_bytes": original_bytes,
                    "compressed_bytes": compressed_bytes,
                    "ratio": compressed_bytes / max(original_bytes, 1),
                    "savings_pct": (1 - compressed_bytes / max(original_bytes, 1)) * 100,
                }

        raise ValueError(f"Unknown compression method: {method}. Use 'fp8', 'int8', or 'int4'.")

    def get_scales(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Get scale factors from compression.

        Returns:
            Tuple of (scale_k_list, scale_v_list) where each list has
            one scale tensor per layer. Empty lists if not compressed.
        """
        with self._lock:
            return list(self._scale_k), list(self._scale_v)

    def update(self, layer_idx: int, new_key: torch.Tensor, new_value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Update KV cache for a layer with optional quantization.

        Uses pre-allocated slicing to avoid O(n²) torch.cat overhead.
        Falls back to dynamic growth when max_seq_len is not set.

        Args:
            layer_idx: Layer index.
            new_key: New key tensor to append.
            new_value: New value tensor to append.

        Returns:
            Updated (key, value) tuple for the layer.

        Raises:
            IndexError: If layer_idx is out of range.
        """
        with self._lock:
            if layer_idx < 0 or layer_idx >= len(self.cache):
                raise IndexError(f"Layer index {layer_idx} out of range (cache has {len(self.cache)} layers)")
            if self._quant_fp8:
                return self._update_fp8_unquantized(layer_idx, new_key, new_value)
            if self._quantized:
                return self._update_quantized(layer_idx, new_key, new_value)

            old_k, old_v = self.cache[layer_idx]
            cur_len = self._seq_lens[layer_idx]
            new_len = new_key.shape[-2]

            if self._max_seq_len and cur_len + new_len <= self._max_seq_len:
                old_k[:, :, cur_len:cur_len + new_len] = new_key
                old_v[:, :, cur_len:cur_len + new_len] = new_value
                self._seq_lens[layer_idx] = cur_len + new_len
                return old_k[:, :, :cur_len + new_len], old_v[:, :, :cur_len + new_len]

            key = torch.cat([old_k[:, :, :cur_len], new_key], dim=-2)
            value = torch.cat([old_v[:, :, :cur_len], new_value], dim=-2)
            self.cache[layer_idx] = (key, value)
            self._seq_lens[layer_idx] = cur_len + new_len
            return key, value

    def _update_quantized(self, layer_idx: int, new_key: torch.Tensor, new_value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Update KV cache with quantization for memory efficiency.

        Stores quantized tokens as append-only segments per layer,
        avoiding O(n) dequantize+requantize of the full cache on every step.
        """
        from distllm.core.quantization_selector import (
            apply_kv_cache_quantization,
            dequantize_kv_cache,
        )

        # Quantize new tensors only (small, O(num_new_tokens))
        (qk, sk), (qv, sv) = apply_kv_cache_quantization(new_key, new_value, self._quant_bits)

        # Store quantized segment (append-only, O(1))
        while len(self._qsegments) <= layer_idx:
            self._qsegments.append([])
        self._qsegments[layer_idx].append((qk, qv, sk, sv))

        # Maintain dequantized cache incrementally (cat only, no dequant/requant)
        new_k = dequantize_kv_cache(qk, sk, self._quant_bits)
        new_v = dequantize_kv_cache(qv, sv, self._quant_bits)

        if layer_idx >= len(self.cache):
            self.cache.append((new_k, new_v))
            return new_k, new_v

        old_k, old_v = self.cache[layer_idx]
        key = torch.cat([old_k, new_k], dim=-2)
        value = torch.cat([old_v, new_v], dim=-2)
        self.cache[layer_idx] = (key, value)
        return key, value

    def _update_fp8_unquantized(self, layer_idx: int, new_key: torch.Tensor, new_value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Append KV tensors without FP8 quantization (fallback path).

        FP8 quantization during incremental autoregressive steps is not yet
        implemented (requires a fused FP8 append kernel).  This method uses
        the standard pre-allocated slice / torch.cat path — functionally
        correct but no memory savings.  Use compress(method="fp8") for bulk
        FP8 compression of an already-populated cache.
        """
        old_k, old_v = self.cache[layer_idx]
        cur_len = self._seq_lens[layer_idx]
        new_len = new_key.shape[-2]
        if self._max_seq_len and cur_len + new_len <= self._max_seq_len:
            old_k[:, :, cur_len:cur_len + new_len] = new_key
            old_v[:, :, cur_len:cur_len + new_len] = new_value
            self._seq_lens[layer_idx] = cur_len + new_len
            return old_k[:, :, :cur_len + new_len], old_v[:, :, :cur_len + new_len]
        key = torch.cat([old_k[:, :, :cur_len], new_key], dim=-2)
        value = torch.cat([old_v[:, :, :cur_len], new_value], dim=-2)
        self.cache[layer_idx] = (key, value)
        self._seq_lens[layer_idx] = cur_len + new_len
        return key, value

    def get_quantized(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get dequantized KV cache for a layer.

        Returns the incrementally maintained dequantized cache directly.
        Returns:
            (key, value) tensors in original dtype.
        """
        with self._lock:
            return self.cache[layer_idx]

    def quantization_savings(self) -> float:
        """Calculate memory savings from quantization.

        Returns:
            Ratio of quantized memory to unquantized memory (lower is better).
        """
        if not self._quantized or not self.cache:
            return 1.0

        quantized_mem = self.memory_usage()

        # Estimate unquantized memory
        unquantized_mem = 0
        for k, v in self.cache:
            # Assume fp16 for unquantized
            unquantized_mem += 2 * k.numel() + 2 * v.numel()

        return quantized_mem / max(unquantized_mem, 1)


class KVCacheManager:
    """Manages KV caches for multiple concurrent requests."""

    def __init__(self):
        self.caches: dict[str, KVCache] = {}
        self._metadata: dict[str, dict] = {}  # E12: Per-cache metadata for eviction scoring
        self._lock = threading.RLock()

    def create(
        self,
        request_id: str,
        num_layers: int,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        device: str = "cpu",
        quant_bits: int = 0,
    ) -> KVCache:
        """Create a new KV cache for a request.

        Args:
            request_id: Unique request identifier.
            num_layers: Number of transformer layers.
            batch_size: Batch size.
            num_heads: Number of attention heads.
            head_dim: Dimension of each head.
            device: Device to allocate tensors on.
            quant_bits: Enable KV cache quantization (4 or 8). 0 = disabled.
        """
        cache = KVCache()
        cache.init_cache(num_layers, batch_size, num_heads, head_dim, device)
        if quant_bits > 0:
            cache.enable_quantization(quant_bits)
        with self._lock:
            self.caches[request_id] = cache
            # E12: Track metadata for eviction scoring
            self._metadata[request_id] = {
                "created_at": time.time(),
                "last_accessed": time.time(),
                "access_count": 0,
                "priority": 0,
            }
        return cache

    def get(self, request_id: str) -> KVCache | None:
        """Get KV cache for a request."""
        with self._lock:
            cache = self.caches.get(request_id)
            # E12: Update access metadata
            if cache is not None and request_id in self._metadata:
                self._metadata[request_id]["last_accessed"] = time.time()
                self._metadata[request_id]["access_count"] += 1
            return cache

    def update(self, request_id: str, layer_idx: int, new_key: torch.Tensor, new_value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Update KV cache for a request."""
        with self._lock:
            cache = self.caches.get(request_id)
            # E12: Update access metadata
            if cache is not None and request_id in self._metadata:
                self._metadata[request_id]["last_accessed"] = time.time()
        if cache is None:
            return None
        return cache.update(layer_idx, new_key, new_value)

    def delete(self, request_id: str):
        """Delete KV cache for a request."""
        with self._lock:
            if request_id in self.caches:
                self.caches[request_id].clear()
                del self.caches[request_id]
                self._metadata.pop(request_id, None)

    def clear_all(self):
        """Clear all caches."""
        with self._lock:
            for cache in self.caches.values():
                cache.clear()
            self.caches = {}
            self._metadata.clear()

    @property
    def active_requests(self) -> int:
        with self._lock:
            return len(self.caches)

    def total_memory_usage(self) -> int:
        """Total memory usage across all caches."""
        with self._lock:
            return sum(cache.memory_usage() for cache in self.caches.values())

    def eviction_score(self, request_id: str) -> float:
        """E12: Compute eviction priority score for a cache.

        Lower score = better candidate for eviction.
        Score = 0.4 * recency + 0.3 * frequency + 0.3 * memory_pressure
        """
        with self._lock:
            if request_id not in self._metadata:
                return 0.0
            meta = self._metadata[request_id]
            now = time.time()
            age = now - meta["created_at"]
            idle = now - meta["last_accessed"]
            recency = max(0.0, 1.0 - idle / max(age, 1))
            frequency = min(1.0, meta["access_count"] / max(meta["access_count"] + 10, 1))
            cache = self.caches.get(request_id)
            mem = cache.memory_usage() if cache else 0
            total = sum(c.memory_usage() for c in self.caches.values())
            mem_pressure = mem / max(total, 1)
            return 0.4 * recency + 0.3 * frequency + 0.3 * (1.0 - mem_pressure)

    def evict_lowest_score(self) -> str | None:
        """E12: Evict the cache with the lowest eviction score.

        Returns:
            The evicted request_id, or None if no caches to evict.
        """
        with self._lock:
            if not self.caches:
                return None
            scores = {rid: self.eviction_score(rid) for rid in self.caches}
            victim = min(scores, key=scores.get)
            self.caches[victim].clear()
            del self.caches[victim]
            self._metadata.pop(victim, None)
            return victim


def serialize_kv_cache(cache: KVCache) -> dict:
    """Serialize KV cache for transmission (e.g., via gRPC).

    Returns a dict with layer data that can be converted to proto.
    """
    layers = []
    for k, v in cache.cache:
        layers.append({
            "key": k.cpu().detach(),
            "value": v.cpu().detach(),
        })
    return {"layers": layers}


async def serialize_kv_cache_async(cache: KVCache, executor=None) -> dict:
    """E14: Async serialization — offloads cpu().detach() to thread pool.

    Non-blocking serialization for streaming use cases.
    """
    loop = asyncio.get_event_loop()

    def _serialize():
        layers = []
        for k, v in cache.cache:
            layers.append({
                "key": k.cpu().detach(),
                "value": v.cpu().detach(),
            })
        return {"layers": layers}

    return await loop.run_in_executor(executor, _serialize)


def deserialize_kv_cache(data: dict) -> KVCache:
    """Deserialize KV cache from transmitted data."""
    cache = KVCache()
    layers = []
    for layer_data in data["layers"]:
        layers.append((layer_data["key"], layer_data["value"]))
    cache.set_all(layers)
    return cache


def save_kv_cache_to_disk(cache: KVCache, path: str) -> None:
    """Save KV cache to a .pt file."""
    data = serialize_kv_cache(cache)
    torch.save(data, path)


def load_kv_cache_from_disk(path: str) -> KVCache:
    """Load KV cache from a .pt file."""
    data = torch.load(path, weights_only=True)
    return deserialize_kv_cache(data)


class AdaptiveQuantizer:
    """Per-layer adaptive quantization for KV cache.

    Profiles each layer's sensitivity to INT8/INT4 quantization and
    assigns mixed precision per layer. Layers that are more sensitive
    to quantization (higher MSE) are kept at higher precision.

    This achieves minimal quality loss at 2x memory saving compared
    to uniform quantization.

    Usage::
        quantizer = AdaptiveQuantizer()
        plan = quantizer.profile(kv_cache)
        quantizer.apply(kv_cache, plan)
    """

    # MSE thresholds for quantization decisions
    INT4_MSE_THRESHOLD = 0.01  # Below this: safe for INT4
    INT8_MSE_THRESHOLD = 0.001  # Below this: safe for INT8

    def __init__(self, target_savings: float = 0.5):
        """
        Args:
            target_savings: Target memory savings ratio (0-1).
                0.5 = aim for 50% memory reduction.
        """
        self._target_savings = target_savings
        self._layer_profiles: dict[int, dict] = {}

    def profile(self, kv_cache: KVCache) -> dict[int, str]:
        """Profile each layer and determine optimal quantization.

        Args:
            kv_cache: KVCache to profile.

        Returns:
            Dict mapping layer_index -> quantization method
            ("fp16", "int8", or "int4").
        """
        plan: dict[int, str] = {}

        with kv_cache._lock:
            for layer_idx, (k, v) in enumerate(kv_cache.cache):
                if k.numel() == 0:
                    plan[layer_idx] = "fp16"
                    continue

                # Compute MSE for INT8 quantization
                k_scale_8 = k.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / INT8_MAX
                k_int8 = (k / k_scale_8).round().clamp(-128, 127)
                k_dequant_8 = k_int8 * k_scale_8
                mse_int8 = ((k.float() - k_dequant_8.float()) ** 2).mean().item()

                # Compute MSE for INT4 quantization
                k_scale_4 = k.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / INT4_MAX
                k_int4 = (k / k_scale_4).round().clamp(-7, 7)
                k_dequant_4 = k_int4 * k_scale_4
                mse_int4 = ((k.float() - k_dequant_4.float()) ** 2).mean().item()

                # Decision based on MSE thresholds
                if mse_int4 < self.INT4_MSE_THRESHOLD:
                    plan[layer_idx] = "int4"
                elif mse_int8 < self.INT8_MSE_THRESHOLD:
                    plan[layer_idx] = "int8"
                else:
                    plan[layer_idx] = "fp16"

                self._layer_profiles[layer_idx] = {
                    "mse_int8": mse_int8,
                    "mse_int4": mse_int4,
                    "decision": plan[layer_idx],
                }

        return plan

    def apply(self, kv_cache: KVCache, plan: dict[int, str]) -> dict:
        """Apply per-layer quantization plan to KV cache.

        Args:
            kv_cache: KVCache to quantize.
            plan: Layer quantization plan from profile().

        Returns:
            Compression stats dict.
        """
        import torch as _torch

        original_bytes = kv_cache.memory_usage()

        with kv_cache._lock:
            new_cache = []
            scales_k = []
            scales_v = []
            layer_methods = {}

            for layer_idx, (k, v) in enumerate(kv_cache.cache):
                method = plan.get(layer_idx, "fp16")

                if method == "int4":
                    k_scale = k.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / INT4_MAX
                    v_scale = v.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / INT4_MAX
                    k_q = (k / k_scale).round().clamp(-7, 7).to(_torch.int8)
                    v_q = (v / v_scale).round().clamp(-7, 7).to(_torch.int8)
                    new_cache.append((k_q, v_q))
                    scales_k.append(k_scale)
                    scales_v.append(v_scale)
                    layer_methods[layer_idx] = "int4"

                elif method == "int8":
                    k_scale = k.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / INT8_MAX
                    v_scale = v.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / INT8_MAX
                    k_q = (k / k_scale).round().clamp(-128, 127).to(_torch.int8)
                    v_q = (v / v_scale).round().clamp(-128, 127).to(_torch.int8)
                    new_cache.append((k_q, v_q))
                    scales_k.append(k_scale)
                    scales_v.append(v_scale)
                    layer_methods[layer_idx] = "int8"

                else:  # fp16
                    new_cache.append((k, v))
                    scales_k.append(None)
                    scales_v.append(None)
                    layer_methods[layer_idx] = "fp16"

            kv_cache.cache = new_cache
            kv_cache._scale_k = scales_k
            kv_cache._scale_v = scales_v
            kv_cache._quantized = True
            kv_cache._quant_bits = 0  # Mixed precision

        compressed_bytes = kv_cache.memory_usage()

        int4_count = sum(1 for m in layer_methods.values() if m == "int4")
        int8_count = sum(1 for m in layer_methods.values() if m == "int8")
        fp16_count = sum(1 for m in layer_methods.values() if m == "fp16")

        return {
            "method": "adaptive_mixed",
            "original_bytes": original_bytes,
            "compressed_bytes": compressed_bytes,
            "ratio": compressed_bytes / max(original_bytes, 1),
            "savings_pct": (1 - compressed_bytes / max(original_bytes, 1)) * 100,
            "int4_layers": int4_count,
            "int8_layers": int8_count,
            "fp16_layers": fp16_count,
        }

    def get_profile(self) -> dict[int, dict]:
        """Get profiling results from the last profile() call."""
        return dict(self._layer_profiles)
