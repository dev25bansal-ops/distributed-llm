"""KV cache management for distributed LLM inference."""

import threading

import numpy as np
import torch
from loguru import logger


class PagedKVCacheBackend:
    """Paged KV cache backend using block-based allocation.

    Wraps PagedAttentionManager to provide a KVCache-compatible interface
    while using paged memory for O(1) allocation and automatic defragmentation.
    """

    def __init__(self, paged_mgr: object | None = None):
        self._paged_mgr = paged_mgr
        self._request_id: str | None = None

    def attach(self, request_id: str) -> None:
        self._request_id = request_id
        if self._paged_mgr is not None:
            self._paged_mgr.create_sequence(request_id)

    def append_kv(self, layer_idx: int, new_key: torch.Tensor, new_value: torch.Tensor) -> None:
        if self._paged_mgr is not None and self._request_id is not None:
            self._paged_mgr.free_layer_kv(self._request_id, layer_idx, new_key, new_value)

    def get_kv(self, request_id: str, layer_idx: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._paged_mgr is not None:
            return self._paged_mgr.gather_kv_for_attention(request_id, layer_idx, seq_len)
        raise RuntimeError("Paged backend not available")

    def free(self, request_id: str) -> None:
        if self._paged_mgr is not None:
            self._paged_mgr.free_sequence(request_id)

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
            if not self.cache:
                return 0
            return self.cache[0][0].shape[-2]

    def clear(self):
        """Clear the cache and quantization state."""
        with self._lock:
            self.cache = []
            self.num_layers = 0
            self._qsegments = []
            self._fp8_segments = []
            self._quant_fp8 = False

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

    def enable_quantization(self, bits: int = 8, use_fp8: bool = False) -> None:
        """Enable KV cache quantization.

        Args:
            bits: Target bit width (4 or 8). Ignored if use_fp8=True.
            use_fp8: Use FP8 E4M3 quantization for 2x capacity vs fp16.
        """
        with self._lock:
            if use_fp8:
                logger.warning("FP8 quantization not available, falling back to int8")
                bits = 8
            if bits not in (4, 8):
                raise ValueError(f"KV cache quantization bits must be 4 or 8, got {bits}")
            self._quantized = True
            self._quant_bits = bits

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
                return self._update_fp8(layer_idx, new_key, new_value)
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

    def _update_fp8(self, layer_idx: int, new_key: torch.Tensor, new_value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """FP8 not available; fall back to non-quantized cat."""
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

    @staticmethod
    def from_proto(proto, device: str = "cpu"):
        """Deserialize KVCache from protobuf format (legacy, gRPC removed)."""
        cache = KVCache()
        if hasattr(proto, 'quant_bits') and proto.quant_bits > 0:
            cache._quantized = True
            cache._quant_bits = proto.quant_bits
        for layer in proto.layers:
            k = _proto_to_tensor(layer.key_states, device)
            v = _proto_to_tensor(layer.value_states, device)
            if cache._quantized and layer.key_states.scale:
                sk = torch.tensor(list(layer.key_states.scale), device=device)
                sv = torch.tensor(list(layer.value_states.scale), device=device)
                cache._qsegments.append([(k, v, sk, sv)])
            cache.cache.append((k, v))
        cache.num_layers = len(cache.cache)
        return cache


class KVCacheManager:
    """Manages KV caches for multiple concurrent requests."""

    def __init__(self):
        self.caches: dict[str, KVCache] = {}
        self._lock = threading.Lock()

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
        return cache

    def get(self, request_id: str) -> KVCache | None:
        """Get KV cache for a request."""
        with self._lock:
            return self.caches.get(request_id)

    def update(self, request_id: str, layer_idx: int, new_key: torch.Tensor, new_value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Update KV cache for a request."""
        with self._lock:
            cache = self.caches.get(request_id)
        if cache is None:
            return None
        return cache.update(layer_idx, new_key, new_value)

    def delete(self, request_id: str):
        """Delete KV cache for a request."""
        with self._lock:
            if request_id in self.caches:
                self.caches[request_id].clear()
                del self.caches[request_id]

    def clear_all(self):
        """Clear all caches."""
        with self._lock:
            for cache in self.caches.values():
                cache.clear()
            self.caches = {}

    @property
    def active_requests(self) -> int:
        with self._lock:
            return len(self.caches)

    def total_memory_usage(self) -> int:
        """Total memory usage across all caches."""
        with self._lock:
            return sum(cache.memory_usage() for cache in self.caches.values())


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


# ── Inline helpers (tensor transport) ────────────────────────────────────

def _tensor_to_bytes(tensor):
    """Convert torch.Tensor to bytes for transport."""
    t = tensor.detach().cpu().contiguous()
    return t.view(torch.uint8).numpy(force=True).tobytes(), list(tensor.shape), str(tensor.dtype)


def _bytes_to_tensor(data, shape, dtype_str, device="cpu"):
    """Convert bytes back to torch.Tensor."""
    dtype_map = {"torch.float32": torch.float32, "torch.float16": torch.float16,
                 "torch.bfloat16": torch.bfloat16, "torch.int64": torch.int64,
                 "torch.int32": torch.int32, "torch.uint8": torch.uint8,
                 "torch.bool": torch.bool}
    torch_dtype = dtype_map.get(dtype_str, torch.float32)
    arr = np.frombuffer(data, dtype=np.uint8)
    return torch.from_numpy(arr).view(torch_dtype).reshape(shape).clone().to(device)
