"""KV cache management for distributed LLM inference."""

from typing import Any

import torch

from distllm.communication.serializers import tensor_to_proto


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
    """

    def __init__(self):
        self.cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.num_layers = 0
        # Quantization state
        self._quantized: bool = False
        self._quant_bits: int = 8
        # Quantized segments stored as list of (qk, qv, sk, sv) per layer
        self._qsegments: list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]] = []

    def init_cache(self, num_layers: int, batch_size: int, num_heads: int, head_dim: int, device: str = "cpu"):
        """Initialize empty KV cache for all layers."""
        self.cache = []
        self.num_layers = num_layers
        for _ in range(num_layers):
            k = torch.zeros(batch_size, num_heads, 0, head_dim, device=device)
            v = torch.zeros(batch_size, num_heads, 0, head_dim, device=device)
            self.cache.append((k, v))

    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Get cached key/value states for a layer."""
        if layer_idx >= len(self.cache):
            return None
        return self.cache[layer_idx]

    def get_all(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Get all cached key/value states."""
        return self.cache

    def set_all(self, cache: list[tuple[torch.Tensor, torch.Tensor]]):
        """Replace entire cache."""
        # Explicitly clear old cache tensors to release GPU memory
        self.cache.clear()
        self.cache = cache
        self.num_layers = len(cache)

    def to(self, device: str) -> "KVCache":
        """Move cache to device."""
        new_cache = KVCache()
        new_cache.cache = [(k.to(device), v.to(device)) for k, v in self.cache]
        new_cache.num_layers = self.num_layers
        return new_cache

    @property
    def sequence_length(self) -> int:
        """Get current sequence length from cache."""
        if not self.cache:
            return 0
        return self.cache[0][0].shape[-2]

    def clear(self):
        """Clear the cache and quantization state."""
        self.cache = []
        self.num_layers = 0
        self._qsegments = []

    def memory_usage(self) -> int:
        """Get memory usage in bytes."""
        total = 0
        for k, v in self.cache:
            total += k.element_size() * k.numel() + v.element_size() * v.numel()
        if self._quantized:
            for segs in self._qsegments:
                for _qk, _qv, sk, sv in segs:
                    total += sk.element_size() * sk.numel()
                    total += sv.element_size() * sv.numel()
        return total

    def enable_quantization(self, bits: int = 8) -> None:
        """Enable KV cache quantization.

        Args:
            bits: Target bit width (4 or 8).
        """
        if bits not in (4, 8):
            raise ValueError(f"KV cache quantization bits must be 4 or 8, got {bits}")
        self._quantized = True
        self._quant_bits = bits

    def update(self, layer_idx: int, new_key: torch.Tensor, new_value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Update KV cache for a layer with optional quantization.

        Args:
            layer_idx: Layer index.
            new_key: New key tensor to append.
            new_value: New value tensor to append.

        Returns:
            Updated (key, value) tuple for the layer.
        """
        if self._quantized:
            return self._update_quantized(layer_idx, new_key, new_value)

        # Original unquantized path

        if layer_idx >= len(self.cache):
            self.cache.append((new_key, new_value))
            return new_key, new_value

        old_k, old_v = self.cache[layer_idx]
        key = torch.cat([old_k, new_key], dim=-2)
        value = torch.cat([old_v, new_value], dim=-2)
        self.cache[layer_idx] = (key, value)
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
        qk, sk = apply_kv_cache_quantization(new_key, None, self._quant_bits)
        qv, sv = apply_kv_cache_quantization(None, new_value, self._quant_bits)

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

    def get_quantized(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get dequantized KV cache for a layer.

        Returns the incrementally maintained dequantized cache directly.
        Returns:
            (key, value) tensors in original dtype.
        """
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

    def to_proto(self):
        """Serialize KVCache to protobuf format.

        Handles both quantized and unquantized caches.
        Quantized caches serialize from stored segments.
        """
        from distllm.communication.node_pb2 import KVCache as ProtoKVCache, KVLayerCache

        layers = []
        if self._quantized:
            for segs in self._qsegments:
                layer_msg = KVLayerCache()
                for qk, qv, sk, sv in segs:
                    pk = tensor_to_proto(qk)
                    pk.scale.extend(sk.flatten().tolist())
                    pv = tensor_to_proto(qv)
                    pv.scale.extend(sv.flatten().tolist())
                    layer_msg.key_states.CopyFrom(pk)
                    layer_msg.value_states.CopyFrom(pv)
                layers.append(layer_msg)
        else:
            for k, v in self.cache:
                layer_msg = KVLayerCache()
                layer_msg.key_states.CopyFrom(tensor_to_proto(k))
                layer_msg.value_states.CopyFrom(tensor_to_proto(v))
                layers.append(layer_msg)
        proto = ProtoKVCache(layers=layers)
        if self._quantized:
            proto.quant_bits = self._quant_bits
        return proto

    @staticmethod
    def from_proto(proto, device: str = "cpu"):
        """Deserialize KVCache from protobuf format.

        Restores quantization state if the serialized cache was quantized.
        """
        from distllm.communication.serializers import proto_to_tensor

        cache = KVCache()
        if hasattr(proto, 'quant_bits') and proto.quant_bits > 0:
            cache._quantized = True
            cache._quant_bits = proto.quant_bits
        for layer in proto.layers:
            k = proto_to_tensor(layer.key_states, device)
            v = proto_to_tensor(layer.value_states, device)
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
        self.caches[request_id] = cache
        return cache

    def get(self, request_id: str) -> KVCache | None:
        """Get KV cache for a request."""
        return self.caches.get(request_id)

    def update(self, request_id: str, layer_idx: int, new_key: torch.Tensor, new_value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Update KV cache for a request."""
        cache = self.caches.get(request_id)
        if cache is None:
            return None
        return cache.update(layer_idx, new_key, new_value)

    def delete(self, request_id: str):
        """Delete KV cache for a request."""
        if request_id in self.caches:
            self.caches[request_id].clear()
            del self.caches[request_id]

    def clear_all(self):
        """Clear all caches."""
        for cache in self.caches.values():
            cache.clear()
        self.caches = {}

    @property
    def active_requests(self) -> int:
        return len(self.caches)

    def total_memory_usage(self) -> int:
        """Total memory usage across all caches."""
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
