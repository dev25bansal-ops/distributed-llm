"""KV cache management for distributed LLM inference."""

import torch
from typing import Dict, Optional, Tuple, List
from loguru import logger


class KVCache:
    """Manages key-value cache for a single generation request.

    Stores past_key_values for each transformer layer, enabling
    efficient autoregressive generation without re-processing tokens.
    """

    def __init__(self):
        self.cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.num_layers = 0
        # Quantization state
        self._quantized: bool = False
        self._quant_bits: int = 8
        self._scales_k: List[torch.Tensor] = []
        self._scales_v: List[torch.Tensor] = []

    def init_cache(self, num_layers: int, batch_size: int, num_heads: int, head_dim: int, device: str = "cpu"):
        """Initialize empty KV cache for all layers."""
        self.cache = []
        self.num_layers = num_layers
        for _ in range(num_layers):
            k = torch.zeros(batch_size, num_heads, 0, head_dim, device=device)
            v = torch.zeros(batch_size, num_heads, 0, head_dim, device=device)
            self.cache.append((k, v))

    def update(self, layer_idx: int, new_key: torch.Tensor, new_value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append new key/value states for a layer, return updated states."""
        if layer_idx >= len(self.cache):
            raise IndexError(f"Layer {layer_idx} out of range for KV cache with {len(self.cache)} layers")

        k_cache, v_cache = self.cache[layer_idx]

        # Concatenate new states
        k = torch.cat([k_cache, new_key], dim=-2)
        v = torch.cat([v_cache, new_value], dim=-2)

        self.cache[layer_idx] = (k, v)
        return k, v

    def get(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get cached key/value states for a layer."""
        if layer_idx >= len(self.cache):
            return None
        return self.cache[layer_idx]

    def get_all(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Get all cached key/value states."""
        return self.cache

    def set_all(self, cache: List[Tuple[torch.Tensor, torch.Tensor]]):
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
        """Clear the cache."""
        self.cache = []
        self.num_layers = 0

    def memory_usage(self) -> int:
        """Get memory usage in bytes."""
        total = 0
        for k, v in self.cache:
            total += k.element_size() * k.numel() + v.element_size() * v.numel()
        if self._quantized:
            # Add scale tensor memory
            for s in self._scales_k:
                total += s.element_size() * s.numel()
            for s in self._scales_v:
                total += s.element_size() * s.numel()
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

    def update(self, layer_idx: int, new_key: torch.Tensor, new_value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        from distllm.core.quantization_selector import apply_kv_cache_quantization, dequantize_kv_cache

        if layer_idx >= len(self.cache):
            self.cache.append((new_key, new_value))
            return new_key, new_value

        old_k, old_v = self.cache[layer_idx]
        key = torch.cat([old_k, new_key], dim=-2)
        value = torch.cat([old_v, new_value], dim=-2)
        self.cache[layer_idx] = (key, value)
        return key, value

    def _update_quantized(self, layer_idx: int, new_key: torch.Tensor, new_value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update KV cache with quantization for memory efficiency."""
        from distllm.core.quantization_selector import apply_kv_cache_quantization, dequantize_kv_cache

        # Quantize new tensors
        qk, sk = apply_kv_cache_quantization(new_key, new_value, self._quant_bits)
        qv, sv = apply_kv_cache_quantization(new_key, new_value, self._quant_bits)

        if layer_idx >= len(self._scales_k):
            # First update for this layer
            self._scales_k.append(sk)
            self._scales_v.append(sv)
            self.cache.append((qk, qv))
        else:
            # Dequantize existing, append, re-quantize
            old_k = dequantize_kv_cache(self.cache[layer_idx][0], self._scales_k[layer_idx], self._quant_bits)
            old_v = dequantize_kv_cache(self.cache[layer_idx][1], self._scales_v[layer_idx], self._quant_bits)

            key = torch.cat([old_k, new_key], dim=-2)
            value = torch.cat([old_v, new_value], dim=-2)

            # Re-quantize the combined tensor
            qk, sk = apply_kv_cache_quantization(key, value, self._quant_bits)
            qv, sv = apply_kv_cache_quantization(key, value, self._quant_bits)

            self.cache[layer_idx] = (qk, qv)
            self._scales_k[layer_idx] = sk
            self._scales_v[layer_idx] = sv

        return key, value

    def get_quantized(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get dequantized KV cache for a layer.

        Returns:
            (key, value) tensors in original dtype.
        """
        from distllm.core.quantization_selector import dequantize_kv_cache

        if not self._quantized:
            return self.cache[layer_idx]

        qk, qv = self.cache[layer_idx]
        key = dequantize_kv_cache(qk, self._scales_k[layer_idx], self._quant_bits)
        value = dequantize_kv_cache(qv, self._scales_v[layer_idx], self._quant_bits)
        return key, value

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

    def clear(self):
        """Clear the cache and quantization state."""
        self.cache = []
        self.num_layers = 0
        self._scales_k = []
        self._scales_v = []


class KVCacheManager:
    """Manages KV caches for multiple concurrent requests."""

    def __init__(self):
        self.caches: Dict[str, KVCache] = {}

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

    def get(self, request_id: str) -> Optional[KVCache]:
        """Get KV cache for a request."""
        return self.caches.get(request_id)

    def update(self, request_id: str, layer_idx: int, new_key: torch.Tensor, new_value: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
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
    data = torch.load(path, weights_only=False)
    return deserialize_kv_cache(data)
