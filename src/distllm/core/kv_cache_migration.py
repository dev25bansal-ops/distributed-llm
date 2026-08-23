"""CPU/GPU migration helpers for KVCache.

Extracted from ``KVCache`` in ``kv_cache.py`` to reduce the file
below the 800-line ceiling.  Each function takes the KVCache instance
as the first parameter (``cache``) so it can access internal state
exactly as the original ``self``-based methods did.
"""

from __future__ import annotations

import gc as _gc

import torch


def offload_to_cpu(cache, non_blocking: bool = True) -> int:
    """Offload KV cache from GPU to CPU RAM.

    Moves all cache tensors to CPU to free GPU memory.

    Returns:
        Bytes offloaded.
    """
    with cache._lock:
        if cache._offloaded:
            return 0

        bytes_offloaded = 0
        cpu_cache = []
        for k, v in cache.cache:
            if k.is_cuda:
                k_cpu = k.to("cpu", non_blocking=non_blocking)
                v_cpu = v.to("cpu", non_blocking=non_blocking)
                cpu_cache.append((k_cpu, v_cpu))
                bytes_offloaded += k.element_size() * k.numel() + v.element_size() * v.numel()
            else:
                cpu_cache.append((k, v))

        cache._cpu_cache = cpu_cache
        cache.cache.clear()
        _gc.collect()
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
        cache._offloaded = True
        return bytes_offloaded


def load_to_gpu(cache, device: str = "cuda", non_blocking: bool = True) -> int:
    """Load KV cache from CPU back to GPU.

    Returns:
        Bytes loaded.
    """
    with cache._lock:
        if not cache._offloaded or cache._cpu_cache is None:
            return 0

        bytes_loaded = 0
        gpu_cache = []
        for k, v in cache._cpu_cache:
            if k.device.type == "cpu":
                k_gpu = k.to(device, non_blocking=non_blocking)
                v_gpu = v.to(device, non_blocking=non_blocking)
                gpu_cache.append((k_gpu, v_gpu))
                bytes_loaded += k.element_size() * k.numel() + v.element_size() * v.numel()
            else:
                gpu_cache.append((k, v))

        cache.cache = gpu_cache
        cache._cpu_cache = None
        cache._offloaded = False
        return bytes_loaded


def pin_memory_for_cache(cache) -> None:
    """Pin CPU memory for faster GPU transfers."""
    with cache._lock:
        if cache._offloaded and cache._cpu_cache is not None:
            cache._cpu_cache = [
                (k.pin_memory(), v.pin_memory()) for k, v in cache._cpu_cache
            ]
        elif not cache._offloaded:
            cache.cache = [
                (k.pin_memory(), v.pin_memory()) for k, v in cache.cache
            ]


def offload_layer(cache, layer_idx: int, non_blocking: bool = True) -> int:
    """Offload a single layer's KV cache from GPU to CPU."""
    with cache._lock:
        if layer_idx >= len(cache.cache):
            return 0
        k, v = cache.cache[layer_idx]
        if not k.is_cuda:
            return 0
        k_cpu = k.to("cpu", non_blocking=non_blocking)
        v_cpu = v.to("cpu", non_blocking=non_blocking)
        cache.cache[layer_idx] = (k_cpu, v_cpu)
        return k.element_size() * k.numel() + v.element_size() * v.numel()


def load_layer(cache, layer_idx: int, device: str = "cuda", non_blocking: bool = True) -> int:
    """Load a single layer's KV cache from CPU to GPU."""
    with cache._lock:
        if layer_idx >= len(cache.cache):
            return 0
        k, v = cache.cache[layer_idx]
        if k.device.type != "cpu":
            return 0
        k_gpu = k.to(device, non_blocking=non_blocking)
        v_gpu = v.to(device, non_blocking=non_blocking)
        cache.cache[layer_idx] = (k_gpu, v_gpu)
        return k.element_size() * k.numel() + v.element_size() * v.numel()


def get_layer_device(cache, layer_idx: int) -> str | None:
    """Return the device of a specific layer, or None if invalid."""
    with cache._lock:
        if layer_idx >= len(cache.cache):
            return None
        return str(cache.cache[layer_idx][0].device)


def offload_layers_to_cpu(cache, layer_indices: list[int], non_blocking: bool = True) -> int:
    """Offload multiple layers to CPU."""
    total = 0
    for idx in layer_indices:
        total += offload_layer(cache, idx, non_blocking=non_blocking)
    return total


def load_layers_to_gpu(cache, layer_indices: list[int], device: str = "cuda", non_blocking: bool = True) -> int:
    """Load multiple layers to GPU."""
    total = 0
    for idx in layer_indices:
        total += load_layer(cache, idx, device=device, non_blocking=non_blocking)
    return total
