from __future__ import annotations

from typing import Any

import torch
from loguru import logger


def extract_kv_cache(coord) -> list[tuple[torch.Tensor, torch.Tensor]] | None:
    """Extract KV cache tensors from a coordinator's model after prefill."""
    try:
        model = coord.model
        if model is None:
            return None

        kv_layers: list[tuple[torch.Tensor, torch.Tensor]] = []
        for name, module in model.named_modules():
            if hasattr(module, "self_attn") and hasattr(module.self_attn, "past_key_value"):
                kv = module.self_attn.past_key_value
                if kv is not None and isinstance(kv, (list, tuple)) and len(kv) >= 2:
                    kv_layers.append((kv[0], kv[1]))

        return kv_layers if kv_layers else None
    except Exception as e:
        logger.debug(f"Failed to extract KV cache: {e}")
        return None


class KVCacheStore:
    """In-memory KV cache store with TTL expiry.

    For distributed deployment, this is replaced by RDMA/NVLink transfer
    from prefill nodes to decode nodes.
    """

    def __init__(self, default_ttl_secs: float = 300.0):
        self._cache: dict[str, Any] = {}
        self._ttl: dict[str, float] = {}
        self._default_ttl_secs = default_ttl_secs
        self._lock = None

    def store(self, request_id: str, kv_cache: Any, ttl_secs: float | None = None) -> None:
        self._cache[request_id] = kv_cache
        self._ttl[request_id] = __import__("time").time() + (ttl_secs or self._default_ttl_secs)

    def get(self, request_id: str) -> Any:
        entry = self._cache.get(request_id)
        if entry is None:
            return None
        if __import__("time").time() > self._ttl.get(request_id, 0):
            self.remove(request_id)
            return None
        return entry

    def remove(self, request_id: str) -> None:
        self._cache.pop(request_id, None)
        self._ttl.pop(request_id, None)

    def sweep_expired(self) -> int:
        now = __import__("time").time()
        expired = [rid for rid, expiry in self._ttl.items() if expiry < now]
        for rid in expired:
            self.remove(rid)
        return len(expired)

    def size(self) -> int:
        self.sweep_expired()
        return len(self._cache)

    def clear(self) -> None:
        self._cache.clear()
        self._ttl.clear()
