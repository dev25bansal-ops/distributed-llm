"""Disaggregated Prefill + Decode — node pools.

Contains ``PrefillPool`` and ``DecodePool`` that manage GPU nodes
dedicated to each phase.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from loguru import logger


class PoolRole(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass
class NodeRegistration:
    node_id: str
    address: str
    role: PoolRole
    max_num_seqs: int = 8
    gpu_memory_gb: float = 0.0
    healthy: bool = True
    active_requests: int = 0
    last_seen: float = field(default_factory=time.time)


class PrefillPool:
    """Pool of GPU nodes dedicated to the prefill phase.

    Prefill is compute-bound — these nodes run with large batch sizes
    (``max_num_seqs=32+``) and low KV cache capacity.
    """

    def __init__(self, max_concurrent_prefills: int = 16):
        self._max_concurrent = max_concurrent_prefills
        self._nodes: dict[str, NodeRegistration] = {}
        self._lock = threading.RLock()
        self._semaphore = asyncio.Semaphore(max_concurrent_prefills)

    def register_node(self, node_id: str, address: str,
                      max_num_seqs: int = 32, gpu_memory_gb: float = 0.0) -> None:
        with self._lock:
            self._nodes[node_id] = NodeRegistration(
                node_id=node_id, address=address,
                role=PoolRole.PREFILL, max_num_seqs=max_num_seqs,
                gpu_memory_gb=gpu_memory_gb,
            )

    def remove_node(self, node_id: str) -> None:
        with self._lock:
            self._nodes.pop(node_id, None)

    def mark_healthy(self, node_id: str) -> None:
        with self._lock:
            node = self._nodes.get(node_id)
            if node:
                node.healthy = True

    def mark_unhealthy(self, node_id: str) -> None:
        with self._lock:
            node = self._nodes.get(node_id)
            if node:
                node.healthy = False

    async def acquire(self, node_id: str) -> bool:
        """Reserve one slot on *node_id* (False if unknown or full)."""
        async with self._semaphore:
            with self._lock:
                node = self._nodes.get(node_id)
                if node is None or not node.healthy or node.active_requests >= node.max_num_seqs:
                    return False
                node.active_requests += 1
                return True

    def release(self, node_id: str) -> None:
        self.release_node(node_id)

    @property
    def total_nodes(self) -> int:
        with self._lock:
            return len(self._nodes)

    @property
    def available_nodes(self) -> int:
        with self._lock:
            return sum(1 for n in self._nodes.values() if n.healthy)

    @property
    def active_count(self) -> int:
        with self._lock:
            return sum(n.active_requests for n in self._nodes.values())

    async def select_node(self) -> Optional[NodeRegistration]:
        with self._lock:
            candidates = [
                n for n in self._nodes.values()
                if n.healthy and n.active_requests < n.max_num_seqs
            ]
            if not candidates:
                return None
            selected = min(candidates, key=lambda n: n.active_requests)
            selected.active_requests += 1
            return selected

    def release_node(self, node_id: str) -> None:
        with self._lock:
            node = self._nodes.get(node_id)
            if node:
                node.active_requests = max(0, node.active_requests - 1)

    @property
    def available_capacity(self) -> int:
        with self._lock:
            return sum(
                max(0, n.max_num_seqs - n.active_requests)
                for n in self._nodes.values() if n.healthy
            )


class DecodePool:
    """Pool of GPU nodes dedicated to the decode phase.

    Decode is memory-bound — these nodes run with small batch sizes
    (``max_num_seqs=4``) but large KV cache capacity (80%+ of GPU memory).
    """

    def __init__(self, max_concurrent_decodes: int = 64):
        self._max_concurrent = max_concurrent_decodes
        self._nodes: dict[str, NodeRegistration] = {}
        self._kv_store: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def register_node(self, node_id: str, address: str,
                      max_num_seqs: int = 4, gpu_memory_gb: float = 80.0) -> None:
        with self._lock:
            self._nodes[node_id] = NodeRegistration(
                node_id=node_id, address=address,
                role=PoolRole.DECODE, max_num_seqs=max_num_seqs,
                gpu_memory_gb=gpu_memory_gb,
            )

    def remove_node(self, node_id: str) -> None:
        with self._lock:
            self._nodes.pop(node_id, None)

    def mark_healthy(self, node_id: str) -> None:
        with self._lock:
            node = self._nodes.get(node_id)
            if node:
                node.healthy = True

    def mark_unhealthy(self, node_id: str) -> None:
        with self._lock:
            node = self._nodes.get(node_id)
            if node:
                node.healthy = False

    def get_node_by_handle(self, node_id: str) -> Optional[NodeRegistration]:
        with self._lock:
            return self._nodes.get(node_id)

    @property
    def total_nodes(self) -> int:
        with self._lock:
            return len(self._nodes)

    @property
    def active_count(self) -> int:
        with self._lock:
            return sum(n.active_requests for n in self._nodes.values())

    async def select_node(self) -> Optional[NodeRegistration]:
        with self._lock:
            candidates = [
                n for n in self._nodes.values()
                if n.healthy and n.active_requests < n.max_num_seqs
            ]
            if not candidates:
                return None
            selected = min(candidates, key=lambda n: n.active_requests)
            selected.active_requests += 1
            return selected

    def store_kv_cache(self, request_id: str, kv_data: dict[str, Any],
                       node_id: str) -> None:
        with self._lock:
            self._kv_store[request_id] = {
                "kv_data": kv_data,
                "node_id": node_id,
                "stored_at": time.time(),
            }

    def lookup_kv_cache(self, request_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            return self._kv_store.get(request_id)

    def release_node(self, node_id: str) -> None:
        with self._lock:
            node = self._nodes.get(node_id)
            if node:
                node.active_requests = max(0, node.active_requests - 1)

    def evict_kv_cache(self, request_id: str) -> None:
        with self._lock:
            self._kv_store.pop(request_id, None)
