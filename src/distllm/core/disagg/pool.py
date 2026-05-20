from __future__ import annotations

import asyncio

from loguru import logger

from distllm.core.disagg.types import PoolNode, PoolStatus


class PrefillPool:
    def __init__(self, min_nodes: int = 1, max_nodes: int = 16):
        self._nodes: dict[str, PoolNode] = {}
        self._min_nodes = min_nodes
        self._max_nodes = max_nodes
        self._pending: asyncio.Queue | None = None
        self._lock = asyncio.Lock()

    @property
    def _pending_queue(self) -> asyncio.Queue:
        if self._pending is None:
            self._pending = asyncio.Queue()
        return self._pending

    async def register_node(self, node_id: str, host: str, port: int, capacity: int = 4) -> None:
        async with self._lock:
            self._nodes[node_id] = PoolNode(
                node_id=node_id, host=host, port=port, capacity=capacity,
            )
            logger.info(f"Prefill node '{node_id}' registered ({capacity} slots)")

    async def unregister_node(self, node_id: str) -> None:
        async with self._lock:
            self._nodes.pop(node_id, None)
            logger.info(f"Prefill node '{node_id}' unregistered")

    async def select_node(self) -> PoolNode | None:
        async with self._lock:
            candidates = [
                n for n in self._nodes.values()
                if n.status == PoolStatus.ACTIVE and n.current_load < n.capacity
            ]
            if not candidates:
                return None
            candidates.sort(key=lambda n: n.current_load / max(n.capacity, 1))
            node = candidates[0]
            node.current_load += 1
            return node

    async def release_node(self, node_id: str) -> None:
        async with self._lock:
            node = self._nodes.get(node_id)
            if node:
                node.current_load = max(0, node.current_load - 1)

    def get_stats(self) -> dict:
        active = sum(1 for n in self._nodes.values() if n.status == PoolStatus.ACTIVE)
        total_load = sum(n.current_load for n in self._nodes.values())
        total_capacity = sum(n.capacity for n in self._nodes.values())
        return {
            "total_nodes": len(self._nodes),
            "active_nodes": active,
            "total_load": total_load,
            "total_capacity": total_capacity,
            "utilization_pct": round(total_load / max(total_capacity, 1) * 100, 1),
        }


class DecodePool:
    def __init__(self, min_nodes: int = 1, max_nodes: int = 32):
        self._nodes: dict[str, PoolNode] = {}
        self._min_nodes = min_nodes
        self._max_nodes = max_nodes
        self._request_node_map: dict[str, str] = {}
        self._lock = asyncio.Lock()

    def nodes(self) -> dict[str, PoolNode]:
        return dict(self._nodes)

    def request_node_map(self) -> dict[str, str]:
        return dict(self._request_node_map)

    async def register_node(self, node_id: str, host: str, port: int, capacity: int = 8) -> None:
        async with self._lock:
            self._nodes[node_id] = PoolNode(
                node_id=node_id, host=host, port=port, capacity=capacity,
            )

    async def unregister_node(self, node_id: str) -> None:
        async with self._lock:
            self._nodes.pop(node_id, None)

    async def assign_request(self, request_id: str, node_id: str | None = None) -> str | None:
        async with self._lock:
            if node_id and node_id in self._nodes:
                node = self._nodes[node_id]
                if node.current_load < node.capacity and node.status == PoolStatus.ACTIVE:
                    node.current_load += 1
                    self._request_node_map[request_id] = node_id
                    return node_id

            candidates = [
                n for n in self._nodes.values()
                if n.status == PoolStatus.ACTIVE and n.current_load < n.capacity
            ]
            if not candidates:
                return None
            candidates.sort(key=lambda n: n.current_load / max(n.capacity, 1))
            chosen = candidates[0]
            chosen.current_load += 1
            self._request_node_map[request_id] = chosen.node_id
            return chosen.node_id

    async def release_request(self, request_id: str) -> None:
        async with self._lock:
            node_id = self._request_node_map.pop(request_id, None)
            if node_id and node_id in self._nodes:
                self._nodes[node_id].current_load = max(0, self._nodes[node_id].current_load - 1)

    def get_node_for_request(self, request_id: str) -> str | None:
        return self._request_node_map.get(request_id)

    def get_stats(self) -> dict:
        active = sum(1 for n in self._nodes.values() if n.status == PoolStatus.ACTIVE)
        total_load = sum(n.current_load for n in self._nodes.values())
        total_capacity = sum(n.capacity for n in self._nodes.values())
        return {
            "total_nodes": len(self._nodes),
            "active_nodes": active,
            "assigned_requests": len(self._request_node_map),
            "total_load": total_load,
            "total_capacity": total_capacity,
            "utilization_pct": round(total_load / max(total_capacity, 1) * 100, 1),
        }
