"""Disaggregated serving with separate prefill and decode pools.

Architecture:
- Prefill pool: GPU nodes optimized for compute-intensive prompt processing.
  Scales based on prompt throughput (tokens/s).
- Decode pool: GPU nodes optimized for memory-bandwidth-bound token generation.
  Scales based on concurrent user count and generation length.
- Router: Distributes requests based on phase, with cross-pool KV cache transfer.
- KV cache transfer: Prefill nodes push KV cache to decode nodes over RDMA/NVLink.

Benefits:
- Independent scaling: prefill and decode can scale to different pod counts.
- Resource efficiency: prefill GPUs can be H100 (high compute), decode GPUs can be
  lower-cost L40S (high memory bandwidth).
- Latency isolation: long decode streams don't block short prefill bursts.
- Co-location avoidance: no need to over-provision for worst-case combined load.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import torch
from loguru import logger


class DisaggPhase(Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    COMPLETE = "complete"


class PoolStatus(Enum):
    ACTIVE = "active"
    DEGRADED = "degraded"
    DRAINING = "draining"


@dataclass
class PrefillRequest:
    """A request to the prefill pool."""
    request_id: str
    prompt_tokens: list[int]
    max_new_tokens: int = 256
    adapter_id: str | None = None
    priority: int = 2
    created_at: float = field(default_factory=time.time)


@dataclass
class PrefillResult:
    """Result from a prefill node, including initial KV cache."""
    request_id: str
    kv_cache: Any  # KV cache tensor(s) to transfer to decode pool
    prompt_len: int
    first_token: int
    prefill_time_ms: float
    prefill_node_id: str


@dataclass
class DecodeRequest:
    """A request to the decode pool for one step."""
    request_id: str
    input_token: int
    kv_cache: Any
    position: int
    adapter_id: str | None = None


@dataclass
class PoolNode:
    """A worker node in a prefill or decode pool."""
    node_id: str
    host: str
    port: int
    capacity: int = 0  # Max concurrent requests
    current_load: int = 0
    status: PoolStatus = PoolStatus.ACTIVE
    metrics: dict[str, float] = field(default_factory=dict)


class PrefillPool:
    """Manages a pool of prefill worker nodes.

    Prefill is compute-bound (large matmuls for prompt processing).
    Workers in this pool are optimized for high FLOPs utilization.
    """

    def __init__(self, min_nodes: int = 1, max_nodes: int = 16):
        self._nodes: dict[str, PoolNode] = {}
        self._min_nodes = min_nodes
        self._max_nodes = max_nodes
        self._pending: asyncio.Queue = asyncio.Queue()
        self._lock = asyncio.Lock()

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
        """Select least-loaded active node."""
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
    """Manages a pool of decode worker nodes.

    Decode is memory-bandwidth-bound (single token generation with large KV cache).
    Workers in this pool are optimized for high memory bandwidth (HBM3).
    """

    def __init__(self, min_nodes: int = 1, max_nodes: int = 32):
        self._nodes: dict[str, PoolNode] = {}
        self._min_nodes = min_nodes
        self._max_nodes = max_nodes
        self._request_node_map: dict[str, str] = {}  # request_id -> node_id
        self._lock = asyncio.Lock()

    async def register_node(self, node_id: str, host: str, port: int, capacity: int = 8) -> None:
        async with self._lock:
            self._nodes[node_id] = PoolNode(
                node_id=node_id, host=host, port=port, capacity=capacity,
            )

    async def unregister_node(self, node_id: str) -> None:
        async with self._lock:
            self._nodes.pop(node_id, None)

    async def assign_request(self, request_id: str, node_id: str | None = None) -> str | None:
        """Assign a request to a decode node.

        If node_id is specified and has capacity, assigns to that node.
        Otherwise selects the least-loaded node.
        Returns the assigned node_id or None if no capacity.
        """
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


class DisaggRouter:
    """Router for disaggregated serving: prefill pool + decode pool.

    Routes requests to the appropriate pool and manages KV cache transfer
    from prefill nodes to decode nodes.

    Usage:
        router = DisaggRouter()
        router.add_prefill_node("pf-1", "10.0.0.1", 50051)
        router.add_decode_node("dc-1", "10.0.0.2", 50052)

        # On request:
        result = await router.prefill(request)
        token = await router.decode(result)
    """

    def __init__(self):
        self.prefill_pool = PrefillPool()
        self.decode_pool = DecodePool()
        self._active_requests: dict[str, DecodeRequest] = {}
        self._kv_cache_store: dict[str, Any] = {}  # request_id -> KV cache
        self._lock = asyncio.Lock()

    async def add_prefill_node(self, node_id: str, host: str, port: int, capacity: int = 4) -> None:
        await self.prefill_pool.register_node(node_id, host, port, capacity)

    async def remove_prefill_node(self, node_id: str) -> None:
        await self.prefill_pool.unregister_node(node_id)

    async def add_decode_node(self, node_id: str, host: str, port: int, capacity: int = 8) -> None:
        await self.decode_pool.register_node(node_id, host, port, capacity)

    async def remove_decode_node(self, node_id: str) -> None:
        await self.decode_pool.unregister_node(node_id)

    async def prefill(self, request: PrefillRequest) -> PrefillResult | None:
        """Route a request to the prefill pool and return KV cache.

        The prefill node processes the prompt, returns the first token
        and the KV cache for subsequent decode steps.
        """
        node = await self.prefill_pool.select_node()
        if node is None:
            logger.warning("No prefill node available")
            return None

        t0 = time.time()
        try:
            result = await self._send_prefill(node, request)
            transfer_time = time.time() - t0

            # Store KV cache for decode pool
            async with self._lock:
                self._kv_cache_store[request.request_id] = result.kv_cache

            # Assign to decode pool
            decode_node_id = await self.decode_pool.assign_request(request.request_id)
            if decode_node_id:
                await self._transfer_kv_to_decode(
                    result, decode_node_id,
                )

            await self.prefill_pool.release_node(node.node_id)
            return result
        except Exception as e:
            logger.error(f"Prefill failed on node '{node.node_id}': {e}")
            await self.prefill_pool.release_node(node.node_id)
            return None

    async def decode(
        self,
        request_id: str,
        input_token: int,
        position: int,
    ) -> int | None:
        """Execute one decode step on the assigned decode node.

        Args:
            request_id: The request ID from the prefill step.
            input_token: The next token to feed.
            position: The current sequence position.

        Returns:
            The next generated token, or None if failed.
        """
        decode_node_id = self.decode_pool.get_node_for_request(request_id)
        if decode_node_id is None:
            logger.warning(f"No decode node assigned for request '{request_id}'")
            return None

        async with self._lock:
            kv_cache = self._kv_cache_store.get(request_id)

        if kv_cache is None:
            logger.warning(f"No KV cache for request '{request_id}'")
            return None

        try:
            token = await self._send_decode(decode_node_id, request_id, input_token, kv_cache, position)
            return token
        except Exception as e:
            logger.error(f"Decode step failed on node '{decode_node_id}': {e}")
            return None

    async def complete_request(self, request_id: str) -> None:
        """Clean up resources for a completed request."""
        await self.decode_pool.release_request(request_id)
        async with self._lock:
            self._kv_cache_store.pop(request_id, None)

    async def _send_prefill(self, node: PoolNode, request: PrefillRequest) -> PrefillResult:
        """Send prefill request to a worker node.

        In production, this would use gRPC to communicate with the worker.
        """
        import copy
        prompt_len = len(request.prompt_tokens)
        fake_kv = {"key": torch.randn(1, 32, prompt_len, 128), "value": torch.randn(1, 32, prompt_len, 128)}
        first_token = 42
        return PrefillResult(
            request_id=request.request_id,
            kv_cache=fake_kv,
            prompt_len=prompt_len,
            first_token=first_token,
            prefill_time_ms=100.0,
            prefill_node_id=node.node_id,
        )

    async def _transfer_kv_to_decode(self, result: PrefillResult, decode_node_id: str) -> None:
        """Transfer KV cache from prefill to decode node.

        In production, uses RDMA or NVLink for zero-copy transfer.
        """
        pass

    async def _send_decode(
        self, node_id: str, request_id: str, input_token: int,
        kv_cache: Any, position: int,
    ) -> int:
        """Send decode step to a worker node.

        In production, this would use gRPC.
        """
        import random
        return random.randint(0, 32000)

    def get_stats(self) -> dict:
        return {
            "prefill": self.prefill_pool.get_stats(),
            "decode": self.decode_pool.get_stats(),
            "active_requests": len(self._active_requests),
            "kv_cached_requests": len(self._kv_cache_store),
        }


class DisaggOrchestrator:
    """Orchestrates full disaggregated serving lifecycle.

    Manages the end-to-end flow:
    1. Receive request -> route to prefill pool
    2. Prefill processes prompt, returns KV cache
    3. KV cache transferred to decode pool node
    4. Decode loop runs on decode node until complete
    5. Resources released

    Supports async scaling: prefill pool can scale independently from decode pool.
    """

    def __init__(self, router: DisaggRouter):
        self.router = router
        self._running_tasks: dict[str, asyncio.Task] = {}
        self._is_healthy = True

    async def submit(self, prompt_tokens: list[int], max_new_tokens: int = 256, **kwargs) -> str:
        """Submit a generation request through the disagg pipeline.

        Returns a request_id that can be used to poll for results.
        """
        request_id = str(uuid.uuid4())
        request = PrefillRequest(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
        task = asyncio.create_task(self._execute_pipeline(request))
        self._running_tasks[request_id] = task
        return request_id

    async def _execute_pipeline(self, request: PrefillRequest) -> list[int]:
        """Execute the full prefill->decode pipeline for one request."""
        result = await self.router.prefill(request)
        if result is None:
            return []

        generated: list[int] = [result.first_token]
        token = result.first_token
        position = result.prompt_len

        for step in range(1, request.max_new_tokens):
            next_token = await self.router.decode(
                request.request_id, token, position,
            )
            if next_token is None:
                break
            generated.append(next_token)
            token = next_token
            position += 1

        await self.router.complete_request(request.request_id)
        return generated

    async def get_result(self, request_id: str, timeout: float = 30.0) -> list[int] | None:
        """Get the result of a generation request."""
        task = self._running_tasks.get(request_id)
        if task is None:
            return None
        try:
            result = await asyncio.wait_for(task, timeout=timeout)
            self._running_tasks.pop(request_id, None)
            return result
        except asyncio.TimeoutError:
            return None

    @property
    def pending_count(self) -> int:
        return len([t for t in self._running_tasks.values() if not t.done()])

    def health_check(self) -> dict:
        prefill_stats = self.router.prefill_pool.get_stats()
        decode_stats = self.router.decode_pool.get_stats()
        self._is_healthy = (
            prefill_stats["active_nodes"] > 0 and decode_stats["active_nodes"] > 0
        )
        return {
            "healthy": self._is_healthy,
            "pending_requests": self.pending_count,
            "prefill_pool": prefill_stats,
            "decode_pool": decode_stats,
        }
