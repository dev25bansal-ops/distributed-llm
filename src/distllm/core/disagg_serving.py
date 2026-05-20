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
from typing import Any

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
    first_token: int | None
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

    def __init__(
        self,
        local_coordinator: Any | None = None,
        local_model_name: str = "default",
        local_dtype: str = "float16",
    ):
        self.prefill_pool = PrefillPool()
        self.decode_pool = DecodePool()
        self._active_requests: dict[str, DecodeRequest] = {}
        self._kv_cache_store: dict[str, Any] = {}  # request_id -> KV cache
        self._kv_cache_ttl: dict[str, float] = {}  # request_id -> expiry timestamp
        self._kv_cache_ttl_secs = 300  # 5 minutes default TTL
        self._lock = asyncio.Lock()
        self._local_coordinator = local_coordinator
        self._local_model_name = local_model_name
        self._local_dtype = local_dtype
        self._local_coord_lock = asyncio.Lock()
        self._local_infer_lock = asyncio.Lock()

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
            if result is None:
                await self.prefill_pool.release_node(node.node_id)
                return None

            # Store KV cache for decode pool
            async with self._lock:
                self._sweep_expired_kv()
                self._kv_cache_store[request.request_id] = result.kv_cache
                self._kv_cache_ttl[request.request_id] = time.time() + self._kv_cache_ttl_secs

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
            self._kv_cache_ttl.pop(request_id, None)

    def _sweep_expired_kv(self) -> int:
        """Remove expired entries from the KV cache store. Returns count removed."""
        now = time.time()
        expired = [rid for rid, expiry in self._kv_cache_ttl.items() if expiry < now]
        for rid in expired:
            self._kv_cache_store.pop(rid, None)
            self._kv_cache_ttl.pop(rid, None)
            self._active_requests.pop(rid, None)
        return len(expired)

    async def _send_prefill(self, node: PoolNode, request: PrefillRequest) -> PrefillResult:
        """Send prefill request to a worker node via gRPC.

        In production, this connects to a remote prefill worker over gRPC.
        For local/single-node deployment, uses the Coordinator directly.
        """
        t0 = time.time()
        try:
            from distllm.communication.grpc_client import NodeClient
            client = NodeClient(node.host, node.port)
            result = await client.prefill(prompt_tokens=request.prompt_tokens,
                                          max_new_tokens=1,
                                          request_id=request.request_id)
            elapsed_ms = (time.time() - t0) * 1000
            await client.close()

            return PrefillResult(
                request_id=request.request_id,
                kv_cache=result.get("kv_cache"),
                prompt_len=len(request.prompt_tokens),
                first_token=result.get("token_id"),
                prefill_time_ms=elapsed_ms,
                prefill_node_id=node.node_id,
            )
        except ImportError:
            logger.debug("gRPC not available, using local prefill")
        except Exception as e:
            logger.warning(f"gRPC prefill failed on '{node.node_id}': {e}, falling back to local")

        # Local fallback using Coordinator
        try:
            coord = await self._get_local_coordinator()
            prompt = coord.tokenizer.decode(request.prompt_tokens)
            start = time.perf_counter()
            async with self._local_infer_lock:
                generated = await asyncio.to_thread(coord.generate, prompt, max_new_tokens=1)
            elapsed_ms = (time.perf_counter() - start) * 1000

            # Extract KV cache from model
            kv_cache = _extract_kv_cache(coord)
            first_token_id = coord.tokenizer.encode(generated)[-1] if generated else None

            return PrefillResult(
                request_id=request.request_id,
                kv_cache=kv_cache,
                prompt_len=len(request.prompt_tokens),
                first_token=first_token_id,
                prefill_time_ms=elapsed_ms,
                prefill_node_id=node.node_id,
            )
        except Exception as e:
            logger.error(f"Local prefill fallback failed: {e}")
            return None

    async def _transfer_kv_to_decode(self, result: PrefillResult, decode_node_id: str) -> None:
        """Transfer KV cache from prefill to decode node.

        In production, uses RDMA or NVLink for zero-copy transfer.
        For local deployment, KV cache is already in shared memory.
        """
        decode_node = self.decode_pool._nodes.get(decode_node_id)
        if decode_node is None:
            return

        try:
            from distllm.communication.grpc_client import NodeClient
            client = NodeClient(decode_node.host, decode_node.port)
            await client.upload_kv_cache(
                request_id=result.request_id,
                kv_cache=result.kv_cache,
            )
            await client.close()
            logger.debug(f"KV cache transferred to decode node '{decode_node_id}'")
        except Exception as e:
            logger.debug(f"KV cache transfer to '{decode_node_id}' failed (local mode): {e}")

    async def _send_decode(
        self, node_id: str, request_id: str, input_token: int,
        kv_cache: Any, position: int,
    ) -> int | None:
        """Send decode step to a worker node via gRPC or local execution.

        Returns:
            The next token ID, or None if decoding failed.
        """
        decode_node = self.decode_pool._nodes.get(node_id)
        if decode_node is None:
            logger.warning(f"Decode node '{node_id}' not found")
            return None

        try:
            from distllm.communication.grpc_client import NodeClient
            client = NodeClient(decode_node.host, decode_node.port)
            result = await client.decode_step(
                request_id=request_id,
                input_token=input_token,
                position=position,
            )
            await client.close()
            return result.get("token_id")
        except ImportError:
            logger.debug("gRPC not available, using local decode")
        except Exception as e:
            logger.warning(f"gRPC decode failed on '{node_id}': {e}, falling back to local")

        # Local fallback
        try:
            coord = await self._get_local_coordinator()
            token_str = coord.tokenizer.decode([input_token])
            start = time.perf_counter()
            async with self._local_infer_lock:
                generated = await asyncio.to_thread(coord.generate, token_str, max_new_tokens=1)
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.debug(f"Local decode: {elapsed_ms:.1f}ms")

            return coord.tokenizer.encode(generated)[-1] if generated else None
        except Exception as e:
            logger.error(f"Local decode fallback failed: {e}")
            return None

    async def _get_local_coordinator(self):
        """Return the shared local fallback coordinator, loading it once."""
        if self._local_coordinator is not None:
            return self._local_coordinator

        async with self._local_coord_lock:
            if self._local_coordinator is None:
                from distllm.core.coordinator import Coordinator

                coord = Coordinator(
                    model_name=self._local_model_name,
                    dtype=self._local_dtype,
                )
                await asyncio.to_thread(coord.load_local_model)
                self._local_coordinator = coord
        return self._local_coordinator

    def get_stats(self) -> dict:
        self._sweep_expired_kv()
        return {
            "prefill": self.prefill_pool.get_stats(),
            "decode": self.decode_pool.get_stats(),
            "active_requests": len(self._active_requests),
            "kv_cached_requests": len(self._kv_cache_store),
            "local_fallback_loaded": self._local_coordinator is not None,
        }


def _extract_kv_cache(coord) -> list[tuple[torch.Tensor, torch.Tensor]] | None:
    """Extract KV cache from a Coordinator's model after prefill.

    Returns list of per-layer (key, value) tuples or None if extraction fails.
    """
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

        if result.first_token is None:
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
