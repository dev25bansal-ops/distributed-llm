from __future__ import annotations

import time
from typing import Any

from loguru import logger

from distllm.core.disagg.types import (
    PrefillRequest,
    PrefillResult,
    PoolNode,
)
from distllm.core.disagg.pool import PrefillPool, DecodePool
from distllm.core.disagg.kv_cache import KVCacheStore, extract_kv_cache


class DisaggRouter:
    """Routes requests between prefill and decode pools.

    Handles KV cache transfer from prefill nodes to decode nodes.
    Supports both remote (gRPC) and local (Coordinator) execution.
    """

    def __init__(
        self,
        local_coordinator: Any | None = None,
        local_model_name: str = "default",
        local_dtype: str = "float16",
        kv_cache_ttl_secs: float = 300.0,
    ):
        self.prefill_pool = PrefillPool()
        self.decode_pool = DecodePool()
        self._active_requests: dict[str, Any] = {}
        self._kv_store = KVCacheStore(default_ttl_secs=kv_cache_ttl_secs)
        self._lock = __import__("asyncio").Lock()
        self._local_coordinator = local_coordinator
        self._local_model_name = local_model_name
        self._local_dtype = local_dtype
        self._local_coord_lock = __import__("asyncio").Lock()
        self._local_infer_lock = __import__("asyncio").Lock()

    # ------------------------------------------------------------------
    # Pool management
    # ------------------------------------------------------------------

    async def add_prefill_node(self, node_id: str, host: str, port: int, capacity: int = 4) -> None:
        await self.prefill_pool.register_node(node_id, host, port, capacity)

    async def remove_prefill_node(self, node_id: str) -> None:
        await self.prefill_pool.unregister_node(node_id)

    async def add_decode_node(self, node_id: str, host: str, port: int, capacity: int = 8) -> None:
        await self.decode_pool.register_node(node_id, host, port, capacity)

    async def remove_decode_node(self, node_id: str) -> None:
        await self.decode_pool.unregister_node(node_id)

    # ------------------------------------------------------------------
    # Request flow
    # ------------------------------------------------------------------

    async def prefill(self, request: PrefillRequest) -> PrefillResult | None:
        """Route a request to the prefill pool and return KV cache."""
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

            self._kv_store.store(request.request_id, result.kv_cache)

            decode_node_id = await self.decode_pool.assign_request(request.request_id)
            if decode_node_id:
                await self._transfer_kv_to_decode(result, decode_node_id)

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
        """Execute one decode step on the assigned decode node."""
        decode_node_id = self.decode_pool.get_node_for_request(request_id)
        if decode_node_id is None:
            logger.warning(f"No decode node assigned for request '{request_id}'")
            return None

        kv_cache = self._kv_store.get(request_id)
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
        self._kv_store.remove(request_id)

    # ------------------------------------------------------------------
    # Remote / local execution
    # ------------------------------------------------------------------

    async def _send_prefill(self, node: PoolNode, request: PrefillRequest) -> PrefillResult | None:
        t0 = time.time()
        try:
            from distllm.communication.grpc_client import NodeClient

            client = NodeClient(node.host, node.port)
            result = await client.prefill(
                prompt_tokens=request.prompt_tokens,
                max_new_tokens=1,
                request_id=request.request_id,
            )
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

        return await self._local_prefill(request, node.node_id)

    async def _local_prefill(self, request: PrefillRequest, node_id: str) -> PrefillResult | None:
        try:
            coord = await self._get_local_coordinator()
            prompt = coord.tokenizer.decode(request.prompt_tokens)
            start = time.perf_counter()
            async with self._local_infer_lock:
                generated = await __import__("asyncio").to_thread(coord.generate, prompt, max_new_tokens=1)
            elapsed_ms = (time.perf_counter() - start) * 1000

            kv_cache = extract_kv_cache(coord)
            first_token_id = coord.tokenizer.encode(generated)[-1] if generated else None

            return PrefillResult(
                request_id=request.request_id,
                kv_cache=kv_cache,
                prompt_len=len(request.prompt_tokens),
                first_token=first_token_id,
                prefill_time_ms=elapsed_ms,
                prefill_node_id=node_id,
            )
        except Exception as e:
            logger.error(f"Local prefill fallback failed: {e}")
            return None

    async def _transfer_kv_to_decode(self, result: PrefillResult, decode_node_id: str) -> None:
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

        return await self._local_decode(input_token)

    async def _local_decode(self, input_token: int) -> int | None:
        try:
            coord = await self._get_local_coordinator()
            token_str = coord.tokenizer.decode([input_token])
            start = time.perf_counter()
            async with self._local_infer_lock:
                generated = await __import__("asyncio").to_thread(coord.generate, token_str, max_new_tokens=1)
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.debug(f"Local decode: {elapsed_ms:.1f}ms")
            return coord.tokenizer.encode(generated)[-1] if generated else None
        except Exception as e:
            logger.error(f"Local decode fallback failed: {e}")
            return None

    async def _get_local_coordinator(self):
        if self._local_coordinator is not None:
            return self._local_coordinator

        async with self._local_coord_lock:
            if self._local_coordinator is None:
                from distllm.core.coordinator import Coordinator

                coord = Coordinator(
                    model_name=self._local_model_name,
                    dtype=self._local_dtype,
                )
                await __import__("asyncio").to_thread(coord.load_local_model)
                self._local_coordinator = coord
        return self._local_coordinator

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        self._kv_store.sweep_expired()
        return {
            "prefill": self.prefill_pool.get_stats(),
            "decode": self.decode_pool.get_stats(),
            "active_requests": len(self._active_requests),
            "kv_cached_requests": self._kv_store.size(),
            "local_fallback_loaded": self._local_coordinator is not None,
        }
