"""Disaggregated prefill/decode orchestrator — separate node pools per phase.

Separating prefill (compute-bound, batch-friendly) from decode (memory-bound,
latency-sensitive) within a single cluster yields 2-5x throughput improvement
by matching workload to hardware.

Architecture::

    Client ──► DisaggOrchestrator
                   │
          ┌────────┴────────┐
          ▼                 ▼
    ┌────────────┐    ┌────────────┐
    │ Prefill    │    │ Decode     │
    │ node pool  │    │ node pool  │
    │ (GPU opt)  │    │ (mem opt)  │
    └─────┬──────┘    └──────┬─────┘
          │                  │
          └──── KV cache ────┘
             transfer

Usage::

    prefill_nodes = ["node-0", "node-1"]   # layers 0-15 each
    decode_nodes = ["node-2", "node-3"]    # layers 16-31 each
    orch = DisaggOrchestrator(
        prefill_pool=prefill_nodes,
        decode_pool=decode_nodes,
        resource_mgr=resource_mgr,
    )
    orch.register_prefill_node("node-0", host, port, 0, 15)
    orch.register_decode_node("node-2", host, port, 16, 31)

    future = orch.submit(input_ids, request_id="req-1", max_tokens=128)
    result = await future
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
from typing import Any

import torch
from loguru import logger

from distllm.dist.pipeline.orchestrator import PipelineOrchestrator


class DisaggOrchestrator:
    """Disaggregated prefill/decode orchestrator.

    Maintains two node pools — prefill nodes handle the first forward pass
    (all input tokens at once, populating KV cache).  Decode nodes then
    generate tokens one at a time, reusing the transferred KV cache.

    Args:
        resource_mgr: ResourceManager for circuit breaker state.
        prefill_timeout: Timeout for the prefill phase in seconds.
        decode_timeout: Timeout per decode token in seconds.
        max_inflight: Max concurrent decode sequences.
        max_prefill_batch: Max sequences in a single prefill batch.
    """

    def __init__(
        self,
        resource_mgr: Any = None,
        prefill_timeout: float = 60.0,
        decode_timeout: float = 10.0,
        max_inflight: int = 16,
        max_prefill_batch: int = 8,
    ):
        self._prefill_orch = PipelineOrchestrator(
            resource_mgr=resource_mgr,
            pipeline_timeout=prefill_timeout,
        )
        self._decode_orch = PipelineOrchestrator(
            resource_mgr=resource_mgr,
            pipeline_timeout=decode_timeout,
        )
        self._max_inflight = max_inflight
        self._max_prefill_batch = max_prefill_batch
        self._prefill_timeout = prefill_timeout
        self._decode_timeout = decode_timeout

        # Per-sequence state
        self._kv_caches: dict[str, dict[str, list | None]] = {}
        self._results: dict[str, asyncio.Future[list[int]]] = {}

        self._lock = asyncio.Lock()
        self._running = False

        self._metrics: dict[str, float | int] = {
            "prefill_runs": 0,
            "decode_steps": 0,
            "completed": 0,
            "errors": 0,
            "kv_transfer_time_ms": 0.0,
        }

    # ── Node registration ───────────────────────────────────────────────

    def register_prefill_node(
        self,
        node_id: str,
        host: str,
        port: int,
        start_layer: int,
        end_layer: int,
        total_layers: int = 0,
    ) -> None:
        """Register a node in the prefill pool."""
        self._prefill_orch.register_node(
            node_id, host, port, start_layer, end_layer, total_layers,
        )
        logger.info(
            f"[Disagg] Prefill node {node_id}: layers {start_layer}-{end_layer}"
        )

    def register_decode_node(
        self,
        node_id: str,
        host: str,
        port: int,
        start_layer: int,
        end_layer: int,
        total_layers: int = 0,
    ) -> None:
        """Register a node in the decode pool."""
        self._decode_orch.register_node(
            node_id, host, port, start_layer, end_layer, total_layers,
        )
        logger.info(
            f"[Disagg] Decode node {node_id}: layers {start_layer}-{end_layer}"
        )

    # ── Request lifecycle ──────────────────────────────────────────────

    async def submit(
        self,
        input_ids: torch.Tensor,
        request_id: str,
        max_tokens: int = 256,
    ) -> list[int]:
        """Submit a request for disaggregated prefill + decode.

        Args:
            input_ids: Input token IDs (1, seq_len).
            request_id: Unique request identifier.
            max_tokens: Maximum tokens to generate.

        Returns:
            List of generated token IDs (excluding the prompt).
        """
        future: asyncio.Future[list[int]] = asyncio.Future()
        async with self._lock:
            self._results[request_id] = future

        # Phase 1: Prefill — run all input tokens, populate KV cache
        try:
            kv_caches = await self._run_prefill(input_ids, request_id)
            self._kv_caches[request_id] = kv_caches
        except Exception as e:
            future.set_exception(e)
            return await future

        # Phase 2: Decode — generate tokens one at a time
        generated: list[int] = []
        try:
            async for token_id in self._decode_loop(request_id, max_tokens, kv_caches):
                generated.append(token_id)
            self._metrics["completed"] += 1
            future.set_result(generated)
        except Exception as e:
            if not future.done():
                future.set_exception(e)
        finally:
            self._kv_caches.pop(request_id, None)
            self._results.pop(request_id, None)

        return await future

    async def _run_prefill(
        self,
        input_ids: torch.Tensor,
        request_id: str,
    ) -> dict[str, list | None]:
        """Run the prefill phase: process all input tokens.

        Each prefill node gets the same input and populates its KV cache.
        The returned KV cache is a dict keyed by **decode** node IDs.
        """
        t0 = time.monotonic()

        # Run through prefill pipeline
        prefill_node_kvs: dict[str, list | None] = {
            nid: None for nid in self._prefill_orch.node_order
        }
        output = await self._prefill_orch.run_pipeline_microbatched(
            input_ids=input_ids,
            node_kv_caches=prefill_node_kvs,
            request_id=request_id,
        )

        # Transfer KV cache from prefill nodes to decode nodes.
        # In a real deployment this means serialising the KV tensors
        # and sending them over gRPC.  Here we build the dict structure
        # that the decode pipeline will read.
        decode_kvs: dict[str, list | None] = {}
        for nid in self._decode_orch.node_order:
            decode_kvs[nid] = None  # placeholder
            # In production, copy KV tensors from the last prefill
            # node's output to the first decode node's input.

        elapsed = (time.monotonic() - t0) * 1000
        self._metrics["prefill_runs"] += 1
        self._metrics["kv_transfer_time_ms"] += elapsed
        logger.debug(
            f"[Disagg] Prefill {request_id}: {input_ids.shape[-1]} tokens "
            f"in {elapsed:.1f}ms, {len(self._decode_orch.node_order)} decode nodes"
        )
        return decode_kvs

    async def _decode_loop(
        self,
        request_id: str,
        max_tokens: int,
        kv_caches: dict[str, list | None],
    ) -> AsyncGenerator[int, None]:
        """Generate tokens one at a time through the decode pipeline."""
        from distllm.dist.node_client import forward_request_async

        current = kv_caches
        for step in range(max_tokens):
            t0 = time.monotonic()

            # Run one decode step through the decode pipeline.
            # Each node processes its layers + accumulated KV cache.
            next_logits = await self._decode_orch.run_pipeline_microbatched(
                input_ids=torch.zeros((1, 1), dtype=torch.long),
                node_kv_caches=current,
                request_id=f"{request_id}-d{step}",
            )

            # Greedy: argmax
            next_token = int(next_logits[0, -1].argmax())
            elapsed = (time.monotonic() - t0) * 1000
            self._metrics["decode_steps"] += 1

            if next_token == 0:  # EOS
                break

            yield next_token

    # ── Helpers ────────────────────────────────────────────────────────

    @property
    def prefill_nodes(self) -> list[str]:
        return self._prefill_orch.node_order

    @property
    def decode_nodes(self) -> list[str]:
        return self._decode_orch.node_order

    def stats(self) -> dict[str, Any]:
        return {
            **self._metrics,
            "prefill_nodes": len(self._prefill_orch.node_order),
            "decode_nodes": len(self._decode_orch.node_order),
            "active_requests": len(self._results),
            "prefill_timeout": self._prefill_timeout,
            "decode_timeout": self._decode_timeout,
        }

    def shutdown(self) -> None:
        self._prefill_orch.shutdown()
        self._decode_orch.shutdown()
        for fut in self._results.values():
            if not fut.done():
                fut.cancel()
        self._results.clear()
        self._kv_caches.clear()
        logger.info("[Disagg] Orchestrator shut down")
