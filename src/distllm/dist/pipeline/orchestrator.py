"""Pipeline orchestrator — coordinates distributed inference across nodes.

Manages the pipeline of worker nodes, routing input tensors through
the correct sequence of layers across nodes via gRPC.

Features:
  - Sequential pipeline routing (token-by-token)
  - Micro-batch pipelining (interleave multiple requests across stages)
  - Straggler-aware dynamic micro-batch sizing

Usage::

    orchestrator = PipelineOrchestrator(resource_mgr=rm)
    orchestrator.register_node("node-0", "10.0.0.1", 50051, 0, 15)
    orchestrator.register_node("node-1", "10.0.0.2", 50051, 16, 31)
    output = orchestrator.run_pipeline(input_ids, kv_caches, "req-123")

    # Micro-batched (higher throughput):
    output = await orchestrator.run_pipeline_microbatched(
        input_ids, kv_caches, "req-123", micro_batch_size=4
    )
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from loguru import logger


@dataclass
class PipelineNode:
    """A registered worker node in the pipeline."""
    node_id: str
    host: str
    port: int
    start_layer: int
    end_layer: int
    total_layers: int = 0
    is_healthy: bool = True
    last_heartbeat: float = field(default_factory=time.time)
    latency_ms: float = 0.0


class PipelineOrchestrator:
    """Coordinates distributed inference across pipeline-parallel nodes.

    Routes input tensors through nodes in layer order, collecting
    hidden states at each stage and passing them to the next.

    Args:
        resource_mgr: ResourceManager for circuit breaker state.
        pipeline_timeout: Timeout for the full pipeline pass in seconds.
        redundancy: Number of redundant copies per layer (for fault tolerance).
        default_micro_batch_size: Default batch size for micro-batched execution.
    """

    def __init__(
        self,
        resource_mgr: Any = None,
        pipeline_timeout: float = 30.0,
        redundancy: int = 1,
        default_micro_batch_size: int = 4,
        max_inflight_micro_batches: int = 8,
    ):
        self._resource_mgr = resource_mgr
        self._timeout = pipeline_timeout
        self._redundancy = redundancy
        self._default_micro_batch_size = default_micro_batch_size
        self._max_inflight = max_inflight_micro_batches
        self._nodes: dict[str, PipelineNode] = {}
        self._node_order: list[str] = []
        self._lock = threading.Lock()
        self._latency_tracker: Any = None
        self._straggler_detector: Any = None
        self._total_layers: int = 0
        self._wan: Any = None
        self._stats = {
            "pipeline_runs": 0,
            "total_latency_ms": 0.0,
            "errors": 0,
            "micro_batched_runs": 0,
            "avg_micro_batch_size": 0.0,
            "micro_batch_count_total": 0,
            "dynamic_batch_adjustments": [],
        }

    @property
    def nodes(self) -> dict[str, Any]:
        """Registered nodes."""
        with self._lock:
            return {nid: {
                "host": n.host,
                "port": n.port,
                "start_layer": n.start_layer,
                "end_layer": n.end_layer,
                "healthy": n.is_healthy,
            } for nid, n in self._nodes.items()}

    @property
    def node_order(self) -> list[str]:
        """Ordered list of node IDs (by layer assignment)."""
        with self._lock:
            return list(self._node_order)

    def register_node(
        self,
        node_id: str,
        host: str,
        port: int,
        start_layer: int,
        end_layer: int,
        total_layers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Register a worker node in the pipeline."""
        with self._lock:
            self._nodes[node_id] = PipelineNode(
                node_id=node_id,
                host=host,
                port=port,
                start_layer=start_layer,
                end_layer=end_layer,
                total_layers=total_layers,
            )
            # Maintain sorted order by start_layer
            self._node_order = sorted(
                self._nodes.keys(),
                key=lambda nid: self._nodes[nid].start_layer,
            )
        logger.info(f"Pipeline: registered {node_id} (layers {start_layer}-{end_layer})")

    def unregister_node(self, node_id: str) -> None:
        """Remove a node from the pipeline."""
        with self._lock:
            self._nodes.pop(node_id, None)
            self._node_order = [n for n in self._node_order if n != node_id]

    def get_node(self, node_id: str) -> PipelineNode | None:
        """Get a registered node by ID."""
        return self._nodes.get(node_id)

    def remove_node(self, node_id: str) -> None:
        """Alias for unregister_node — remove a node from the pipeline."""
        self.unregister_node(node_id)

    def validate_layer_assignment(self, node_id: str, start_layer: int, end_layer: int) -> None:
        """Validate that a layer assignment doesn't overlap with existing nodes."""
        with self._lock:
            for nid, node in self._nodes.items():
                if nid == node_id:
                    continue
                if start_layer <= node.end_layer and end_layer >= node.start_layer:
                    raise ValueError(
                        f"Layer overlap: {node_id}[{start_layer}-{end_layer}] "
                        f"overlaps with {nid}[{node.start_layer}-{node.end_layer}]"
                    )

    # ── Sequential pipeline ────────────────────────────────────────────

    def run_pipeline(
        self,
        input_ids: torch.Tensor,
        node_kv_caches: dict[str, list | None],
        request_id: str,
    ) -> torch.Tensor:
        """Run input through all nodes sequentially via gRPC.

        Routes the input tensor through each node in layer order.
        Each node processes its assigned layers and returns hidden
        states for the next node.

        Args:
            input_ids: Input token IDs (1, seq_len).
            node_kv_caches: Per-node KV cache lists.
            request_id: Request ID for tracking.

        Returns:
            Output logits from the last node.
        """
        start_time = time.time()
        self._stats["pipeline_runs"] += 1

        with self._lock:
            ordered_nodes = [
                self._nodes[nid] for nid in self._node_order
                if nid in self._nodes and self._nodes[nid].is_healthy
            ]

        if not ordered_nodes:
            raise RuntimeError("No healthy nodes in pipeline")

        current_tensor = input_ids

        for node in ordered_nodes:
            node_start = time.time()
            try:
                # Route the tensor through the node's Forward RPC
                from distllm.dist.node_client import forward_request
                kv_cache = node_kv_caches.get(node.node_id)

                current_tensor = forward_request(
                    host=node.host,
                    port=node.port,
                    hidden_states=current_tensor,
                    kv_cache=kv_cache,
                    request_id=request_id,
                )
                if current_tensor is None:
                    raise RuntimeError(f"Node {node.node_id} returned None")

                # Track latency
                latency_ms = (time.time() - node_start) * 1000
                node.latency_ms = latency_ms

                if self._resource_mgr:
                    self._resource_mgr.record_success(node.node_id)

            except Exception as e:
                logger.error(f"Pipeline error at node {node.node_id}: {e}")
                self._stats["errors"] += 1
                if self._resource_mgr:
                    self._resource_mgr.record_failure(node.node_id)
                raise

        total_ms = (time.time() - start_time) * 1000
        self._stats["total_latency_ms"] += total_ms

        return current_tensor

    # ── Micro-batched pipeline ─────────────────────────────────────────

    async def run_pipeline_microbatched(
        self,
        input_ids: torch.Tensor,
        node_kv_caches: dict[str, list | None],
        request_id: str,
        micro_batch_size: int | None = None,
    ) -> torch.Tensor:
        """Run input through the pipeline with micro-batching across stages.

        Splits the input into micro-batches and interleaves their execution
        across pipeline stages. While batch N is at stage 2, batch N-1
        can be at stage 1 — overlapping communication and computation.

        This transforms pipeline depth latency from O(num_nodes * num_batches)
        to O(num_nodes + num_batches - 1) sequential steps, significantly
        improving throughput for large inputs.

        Args:
            input_ids: Input token IDs (batch, seq_len).
            node_kv_caches: Per-node KV cache lists.
            request_id: Request ID for tracking.
            micro_batch_size: Number of tokens per micro-batch.
                              Defaults to constructor default (4).

        Returns:
            Output logits from the last node.
        """
        start_time = time.time()
        self._stats["micro_batched_runs"] += 1

        if micro_batch_size is None:
            micro_batch_size = self._default_micro_batch_size

        with self._lock:
            ordered_nodes = [
                self._nodes[nid] for nid in self._node_order
                if nid in self._nodes and self._nodes[nid].is_healthy
            ]

        if not ordered_nodes:
            raise RuntimeError("No healthy nodes in pipeline")

        num_stages = len(ordered_nodes)

        # Split input into micro-batches along the batch dimension
        total_tokens = input_ids.size(0)

        # ── Dynamic micro-batch sizing from StragglerDetector feedback ──
        if self._straggler_detector is not None:
            try:
                reports = self._straggler_detector.get_reports()
                has_moderate_or_worse = any(
                    r.severity.value in ("moderate", "severe") for r in reports
                )

                adjustment: str | None = None
                if has_moderate_or_worse:
                    micro_batch_size = max(1, int(micro_batch_size * 0.5))
                    adjustment = "straggler_reduce_50"
                elif not reports:
                    # No stragglers — check average node latency via stats
                    det_stats = self._straggler_detector.stats()
                    node_infos = det_stats.get("nodes", {})
                    if node_infos:
                        avg_latencies = [
                            n["avg_latency"] for n in node_infos.values()
                            if n.get("avg_latency", 0) > 0
                        ]
                        if avg_latencies:
                            overall_avg = sum(avg_latencies) / len(avg_latencies)
                            if overall_avg < 50.0:  # low latency threshold
                                micro_batch_size = int(micro_batch_size * 1.25)
                                adjustment = "low_latency_increase_25"

                if adjustment:
                    self._stats["dynamic_batch_adjustments"].append({
                        "request_id": request_id,
                        "adjustment": adjustment,
                        "micro_batch_size": micro_batch_size,
                        "straggler_count": len(reports),
                    })
            except Exception:
                logger.warning("StragglerDetector query failed, using default batch size")

        micro_batch_size = max(1, min(micro_batch_size, total_tokens // 2))
        micro_batches = list(torch.split(input_ids, micro_batch_size, dim=0))
        num_batches = len(micro_batches)

        # Track micro-batch stats
        self._stats["micro_batch_count_total"] += num_batches
        n = self._stats["micro_batched_runs"]
        self._stats["avg_micro_batch_size"] = (
            (self._stats["avg_micro_batch_size"] * (n - 1) + micro_batch_size) / n
        )

        # Interleave execution across pipeline stages
        # In a steady-state pipeline: each stage processes a different micro-batch.
        # The warm-up and cool-down phases are num_stages - 1 steps each.
        # Steady-state throughput = micro_batch_size / max(stage_latency) tokens/sec.

        from distllm.dist.node_client import forward_request_async

        # Results storage: results[stage_idx][batch_idx] = output_tensor
        results: list[list[torch.Tensor | None]] = [
            [None] * num_batches for _ in range(num_stages)
        ]

        async def run_stage(
            stage_idx: int,
            batch_idx: int,
            input_tensor: torch.Tensor,
        ) -> torch.Tensor | None:
            """Execute one micro-batch through one pipeline stage."""
            node = ordered_nodes[stage_idx]
            kv_cache = node_kv_caches.get(node.node_id)
            try:
                result = await forward_request_async(
                    host=node.host,
                    port=node.port,
                    hidden_states=input_tensor,
                    kv_cache=kv_cache,
                )
                if self._resource_mgr:
                    self._resource_mgr.record_success(node.node_id)
                return result
            except Exception as e:
                logger.error(
                    f"Micro-batch error: stage {stage_idx}, "
                    f"batch {batch_idx}, node {node.node_id}: {e}"
                )
                self._stats["errors"] += 1
                if self._resource_mgr:
                    self._resource_mgr.record_failure(node.node_id)
                return None

        # ── 1F1B Pipeline scheduling (bubble-optimal) ──────────────────
        #
        # Three phases:
        #   Warmup (num_stages steps):  fill the pipeline
        #   Steady-state:               one stage-0 batch enters, one finishes
        #   Cooldown (num_stages):      drain remaining batches
        #
        # This reduces pipeline bubbles from O(num_stages) to O(num_stages / micro_batch_size).
        # For inference (forward only), this is equivalent to classic pipeline parallelism
        # scheduling — each step executes one micro-batch on one stage.

        # Per-stage dependency: stage s can process batch b only after
        # stage s-1 has processed batch b (or after stage 0 starts batch b+s).
        # We track this via an event per (stage, batch) pair.
        stage_batch_ready: list[list[asyncio.Event]] = [
            [asyncio.Event() for _ in range(num_batches)]
            for _ in range(num_stages)
        ]

        # Stage 0 has no dependencies — it can always start
        for b in range(num_batches):
            stage_batch_ready[0][b].set()

        outputs: list[torch.Tensor | None] = [None] * num_batches

        async def execute_pipeline_step(
            stage_idx: int,
            batch_idx: int,
        ) -> None:
            """Execute one micro-batch on one pipeline stage.

            Waits until the previous stage has produced output for this batch,
            then runs the forward pass and signals the next stage.
            """
            try:
                await asyncio.wait_for(
                    stage_batch_ready[stage_idx][batch_idx].wait(),
                    timeout=self._timeout,
                )
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"Pipeline stage {stage_idx} batch {batch_idx} timed out "
                    f"waiting for previous stage output"
                )

            input_tensor: torch.Tensor
            if stage_idx == 0:
                input_tensor = micro_batches[batch_idx]
            else:
                # Wait for previous stage's output (may already be set)
                prev = results[stage_idx - 1][batch_idx]
                if prev is None:
                    raise RuntimeError(
                        f"Pipeline dependency broken: "
                        f"stage {stage_idx - 1} batch {batch_idx} not ready"
                    )
                input_tensor = prev

            node = ordered_nodes[stage_idx]
            kv_cache = node_kv_caches.get(node.node_id)
            try:
                result = await forward_request_async(
                    host=node.host,
                    port=node.port,
                    hidden_states=input_tensor,
                    kv_cache=kv_cache,
                    request_id=f"{request_id}-s{stage_idx}b{batch_idx}",
                )
                if result is None:
                    raise RuntimeError(f"Node {node.node_id} returned None")

                results[stage_idx][batch_idx] = result

                # Signal next stage that this batch is ready
                if stage_idx + 1 < num_stages:
                    stage_batch_ready[stage_idx + 1][batch_idx].set()

                if stage_idx == num_stages - 1:
                    # Last stage — store final output
                    outputs[batch_idx] = result

                if self._resource_mgr:
                    self._resource_mgr.record_success(node.node_id)

            except Exception as e:
                logger.error(
                    f"Pipeline step error: stage {stage_idx}, "
                    f"batch {batch_idx}, node {node.node_id}: {e}"
                )
                self._stats["errors"] += 1
                if self._resource_mgr:
                    self._resource_mgr.record_failure(node.node_id)

        # Semaphore to cap in-flight micro-batches (backpressure).
        # Prevents OOM when a slow downstream stage causes micro-batches
        # to accumulate in the pipeline buffers.
        inflight_sem = asyncio.Semaphore(self._max_inflight)

        def schedule_steps() -> list[tuple[int, int]]:
            """Generate (stage, batch) schedule using 1F1B ordering.

            Warmup:  first num_stages steps fill the pipeline
            Steady:  one new batch enters as one completes
            Cooldown: drain remaining batches from pipeline
            """
            steps: list[tuple[int, int]] = []

            # Warmup: fill pipeline — stage i processes batch 0
            for s in range(min(num_stages, num_batches)):
                steps.append((s, 0))

            # Steady state: interleave new batches behind their predecessors
            for b in range(1, num_batches):
                for s in range(min(num_stages, num_batches)):
                    steps.append((s, b))

            return steps

        # Execute the scheduled steps with asyncio tasks.
        # Each step is a single (stage, batch) unit — asyncio schedules
        # them concurrently, and the events handle the data dependencies.
        steps = schedule_steps()

        async def throttled_step(s: int, b: int) -> None:
            """Execute one step with backpressure via inflight semaphore."""
            async with inflight_sem:
                await execute_pipeline_step(s, b)

        tasks = [
            throttled_step(s, b) for s, b in steps
        ]

        # Create them in batches to avoid overwhelming the event loop
        # while still allowing interleaving
        await asyncio.gather(*tasks)

        # Filter out failures
        valid_outputs = [o for o in outputs if o is not None]
        if not valid_outputs:
            raise RuntimeError("All micro-batches failed in pipeline")

        total_ms = (time.time() - start_time) * 1000
        self._stats["total_latency_ms"] += total_ms

        return torch.cat(valid_outputs, dim=0)

    # ── Utility ───────────────────────────────────────────────────────

    def get_healthy_nodes(self) -> list[str]:
        """Return list of healthy node IDs."""
        with self._lock:
            return [nid for nid, n in self._nodes.items() if n.is_healthy]

    def mark_node_healthy(self, node_id: str) -> None:
        with self._lock:
            if node_id in self._nodes:
                self._nodes[node_id].is_healthy = True

    def mark_node_unhealthy(self, node_id: str) -> None:
        with self._lock:
            if node_id in self._nodes:
                self._nodes[node_id].is_healthy = False

    def stats(self) -> dict:
        with self._lock:
            runs = self._stats["pipeline_runs"]
            m_runs = self._stats["micro_batched_runs"]
            return {
                **self._stats,
                "node_count": len(self._nodes),
                "healthy_nodes": sum(1 for n in self._nodes.values() if n.is_healthy),
                "avg_latency_ms": (
                    self._stats["total_latency_ms"] / runs if runs > 0 else 0
                ),
                "total_layers": self._total_layers,
                "micro_batched_enabled": True,
            }

    # ── Setter methods expected by Coordinator ────────────────────────

    def set_latency_tracker(self, tracker: Any) -> None:
        """Set the latency tracker for pipeline monitoring."""
        self._latency_tracker = tracker

    def set_straggler_detector(self, detector: Any) -> None:
        """Set the straggler detector for pipeline monitoring."""
        self._straggler_detector = detector

    @property
    def total_layers(self) -> int:
        """Total number of layers across all nodes."""
        with self._lock:
            if self._total_layers > 0:
                return self._total_layers
            if not self._nodes:
                return 0
            return max(n.end_layer for n in self._nodes.values()) + 1

    @total_layers.setter
    def total_layers(self, value: int) -> None:
        self._total_layers = value

    @property
    def pipeline_timeout(self) -> float:
        return self._timeout

    @pipeline_timeout.setter
    def pipeline_timeout(self, value: float) -> None:
        self._timeout = value

    @property
    def wan(self) -> Any:
        return self._wan

    @wan.setter
    def wan(self, value: Any) -> None:
        self._wan = value

    def shutdown(self) -> None:
        """Shutdown the pipeline orchestrator."""
        with self._lock:
            self._nodes.clear()
            self._node_order.clear()
        logger.info("Pipeline orchestrator shut down")
