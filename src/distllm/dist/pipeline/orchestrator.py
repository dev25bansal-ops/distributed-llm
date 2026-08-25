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
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from loguru import logger

from distllm.dist.config import WideAreaConfig
from distllm.dist.latency import LatencyTracker
from distllm.dist.straggler import StragglerDetector


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
    # Injected transport doubles (tests) or cached node clients.
    client: Any = None
    async_client: Any = None
    kv_cache: Any = None  # replicated KV state (set on standby promotion)


class PipelineError(RuntimeError):
    """One or more micro-batches failed during a pipeline run.

    Raised by ``run_pipeline_microbatched`` (default
    ``on_partial_failure="raise"``) instead of silently returning a
    shrunken response batch.  Subclasses ``RuntimeError`` so broad
    existing handlers keep working.

    Attributes:
        failed_sequences: Input row indices (batch dimension) that
            produced no output, in ascending order.
        failed_micro_batches: Micro-batch indices that failed.
        errors: Micro-batch index -> underlying error message.
    """

    def __init__(
        self,
        message: str,
        failed_sequences: list[int] | None = None,
        failed_micro_batches: list[int] | None = None,
        errors: dict[int, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.failed_sequences = list(failed_sequences or [])
        self.failed_micro_batches = list(failed_micro_batches or [])
        self.errors = dict(errors or {})


class PipelineOrchestrator:
    """Coordinates distributed inference across pipeline-parallel nodes.

    Routes input tensors through nodes in layer order, collecting
    hidden states at each stage and passing them to the next.

    Args:
        resource_mgr: ResourceManager for circuit breaker state.
        pipeline_timeout: Timeout for the full pipeline pass in seconds.
        redundancy: Number of redundant copies per layer (for fault tolerance).
        default_micro_batch_size: Default batch size for micro-batched execution.
        on_partial_failure: What to do when some (not all) micro-batches
            fail in the micro-batched path.  ``"raise"`` (default) raises
            :class:`PipelineError` naming the failed sequences;
            ``"drop"`` preserves the legacy behaviour of returning only
            the successful rows but attaches explicit
            ``failed_sequences`` metadata to the returned tensor (and to
            ``stats()["last_failed_sequences"]``) so callers must see it.
    """

    _PARTIAL_FAILURE_POLICIES = ("raise", "drop")

    def __init__(
        self,
        resource_mgr: Any = None,
        pipeline_timeout: float = 30.0,
        redundancy: int = 1,
        default_micro_batch_size: int = 4,
        max_inflight_micro_batches: int = 8,
        use_tls: bool = False,
        ca_cert: str | None = None,
        total_layers: int = 0,
        on_partial_failure: str = "raise",
    ):
        if on_partial_failure not in self._PARTIAL_FAILURE_POLICIES:
            raise ValueError(
                f"on_partial_failure must be one of "
                f"{self._PARTIAL_FAILURE_POLICIES}, got {on_partial_failure!r}"
            )
        self._resource_mgr = resource_mgr
        self._timeout = pipeline_timeout
        self._redundancy = redundancy
        self._default_micro_batch_size = default_micro_batch_size
        self._max_inflight = max_inflight_micro_batches
        # Encrypt pipeline-parallel gRPC (activations/KV carry prompt content).
        # Enable via DISTLLM_PIPELINE_TLS=1 and point DISTLLM_TLS_CA_CERT(_FILE)
        # at the cluster CA.
        self._use_tls = use_tls or os.environ.get("DISTLLM_PIPELINE_TLS", "0") == "1"
        self._on_partial_failure = on_partial_failure
        self._ca_cert = (
            ca_cert
            or os.environ.get("DISTLLM_TLS_CA_CERT_FILE")
            or os.environ.get("DISTLLM_TLS_CA_CERT")
        )
        self._nodes: dict[str, PipelineNode] = {}
        self._node_order: list[str] = []
        self._tensor_transport: Any = None
        self._lock = threading.Lock()
        self._latency_tracker: Any = None
        self._straggler_detector: Any = None
        self._total_layers: int = total_layers
        self._wan: Any = None
        self._stats = {
            "pipeline_runs": 0,
            "total_latency_ms": 0.0,
            "errors": 0,
            "micro_batched_runs": 0,
            "avg_micro_batch_size": 0.0,
            "micro_batch_count_total": 0,
            "dynamic_batch_adjustments": [],
            "last_failed_sequences": [],
        }

    @property
    def nodes(self) -> dict[str, Any]:
        """Live mapping of node_id -> PipelineNode.

        Returns the internal mapping (not a copy): callers register,
        inject clients, and flip health flags in place.  Structural
        changes go through register_node/unregister_node.
        """
        with self._lock:
            return self._nodes

    @nodes.setter
    def nodes(self, value: dict[str, Any]) -> None:
        """Replace the registered node set (full node mapping)."""
        with self._lock:
            self._nodes = dict(value)

    @property
    def node_order(self) -> list[str]:
        """Ordered list of node IDs (by layer assignment)."""
        with self._lock:
            return list(self._node_order)

    @node_order.setter
    def node_order(self, value: list[str]) -> None:
        """Replace the node ordering (used on node removal/health recovery)."""
        with self._lock:
            self._node_order = list(value)

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

    def create_node_kv_caches(self) -> dict[str, Any]:
        """Allocate one KV-cache slot per registered node.

        Slots start as ``None`` (= no cache); callers populate them with
        per-node cache objects as sequences are processed.
        """
        with self._lock:
            return {nid: None for nid in self._nodes}

    def set_tensor_transport(self, transport: Any) -> None:
        """Register a pluggable tensor transport (NCCL/RAIL/mock).

        When set, per-hop tensor sends may go through
        ``transport.send_tensor(node_id, tensor, tag)`` instead of the
        default gRPC client path.
        """
        self._tensor_transport = transport

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
                kv_cache = node_kv_caches.get(node.node_id)

                # Prefer an injected per-node client (tests / pooled
                # connections) over dialing a fresh gRPC channel.
                if getattr(node, "client", None) is not None and hasattr(
                    node.client, "forward"
                ):
                    current_tensor = node.client.forward(
                        hidden_states=current_tensor,
                        kv_cache=kv_cache,
                        request_id=request_id,
                    )
                else:
                    # Route the tensor through the node's Forward RPC
                    from distllm.dist.node_client import forward_request

                    current_tensor = forward_request(
                        host=node.host,
                        port=node.port,
                        hidden_states=current_tensor,
                        kv_cache=kv_cache,
                        request_id=request_id,
                        use_tls=self._use_tls,
                        ca_cert=self._ca_cert,
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

    def run_pipeline_overlap(
        self,
        input_ids: torch.Tensor,
        node_kv_caches: dict[str, list | None],
        request_id: str,
        micro_batch_size: int | None = None,
    ) -> torch.Tensor:
        """Synchronous entry point for the micro-batched (overlapped) pipeline.

        Overlaps stage compute with transport; not callable from within a
        running event loop (use ``await run_pipeline_microbatched`` there).
        """
        return asyncio.run(
            self.run_pipeline_microbatched(
                input_ids, node_kv_caches, request_id,
                micro_batch_size=micro_batch_size,
            )
        )

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

        # Failure registry for this run: batch_idx -> first error message.
        # A failed batch is cascaded to downstream stages via their ready
        # events so they unwind immediately instead of blocking until the
        # pipeline timeout — and instead of invoking workers with a missing
        # input tensor.
        step_failures: dict[int, str] = {}

        async def execute_pipeline_step(
            stage_idx: int,
            batch_idx: int,
        ) -> None:
            """Execute one micro-batch on one pipeline stage.

            Waits until the previous stage has produced output for this batch,
            then runs the forward pass and signals the next stage.

            Step failures are recorded in ``step_failures`` (never silently
            dropped) and propagated to the next stage, whose step returns
            without issuing an RPC.
            """
            try:
                await asyncio.wait_for(
                    stage_batch_ready[stage_idx][batch_idx].wait(),
                    timeout=self._timeout,
                )
            except asyncio.TimeoutError:
                # Genuine hang upstream (neither output nor failure arrived):
                # keep the loud, immediate TimeoutError rather than blocking
                # on siblings that may never finish.
                raise TimeoutError(
                    f"Pipeline stage {stage_idx} batch {batch_idx} timed out "
                    f"waiting for previous stage output"
                )

            node = ordered_nodes[stage_idx]

            def _cascade() -> None:
                if stage_idx + 1 < num_stages:
                    stage_batch_ready[stage_idx + 1][batch_idx].set()

            # An earlier stage already failed this batch: unwind without an
            # RPC so workers never receive a missing input.
            if batch_idx in step_failures:
                _cascade()
                return

            input_tensor: torch.Tensor
            if stage_idx == 0:
                input_tensor = micro_batches[batch_idx]
            else:
                # Wait for previous stage's output (may already be set)
                prev = results[stage_idx - 1][batch_idx]
                if prev is None:
                    step_failures.setdefault(
                        batch_idx,
                        f"Pipeline dependency broken: "
                        f"stage {stage_idx - 1} batch {batch_idx} not ready",
                    )
                    _cascade()
                    return
                input_tensor = prev

            kv_cache = node_kv_caches.get(node.node_id)
            try:
                # Prefer an injected per-node client (tests / pooled
                # connections) over dialing a fresh gRPC channel.
                injected_sync = getattr(node, "client", None)
                if injected_sync is not None and hasattr(injected_sync, "forward"):
                    result = injected_sync.forward(
                        hidden_states=input_tensor,
                        kv_cache=kv_cache,
                        request_id=f"{request_id}-s{stage_idx}b{batch_idx}",
                    )
                else:
                    result = await forward_request_async(
                        host=node.host,
                        port=node.port,
                        hidden_states=input_tensor,
                        kv_cache=kv_cache,
                        request_id=f"{request_id}-s{stage_idx}b{batch_idx}",
                        use_tls=self._use_tls,
                        ca_cert=self._ca_cert,
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
                step_failures.setdefault(batch_idx, str(e))
                self._stats["errors"] += 1
                if self._resource_mgr:
                    self._resource_mgr.record_failure(node.node_id)
                # Wake the next stage so it cascades instead of stalling
                # until the pipeline timeout.
                _cascade()

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

            # Warmup: fill pipeline — EVERY stage must process batch 0
            # (capping stages by num_batches silently dropped the final
            # stage whenever a request produced fewer micro-batches than
            # there are stages — e.g. every single-sequence request —
            # so the output-producing stage never ran).
            for s in range(num_stages):
                steps.append((s, 0))

            # Steady state: interleave new batches behind their predecessors
            for b in range(1, num_batches):
                for s in range(num_stages):
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

        # ── Result assembly with explicit failure semantics ────────────
        #
        # Partial failures are never silent: default policy raises
        # PipelineError naming the failed input rows; the opt-in "drop"
        # policy keeps legacy shapes but attaches machine-readable
        # failed_sequences metadata callers must acknowledge.
        failed_micro_batches = sorted(step_failures)
        failed_sequences = sorted(
            row
            for b in failed_micro_batches
            for row in range(
                b * micro_batch_size,
                min((b + 1) * micro_batch_size, total_tokens),
            )
        )
        self._stats["last_failed_sequences"] = list(failed_sequences)

        valid_outputs = [o for o in outputs if o is not None]

        if failed_micro_batches:
            if self._on_partial_failure == "drop":
                if not valid_outputs:
                    raise RuntimeError("All micro-batches failed in pipeline")
                logger.warning(
                    f"Micro-batch partial failures dropped for "
                    f"{request_id}: rows {failed_sequences}"
                )
            else:
                if len(failed_micro_batches) == num_batches:
                    raise PipelineError(
                        f"All micro-batches failed in pipeline "
                        f"(request {request_id}); failed sequences "
                        f"{failed_sequences}",
                        failed_sequences=failed_sequences,
                        failed_micro_batches=failed_micro_batches,
                        errors=dict(step_failures),
                    )
                raise PipelineError(
                    f"Pipeline partial failure (request {request_id}): "
                    f"{len(failed_micro_batches)}/{num_batches} micro-batches "
                    f"failed — sequences {failed_sequences}. First error: "
                    f"{step_failures[failed_micro_batches[0]]}",
                    failed_sequences=failed_sequences,
                    failed_micro_batches=failed_micro_batches,
                    errors=dict(step_failures),
                )

        if not valid_outputs:
            raise RuntimeError("All micro-batches failed in pipeline")

        total_ms = (time.time() - start_time) * 1000
        self._stats["total_latency_ms"] += total_ms

        result = torch.cat(valid_outputs, dim=0)
        if failed_micro_batches and self._on_partial_failure == "drop":
            # Attach to the FINAL tensor (cat builds a new one) so the
            # shrinkage is explicit and machine-readable at the call site.
            result.failed_sequences = tuple(failed_sequences)
        return result

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

    def set_latency_tracker(self, tracker: LatencyTracker) -> None:
        """Set the latency tracker for pipeline monitoring."""
        self._latency_tracker = tracker

    def set_straggler_detector(self, detector: StragglerDetector) -> None:
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
    def wan(self) -> WideAreaConfig | None:
        return self._wan

    @wan.setter
    def wan(self, value: WideAreaConfig) -> None:
        self._wan = value

    def shutdown(self) -> None:
        """Shutdown the pipeline orchestrator."""
        with self._lock:
            self._nodes.clear()
            self._node_order.clear()
        logger.info("Pipeline orchestrator shut down")
