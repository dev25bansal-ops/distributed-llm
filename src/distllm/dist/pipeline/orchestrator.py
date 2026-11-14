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

from loguru import logger

# Optional dependency detection for torch
_TORCH_AVAILABLE: bool
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _require_torch() -> None:
    """Raise a helpful ImportError if torch is not installed."""
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "torch is required for pipeline orchestrator; "
            "install with pip install distributed-llm[self-hosted]"
        )


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
    # Populated by the most recent health_check() call.
    gpu_memory_free: int = 0
    gpu_memory_total: int = 0
    last_health_time: float = 0.0

    def health_check(self) -> bool:
        """Probe the node's gRPC HealthCheck RPC.

        On success, refreshes ``is_healthy=True`` plus the memory fields
        and returns True. On ANY failure (network error, malformed
        response, healthy=False) marks the node unhealthy and returns
        False — a partition must never read as healthy.  ``is_healthy``
        is the canonical spelling on PipelineNode (``healthy`` is
        NodeRegistration's field; a distinct attr here breaks the
        hasattr contract pinned by tests/dist/test_wan_attrs.py).
        """
        client = getattr(self, "client", None)
        if client is None or getattr(client, "stub", None) is None:
            # No transport attached — cannot verify liveness.
            self.is_healthy = False
            return False
        try:
            resp = client.stub.HealthCheck(self._build_health_request())
            self.is_healthy = bool(getattr(resp, "healthy", False))
            self.gpu_memory_free = int(getattr(resp, "memory_free_bytes", 0) or 0)
            self.gpu_memory_total = int(getattr(resp, "memory_total_bytes", 0) or 0)
            self.last_health_time = time.time()
            self.last_heartbeat = self.last_health_time
        except Exception:
            # CQ-08: intentional fail-open: fallback default assigned on failure.
            self.is_healthy = False
            self.last_health_time = time.time()
        return self.is_healthy

    def _build_health_request(self) -> Any:
        """Build the HealthCheckRequest proto (node_pb2)."""
        from distllm.dist import node_pb2
        return node_pb2.HealthCheckRequest()


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
        cluster_key: Optional shared cluster auth secret.  Stamped onto
            every ForwardPass request so workers running fail-closed auth
            (``NodeServicer._check_auth``) accept our RPCs.  Resolution:
            explicit argument > ``DISTLLM_CLUSTER_KEY`` env var > ``None``.
            ``None`` keeps insecure deployments working (requests carry an
            empty key; workers without a configured key accept them).
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
        cluster_key: str | None = None,
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
        # Shared cluster auth secret — explicit arg > DISTLLM_CLUSTER_KEY
        # env var > None.  None is valid (insecure deployments); every
        # dial-through forward call then sends an empty ``cluster_key``
        # protobuf field, which unauthenticated workers accept.
        self._cluster_key = cluster_key or os.environ.get("DISTLLM_CLUSTER_KEY") or None
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
        """Validate that a layer assignment doesn't overlap with existing nodes.

        M-06: delegates to the shared ``TopologyRegistry`` so the overlap
        invariant lives in exactly one place (core and dist both consume
        it). Lock is held here for the snapshot; the validation math is pure.
        """
        from distllm.core.topology_registry import NodeRange, validate_assignment

        with self._lock:
            existing = [
                NodeRange(nid, node.start_layer, node.end_layer)
                for nid, node in self._nodes.items()
            ]
        validate_assignment(node_id, start_layer, end_layer, existing)

    # ── Failure resilience (chaos seam) ────────────────────────────────

    def _find_fallback_node(self, failed_id: str, failed_node: Any) -> Any | None:
        """Find a healthy node whose layer range covers the failed node's.

        Used when a node dies mid-pipeline: a covering node can take over
        the failed layer range without pipeline reconfiguration. With
        disjoint (non-overlapping) layer assignments there is no fallback —
        returns None and the caller degrades instead of silently computing
        wrong layers on the wrong node.
        """
        fallback = None
        for nid, node in self._nodes.items():
            if nid == failed_id:
                continue
            if not node.is_healthy:
                continue
            if (node.start_layer <= failed_node.start_layer
                    and node.end_layer >= failed_node.end_layer):
                fallback = node
                break
        return fallback

    def _execute_node_grpc(self, ctx: Any) -> Any:
        """Execute one node's forward pass via gRPC, with degraded passthrough.

        Contract (pinned by the chaos suite):
        * node NOT registered / circuit breaker open → return the incoming
          hidden state unchanged (``passthrough``) so a single-node failure
          degrades quality rather than killing the whole request;
        * registered + RPC fails (partition, timeout, internal error) →
          record the failure and re-raise — silent data loss is worse.
        """
        node = self._nodes.get(getattr(ctx, "node_id", None))
        breaker_open = False
        if self._resource_mgr is not None:
            breaker_open = bool(
                self._resource_mgr.check_circuit_breaker(ctx.node_id)
            )
        if node is None or breaker_open:
            logger.warning(
                f"Node {getattr(ctx, 'node_id', '?')} unavailable "
                f"(registered={node is not None}, breaker_open={breaker_open}) "
                "— passing hidden state through unchanged"
            )
            return getattr(ctx, "current_hidden", None)

        client = getattr(node, "client", None)
        stub = getattr(client, "stub", None)
        try:
            if stub is not None:
                resp = stub.ForwardPass(None)
                if not getattr(resp, "success", True):
                    raise RuntimeError(
                        getattr(resp, "error_message", "forward pass failed")
                    )
                if self._resource_mgr is not None:
                    self._resource_mgr.record_success(node.node_id)
                return getattr(resp, "output", None)
            raise RuntimeError(f"Node {node.node_id} has no gRPC stub attached")
        except Exception:
            if self._resource_mgr is not None:
                self._resource_mgr.record_failure(node.node_id)
            raise

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

        # Redundancy engages here: with redundancy > 1 the same stage is
        # fanned out to the peers covering its layer range and the first
        # successful response wins.  Before this wiring existed, the
        # ``redundancy`` config knob (coordinator --redundancy) reached no
        # code path and every stage ran exactly once.
        if self._redundancy > 1:
            from distllm.dist.redundant import RedundantExecutor

            try:
                result = RedundantExecutor(
                    self,
                    redundancy=self._redundancy,
                    timeout_s=self._timeout,
                ).run_pipeline(input_ids, node_kv_caches, request_id)
            except Exception:
                self._stats["errors"] += 1
                raise
            self._stats["total_latency_ms"] += (time.time() - start_time) * 1000
            return result

        # A draining node keeps finishing in-flight requests but must not
        # receive NEW work. The serving path used to filter on is_healthy
        # only, so `admin drain` marked the node in the ResourceManager and
        # nothing consulted it — drained nodes kept taking fresh requests
        # and drain could never actually quiesce a node for removal.
        ordered_nodes = [
            node for node in (
                self._nodes.get(nid) for nid in self.serving_node_ids()
            ) if node is not None
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
                        cluster_key=self._cluster_key,
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

        # Same serving filter as the sequential path (``run_pipeline``):
        # healthy AND not draining. The overlap/micro-batched path is what
        # request_pipeline actually calls when overlap or 1F1B is enabled,
        # so without this a drained node kept receiving NEW micro-batches
        # even though `admin drain` and the sequential path honoured it.
        ordered_nodes = [
            node for node in (
                self._nodes.get(nid) for nid in self.serving_node_ids()
            ) if node is not None
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
                        cluster_key=self._cluster_key,
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

    def _draining_node_ids(self) -> set[str]:
        """Node IDs the ResourceManager has marked draining.

        Every serving path filters these out of its node set (in-flight
        requests still complete on the node they started on, because the
        node list is snapshotted per request). Missing/degraded resource
        managers yield an empty set — draining is operator intent and must
        never crash a request path.
        """
        if self._resource_mgr is None or not hasattr(self._resource_mgr, "get_draining_nodes"):
            return set()
        try:
            return set(self._resource_mgr.get_draining_nodes())
        except Exception:
            return set()

    def serving_node_ids(self) -> list[str]:
        """Node IDs eligible for NEW work: healthy and not draining.

        The single source of truth for the serving filter, shared by the
        sequential and the overlapped/micro-batched paths so `admin drain`
        means the same thing on both.
        """
        draining = self._draining_node_ids()
        with self._lock:
            return [
                nid for nid in self._node_order
                if nid in self._nodes
                and self._nodes[nid].is_healthy
                and nid not in draining
            ]

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

    @property
    def cluster_key(self) -> str | None:
        """Shared cluster auth secret forwarded on every pipeline RPC.

        ``None`` when unset (insecure deployments send a keyless request
        that unauthenticated workers accept).
        """
        return self._cluster_key

    @cluster_key.setter
    def cluster_key(self, value: str | None) -> None:
        """Post-construction wiring (ClusterManager/coordinator push)."""
        self._cluster_key = value or None

    def shutdown(self) -> None:
        """Shutdown the pipeline orchestrator."""
        with self._lock:
            self._nodes.clear()
            self._node_order.clear()
        logger.info("Pipeline orchestrator shut down")
