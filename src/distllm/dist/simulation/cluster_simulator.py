"""Cluster-level simulator for distributed pipeline inference.

Models N nodes with configurable hardware specs and simulates
pipeline-parallel execution to estimate latency, throughput, and
resource utilization before deploying to real hardware.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class NodeSpec:
    """Hardware specification for a single simulation node."""

    node_id: str
    gpu_name: str = "A100-80GB"
    gpu_count: int = 1
    compute_tflops: float = 312.0  # FP16 TFLOPS per GPU
    memory_gb: float = 80.0
    memory_bandwidth_gbps: float = 2039.0
    interconnect_gbps: float = 50.0  # NIC / fabric bandwidth
    intra_node_bw_gbps: float = 600.0  # NVLink bandwidth


@dataclass(frozen=True)
class ModelConfig:
    """Model architecture description for simulation."""

    name: str
    num_layers: int
    hidden_dim: int
    num_heads: int
    head_dim: int
    intermediate_dim: int
    vocab_size: int
    flops_per_layer: float  # Estimated FLOPS per layer per token
    params_per_layer: int  # Number of parameters per layer


@dataclass
class SimulatedPipelineResult:
    """Result of a single pipeline simulation run."""

    latency_ms: float
    throughput_tok_s: float
    compute_time_ms: float
    comm_time_ms: float
    bubble_overhead_ms: float
    bottleneck_node: str
    stage_times_ms: dict[str, float]
    memory_utilization: dict[str, float]
    compute_utilization: dict[str, float]

    @property
    def total_utilization(self) -> float:
        """Average utilization across all nodes."""
        values = list(self.compute_utilization.values())
        return sum(values) / len(values) if values else 0.0


@dataclass
class PerfMetrics:
    """Aggregated performance metrics over multiple runs."""

    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    avg_throughput_tok_s: float = 0.0
    peak_throughput_tok_s: float = 0.0
    avg_compute_utilization: float = 0.0
    avg_memory_utilization: float = 0.0
    node_utilization: dict[str, float] = field(default_factory=dict)


# ── Default model presets ──────────────────────────────────────────────

_MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "LLaMA-70B": {
        "num_layers": 80,
        "hidden_dim": 8192,
        "num_heads": 64,
        "head_dim": 128,
        "intermediate_dim": 28672,
        "vocab_size": 32000,
        "flops_per_layer": 4.0e12,
        "params_per_layer": 875_000_000,
    },
    "LLaMA-13B": {
        "num_layers": 40,
        "hidden_dim": 5120,
        "num_heads": 40,
        "head_dim": 128,
        "intermediate_dim": 13824,
        "vocab_size": 32000,
        "flops_per_layer": 7.0e11,
        "params_per_layer": 325_000_000,
    },
    "LLaMA-7B": {
        "num_layers": 32,
        "hidden_dim": 4096,
        "num_heads": 32,
        "head_dim": 128,
        "intermediate_dim": 11008,
        "vocab_size": 32000,
        "flops_per_layer": 3.5e11,
        "params_per_layer": 218_000_000,
    },
    "GPT-3-175B": {
        "num_layers": 96,
        "hidden_dim": 12288,
        "num_heads": 96,
        "head_dim": 128,
        "intermediate_dim": 49152,
        "vocab_size": 50257,
        "flops_per_layer": 1.0e13,
        "params_per_layer": 1_822_000_000,
    },
}


def get_model_preset(name: str) -> ModelConfig:
    """Resolve a model preset by name, falling back to a reasonable default."""
    raw = _MODEL_PRESETS.get(name)
    if raw is not None:
        return ModelConfig(name=name, **raw)
    # Fallback: create a generic config
    return ModelConfig(
        name=name,
        num_layers=32,
        hidden_dim=4096,
        num_heads=32,
        head_dim=128,
        intermediate_dim=11008,
        vocab_size=32000,
        flops_per_layer=3.5e11,
        params_per_layer=218_000_000,
    )


class ClusterSimulator:
    """Simulate distributed pipeline inference on a cluster of nodes.

    Models compute time, communication overhead, pipeline bubbles, and
    memory constraints to estimate end-to-end latency and throughput.
    Supports pipeline parallelism, tensor parallelism, and micro-batching.
    """

    def __init__(self) -> None:
        self._last_result: SimulatedPipelineResult | None = None
        self._run_history: list[SimulatedPipelineResult] = []
        self._nodes: dict[str, NodeSpec] = {}

    def add_node(self, spec: NodeSpec) -> None:
        """Register a node in the simulation cluster."""
        self._nodes[spec.node_id] = spec

    def add_nodes(self, *specs: NodeSpec) -> None:
        """Register multiple nodes."""
        for spec in specs:
            self._nodes[spec.node_id] = spec

    def clear_nodes(self) -> None:
        """Remove all registered nodes."""
        self._nodes.clear()

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    # ── Analytical cost helpers ──────────────────────────────────────────

    @staticmethod
    def _compute_time_per_layer(
        model: ModelConfig,
        batch_size: int,
        seq_len: int,
        compute_tflops: float,
        memory_bandwidth_gbps: float,
        memory_gb: float,
        tensor_parallelism: int = 1,
    ) -> tuple[float, float]:
        """Estimate compute and memory-bound time for one layer.

        Returns (compute_ms, memory_ms).
        """
        tp = max(tensor_parallelism, 1)

        # Compute-bound time: FLOPS scaled by batch * seq
        per_gpu_flops = model.flops_per_layer / tp
        total_flops = per_gpu_flops * batch_size * seq_len
        compute_ms = (total_flops / (compute_tflops * 1e12)) * 1000.0

        # Memory-bound time: weight loading + KV cache reads
        weights_bytes = model.params_per_layer * 2  # FP16 = 2 bytes
        kv_bytes = (
            batch_size * seq_len * model.num_heads * model.head_dim * 2 * 2
        )  # K + V, FP16
        total_mem_bytes = (weights_bytes + kv_bytes) / tp

        mem_bw_bytes_ms = memory_bandwidth_gbps * 1e9 / 8 / 1000
        if mem_bw_bytes_ms > 0:
            memory_ms = total_mem_bytes / mem_bw_bytes_ms
        else:
            memory_ms = 0.0

        return compute_ms, memory_ms

    @staticmethod
    def _comm_time(
        model: ModelConfig,
        batch_size: int,
        seq_len: int,
        interconnect_gbps: float,
    ) -> float:
        """Estimate hidden-state communication time in ms between stages."""
        bytes_per_elem = 2  # FP16
        total_bytes = batch_size * seq_len * model.hidden_dim * bytes_per_elem
        if interconnect_gbps <= 0:
            return 0.0
        return (total_bytes * 8) / (interconnect_gbps * 1e9) * 1000.0

    @staticmethod
    def _bubble_overhead(
        num_stages: int,
        stage_time_ms: float,
        num_micro_batches: int,
    ) -> float:
        """Compute pipeline bubble overhead using 1F1B scheduling.

        Bubble ratio = (N - 1) / (M + N - 1) where N is stages and
        M is micro-batches.  The absolute overhead is bubble_ratio * latency.
        """
        if num_stages <= 1:
            return 0.0
        num_mb = max(num_micro_batches, 1)
        bubble_ratio = (num_stages - 1) / (num_mb + num_stages - 1)
        return bubble_ratio * stage_time_ms * num_stages

    # ── Pipeline simulation ──────────────────────────────────────────────

    def run_pipeline(
        self,
        model: str | ModelConfig,
        nodes: list[NodeSpec | str] | None = None,
        batch_size: int = 1,
        seq_len: int = 2048,
        pipeline_depth: int | None = None,
        micro_batch_size: int | None = None,
        tensor_parallelism: int = 1,
    ) -> SimulatedPipelineResult:
        """Simulate a single pipeline-parallel inference run.

        Args:
            model: Model name (preset) or ModelConfig.
            nodes: Node specs or node IDs to use. Defaults to all registered.
            batch_size: Total batch size.
            seq_len: Sequence length per sample.
            pipeline_depth: Number of pipeline stages. Defaults to node count.
            micro_batch_size: Micro-batch size for 1F1B. Defaults to
                              min(batch_size, 2).
            tensor_parallelism: Tensor parallelism degree within a node.

        Returns:
            SimulatedPipelineResult with latency, throughput, and utilization.
        """
        # Resolve model config
        model_cfg = (
            get_model_preset(model) if isinstance(model, str) else model
        )

        # Resolve node list
        if nodes is not None:
            resolved_nodes: list[NodeSpec] = []
            for n in nodes:
                if isinstance(n, NodeSpec):
                    resolved_nodes.append(n)
                elif isinstance(n, str) and n in self._nodes:
                    resolved_nodes.append(self._nodes[n])
                else:
                    # Create a default spec for unknown node references
                    resolved_nodes.append(NodeSpec(node_id=str(n)))
        else:
            resolved_nodes = list(self._nodes.values())

        if not resolved_nodes:
            resolved_nodes = [NodeSpec(node_id="default-node")]

        num_stages = pipeline_depth or len(resolved_nodes)
        micro_bs = micro_batch_size or min(batch_size, 2)
        num_micro_batches = max(1, batch_size // max(micro_bs, 1))

        # Assign layers to stages (equal split)
        layers_per_stage = model_cfg.num_layers / max(num_stages, 1)

        stage_times: dict[str, float] = {}
        memory_util: dict[str, float] = {}
        compute_util: dict[str, float] = {}

        max_stage_time = 0.0
        bottleneck_node = resolved_nodes[0].node_id

        for idx, node in enumerate(resolved_nodes):
            if idx >= num_stages:
                break

            effective_gpus = max(node.gpu_count, 1)
            tp_effective = min(tensor_parallelism, effective_gpus)

            # Compute time per layer on this node
            comp_ms, mem_ms = self._compute_time_per_layer(
                model_cfg,
                batch_size=micro_bs,
                seq_len=seq_len,
                compute_tflops=node.compute_tflops * tp_effective,
                memory_bandwidth_gbps=node.memory_bandwidth_gbps,
                memory_gb=node.memory_gb,
                tensor_parallelism=tp_effective,
            )

            # Total compute for the layers assigned to this stage
            stage_compute_ms = comp_ms * layers_per_stage
            stage_mem_ms = mem_ms * layers_per_stage

            # Communication time between stages
            comm_to_next = self._comm_time(
                model_cfg,
                batch_size=micro_bs,
                seq_len=seq_len,
                interconnect_gbps=node.interconnect_gbps,
            )

            # Total time per micro-batch at this stage
            # max(compute, memory) + communication (partially overlapped)
            stage_total = max(stage_compute_ms, stage_mem_ms) + comm_to_next
            stage_times[node.node_id] = round(stage_total, 2)

            # Utilization
            total_memory_gb = node.memory_gb * effective_gpus
            weights_mem_gb = (
                model_cfg.params_per_layer
                * layers_per_stage
                * 2
                / (1024**3)
            )
            kv_mem_gb = (
                batch_size
                * seq_len
                * model_cfg.num_heads
                * model_cfg.head_dim
                * 2
                * 2
                * layers_per_stage
                / (1024**3)
            )
            used_gb = (weights_mem_gb + kv_mem_gb) / tp_effective
            memory_util[node.node_id] = (
                min(used_gb / max(total_memory_gb, 0.001), 1.0)
                if total_memory_gb > 0
                else 0.0
            )
            compute_util[node.node_id] = (
                min(
                    stage_compute_ms / max(stage_compute_ms + comm_to_next, 0.001),
                    1.0,
                )
                if stage_compute_ms > 0
                else 0.0
            )

            if stage_total > max_stage_time:
                max_stage_time = stage_total
                bottleneck_node = node.node_id

        # Bubble overhead (1F1B scheduling)
        bubble_ms = self._bubble_overhead(
            num_stages, max_stage_time, num_micro_batches,
        )

        # Latency = pipeline fill + steady state + pipeline drain
        # Single micro-batch: N * stage_time
        # With micro-batching: (N + M - 1) * stage_time / M (approx)
        if num_micro_batches <= 1:
            latency_ms = num_stages * max_stage_time
        else:
            latency_ms = max_stage_time * (
                1 + (num_stages - 1) / num_micro_batches
            )

        # Throughput (tokens/second)
        total_tokens = batch_size * seq_len
        throughput_tok_s = (
            (total_tokens / latency_ms * 1000.0) if latency_ms > 0 else 0.0
        )

        result = SimulatedPipelineResult(
            latency_ms=round(latency_ms, 2),
            throughput_tok_s=round(throughput_tok_s, 2),
            compute_time_ms=round(
                sum(stage_times.values()) / max(len(stage_times), 1), 2
            ),
            comm_time_ms=round(
                self._comm_time(
                    model_cfg, micro_bs, seq_len,
                    resolved_nodes[0].interconnect_gbps,
                )
                * num_stages,
                2,
            ),
            bubble_overhead_ms=round(bubble_ms, 2),
            bottleneck_node=bottleneck_node,
            stage_times_ms=dict(stage_times),
            memory_utilization=dict(memory_util),
            compute_utilization=dict(compute_util),
        )

        self._last_result = result
        self._run_history.append(result)
        return result

    # ── Simulation helpers ───────────────────────────────────────────────

    def run_pipeline_1f1b(
        self,
        model: str | ModelConfig,
        nodes: list[NodeSpec | str] | None = None,
        batch_size: int = 1,
        seq_len: int = 2048,
        num_micro_batches: int = 4,
        tensor_parallelism: int = 1,
    ) -> SimulatedPipelineResult:
        """Simulate using explicit 1F1B (one-forward-one-backward) scheduling.

        This variant explicitly models the 1F1B steady-state where each
        stage alternates between forward and backward passes after the
        pipeline fills.

        Returns results with the bubble overhead computed via the standard
        1F1B formula.
        """
        resolved_nodes: list[NodeSpec] = []
        if nodes is not None:
            for n in nodes:
                if isinstance(n, NodeSpec):
                    resolved_nodes.append(n)
                elif isinstance(n, str) and n in self._nodes:
                    resolved_nodes.append(self._nodes[n])
                else:
                    resolved_nodes.append(NodeSpec(node_id=str(n)))
        else:
            resolved_nodes = list(self._nodes.values())

        if not resolved_nodes:
            resolved_nodes = [NodeSpec(node_id="default-node")]

        result = self.run_pipeline(
            model=model,
            nodes=resolved_nodes,
            batch_size=batch_size,
            seq_len=seq_len,
            pipeline_depth=len(resolved_nodes),
            micro_batch_size=(
                batch_size // max(num_micro_batches, 1)
            ),
            tensor_parallelism=tensor_parallelism,
        )
        return result

    def run_pipeline_tensor_parallel(
        self,
        model: str | ModelConfig,
        nodes: list[NodeSpec | str] | None = None,
        batch_size: int = 1,
        seq_len: int = 2048,
        tp_degree: int = 2,
    ) -> SimulatedPipelineResult:
        """Simulate with tensor parallelism across GPUs within each node.

        Tensor parallelism splits each layer's weight matrices across
        GPUs, reducing per-GPU memory and compute at the cost of
        all-reduce communication within the node.
        """
        return self.run_pipeline(
            model=model,
            nodes=nodes,
            batch_size=batch_size,
            seq_len=seq_len,
            tensor_parallelism=tp_degree,
        )

    def simulate_fsdp(
        self,
        model: str | ModelConfig,
        nodes: list[NodeSpec | str] | None = None,
        batch_size: int = 1,
        seq_len: int = 2048,
        shard_degree: int = 1,
    ) -> SimulatedPipelineResult:
        """Simulate with FSDP (Fully Sharded Data Parallelism) weight sharding.

        FSDP shards optimizer states, gradients, and parameters across
        GPUs, gathering them before each forward/backward pass.
        This adds all-gather communication overhead but reduces per-GPU
        memory footprint.
        """
        model_cfg = (
            get_model_preset(model) if isinstance(model, str) else model
        )
        resolved_nodes: list[NodeSpec] = []
        if nodes is not None:
            for n in nodes:
                if isinstance(n, NodeSpec):
                    resolved_nodes.append(n)
                elif isinstance(n, str) and n in self._nodes:
                    resolved_nodes.append(self._nodes[n])
                else:
                    resolved_nodes.append(NodeSpec(node_id=str(n)))
        else:
            resolved_nodes = list(self._nodes.values())

        if not resolved_nodes:
            resolved_nodes = [NodeSpec(node_id="default-node")]

        # FSDP all-gather cost per layer
        # Each all-gather communicates shard_degree * params_per_layer bytes
        sd = max(shard_degree, 1)
        params_per_stage = (
            model_cfg.params_per_layer
            * model_cfg.num_layers
            / max(len(resolved_nodes), 1)
        )
        all_gather_bytes = params_per_stage * 2 * (sd - 1) / sd  # FP16

        base_result = self.run_pipeline(
            model=model_cfg,
            nodes=resolved_nodes,
            batch_size=batch_size,
            seq_len=seq_len,
        )

        # Additional FSDP communication overhead
        avg_bw = (
            sum(n.interconnect_gbps for n in resolved_nodes)
            / max(len(resolved_nodes), 1)
        )
        fsdp_comm_ms = (
            (all_gather_bytes * 8) / (avg_bw * 1e9) * 1000.0
            if avg_bw > 0.0
            else 0.0
        )

        # FSDP reduces memory (weights are sharded)
        reduced_mem = 1.0 - (1.0 / sd)
        for node_id in base_result.memory_utilization:
            base_result.memory_utilization[node_id] = max(
                base_result.memory_utilization[node_id] - reduced_mem * 0.5,
                0.0,
            )

        adjusted_latency = base_result.latency_ms + fsdp_comm_ms
        total_tokens = batch_size * seq_len
        adjusted_throughput = (
            (total_tokens / adjusted_latency * 1000.0)
            if adjusted_latency > 0
            else 0.0
        )

        return SimulatedPipelineResult(
            latency_ms=round(adjusted_latency, 2),
            throughput_tok_s=round(adjusted_throughput, 2),
            compute_time_ms=base_result.compute_time_ms,
            comm_time_ms=round(base_result.comm_time_ms + fsdp_comm_ms, 2),
            bubble_overhead_ms=base_result.bubble_overhead_ms,
            bottleneck_node=base_result.bottleneck_node,
            stage_times_ms=base_result.stage_times_ms,
            memory_utilization=base_result.memory_utilization,
            compute_utilization=base_result.compute_utilization,
        )

    # ── Metrics ──────────────────────────────────────────────────────────

    def get_perf_metrics(
        self,
        model: str | ModelConfig = "LLaMA-7B",
        batch_sizes: list[int] | None = None,
        seq_len: int = 2048,
    ) -> PerfMetrics:
        """Aggregate performance metrics across multiple batch sizes.

        Runs the pipeline simulation across the provided batch sizes
        and returns a summary of latency, throughput, and utilization.
        """
        nodes = list(self._nodes.values())
        if not nodes:
            nodes = [NodeSpec(node_id="default-node")]

        batch_sizes = batch_sizes or [1, 2, 4, 8]
        latencies: list[float] = []
        throughputs: list[float] = []

        for bs in batch_sizes:
            result = self.run_pipeline(
                model=model,
                nodes=nodes,
                batch_size=bs,
                seq_len=seq_len,
            )
            latencies.append(result.latency_ms)
            throughputs.append(result.throughput_tok_s)

        avg_lat = sum(latencies) / max(len(latencies), 1)
        sorted_lat = sorted(latencies)
        p50 = sorted_lat[len(sorted_lat) // 2] if sorted_lat else 0.0
        p99 = (
            sorted_lat[int(len(sorted_lat) * 0.99)]
            if len(sorted_lat) >= 100
            else sorted_lat[-1]
            if sorted_lat
            else 0.0
        )

        return PerfMetrics(
            avg_latency_ms=round(avg_lat, 2),
            p50_latency_ms=round(p50, 2),
            p99_latency_ms=round(p99, 2),
            avg_throughput_tok_s=round(
                sum(throughputs) / max(len(throughputs), 1), 2
            ),
            peak_throughput_tok_s=round(max(throughputs), 2),
            avg_compute_utilization=0.0,
            avg_memory_utilization=0.0,
            node_utilization={},
        )

    def summary(self) -> str:
        """Human-readable summary of registered nodes and last result."""
        lines: list[str] = [
            f"ClusterSimulator: {len(self._nodes)} nodes registered"
        ]
        for node_id, spec in self._nodes.items():
            lines.append(
                f"  {node_id}: {spec.gpu_name} x{spec.gpu_count} "
                f"({spec.compute_tflops} TFLOPS, {spec.memory_gb} GB)"
            )
        if self._last_result is not None:
            lines.append("Last simulation:")
            lines.append(
                f"  Latency: {self._last_result.latency_ms} ms"
            )
            lines.append(
                f"  Throughput: {self._last_result.throughput_tok_s} tok/s"
            )
            lines.append(
                f"  Bottleneck: {self._last_result.bottleneck_node}"
            )
        else:
            lines.append("No simulations run yet")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ClusterSimulator(nodes={len(self._nodes)}, "
            f"runs={len(self._run_history)})"
        )
