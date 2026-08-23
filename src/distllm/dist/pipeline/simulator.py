"""Pipeline simulator for analytical cost modeling."""

from __future__ import annotations

import math


class PipelineSimulator:
    """Simulate pipeline performance without real nodes.

    Uses analytical cost models to estimate latency, throughput, and
    bottlenecks for different pipeline strategies.
    """

    FLOPS_PER_LAYER: dict[str, float] = {
        "70B": 4.0e12,
        "13B": 7.0e11,
        "7B": 3.5e11,
        "3B": 1.5e11,
        "1B": 5.0e10,
    }

    def __init__(
        self,
        model_size: str = "7B",
        gpu_tflops: float = 312.0,
        gpu_bandwidth_gbps: float = 600.0,
        interconnect_gbps: float = 50.0,
        hidden_dim: int = 4096,
        num_heads: int = 32,
        head_dim: int = 128,
        vocab_size: int = 32000,
        activation_ratio: float = 3.5,
    ):
        self.model_size = model_size
        self.gpu_tflops = gpu_tflops
        self.gpu_bandwidth_gbps = gpu_bandwidth_gbps
        self.interconnect_gbps = interconnect_gbps
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.vocab_size = vocab_size
        self.activation_ratio = activation_ratio

    def _compute_time_per_layer(self, seq_len: int, batch_size: int) -> float:
        """Estimate compute time for one layer in milliseconds."""
        flops_per_layer = self.FLOPS_PER_LAYER.get(self.model_size, 3.5e11)
        total_flops = flops_per_layer * batch_size * seq_len
        compute_ms = (total_flops / (self.gpu_tflops * 1e12)) * 1000
        return max(compute_ms, 0.01)

    def _comm_time_hidden(self, batch_size: int) -> float:
        """Estimate hidden-state communication time in milliseconds."""
        bytes_per_elem = 2
        total_bytes = batch_size * self.hidden_dim * bytes_per_elem
        comm_ms = (total_bytes / (self.interconnect_gbps * 1e9 / 8)) * 1000
        return comm_ms

    def _comm_time_kv(self, seq_len: int, batch_size: int, num_layers_per_node: int) -> float:
        """Estimate KV cache transfer time in milliseconds."""
        bytes_per_elem = 2
        layers = max(num_layers_per_node, 1)
        kv_per_layer = batch_size * seq_len * self.num_heads * self.head_dim * 2
        total_bytes = kv_per_layer * layers * bytes_per_elem
        comm_ms = (total_bytes / (self.gpu_bandwidth_gbps * 1e9 / 8)) * 1000
        return comm_ms

    def simulate(
        self,
        num_nodes: int,
        num_layers: int,
        batch_size: int = 1,
        seq_len: int = 2048,
        num_stages: int | None = None,
        num_micro_batches: int = 4,
    ) -> dict:
        """Simulate pipeline performance and compare strategies."""
        if num_stages is None:
            if num_nodes <= 4:
                num_stages = max(1, num_nodes)
            elif num_nodes <= 16:
                num_stages = max(1, int(num_nodes**0.5))
            else:
                num_stages = max(2, int(math.log2(num_nodes)))

        layers_per_node = num_layers / max(num_nodes, 1)
        t_compute = self._compute_time_per_layer(seq_len, batch_size)
        t_comm_hidden = self._comm_time_hidden(batch_size)
        t_comm_kv = self._comm_time_kv(seq_len, batch_size, int(layers_per_node))

        seq_latency_ms = num_nodes * (t_compute + t_comm_hidden + t_comm_kv)
        seq_throughput = 1000.0 / (seq_latency_ms / batch_size) if seq_latency_ms > 0 else 0
        seq_bottleneck = "compute" if t_compute > t_comm_hidden + t_comm_kv else "communication"

        overlap_latency_ms = t_compute + t_comm_hidden + t_comm_kv + (num_nodes - 1) * max(t_compute, t_comm_hidden + t_comm_kv)
        overlap_throughput = 1000.0 / (overlap_latency_ms / batch_size) if overlap_latency_ms > 0 else 0
        overlap_bottleneck = "compute" if t_compute > t_comm_hidden + t_comm_kv else "communication"

        bubble_ratio = (num_nodes - 1) / (num_micro_batches + num_nodes - 1)
        async_latency_ms = seq_latency_ms * (1 - bubble_ratio * 0.5)
        async_throughput = num_micro_batches * 1000.0 / async_latency_ms if async_latency_ms > 0 else 0

        nodes_per_stage = max(1, num_nodes // max(num_stages, 1))
        stage_compute = nodes_per_stage * t_compute
        stage_comm = t_comm_hidden + t_comm_kv
        staged_latency_ms = stage_compute + stage_comm * num_stages + (num_stages - 1) * max(stage_compute, stage_comm)
        staged_throughput = 1000.0 / (staged_latency_ms / batch_size) if staged_latency_ms > 0 else 0

        per_node_time = t_compute + t_comm_hidden + t_comm_kv

        return {
            "config": {
                "model": f"LLaMA-{self.model_size}",
                "num_nodes": num_nodes,
                "num_layers": num_layers,
                "layers_per_node": layers_per_node,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "num_stages": num_stages,
                "num_micro_batches": num_micro_batches,
                "gpu_tflops": self.gpu_tflops,
                "interconnect_gbps": self.interconnect_gbps,
            },
            "per_node_estimate_ms": {
                "compute": round(t_compute, 2),
                "comm_hidden": round(t_comm_hidden, 2),
                "comm_kv": round(t_comm_kv, 2),
                "total": round(per_node_time, 2),
            },
            "strategies": {
                "sequential": {
                    "latency_ms": round(seq_latency_ms, 2),
                    "throughput_tok_s": round(seq_throughput, 2),
                    "bottleneck": seq_bottleneck,
                },
                "overlap": {
                    "latency_ms": round(overlap_latency_ms, 2),
                    "throughput_tok_s": round(overlap_throughput, 2),
                    "bottleneck": overlap_bottleneck,
                },
                "async_1f1b": {
                    "latency_ms": round(async_latency_ms, 2),
                    "throughput_tok_s": round(async_throughput, 2),
                    "bubble_ratio": round(bubble_ratio, 3),
                },
                "staged": {
                    "latency_ms": round(staged_latency_ms, 2),
                    "throughput_tok_s": round(staged_throughput, 2),
                    "stages": num_stages,
                    "nodes_per_stage": nodes_per_stage,
                },
            },
            "recommendation": self._recommend_strategy(
                seq_throughput, overlap_throughput, async_throughput, staged_throughput,
                num_nodes=num_nodes,
            ),
        }

    def _recommend_strategy(self, *throughputs: float, num_nodes: int) -> str:
        """Recommend best strategy based on estimated throughput."""
        labels = ["sequential", "overlap", "async_1f1b", "staged"]
        best_idx = max(range(len(throughputs)), key=lambda i: throughputs[i])
        best = labels[best_idx] if best_idx < len(labels) else "sequential"
        if num_nodes <= 2:
            if throughputs[1] > throughputs[0] * 1.1:
                return f"{best} (recommended for {num_nodes} nodes)"
            return f"sequential or overlap (simple setup, {num_nodes} nodes)"
        if num_nodes <= 4:
            return f"{best} (good balance for {num_nodes} nodes)"
        if num_nodes > 8:
            return f"{best} (staged or async recommended for {num_nodes}+ nodes)"
        return f"{best} (recommended for {num_nodes} nodes)"
