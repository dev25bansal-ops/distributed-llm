from __future__ import annotations

import math
from dataclasses import dataclass

from distllm.dist.partition.profiles import GPUProfile, LayerWeights
from distllm.dist.partition.topology import TopologyGraph


@dataclass
class NodeCost:
    node_id: str
    start_layer: int
    end_layer: int
    compute_time_ms: float = 0.0
    communication_time_ms: float = 0.0
    total_time_ms: float = 0.0
    memory_bytes: int = 0
    memory_available_bytes: int = 0
    fits_in_memory: bool = True

    @property
    def memory_utilization(self) -> float:
        if self.memory_available_bytes == 0:
            return 0.0
        return self.memory_bytes / self.memory_available_bytes


# Memory fragmentation model: as utilization increases, fragmentation
# reduces effective available memory.  These factors represent the
# fraction of VRAM that remains usable at a given utilization band.
_FRAGMENTATION_FACTORS: list[tuple[float, float]] = [
    (0.50, 1.00),  # Up to 50% utilized → no fragmentation loss
    (0.60, 0.98),  # 50-60% → 2% loss
    (0.70, 0.95),  # 60-70% → 5% loss
    (0.80, 0.90),  # 70-80% → 10% loss
    (0.90, 0.82),  # 80-90% → 18% loss
    (1.00, 0.70),  # 90-100% → 30% loss
]


def _fragmentation_factor(utilization: float) -> float:
    """Return the fraction of memory that remains usable given utilization.

    Models real GPU memory fragmentation: high utilization causes
    non-contiguous free blocks that reduce effective capacity.
    """
    if utilization <= 0.5:
        return 1.0
    for threshold, factor in _FRAGMENTATION_FACTORS:
        if utilization <= threshold:
            return factor
    return 0.70


class PartitionCostModel:
    def __init__(
        self,
        gpu_profiles: list[GPUProfile] | dict[str, GPUProfile],
        layer_weights: list[LayerWeights],
        topology: TopologyGraph,
        pipeline_node_order: list[str] | None = None,
    ):
        if isinstance(gpu_profiles, dict):
            self._gpu_profiles = {str(k): v for k, v in gpu_profiles.items()}
        else:
            self._gpu_profiles = {str(p.gpu_id): p for p in gpu_profiles}
        self._layer_weights = layer_weights
        self._topology = topology
        # M1: Allow explicit pipeline ordering for non-linear topologies
        self._pipeline_order = pipeline_node_order

    def evaluate(
        self,
        node_id: str,
        start_layer_id: int,
        end_layer_id: int,
        batch_size: int = 1,
        seq_len: int = 4096,
        prev_node_id: str | None = None,
    ) -> NodeCost:
        layers = self._layer_weights[start_layer_id:end_layer_id]
        if not layers:
            return NodeCost(
                node_id=node_id,
                start_layer=start_layer_id,
                end_layer=end_layer_id,
                fits_in_memory=True,
            )

        gpu = self._gpu_profiles.get(node_id)
        if gpu is None:
            cpu_ms = self._estimate_compute_cpu(layers, batch_size, seq_len)
            return NodeCost(
                node_id=node_id,
                start_layer=start_layer_id,
                end_layer=end_layer_id,
                compute_time_ms=round(cpu_ms, 2),
                total_time_ms=round(cpu_ms, 2),
                fits_in_memory=True,
            )

        tflops = gpu.compute_tflops
        mem_available = gpu.total_memory_bytes

        # 3.1: Apply utilization factor to peak TFLOPS
        hidden_size = 0
        for l in layers:
            if l.layer_type == "transformer" and l.weight_memory_bytes > 0:
                h_sq = l.weight_memory_bytes / (7 * 2)
                hidden_size = max(int(h_sq ** 0.5) if h_sq > 0 else 0, hidden_size)
                break
        util_factor = self._compute_utilization_factor(hidden_size, batch_size, seq_len, tflops)
        effective_tflops = tflops * util_factor

        # 3.1: Separate attention (memory-bound) and MLP (compute-bound)
        attn_flops, mlp_flops = self._estimate_compute_split(layers, batch_size, seq_len)
        total_flops = attn_flops + mlp_flops

        # 3.1: Compute with memory latency overlap model
        # MLP is compute-bound, attention is partially memory-bound
        compute_ms = self._flops_to_ms(total_flops, effective_tflops)

        # 3.1: Memory latency (partially overlapped with compute)
        weights_mem = sum(l.weight_memory_bytes for l in layers)
        kv_mem = sum(l.kv_cache_bytes_per_token for l in layers) * batch_size * seq_len
        act_per_token = max(
            (l.activation_memory_bytes for l in layers if l.activation_memory_bytes > 0),
            default=seq_len * 2,
        )
        activation_mem = act_per_token * batch_size * seq_len
        total_mem = weights_mem + kv_mem + activation_mem

        # Memory transfer time (partially overlapped with compute)
        mem_bw_gbps = gpu.memory_bandwidth_gbps
        if mem_bw_gbps > 0:
            # Weight loading: not overlapped (must happen first)
            weight_load_ms = (weights_mem * 8) / (mem_bw_gbps * 1e9) * 1000
            # KV cache + activation: overlapped with compute at ~60%
            kv_act_ms = ((kv_mem + activation_mem) * 8) / (mem_bw_gbps * 1e9) * 1000
            memory_ms = weight_load_ms + kv_act_ms * 0.4
            # Total: max(compute, memory) with partial overlap
            compute_ms = max(compute_ms, memory_ms)

        # M6: Apply fragmentation-aware memory check
        raw_utilization = total_mem / mem_available if mem_available > 0 else 1.0
        frag_factor = _fragmentation_factor(raw_utilization)
        effective_available = mem_available * frag_factor
        fits = total_mem <= effective_available * 0.9

        # M1: Communication cost using topology graph (not just linear prev→next)
        # 3.1: Apply contention factor for shared links
        comm_ms = 0.0
        prev = prev_node_id or self._get_prev_node(node_id, layers)
        if prev and layers:
            last_layer = layers[-1]
            act_size_per_token = last_layer.activation_memory_bytes or seq_len * 2
            activation_bytes = act_size_per_token * batch_size * seq_len
            bw = self._topology.get_bandwidth(prev, node_id)
            if bw > 0:
                contention = self._estimate_contention_factor(prev, node_id)
                effective_bw = bw * contention
                comm_ms = (activation_bytes * 8) / (effective_bw * 1e9) * 1000

        total_ms = compute_ms + comm_ms

        return NodeCost(
            node_id=node_id,
            start_layer=start_layer_id,
            end_layer=end_layer_id,
            compute_time_ms=round(compute_ms, 2),
            communication_time_ms=round(comm_ms, 2),
            total_time_ms=round(total_ms, 2),
            memory_bytes=total_mem,
            memory_available_bytes=mem_available,
            fits_in_memory=fits,
        )

    def weights_memory_bytes(self, start_layer_id: int, end_layer_id: int) -> int:
        """Total weight (model parameter) memory for a layer range, in bytes.

        Used by quantization-aware cost models so that weight-quantization
        reductions are applied only to the weights portion of the footprint
        (KV cache and activations are not shrunk by weight quantization).
        """
        return int(
            sum(
                l.weight_memory_bytes
                for l in self._layer_weights[start_layer_id:end_layer_id]
            )
        )

    def evaluate_partition(
        self,
        partition: list[tuple[str, int, int]],
        batch_size: int = 1,
        seq_len: int = 4096,
    ) -> list[NodeCost]:
        costs: list[NodeCost] = []
        for idx, (node_id, start, end) in enumerate(partition):
            prev = partition[idx - 1][0] if idx > 0 else None
            cost = self.evaluate(node_id, start, end, batch_size, seq_len, prev_node_id=prev)
            costs.append(cost)
        return costs

    def max_latency(
        self,
        partition: list[tuple[str, int, int]],
        batch_size: int = 1,
        seq_len: int = 4096,
    ) -> float:
        costs = self.evaluate_partition(partition, batch_size, seq_len)
        return max(c.total_time_ms for c in costs) if costs else 0.0

    def combined_throughput(
        self,
        partition: list[tuple[str, int, int]],
        batch_size: int = 1,
        seq_len: int = 4096,
    ) -> float:
        costs = self.evaluate_partition(partition, batch_size, seq_len)
        if not costs:
            return 0.0
        bottleneck_ms = max(c.total_time_ms for c in costs)
        if bottleneck_ms <= 0:
            return 0.0
        return (batch_size * seq_len) / (bottleneck_ms / 1000.0)

    def pipeline_latency(
        self,
        partition: list[tuple[str, int, int]],
        batch_size: int = 1,
        seq_len: int = 4096,
        num_pipeline_stages: int | None = None,
    ) -> float:
        """M2: Compute pipeline latency including bubble overhead.

        Pipeline execution has warm-up and cool-down phases that create
        "bubbles" where some stages are idle.  For N stages processing
        a single micro-batch:

            bubble_overhead = (N - 1) * stage_time

        For continuous batching with M micro-batches, the effective
        overhead is amortized:

            effective_latency = stage_time + (N - 1) * stage_time / M

        Args:
            partition: List of (node_id, start_layer, end_layer).
            batch_size: Micro-batch size.
            seq_len: Sequence length.
            num_pipeline_stages: Override for number of stages.
        """
        costs = self.evaluate_partition(partition, batch_size, seq_len)
        if not costs:
            return 0.0

        n_stages = num_pipeline_stages or len(costs)
        stage_time = max(c.total_time_ms for c in costs)

        if n_stages <= 1:
            return stage_time

        # Bubble overhead: (N-1) stages warm up + (N-1) stages cool down
        # For a single micro-batch, this is the dominant cost
        bubble_overhead = (n_stages - 1) * stage_time

        # With continuous batching (batch_size > 1), amortize over micro-batches
        num_micro_batches = max(1, batch_size)
        effective_latency = stage_time + bubble_overhead / num_micro_batches

        return effective_latency

    def _estimate_compute_cpu(
        self, layers: list[LayerWeights], batch_size: int, seq_len: int
    ) -> float:
        total_flops = sum(l.flops_per_seq for l in layers) * batch_size * seq_len
        return self._flops_to_ms(total_flops, tflops=1.0) * 50

    def _flops_to_ms(self, flops: int, tflops: float) -> float:
        if tflops <= 0:
            return 0.0
        return (flops / (tflops * 1e12)) * 1000.0

    def _compute_utilization_factor(
        self, hidden_size: int, batch_size: int, seq_len: int, tflops: float,
    ) -> float:
        """3.1: GPU utilization factor based on problem size.

        Real GPUs don't achieve peak TFLOPS — utilization depends on
        matrix dimensions, batch size, and memory access patterns.

        Returns a multiplier in (0.3, 0.95] applied to peak TFLOPS.
        """
        # Larger hidden_size → better matrix multiply efficiency
        # Diminishing returns past 4096
        dim_efficiency = min(hidden_size / 4096, 1.0) * 0.3 + 0.5

        # Larger batch → better compute utilization
        batch_efficiency = min(batch_size / 8, 1.0) * 0.2 + 0.7

        # Very long sequences can reduce efficiency due to memory pressure
        seq_penalty = 1.0
        if seq_len > 8192:
            seq_penalty = max(0.85, 1.0 - (seq_len - 8192) / 100000)

        return dim_efficiency * batch_efficiency * seq_penalty

    def _estimate_compute_split(
        self, layers: list[LayerWeights], batch_size: int, seq_len: int,
    ) -> tuple[float, float]:
        """3.1: Split compute into attention (memory-bound) and MLP (compute-bound).

        Attention is typically memory-bound (scales with seq² but has
        low arithmetic intensity).  MLP is compute-bound (scales linearly
        with high arithmetic intensity).

        Returns (attention_ms, mlp_ms).
        """
        attn_flops = 0
        mlp_flops = 0

        for layer in layers:
            if layer.layer_type != "transformer":
                continue
            # Attention: QKV projection + attention computation + O projection
            # Approximate: attention FLOPS ≈ 2 * h * seq² per head (memory-bound)
            # MLP FLOPS ≈ 3 * 2 * h * intermediate (compute-bound)
            # Use rough 30/70 split of total transformer FLOPS
            total = layer.flops_per_seq
            attn_flops += int(total * 0.3)
            mlp_flops += int(total * 0.7)

        total_attn = attn_flops * batch_size * seq_len
        total_mlp = mlp_flops * batch_size * seq_len

        return float(total_attn), float(total_mlp)

    def _estimate_contention_factor(
        self, prev_node: str, node_id: str, num_active_links: int = 1,
    ) -> float:
        """3.1: Bandwidth contention factor for shared links.

        When multiple pipeline stages communicate simultaneously on
        shared NIC/PCIe lanes, effective bandwidth is reduced.

        Returns a multiplier in (0.3, 1.0].
        """
        link = None
        for l in self._topology.links:
            if (l.source == prev_node and l.target == node_id) or \
               (l.source == node_id and l.target == prev_node):
                link = l
                break

        if link is None:
            return 0.8  # Default 20% contention for unknown links

        # NVLink: minimal contention (dedicated links)
        if link.is_nvlink:
            return 0.95

        # Infiniband: moderate contention (shared fabric)
        if link.is_infiniband:
            return max(0.6, 1.0 - 0.1 * num_active_links)

        # Ethernet: high contention (shared NIC)
        return max(0.3, 1.0 - 0.15 * num_active_links)

    def _get_prev_node(self, node_id: str, layers: list[LayerWeights] | None = None) -> str | None:
        # M1: Use explicit pipeline order if available
        if self._pipeline_order:
            try:
                idx = self._pipeline_order.index(node_id)
                return self._pipeline_order[idx - 1] if idx > 0 else None
            except ValueError:
                return None

        # Fallback: linear pipeline based on topology node ordering
        nodes = self._topology.node_ids
        try:
            idx = nodes.index(node_id)
            return nodes[idx - 1] if idx > 0 else None
        except ValueError:
            return None

    def cost_summary(
        self, node_id: str, start: int, end: int,
    ) -> str:
        cost = self.evaluate(node_id, start, end)
        return (
            f"Node {cost.node_id} layers [{cost.start_layer}, {cost.end_layer}): "
            f"compute={cost.compute_time_ms}ms, "
            f"comm={cost.communication_time_ms}ms, "
            f"total={cost.total_time_ms}ms, "
            f"mem={cost.memory_bytes/(1024**2):.0f}MB/"
            f"{cost.memory_available_bytes/(1024**3):.1f}GB"
            f"{' [OK]' if cost.fits_in_memory else ' [OOM]'}"
        )

    # ── 4.5: Token-Aware Cost Model ──────────────────────────────────────

    def evaluate_token_aware(
        self,
        node_id: str,
        start_layer_id: int,
        end_layer_id: int,
        batch_size: int = 1,
        prefill_seq_len: int = 4096,
        num_decode_tokens: int = 1,
        prev_node_id: str | None = None,
    ) -> NodeCost:
        """4.5: Token-aware cost model with separate prefill/decode phases.

        Real LLM inference has two distinct phases:
        - Prefill: Compute-bound, processes all input tokens at once.
          FLOPS scale as O(batch * seq_len²) for attention, O(batch * seq_len) for MLP.
        - Decode: Memory-bound, generates one token at a time.
          Latency dominated by KV cache reads + weight loading.

        This method models both phases separately and sums them.

        Args:
            node_id: Target node.
            start_layer_id: First layer index.
            end_layer_id: Last layer index (exclusive).
            batch_size: Batch size.
            prefill_seq_len: Input sequence length for prefill.
            num_decode_tokens: Number of tokens to generate.
            prev_node_id: Previous node in pipeline (for comm cost).
        """
        layers = self._layer_weights[start_layer_id:end_layer_id]
        if not layers:
            return NodeCost(
                node_id=node_id, start_layer=start_layer_id,
                end_layer=end_layer_id, fits_in_memory=True,
            )

        gpu = self._gpu_profiles.get(node_id)
        if gpu is None:
            cpu_ms = self._estimate_compute_cpu(layers, batch_size, prefill_seq_len)
            return NodeCost(
                node_id=node_id, start_layer=start_layer_id,
                end_layer=end_layer_id, compute_time_ms=round(cpu_ms, 2),
                total_time_ms=round(cpu_ms, 2), fits_in_memory=True,
            )

        tflops = gpu.compute_tflops
        mem_bw_gbps = gpu.memory_bandwidth_gbps
        mem_available = gpu.total_memory_bytes

        # Memory footprint
        weights_mem = sum(l.weight_memory_bytes for l in layers)
        kv_per_token = sum(l.kv_cache_bytes_per_token for l in layers)
        kv_mem_prefill = kv_per_token * batch_size * prefill_seq_len
        kv_mem_decode = kv_per_token * batch_size * (prefill_seq_len + num_decode_tokens)
        total_mem = weights_mem + kv_mem_decode

        # ── Prefill phase ─────────────────────────────────────────────────
        # Attention: O(batch * seq_len² * hidden) — compute-bound
        # MLP: O(batch * seq_len * hidden * intermediate) — compute-bound
        attn_flops, mlp_flops = self._estimate_compute_split(layers, batch_size, prefill_seq_len)
        prefill_flops = attn_flops + mlp_flops

        util = self._compute_utilization_factor(0, batch_size, prefill_seq_len, tflops)
        effective_tflops = tflops * util

        prefill_compute_ms = self._flops_to_ms(prefill_flops, effective_tflops)

        # Prefill memory: load weights + write KV cache
        if mem_bw_gbps > 0:
            prefill_mem_bytes = weights_mem + kv_mem_prefill
            prefill_mem_ms = (prefill_mem_bytes * 8) / (mem_bw_gbps * 1e9) * 1000
            prefill_ms = max(prefill_compute_ms, prefill_mem_ms)
        else:
            prefill_ms = prefill_compute_ms

        # ── Decode phase ──────────────────────────────────────────────────
        # Each decode step: load weights + read full KV cache + small compute
        # Compute is tiny (one token), memory is the bottleneck
        decode_compute_ms = self._flops_to_ms(
            sum(l.flops_per_seq for l in layers if l.layer_type == "transformer"),
            effective_tflops,
        )

        if mem_bw_gbps > 0:
            # Per-step: read weights + read KV cache (all prior tokens)
            avg_kv_read = kv_per_token * batch_size * (prefill_seq_len + num_decode_tokens / 2)
            decode_mem_per_step = weights_mem + avg_kv_read
            decode_mem_ms = (decode_mem_per_step * 8) / (mem_bw_gbps * 1e9) * 1000
            decode_per_step_ms = max(decode_compute_ms, decode_mem_ms)
        else:
            decode_per_step_ms = decode_compute_ms

        decode_total_ms = decode_per_step_ms * num_decode_tokens

        # ── Communication cost ────────────────────────────────────────────
        comm_ms = 0.0
        prev = prev_node_id or self._get_prev_node(node_id, layers)
        if prev and layers:
            last_layer = layers[-1]
            act_size = last_layer.activation_memory_bytes or prefill_seq_len * 2
            act_bytes = act_size * batch_size * prefill_seq_len
            bw = self._topology.get_bandwidth(prev, node_id)
            if bw > 0:
                contention = self._estimate_contention_factor(prev, node_id)
                comm_ms = (act_bytes * 8) / (bw * contention * 1e9) * 1000

        total_ms = prefill_ms + decode_total_ms + comm_ms

        raw_util = total_mem / mem_available if mem_available > 0 else 1.0
        frag_factor = _fragmentation_factor(raw_util)
        fits = total_mem <= mem_available * frag_factor * 0.9

        return NodeCost(
            node_id=node_id,
            start_layer=start_layer_id,
            end_layer=end_layer_id,
            compute_time_ms=round(prefill_ms + decode_total_ms, 2),
            communication_time_ms=round(comm_ms, 2),
            total_time_ms=round(total_ms, 2),
            memory_bytes=total_mem,
            memory_available_bytes=mem_available,
            fits_in_memory=fits,
        )

    def combined_throughput_token_aware(
        self,
        partition: list[tuple[str, int, int]],
        batch_size: int = 1,
        prefill_seq_len: int = 4096,
        num_decode_tokens: int = 1,
    ) -> float:
        """4.5: Throughput using token-aware cost model.

        Returns tokens/second for the decode phase (the steady-state metric).
        """
        costs: list[NodeCost] = []
        for idx, (node_id, start, end) in enumerate(partition):
            prev = partition[idx - 1][0] if idx > 0 else None
            cost = self.evaluate_token_aware(
                node_id, start, end, batch_size,
                prefill_seq_len, num_decode_tokens, prev,
            )
            costs.append(cost)

        if not costs:
            return 0.0
        bottleneck_ms = max(c.total_time_ms for c in costs)
        if bottleneck_ms <= 0:
            return 0.0
        return (batch_size * num_decode_tokens) / (bottleneck_ms / 1000.0)
