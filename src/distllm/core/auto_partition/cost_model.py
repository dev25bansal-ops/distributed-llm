from __future__ import annotations

from dataclasses import dataclass

from distllm.core.auto_partition.profiles import GPUProfile, LayerWeights
from distllm.core.auto_partition.topology import TopologyGraph


@dataclass
class NodeCost:
    """Estimated cost for running a set of layers on a specific node."""
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


class PartitionCostModel:
    """Estimates per-node latency for a given partition assignment.

    Given the hardware profile and layer weights, computes:
    - Compute time: total FLOPs assigned / GPU TFLOPS
    - Communication time: activation size / link bandwidth
    - Memory check: weights + KV cache < GPU memory

    Usage:
        cost_model = PartitionCostModel(gpu_profiles, layer_weights, topology)
        cost = cost_model.evaluate(node_id="node-0", start=0, end=10, batch_size=1, seq_len=4096)
    """

    def __init__(
        self,
        gpu_profiles: list[GPUProfile] | dict[str, GPUProfile],
        layer_weights: list[LayerWeights],
        topology: TopologyGraph,
    ):
        if isinstance(gpu_profiles, dict):
            self._gpu_profiles = {str(k): v for k, v in gpu_profiles.items()}
        else:
            self._gpu_profiles = {str(p.gpu_id): p for p in gpu_profiles}
        self._layer_weights = layer_weights
        self._topology = topology

    def evaluate(
        self,
        node_id: str,
        start_layer_id: int,
        end_layer_id: int,
        batch_size: int = 1,
        seq_len: int = 4096,
    ) -> NodeCost:
        """Estimate the cost of assigning layers [start, end) to a node.

        Args:
            node_id: The target node identifier.
            start_layer_id: Index of first layer (inclusive).
            end_layer_id: Index of last layer (exclusive).
            batch_size: Batch size for activation sizing.
            seq_len: Sequence length for activation sizing.

        Returns:
            NodeCost with compute, communication, and memory estimates.
        """
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
        mem_bw = gpu.memory_bandwidth_gbps
        mem_available = gpu.total_memory_bytes

        # Compute time: flops_per_seq is flops per token, multiply by seq_len
        total_flops_per_token = sum(l.flops_per_seq for l in layers if l.layer_type == "transformer")
        total_flops = total_flops_per_token * batch_size * seq_len
        compute_ms = self._flops_to_ms(total_flops, tflops)

        # Memory check: weights + KV cache + activations must fit
        weights_mem = sum(l.weight_memory_bytes for l in layers)
        kv_mem = sum(l.kv_cache_bytes_per_token for l in layers) * batch_size * seq_len
        act_per_token = max(
            (l.activation_memory_bytes for l in layers if l.activation_memory_bytes > 0),
            default=seq_len * 2,
        )
        activation_mem = act_per_token * batch_size * seq_len
        total_mem = weights_mem + kv_mem + activation_mem

        fits = total_mem <= mem_available * 0.9

        # Communication time: send hidden state activations to next pipeline stage
        comm_ms = 0.0
        prev_node = self._get_prev_node(node_id)
        if prev_node and layers:
            last_layer = layers[-1]
            act_size_per_token = last_layer.activation_memory_bytes or seq_len * 2
            activation_bytes = act_size_per_token * batch_size * seq_len
            bw = self._topology.get_bandwidth(prev_node, node_id)
            if bw > 0:
                comm_ms = (activation_bytes * 8) / (bw * 1e9) * 1000

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

    def evaluate_partition(
        self,
        partition: list[tuple[str, int, int]],
        batch_size: int = 1,
        seq_len: int = 4096,
    ) -> list[NodeCost]:
        """Evaluate a full partition assignment.

        Args:
            partition: List of (node_id, start_layer, end_layer) tuples.
            batch_size: Batch size.
            seq_len: Sequence length.

        Returns:
            List of NodeCost, one per node.
        """
        costs: list[NodeCost] = []
        for node_id, start, end in partition:
            cost = self.evaluate(node_id, start, end, batch_size, seq_len)
            costs.append(cost)
        return costs

    def max_latency(
        self,
        partition: list[tuple[str, int, int]],
        batch_size: int = 1,
        seq_len: int = 4096,
    ) -> float:
        """Compute the maximum per-node latency for a partition."""
        costs = self.evaluate_partition(partition, batch_size, seq_len)
        return max(c.total_time_ms for c in costs) if costs else 0.0

    def combined_throughput(
        self,
        partition: list[tuple[str, int, int]],
        batch_size: int = 1,
        seq_len: int = 4096,
    ) -> float:
        """Estimate throughput in tokens/sec for a partition."""
        costs = self.evaluate_partition(partition, batch_size, seq_len)
        if not costs:
            return 0.0
        bottleneck_ms = max(c.total_time_ms for c in costs)
        if bottleneck_ms <= 0:
            return 0.0
        return (batch_size * seq_len) / (bottleneck_ms / 1000.0)

    def _estimate_compute_cpu(
        self, layers: list[LayerWeights], batch_size: int, seq_len: int
    ) -> float:
        total_flops = sum(l.flops_per_seq for l in layers) * batch_size * seq_len
        return self._flops_to_ms(total_flops, tflops=1.0) * 50

    def _flops_to_ms(self, flops: int, tflops: float) -> float:
        if tflops <= 0:
            return 0.0
        return (flops / (tflops * 1e12)) * 1000.0

    def _get_prev_node(self, node_id: str) -> str | None:
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
