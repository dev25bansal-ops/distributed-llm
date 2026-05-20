from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from loguru import logger

from distllm.core.auto_partition.profiles import GPUProfiler, LayerWeights
from distllm.core.auto_partition.topology import TopologyProber, TopologyGraph
from distllm.core.auto_partition.cost_model import PartitionCostModel
from distllm.core.auto_partition.optimizer import PartitionOptimizer, PartitionSolution


class HardwareAwarePartitioner:
    """Orchestrates the full hardware-aware partitioning pipeline.

    Flow:
        1. Profile all GPUs (compute, memory, bandwidth)
        2. Profile inter-node topology (latency, bandwidth)
        3. Estimate layer weights from model config
        4. Build cost model from profiles + weights
        5. Run DP optimizer to find min-max latency partition
        6. Optionally persist plan

    Usage:
        partitioner = HardwareAwarePartitioner()
        solution = await partitioner.partition(
            model_name="llama-2-7b",
            node_ids=["node-0", "node-1", "node-2"],
            gpu_counts={"node-0": 1, "node-1": 1, "node-2": 1},
            hidden_size=4096, num_layers=32,
        )
        print(solution.summary())
    """

    def __init__(
        self,
        batch_size: int = 1,
        seq_len: int = 4096,
        allow_oom: bool = False,
        profile_dir: str | None = None,
    ):
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._allow_oom = allow_oom
        self._profile_dir = (
            Path(profile_dir).expanduser().resolve()
            if profile_dir
            else Path.home() / ".distllm" / "partitions"
        )
        self._profile_dir.mkdir(parents=True, exist_ok=True)

        self._gpu_profiler = GPUProfiler()
        self._topology_prober = TopologyProber()

        # Cached results
        self._gpu_profiles: list[Any] = []
        self._topology: TopologyGraph | None = None
        self._layer_weights: list[LayerWeights] = []
        self._solution: PartitionSolution | None = None
        self._last_partition_time: float = 0.0

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def partition(
        self,
        model_name: str | None = None,
        node_ids: list[str] | None = None,
        gpu_counts: dict[str, int] | None = None,
        hostnames: dict[str, str] | None = None,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_layers: int = 32,
        num_heads: int = 32,
        head_dim: int = 128,
        vocab_size: int = 32000,
        max_seq_len: int = 4096,
    ) -> PartitionSolution:
        """Run the full hardware-aware partitioning pipeline.

        Args:
            model_name: Optional model name for cache/profile lookup.
            node_ids: List of node identifiers.
            gpu_counts: Optional mapping of node_id -> GPU count.
            hostnames: Optional mapping of node_id -> hostname/IP.
            hidden_size: Model hidden dimension.
            intermediate_size: MLP intermediate dimension.
            num_layers: Number of transformer layers.
            num_heads: Number of attention heads.
            head_dim: Attention head dimension.
            vocab_size: Vocabulary size.
            max_seq_len: Maximum sequence length.

        Returns:
            PartitionSolution with optimal assignments.
        """
        t0 = time.time()

        # Step 1: Profile hardware
        logger.info("Profiling GPUs...")
        self._gpu_profiles = self._gpu_profiler.profile_all_gpus()

        nodes = node_ids or [f"node-{i}" for i in range(len(self._gpu_profiles))]
        gpu_cnt = gpu_counts or {n: 1 for n in nodes}

        logger.info(f"Profiling inter-node topology ({len(nodes)} nodes)...")
        self._topology = await self._topology_prober.probe(
            node_ids=nodes,
            hostnames=hostnames,
            gpu_counts=gpu_cnt,
            gpu_profiles={
                n: [self._gpu_profiler.profile_to_dict(p) for p in self._gpu_profiles]
                for n in nodes
            },
        )

        # Step 2: Estimate layer weights
        logger.info("Estimating layer weights...")
        self._layer_weights = self._gpu_profiler.estimate_layer_weights(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
        )

        # Step 3: Build cost model
        cost_model = PartitionCostModel(
            self._gpu_profiles, self._layer_weights, self._topology,
        )

        # Step 4: Run optimizer
        logger.info("Running partition optimizer...")
        optimizer = PartitionOptimizer(
            cost_model=cost_model,
            node_ids=nodes,
            batch_size=self._batch_size,
            seq_len=self._seq_len,
            allow_oom=self._allow_oom,
        )

        self._solution = optimizer.solve(num_layers + 2)  # + embed + lm_head

        self._last_partition_time = time.time() - t0

        logger.info(
            f"Partition found in {self._last_partition_time:.1f}s: "
            f"{self._solution.num_nodes} nodes, "
            f"max latency {self._solution.max_node_time_ms:.1f}ms, "
            f"throughput {self._solution.estimated_throughput_tok_s:.0f} tok/s"
        )

        # Step 5: Persist
        self._save_plan(model_name)
        return self._solution

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare_to_baselines(self) -> dict[str, Any] | None:
        """Compare the DP solution to equal and proportional splits."""
        if not self._solution:
            return None

        cost_model = PartitionCostModel(
            self._gpu_profiles, self._layer_weights, self._topology,
        )

        nodes = self._topology.node_ids if self._topology else []
        optimizer = PartitionOptimizer(
            cost_model=cost_model,
            node_ids=nodes,
            batch_size=self._batch_size,
            seq_len=self._seq_len,
            allow_oom=self._allow_oom,
        )

        total = len(self._layer_weights)
        return optimizer.compare_strategies(total, self._batch_size, self._seq_len)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def solution(self) -> PartitionSolution | None:
        return self._solution

    def get_layer_assignments(self) -> list[dict[str, Any]] | None:
        """Get per-layer assignments in a flat format."""
        if not self._solution:
            return None
        result: list[dict[str, Any]] = []
        for pt in self._solution.points:
            for layer_id in range(pt.start_layer, pt.end_layer):
                result.append({
                    "layer_id": layer_id,
                    "node_id": pt.node_id,
                    "layer_type": (
                        self._layer_weights[layer_id].layer_type
                        if layer_id < len(self._layer_weights)
                        else "unknown"
                    ),
                })
        return result

    def get_node_summaries(self) -> list[dict[str, Any]] | None:
        if not self._solution:
            return None
        cost_model = PartitionCostModel(
            self._gpu_profiles, self._layer_weights, self._topology,
        )
        summaries = []
        for pt in self._solution.points:
            cost = cost_model.evaluate(
                pt.node_id, pt.start_layer, pt.end_layer,
                self._batch_size, self._seq_len,
            )
            summaries.append({
                "node_id": pt.node_id,
                "layers": f"[{pt.start_layer}, {pt.end_layer})",
                "num_layers": pt.end_layer - pt.start_layer,
                "compute_time_ms": cost.compute_time_ms,
                "comm_time_ms": cost.communication_time_ms,
                "total_time_ms": cost.total_time_ms,
                "memory_bytes": cost.memory_bytes,
                "memory_gb": round(cost.memory_bytes / (1024**3), 2),
                "fits_in_memory": cost.fits_in_memory,
            })
        return summaries

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_plan(self, model_name: str | None) -> None:
        if not self._solution:
            return
        name = (model_name or "unknown").replace("/", "_")
        path = self._profile_dir / f"{name}_partition_plan.json"
        try:
            data = {
                "created_at": time.time(),
                "model": model_name,
                "partition_time_s": round(self._last_partition_time, 2),
                "batch_size": self._batch_size,
                "seq_len": self._seq_len,
                "solution": {
                    "max_node_time_ms": self._solution.max_node_time_ms,
                    "estimated_throughput_tok_s": self._solution.estimated_throughput_tok_s,
                    "num_oom_nodes": self._solution.num_oom_nodes,
                    "assignments": [
                        {
                            "node_id": p.node_id,
                            "start_layer": p.start_layer,
                            "end_layer": p.end_layer,
                            "estimated_time_ms": p.estimated_time_ms,
                        }
                        for p in self._solution.points
                    ],
                },
                "gpu_profiles": [
                    self._gpu_profiler.profile_to_dict(p)
                    for p in self._gpu_profiles
                ],
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Partition plan saved to {path}")
        except Exception as e:
            logger.debug(f"Failed to save partition plan: {e}")

    def load_plan(self, model_name: str) -> dict[str, Any] | None:
        name = model_name.replace("/", "_")
        path = self._profile_dir / f"{name}_partition_plan.json"
        if not path.exists():
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return None

    def summary(self) -> str:
        lines = [
            "HardwareAwarePartitioner",
            f"  GPUs: {len(self._gpu_profiles)} profiled",
            f"  Nodes: {len(self._topology.node_ids) if self._topology else 0}",
            f"  Layers: {len(self._layer_weights)} estimated",
        ]
        if self._solution:
            lines.append(f"  Partition: {self._solution.summary()}")
        else:
            lines.append("  No partition computed yet")
        lines.append(f"  Profile dir: {self._profile_dir}")
        return "\n".join(lines)
