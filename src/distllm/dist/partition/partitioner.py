from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from loguru import logger

from distllm.dist.partition.profiles import GPUProfiler, LayerWeights
from distllm.dist.partition.topology import TopologyProber, TopologyGraph
from distllm.dist.partition.cost_model import PartitionCostModel
from distllm.dist.partition.optimizer import PartitionOptimizer, PartitionSolution


class HardwareAwarePartitioner:
    def __init__(
        self,
        batch_size: int = 1,
        seq_len: int = 4096,
        allow_oom: bool = False,
        profile_dir: str | None = None,
        enable_quant_tuning: bool = False,
        max_quality_loss: float = 0.05,
        prefer_speed: bool = False,
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

        self._gpu_profiles: list[Any] = []
        self._topology: TopologyGraph | None = None
        self._layer_weights: list[LayerWeights] = []
        self._solution: PartitionSolution | None = None
        self._last_partition_time: float = 0.0
        # 3.4: Cache invalidation — track config hash
        self._last_config_hash: str = ""
        self._last_model_name: str | None = None
        # APO integration
        self._enable_quant_tuning = enable_quant_tuning
        self._max_quality_loss = max_quality_loss
        self._prefer_speed = prefer_speed

    def _compute_config_hash(
        self,
        model_name: str | None,
        node_ids: list[str] | None,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
        vocab_size: int,
    ) -> str:
        """3.4: Compute a hash of the partitioning config for cache invalidation."""
        config_str = (
            f"{model_name}:{sorted(node_ids or [])}:{hidden_size}:"
            f"{num_layers}:{num_heads}:{intermediate_size}:{vocab_size}:"
            f"{self._batch_size}:{self._seq_len}:{self._allow_oom}"
        )
        return hashlib.md5(config_str.encode()).hexdigest()

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
        t0 = time.time()

        # 3.4: Cache invalidation — skip recomputation if config unchanged
        config_hash = self._compute_config_hash(
            model_name, node_ids, hidden_size, num_layers, num_heads,
            intermediate_size, vocab_size,
        )
        if (self._solution is not None and
                config_hash == self._last_config_hash and
                model_name == self._last_model_name):
            logger.info("Using cached partition (config unchanged)")
            return self._solution

        nodes = node_ids or [f"node-{i}" for i in range(len(self._gpu_profiles) or 1)]
        gpu_cnt = gpu_counts or {n: 1 for n in nodes}

        # 3.3: Parallel GPU profiling
        logger.info("Profiling GPUs...")
        self._gpu_profiles = await self._profile_gpus_parallel()

        # 3.4: Graceful degradation for topology probing
        logger.info(f"Profiling inter-node topology ({len(nodes)} nodes)...")
        try:
            self._topology = await self._topology_prober.probe(
                node_ids=nodes,
                hostnames=hostnames,
                gpu_counts=gpu_cnt,
                gpu_profiles={
                    n: [self._gpu_profiler.profile_to_dict(p) for p in self._gpu_profiles]
                    for n in nodes
                },
            )
        except Exception as e:
            logger.warning(f"Topology probing failed, using fallback: {e}")
            self._topology = TopologyProber.make_fallback_topology(
                num_nodes=len(nodes), gpus_per_node=1,
            )
            self._topology.node_ids = list(nodes)

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

        cost_model = PartitionCostModel(
            self._gpu_profiles, self._layer_weights, self._topology,
        )

        # Build APO tuner and node infos if quant tuning is enabled
        quant_tuner = None
        node_infos = []
        model_size_bytes = sum(l.weight_memory_bytes for l in self._layer_weights)
        inter_bw = None

        if self._enable_quant_tuning:
            from distllm.dist.partition.quantization_tuner import (
                QuantizationAutoTuner, NodeInfo,
            )
            quant_tuner = QuantizationAutoTuner(
                max_quality_loss=self._max_quality_loss,
                prefer_speed=self._prefer_speed,
            )
            for i, gp in enumerate(self._gpu_profiles):
                node_infos.append(NodeInfo.from_gpu_profile(
                    gp, node_id=nodes[i] if i < len(nodes) else f"node-{i}",
                ))
            # Estimate inter-node bandwidth from topology
            if self._topology and len(self._topology.links) > 0:
                bw_values = [l.bandwidth_gbps for l in self._topology.links if l.bandwidth_gbps > 0]
                if bw_values:
                    inter_bw = min(bw_values)

        logger.info("Running partition optimizer...")
        optimizer = PartitionOptimizer(
            cost_model=cost_model,
            node_ids=nodes,
            batch_size=self._batch_size,
            seq_len=self._seq_len,
            allow_oom=self._allow_oom,
            gpu_counts=gpu_cnt,
            quant_tuner=quant_tuner,
            node_infos=node_infos,
            model_size_bytes=model_size_bytes,
            inter_node_bandwidth_gbps=inter_bw,
        )

        self._solution = optimizer.solve(len(self._layer_weights))

        self._last_partition_time = time.time() - t0
        self._last_config_hash = config_hash
        self._last_model_name = model_name

        logger.info(
            f"Partition found in {self._last_partition_time:.1f}s: "
            f"{self._solution.num_nodes} nodes, "
            f"max latency {self._solution.max_node_time_ms:.1f}ms, "
            f"pipeline latency {self._solution.pipeline_latency_ms:.1f}ms, "
            f"throughput {self._solution.estimated_throughput_tok_s:.0f} tok/s"
            + (f", quant plan: {self._solution.quant_plan.strategy}"
               if self._solution.quant_plan else "")
        )

        self._save_plan(model_name)
        return self._solution

    async def _profile_gpus_parallel(self) -> list[Any]:
        """3.3: Profile GPUs in parallel using thread pool."""
        num_gpus = self._gpu_profiler._device_count()
        if num_gpus <= 1:
            return self._gpu_profiler.profile_all_gpus()

        profiles: list[Any] = [None] * num_gpus  # type: ignore[assignment]
        errors: list[str] = []

        def _profile_one(gpu_id: int) -> tuple[int, Any]:
            try:
                profile = self._gpu_profiler._profile_single_gpu(gpu_id)
                return gpu_id, profile
            except Exception as e:
                return gpu_id, e

        with ThreadPoolExecutor(max_workers=min(num_gpus, 4)) as executor:
            futures = {executor.submit(_profile_one, i): i for i in range(num_gpus)}
            for future in as_completed(futures):
                gpu_id, result = future.result()
                if isinstance(result, Exception):
                    errors.append(f"GPU {gpu_id}: {result}")
                    profiles[gpu_id] = self._gpu_profiler._profile_single_gpu.__wrapped__(self._gpu_profiler, gpu_id)  # type: ignore[attr-defined]
                else:
                    profiles[gpu_id] = result

        if errors:
            logger.warning(f"Some GPU profiling failed (fell back to defaults): {errors}")

        if not profiles:
            profiles = [self._gpu_profiler._profile_single_gpu(0)]

        return profiles

    def compare_to_baselines(self) -> dict[str, Any] | None:
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

    def solution(self) -> PartitionSolution | None:
        return self._solution

    def get_layer_assignments(self) -> list[dict[str, Any]] | None:
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
                "quant_tuning_enabled": self._enable_quant_tuning,
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
                            "quant_method": p.quant_method,
                        }
                        for p in self._solution.points
                    ],
                },
                "quant_plan": (
                    self._solution.quant_plan.to_dict()
                    if self._solution.quant_plan else None
                ),
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
