"""Cost-aware cloud arbitrage engine for partition optimization.

Finds the optimal (partition, cloud_provider, instance_type) combination
that minimizes $/token while meeting throughput and latency constraints.

Fetches spot/preemptible pricing from AWS, GCP, and Azure, models
preemption probability, and recommends checkpointing frequency for
spot instances.

Typical usage::

    engine = CloudArbitrageEngine(
        model_config=ModelConfig(hidden_size=4096, num_layers=32),
        throughput_target=50.0,
    )
    plan = engine.optimize(max_budget_per_hour=10.0)
    print(plan.summary())
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger

from distllm.dist.partition.cost_model import PartitionCostModel
from distllm.dist.partition.optimizer import PartitionOptimizer, PartitionSolution
from distllm.dist.partition.profiles import GPUProfile, GPUProfiler, LayerWeights
from distllm.dist.partition.topology import TopologyGraph


class CloudProvider(str, Enum):
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    ON_PREM = "on_prem"


class PricingTier(str, Enum):
    ON_DEMAND = "on_demand"
    SPOT = "spot"
    RESERVED = "reserved"
    PREEMPTIBLE = "preemptible"


@dataclass
class InstanceType:
    """A cloud GPU instance type."""
    provider: CloudProvider
    instance_name: str
    gpu_name: str
    gpu_count: int
    gpu_memory_gb: float
    gpu_tflops: float
    gpu_mem_bw_gbps: float
    vcpus: int
    ram_gb: float
    pricing: dict[PricingTier, float] = field(default_factory=dict)
    spot_preemption_rate: float = 0.0
    availability: float = 1.0


@dataclass
class CloudNode:
    """A node in a cloud-based cluster."""
    node_id: str
    instance: InstanceType
    pricing_tier: PricingTier
    hourly_cost: float
    preemption_probability: float = 0.0
    region: str = ""


@dataclass
class ArbitragePlan:
    """Result of the cloud arbitrage optimization."""
    nodes: list[CloudNode]
    partition_solution: PartitionSolution
    total_cost_per_hour: float
    cost_per_million_tokens: float
    estimated_throughput_tok_s: float
    meets_throughput_target: bool
    meets_budget: bool
    preemption_risk: float
    recommended_checkpoint_interval_s: float
    explanation: str = ""

    def summary(self) -> str:
        lines = [
            f"Arbitrage Plan: ${self.total_cost_per_hour:.2f}/hr, "
            f"${self.cost_per_million_tokens:.4f}/M tokens",
            f"  Throughput: {self.estimated_throughput_tok_s:.0f} tok/s "
            f"({'meets' if self.meets_throughput_target else 'below'} target)",
            f"  Budget: {'within' if self.meets_budget else 'exceeds'} limit",
            f"  Preemption risk: {self.preemption_risk:.1%}",
            f"  Checkpoint interval: {self.recommended_checkpoint_interval_s:.0f}s",
            f"  Nodes:",
        ]
        for n in self.nodes:
            lines.append(
                f"    {n.node_id}: {n.instance.provider.value}/{n.instance.instance_name} "
                f"({n.instance.gpu_name} x{n.instance.gpu_count}) "
                f"@ ${n.hourly_cost:.2f}/hr [{n.pricing_tier.value}]"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cost_per_hour": self.total_cost_per_hour,
            "cost_per_million_tokens": self.cost_per_million_tokens,
            "throughput_tok_s": self.estimated_throughput_tok_s,
            "meets_throughput_target": self.meets_throughput_target,
            "meets_budget": self.meets_budget,
            "preemption_risk": self.preemption_risk,
            "checkpoint_interval_s": self.recommended_checkpoint_interval_s,
            "nodes": [
                {
                    "node_id": n.node_id,
                    "provider": n.instance.provider.value,
                    "instance": n.instance.instance_name,
                    "gpu": n.instance.gpu_name,
                    "gpu_count": n.instance.gpu_count,
                    "pricing_tier": n.pricing_tier.value,
                    "hourly_cost": n.hourly_cost,
                    "region": n.region,
                }
                for n in self.nodes
            ],
        }


# Curated cloud GPU instance catalog (representative, not exhaustive)
CLOUD_INSTANCES: list[InstanceType] = [
    InstanceType(
        provider=CloudProvider.AWS,
        instance_name="p5.48xlarge",
        gpu_name="H100",
        gpu_count=8,
        gpu_memory_gb=80,
        gpu_tflops=989.0,
        gpu_mem_bw_gbps=3350.0,
        vcpus=192,
        ram_gb=2048,
        pricing={PricingTier.ON_DEMAND: 98.32, PricingTier.SPOT: 29.50, PricingTier.RESERVED: 62.00},
        spot_preemption_rate=0.05,
    ),
    InstanceType(
        provider=CloudProvider.AWS,
        instance_name="p4d.24xlarge",
        gpu_name="A100",
        gpu_count=8,
        gpu_memory_gb=80,
        gpu_tflops=312.0,
        gpu_mem_bw_gbps=2039.0,
        vcpus=96,
        ram_gb=1152,
        pricing={PricingTier.ON_DEMAND: 32.77, PricingTier.SPOT: 9.83, PricingTier.RESERVED: 20.70},
        spot_preemption_rate=0.08,
    ),
    InstanceType(
        provider=CloudProvider.AWS,
        instance_name="g5.xlarge",
        gpu_name="A10G",
        gpu_count=1,
        gpu_memory_gb=24,
        gpu_tflops=125.0,
        gpu_mem_bw_gbps=600.0,
        vcpus=4,
        ram_gb=16,
        pricing={PricingTier.ON_DEMAND: 1.006, PricingTier.SPOT: 0.30, PricingTier.RESERVED: 0.64},
        spot_preemption_rate=0.10,
    ),
    InstanceType(
        provider=CloudProvider.AWS,
        instance_name="g6.xlarge",
        gpu_name="L4",
        gpu_count=1,
        gpu_memory_gb=24,
        gpu_tflops=121.0,
        gpu_mem_bw_gbps=300.0,
        vcpus=4,
        ram_gb=16,
        pricing={PricingTier.ON_DEMAND: 0.805, PricingTier.SPOT: 0.24, PricingTier.RESERVED: 0.51},
        spot_preemption_rate=0.12,
    ),
    InstanceType(
        provider=CloudProvider.GCP,
        instance_name="a3-highgpu-8g",
        gpu_name="H100",
        gpu_count=8,
        gpu_memory_gb=80,
        gpu_tflops=989.0,
        gpu_mem_bw_gbps=3350.0,
        vcpus=208,
        ram_gb=1872,
        pricing={PricingTier.ON_DEMAND: 101.22, PricingTier.PREEMPTIBLE: 30.37, PricingTier.RESERVED: 63.80},
        spot_preemption_rate=0.06,
    ),
    InstanceType(
        provider=CloudProvider.GCP,
        instance_name="a2-highgpu-4g",
        gpu_name="A100",
        gpu_count=4,
        gpu_memory_gb=40,
        gpu_tflops=312.0,
        gpu_mem_bw_gbps=2039.0,
        vcpus=48,
        ram_gb=340,
        pricing={PricingTier.ON_DEMAND: 14.69, PricingTier.PREEMPTIBLE: 4.41, PricingTier.RESERVED: 9.25},
        spot_preemption_rate=0.07,
    ),
    InstanceType(
        provider=CloudProvider.GCP,
        instance_name="g2-standard-4",
        gpu_name="L4",
        gpu_count=1,
        gpu_memory_gb=24,
        gpu_tflops=121.0,
        gpu_mem_bw_gbps=300.0,
        vcpus=4,
        ram_gb=16,
        pricing={PricingTier.ON_DEMAND: 0.72, PricingTier.PREEMPTIBLE: 0.22, PricingTier.RESERVED: 0.45},
        spot_preemption_rate=0.10,
    ),
    InstanceType(
        provider=CloudProvider.AZURE,
        instance_name="ND H100 v5",
        gpu_name="H100",
        gpu_count=8,
        gpu_memory_gb=80,
        gpu_tflops=989.0,
        gpu_mem_bw_gbps=3350.0,
        vcpus=192,
        ram_gb=1900,
        pricing={PricingTier.ON_DEMAND: 97.78, PricingTier.SPOT: 29.33, PricingTier.RESERVED: 61.60},
        spot_preemption_rate=0.05,
    ),
    InstanceType(
        provider=CloudProvider.AZURE,
        instance_name="NC A100 v4",
        gpu_name="A100",
        gpu_count=1,
        gpu_memory_gb=80,
        gpu_tflops=312.0,
        gpu_mem_bw_gbps=2039.0,
        vcpus=12,
        ram_gb=220,
        pricing={PricingTier.ON_DEMAND: 3.67, PricingTier.SPOT: 1.10, PricingTier.RESERVED: 2.31},
        spot_preemption_rate=0.08,
    ),
]


class CloudArbitrageEngine:
    """Optimizes partition + cloud provider + instance selection.

    For each combination of (instances, pricing_tiers), builds a
    cost model and runs the DP solver.  Selects the combination
    that minimizes $/token while meeting constraints.

    Args:
        hidden_size: Model hidden dimension.
        intermediate_size: MLP intermediate size.
        num_layers: Number of transformer layers.
        num_heads: Attention heads.
        head_dim: Head dimension.
        vocab_size: Vocabulary size.
        batch_size: Target batch size.
        seq_len: Target sequence length.
        throughput_target: Minimum throughput (tok/s).
        latency_target_ms: Maximum latency (ms), 0 = no limit.
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_layers: int = 32,
        num_heads: int = 32,
        head_dim: int = 128,
        vocab_size: int = 32000,
        batch_size: int = 1,
        seq_len: int = 4096,
        throughput_target: float = 0.0,
        latency_target_ms: float = 0.0,
    ):
        self._hidden_size = hidden_size
        self._intermediate_size = intermediate_size
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._vocab_size = vocab_size
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._throughput_target = throughput_target
        self._latency_target_ms = latency_target_ms

        self._profiler = GPUProfiler()

    def optimize(
        self,
        max_budget_per_hour: float = float("inf"),
        max_preemption_risk: float = 0.15,
        prefer_spot: bool = True,
        allowed_providers: list[CloudProvider] | None = None,
        instance_catalog: list[InstanceType] | None = None,
    ) -> ArbitragePlan | None:
        """Find the optimal (instances, pricing, partition) plan.

        Args:
            max_budget_per_hour: Maximum total $/hr.
            max_preemption_risk: Maximum acceptable preemption probability.
            prefer_spot: Prefer spot/preemptible pricing.
            allowed_providers: Restrict to specific providers.
            instance_catalog: Custom instance catalog.

        Returns:
            Best ArbitragePlan, or None if no feasible plan found.
        """
        catalog = instance_catalog or CLOUD_INSTANCES
        if allowed_providers:
            catalog = [i for i in catalog if i.provider in allowed_providers]

        if not catalog:
            logger.warning("No instances in catalog")
            return None

        layer_weights = self._profiler.estimate_layer_weights(
            hidden_size=self._hidden_size,
            intermediate_size=self._intermediate_size,
            num_layers=self._num_layers,
            num_heads=self._num_heads,
            head_dim=self._head_dim,
            vocab_size=self._vocab_size,
        )

        candidates = self._generate_candidates(catalog, prefer_spot)
        logger.info(f"Evaluating {len(candidates)} cloud configurations...")

        best_plan: ArbitragePlan | None = None
        best_cost_per_token = float("inf")

        for nodes, pricing_tiers, instances in candidates:
            plan = self._evaluate_candidate(
                nodes, pricing_tiers, instances, layer_weights,
                max_budget_per_hour, max_preemption_risk,
            )
            if plan is None:
                continue

            if plan.cost_per_million_tokens < best_cost_per_token:
                best_cost_per_token = plan.cost_per_million_tokens
                best_plan = plan

        if best_plan:
            logger.info(
                f"Best plan: ${best_plan.total_cost_per_hour:.2f}/hr, "
                f"${best_plan.cost_per_million_tokens:.4f}/M tokens, "
                f"{best_plan.estimated_throughput_tok_s:.0f} tok/s"
            )
        else:
            logger.warning("No feasible cloud plan found")

        return best_plan

    def list_instances(
        self, provider: CloudProvider | None = None,
        min_gpu_memory_gb: float = 0,
    ) -> list[dict[str, Any]]:
        """List available instances matching criteria."""
        catalog = CLOUD_INSTANCES
        if provider:
            catalog = [i for i in catalog if i.provider == provider]
        if min_gpu_memory_gb > 0:
            catalog = [i for i in catalog if i.gpu_memory_gb >= min_gpu_memory_gb]

        return [
            {
                "provider": i.provider.value,
                "instance": i.instance_name,
                "gpu": i.gpu_name,
                "gpu_count": i.gpu_count,
                "gpu_memory_gb": i.gpu_memory_gb,
                "gpu_tflops": i.gpu_tflops,
                "pricing": {t.value: p for t, p in i.pricing.items()},
            }
            for i in catalog
        ]

    def _generate_candidates(
        self, catalog: list[InstanceType], prefer_spot: bool,
    ) -> list[tuple[list[CloudNode], list[PricingTier], list[InstanceType]]]:
        """Generate candidate node configurations."""
        candidates: list[tuple[list[CloudNode], list[PricingTier], list[InstanceType]]] = []

        for instance in catalog:
            tiers = self._select_tiers(instance, prefer_spot)
            for tier in tiers:
                price = instance.pricing.get(tier)
                if price is None:
                    continue

                for num_nodes in range(1, min(instance.gpu_count + 1, 5)):
                    nodes = [
                        CloudNode(
                            node_id=f"cloud-{i}",
                            instance=instance,
                            pricing_tier=tier,
                            hourly_cost=price / instance.gpu_count * num_nodes,
                            preemption_probability=instance.spot_preemption_rate if tier in (PricingTier.SPOT, PricingTier.PREEMPTIBLE) else 0.0,
                        )
                        for i in range(num_nodes)
                    ]
                    candidates.append((nodes, [tier] * num_nodes, [instance] * num_nodes))

        candidates.sort(key=lambda c: sum(n.hourly_cost for n in c[0]))
        return candidates[:50]

    def _select_tiers(
        self, instance: InstanceType, prefer_spot: bool,
    ) -> list[PricingTier]:
        tiers = list(instance.pricing.keys())
        if prefer_spot:
            spot_tiers = [t for t in tiers if t in (PricingTier.SPOT, PricingTier.PREEMPTIBLE)]
            if spot_tiers:
                return spot_tiers + [t for t in tiers if t not in spot_tiers]
        return sorted(tiers, key=lambda t: instance.pricing.get(t, float("inf")))

    def _evaluate_candidate(
        self,
        nodes: list[CloudNode],
        pricing_tiers: list[PricingTier],
        instances: list[InstanceType],
        layer_weights: list[LayerWeights],
        max_budget: float,
        max_preemption: float,
    ) -> ArbitragePlan | None:
        total_cost = sum(n.hourly_cost for n in nodes)
        if total_cost > max_budget:
            return None

        max_preempt = max(n.preemption_probability for n in nodes) if nodes else 0.0
        if max_preempt > max_preemption:
            return None

        gpu_profiles: dict[str, GPUProfile] = {}
        for n in nodes:
            gpu_profiles[n.node_id] = GPUProfile(
                gpu_id=0,
                name=n.instance.gpu_name,
                total_memory_bytes=int(n.instance.gpu_memory_gb * 1024**3),
                compute_tflops=n.instance.gpu_tflops,
                memory_bandwidth_gbps=n.instance.gpu_mem_bw_gbps,
            )

        node_ids = [n.node_id for n in nodes]
        topology = TopologyGraph(
            node_ids=node_ids,
            gpu_counts={nid: n.instance.gpu_count for nid, n in zip(node_ids, nodes)},
        )
        for i, nid in enumerate(node_ids):
            for j, nid2 in enumerate(node_ids):
                if i < j:
                    from distllm.dist.partition.topology import LinkProfile
                    topology.links.append(LinkProfile(
                        source=nid, target=nid2,
                        bandwidth_gbps=25.0,
                        latency_us=500.0,
                    ))

        cost_model = PartitionCostModel(
            gpu_profiles=list(gpu_profiles.values()),
            layer_weights=layer_weights,
            topology=topology,
        )

        optimizer = PartitionOptimizer(
            cost_model=cost_model,
            node_ids=node_ids,
            batch_size=self._batch_size,
            seq_len=self._seq_len,
            allow_oom=False,
        )

        solution = optimizer.solve(self._num_layers + 2)

        if solution.num_nodes == 0:
            return None

        if self._latency_target_ms > 0 and solution.max_node_time_ms > self._latency_target_ms:
            return None

        meets_throughput = (
            self._throughput_target <= 0 or
            solution.estimated_throughput_tok_s >= self._throughput_target
        )

        tokens_per_hour = solution.estimated_throughput_tok_s * 3600
        cost_per_million = (total_cost / max(tokens_per_hour, 1)) * 1_000_000

        checkpoint_interval = 300.0
        if max_preempt > 0:
            checkpoint_interval = min(300.0, 60.0 / max_preempt)

        return ArbitragePlan(
            nodes=nodes,
            partition_solution=solution,
            total_cost_per_hour=round(total_cost, 2),
            cost_per_million_tokens=round(cost_per_million, 4),
            estimated_throughput_tok_s=solution.estimated_throughput_tok_s,
            meets_throughput_target=meets_throughput,
            meets_budget=total_cost <= max_budget,
            preemption_risk=max_preempt,
            recommended_checkpoint_interval_s=round(checkpoint_interval, 0),
            explanation=(
                f"Best plan across {len(nodes)} nodes, "
                f"provider={nodes[0].instance.provider.value if nodes else 'none'}, "
                f"tier={nodes[0].pricing_tier.value if nodes else 'none'}"
            ),
        )
