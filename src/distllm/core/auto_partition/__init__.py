"""Hardware-Aware Auto-Partitioner.

Profiles each GPU (memory, TFLOPS, bandwidth), profiles inter-node
latency, then solves an optimization problem to find the partition that
minimizes max per-node latency. Supports non-uniform layer sizes.

Package structure:
    profiles.py   — GPUProfiler: benchmarks compute, memory, bandwidth
    topology.py   — TopologyProber: inter-node latency and bandwidth
    cost_model.py — PartitionCostModel: estimates per-node latency
    optimizer.py  — PartitionOptimizer: DP solver for min-max partition
    partitioner.py— HardwareAwarePartitioner: orchestrator
    config.py     — Configuration models
"""

from distllm.core.auto_partition.profiles import (
    GPUProfile,
    GPUProfiler,
    LayerWeights,
)
from distllm.core.auto_partition.topology import (
    LinkProfile,
    TopologyGraph,
    TopologyProber,
)
from distllm.core.auto_partition.cost_model import (
    NodeCost,
    PartitionCostModel,
)
from distllm.core.auto_partition.optimizer import (
    PartitionPoint,
    PartitionSolution,
    PartitionOptimizer,
)
from distllm.core.auto_partition.partitioner import (
    HardwareAwarePartitioner,
)
from distllm.core.auto_partition.config import (
    AutoPartitionConfig,
    ProfilerConfig,
    OptimizerConfig,
    ModelConfig,
)

__all__ = [
    "GPUProfile",
    "GPUProfiler",
    "LayerWeights",
    "LinkProfile",
    "TopologyGraph",
    "TopologyProber",
    "NodeCost",
    "PartitionCostModel",
    "PartitionPoint",
    "PartitionSolution",
    "PartitionOptimizer",
    "HardwareAwarePartitioner",
    "AutoPartitionConfig",
    "ProfilerConfig",
    "OptimizerConfig",
    "ModelConfig",
]
