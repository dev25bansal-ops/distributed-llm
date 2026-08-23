"""Hardware-Aware Auto-Partitioner.

Profiles each GPU (memory, TFLOPS, bandwidth), profiles inter-node
latency, then solves an optimization problem to find the partition that
minimizes max per-node latency. Supports non-uniform layer sizes.

Package structure:
    profiles.py       — GPUProfiler: benchmarks compute, memory, bandwidth
    topology.py       — TopologyProber: inter-node latency and bandwidth
    cost_model.py     — PartitionCostModel: estimates per-node latency
    optimizer.py      — PartitionOptimizer: DP solver for min-max partition
    partitioner.py    — HardwareAwarePartitioner: orchestrator
    config.py         — Configuration models
    quantization_tuner.py — Adaptive Precision Optimizer (APO)
    quant_bench.py    — Live quantization benchmark suite
    quant_cost.py     — Quantization-aware cost model extension
    quant_report.py   — Cluster-wide quantization report
"""

from __future__ import annotations
from distllm.dist.partition.profiles import (
    GPUProfile,
    GPUProfiler,
    LayerWeights,
)
from distllm.dist.partition.topology import (
    LinkProfile,
    TopologyGraph,
    TopologyProber,
)
from distllm.dist.partition.cost_model import (
    NodeCost,
    PartitionCostModel,
)
from distllm.dist.partition.optimizer import (
    PartitionPoint,
    PartitionSolution,
    PartitionOptimizer,
)
from distllm.dist.partition.partitioner import (
    HardwareAwarePartitioner,
)
from distllm.dist.partition.config import (
    AutoPartitionConfig,
    ProfilerConfig,
    OptimizerConfig,
    ModelConfig,
)
from distllm.dist.partition.quantization_tuner import (
    QuantMethod,
    QuantProfile,
    QuantizationPlan,
    QuantizationAutoTuner,
    NodeQuantRecommendation,
    NodeInfo,
    ScoreWeights,
    ActivationQuantMethod,
    KVCacheBits,
    MixedPrecisionPlan,
    LayerQuantPlan,
    SensitivityAnalyzer,
    QUANT_PROFILES,
    select_for_node,
)
from distllm.dist.partition.quant_bench import (
    QuantBenchmarker,
    QuantBenchmarkSuite,
    QuantBenchmarkResult,
)
from distllm.dist.partition.quant_cost import (
    QuantizationAwareCostModel,
    QuantNodeCost,
)
from distllm.dist.partition.quant_report import (
    ReportGenerator,
    QuantizationReport,
    NodeReport,
    ConflictWarning,
)
from distllm.dist.partition.quant_calibrate import (
    QualityCalibrator,
    CalibrationResult,
    CalibrationReport,
)
from distllm.dist.partition.quant_coordinator import (
    QuantizationCoordinator,
    NodeProfile,
    NodeQuantAssignment,
    CoordinatorState,
)

__all__ = [
    # Core partitioning
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
    # Adaptive Precision Optimizer (APO)
    "QuantMethod",
    "QuantProfile",
    "QuantizationPlan",
    "QuantizationAutoTuner",
    "NodeQuantRecommendation",
    "NodeInfo",
    "ScoreWeights",
    "ActivationQuantMethod",
    "KVCacheBits",
    "MixedPrecisionPlan",
    "LayerQuantPlan",
    "SensitivityAnalyzer",
    "QUANT_PROFILES",
    "select_for_node",
    # Benchmark suite
    "QuantBenchmarker",
    "QuantBenchmarkSuite",
    "QuantBenchmarkResult",
    # Quantization-aware cost model
    "QuantizationAwareCostModel",
    "QuantNodeCost",
    # Report generator
    "ReportGenerator",
    "QuantizationReport",
    "NodeReport",
    "ConflictWarning",
    # Quality calibration
    "QualityCalibrator",
    "CalibrationResult",
    "CalibrationReport",
    # Distributed coordinator
    "QuantizationCoordinator",
    "NodeProfile",
    "NodeQuantAssignment",
    "CoordinatorState",
]
