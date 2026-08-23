"""Simulation subpackage for distributed inference.

Provides pipeline simulation, cluster modeling, latency prediction,
topology optimization, what-if analysis, and chaos/resilience testing
for capacity planning and distributed topology optimization.
"""

from __future__ import annotations

from distllm.dist.simulation.chaos_simulator import (
    ChaosSimulator,
    ResilienceReport,
    ScenarioResult,
    ScenarioType,
)
from distllm.dist.simulation.cluster_simulator import (
    ClusterSimulator,
    ModelConfig,
    NodeSpec,
    PerfMetrics,
    SimulatedPipelineResult,
    get_model_preset,
)
from distllm.dist.simulation.topology_optimizer import (
    Constraints,
    PartitionSolution,
    TopologyOptimizer,
)
from distllm.dist.simulation.what_if import (
    ProjectedChange,
    WhatIfEngine,
)

__all__ = [
    # Cluster simulator
    "ClusterSimulator",
    "NodeSpec",
    "ModelConfig",
    "SimulatedPipelineResult",
    "PerfMetrics",
    "get_model_preset",
    # Topology optimizer
    "TopologyOptimizer",
    "Constraints",
    "PartitionSolution",
    # What-if engine
    "WhatIfEngine",
    "ProjectedChange",
    # Chaos simulator
    "ChaosSimulator",
    "ScenarioType",
    "ScenarioResult",
    "ResilienceReport",
]
