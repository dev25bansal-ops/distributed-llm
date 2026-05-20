"""System-level facades for the Coordinator.

Each system groups related components with clear boundaries:
- NodeSystem: node lifecycle, health, registration
- CacheSystem: prefix cache, KV cache, gossip, persistence
- PipelineSystem: distributed pipeline execution, transport
- GenerationSystem: batching, sampling, preemption
- ModelSystem: model loading, multi-model, adapters

The CoordinatorFacade composes these systems while maintaining
backward compatibility with the existing Coordinator API.
"""

from distllm.core.systems.node_system import NodeSystem
from distllm.core.systems.cache_system import CacheSystem
from distllm.core.systems.pipeline_system import PipelineSystem
from distllm.core.systems.generation_system import GenerationSystem
from distllm.core.systems.model_system import ModelSystem

__all__ = [
    "NodeSystem",
    "CacheSystem",
    "PipelineSystem",
    "GenerationSystem",
    "ModelSystem",
]
