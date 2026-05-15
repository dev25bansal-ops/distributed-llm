"""Operator CRDs."""

from distllm.operator.crds.distributed_llm_cluster import (
    DistributedLLMClusterSpec,
    ModelSpec,
    CoordinatorSpec,
    NodePoolSpec,
    HPASpec,
    ResourceSpec,
)

__all__ = [
    "DistributedLLMClusterSpec",
    "ModelSpec",
    "CoordinatorSpec",
    "NodePoolSpec",
    "HPASpec",
    "ResourceSpec",
]
