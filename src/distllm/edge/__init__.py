"""Edge deployment module for distributed-llm.

Supports lightweight, quantized inference for edge environments
with fallback to cloud/central cluster when edge is overloaded.
"""

from distllm.edge.models import (
    EdgeConfig,
    QuantizationType,
    ModelShard,
    EdgeHealth,
    EdgeNodeInfo,
)
from distllm.edge.serving import EdgeInferenceServer
from distllm.edge.quantized import QuantizedModel, QuantizationBackend
from distllm.edge.sharding import ModelShardManager
from distllm.edge.routing import EdgeRouter, EdgeRouteDecision

__all__ = [
    "EdgeConfig",
    "QuantizationType",
    "ModelShard",
    "EdgeHealth",
    "EdgeNodeInfo",
    "EdgeInferenceServer",
    "QuantizedModel",
    "QuantizationBackend",
    "ModelShardManager",
    "EdgeRouter",
    "EdgeRouteDecision",
]
