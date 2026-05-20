"""Core components for distributed LLM inference.

Public API surface — only exports what external consumers need.
Internal types are importable via their full module path.
"""

# Note: We avoid importing Coordinator here at module level because it creates
# a circular import: core/__init__ -> coordinator -> models.partitioner -> core.kv_cache -> core/__init__
# Instead, import directly from distllm.core.coordinator when needed.
from distllm.core.kv_cache import KVCache, KVCacheManager
from distllm.core.batch_scheduler import BatchScheduler, Sequence, SequenceStatus, ScheduledBatch
from distllm.core.structured_output import JSONSchemaConstraint
from distllm.core.monitor import SystemMonitor
from distllm.core.resource_manager import ResourceManager, NodeRegistration, CircuitBreakerConfig
from distllm.core.cache_manager import CacheManager
from distllm.core.token_generator import TokenGenerator
from distllm.core.pipeline_orchestrator import PipelineOrchestrator
from distllm.core.protocols import (
    INodeClient, ITokenizer, IModelPartitioner, ICacheBackend,
    IMetricsExporter, INodeFactory, IResourceManager, ICacheManager,
    ITokenGenerator, IPipelineOrchestrator,
)
from distllm.core.speculative_decoder import SpeculativeDecoder

__all__ = [
    "KVCache",
    "KVCacheManager",
    "BatchScheduler",
    "Sequence",
    "SequenceStatus",
    "ScheduledBatch",
    "JSONSchemaConstraint",
    "SystemMonitor",
    "ResourceManager",
    "NodeRegistration",
    "CircuitBreakerConfig",
    "CacheManager",
    "TokenGenerator",
    "PipelineOrchestrator",
    "INodeClient",
    "ITokenizer",
    "IModelPartitioner",
    "ICacheBackend",
    "IMetricsExporter",
    "INodeFactory",
    "IResourceManager",
    "ICacheManager",
    "ITokenGenerator",
    "IPipelineOrchestrator",
    "SpeculativeDecoder",
]
