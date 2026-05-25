"""Core components for distributed LLM inference (Path A: Inference Control Plane).

Public API surface — only exports what external consumers need.
Legacy/non-core modules moved to _legacy/ for reference.
"""

from distllm.core.kv_cache import KVCache, KVCacheManager
from distllm.core.batch_scheduler import BatchScheduler, Sequence, SequenceStatus, ScheduledBatch
from distllm.core.monitor import SystemMonitor
from distllm.core.resource_manager import ResourceManager, NodeRegistration, CircuitBreakerConfig
from distllm.core.cache_manager import CacheManager
from distllm.core.token_generator import TokenGenerator
from distllm.core.protocols import (
    INodeClient, ITokenizer, IModelPartitioner, ICacheBackend,
    IMetricsExporter, INodeFactory, IResourceManager, ICacheManager,
    ITokenGenerator, IPipelineOrchestrator,
)

__all__ = [
    "KVCache", "KVCacheManager",
    "BatchScheduler", "Sequence", "SequenceStatus", "ScheduledBatch",
    "SystemMonitor",
    "ResourceManager", "NodeRegistration", "CircuitBreakerConfig",
    "CacheManager", "TokenGenerator",
    "INodeClient", "ITokenizer", "IModelPartitioner", "ICacheBackend",
    "IMetricsExporter", "INodeFactory", "IResourceManager", "ICacheManager",
    "ITokenGenerator", "IPipelineOrchestrator",
]
