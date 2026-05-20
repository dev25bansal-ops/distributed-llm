"""Core components for distributed LLM inference."""

# Note: We avoid importing Coordinator here at module level because it creates
# a circular import: core/__init__ -> coordinator -> models.partitioner -> core.kv_cache -> core/__init__
# Instead, import directly from distllm.core.coordinator when needed.
from distllm.core.kv_cache import KVCache, KVCacheManager
from distllm.core.tls import generate_self_signed_certs, load_tls_credentials, load_tls_channel_credentials
from distllm.core.batch_scheduler import BatchScheduler, Sequence, SequenceStatus, ScheduledBatch
from distllm.core.prefix_cache import PrefixCache
from distllm.core.chunked_prefill import ChunkState, maybe_chunk
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
from distllm.core.plugin import (
    PluginManager,
    HookRegistry,
    IPlugin,
    HookPoint,
    RequestLoggingPlugin,
    MetricsPlugin,
    HealthCheckPlugin,
    BUILTIN_PLUGINS,
)
from distllm.core.self_optimizing_engine import SelfOptimizingEngine, OpType, TunableParams
from distllm.core.cuda_graph import CUDAGraphPool, GraphBuffers
from distllm.core.compile_support import compile_model
from distllm.core.grammar_decoder import GBNFFSM, GBNFParser
from distllm.core.sloRa_adapter import SLoRAManager
from distllm.core.rag_pipeline import RAGPipeline
from distllm.core.agent_loop import AgentLoop
from distllm.core.disagg_serving import DisaggOrchestrator, DisaggRouter
from distllm.core.speculative_decoder import SpeculativeDecoder
from distllm.core.drafters import NgramMatcher, MedusaHeads, EAGLEGenerator, TrainedEAGLEHeads, EAGLE2Heads

__all__ = [
    "KVCache",
    "KVCacheManager",
    "generate_self_signed_certs",
    "load_tls_credentials",
    "load_tls_channel_credentials",
    "BatchScheduler",
    "Sequence",
    "SequenceStatus",
    "ScheduledBatch",
    "PrefixCache",
    "ChunkState",
    "maybe_chunk",
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
    "PluginManager",
    "HookRegistry",
    "IPlugin",
    "HookPoint",
    "RequestLoggingPlugin",
    "MetricsPlugin",
    "HealthCheckPlugin",
    "BUILTIN_PLUGINS",
    "SelfOptimizingEngine",
    "OpType",
    "TunableParams",
    "CUDAGraphPool",
    "GraphBuffers",
    "compile_model",
    "GBNFFSM",
    "GBNFParser",
    "SLoRAManager",
    "RAGPipeline",
    "AgentLoop",
    "DisaggOrchestrator",
    "DisaggRouter",
    "SpeculativeDecoder",
    "NgramMatcher",
    "MedusaHeads",
    "EAGLEGenerator",
    "TrainedEAGLEHeads",
    "EAGLE2Heads",
]
