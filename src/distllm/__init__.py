"""Distributed LLM Inference System."""

__version__ = "0.4.0"

# Import in dependency order to avoid circular imports:
# kv_cache has no internal deps -> models needs kv_cache -> coordinator needs models+kv_cache+communication -> node needs models+communication
from distllm.core.kv_cache import KVCache, KVCacheManager
from distllm.core.tls import generate_self_signed_certs, load_tls_credentials, load_tls_channel_credentials
from distllm.models.partitioner import ModelPartitioner, partition_model_across_nodes, get_model_info
from distllm.models.adapter import AdapterManager
from distllm.communication.serializers import tensor_to_proto, proto_to_tensor, kv_cache_to_proto, proto_to_kv_cache
from distllm.communication.grpc import NodeService, CoordinatorService, GRPCServer, NodeClient
from distllm.core.coordinator import Coordinator
from distllm.core.resource_manager import NodeRegistration
from distllm.core.node import WorkerNode
from distllm.core.batch_scheduler import BatchScheduler, Sequence, SequenceStatus, ScheduledBatch
from distllm.core.prefix_cache import PrefixCache
from distllm.core.radix_tree_cache import RadixTreeCache, RadixNode
from distllm.core.chunked_prefill import ChunkState, maybe_chunk
from distllm.core.structured_output import JSONSchemaConstraint
from distllm.core.constrained_decoder import (
    SchemaConstrainedDecoder,
    ConstrainedConstraint,
    TokenIndex,
    JSONSchemaFSM,
    RegexFSM,
)
from distllm.core.monitor import SystemMonitor
from distllm.core.tp_launcher import launch_tp_workers
from distllm.core.moe_router import MoERouter
from distllm.core.speculative_decoder import NgramMatcher, MedusaHeads, EAGLEGenerator, SpeculativeDecoder
from distllm.core.gossip_protocol import GossipClient
from distllm.core.gossip_transport import GossipTransport, KVCacheTransfer
from distllm.core.tool_engine import ToolCallingEngine, ToolCall, ToolResult
from distllm.core.quantization_selector import apply_kv_cache_quantization, dequantize_kv_cache
from distllm.models.adapter import AdapterManager, AdapterPool, AdapterInfo
from distllm.core.embedding_loader import EmbeddingModelLoader
from distllm.deploy.version_manager import VersionManager, ModelVersion, VersionStatus, VersionMetrics, StatisticalAnalyzer
from distllm.core.paged_attention import PagedAttentionManager, BlockPool, BlockTable, Block
try:
    from distllm.core.multi_model_serving import ModelHotSwapManager, ModelMemoryBudget, ModelInstance
except ImportError:
    ModelHotSwapManager = ModelMemoryBudget = ModelInstance = None
try:
    from distllm.core.flash_attention import FlashAttentionWrapper, apply_flash_attention_to_model
except ImportError:
    FlashAttentionWrapper = apply_flash_attention_to_model = None
try:
    from distllm.core.vlm_pipeline import VLMPipeline, VisionTower, ImageContent
except ImportError:
    VLMPipeline = VisionTower = ImageContent = None

__all__ = [
    "__version__",
    "Coordinator",
    "KVCache",
    "KVCacheManager",
    "ModelPartitioner",
    "partition_model_across_nodes",
    "BatchScheduler",
    "SpeculativeDecoder",
    "SystemMonitor",
]
