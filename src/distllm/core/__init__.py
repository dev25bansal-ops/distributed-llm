"""Core components for distributed LLM inference (Path A: Inference Control Plane).

Public API surface — only exports what external consumers need.
Legacy/non-core modules moved to _legacy/ for reference.
"""

from distllm.core.api_key_store import (
    ApiKeyStore,
    get_api_key_store,
    reset_api_key_store,
    role_satisfies,
)
from distllm.core.backup_manager import BackupEntry, BackupManager, BackupManifest
from distllm.core.batch_scheduler import (
    BatchScheduler,
    ScheduledBatch,
    Sequence,
    SequenceStatus,
)
from distllm.core.cache_manager import CacheManager
from distllm.core.certificate_manager import CertificateInfo, CertificateManager
from distllm.core.cluster_state_store import (
    ClusterNodeState,
    ClusterState,
    ClusterStateStore,
)
from distllm.core.distributed_speculative import (
    DistributedSpeculativeDecoder,
    DraftLatencyStats,
    DraftTokenResult,
    RemoteDraftConfig,
    RemoteDraftModel,
)
from distllm.core.draft_model_router import (
    DraftModelFleet,
    DraftModelRouter,
    DraftModelSpec,
    RoutingConstraints,
    RoutingDecision,
)
from distllm.core.gpu_resource_manager import (
    Allocation,
    GPUMemorySnapshot,
    GPUResourceManager,
    MemoryPriority,
    get_gpu_resource_manager,
)
from distllm.core.kv_cache import KVCache, KVCacheManager
from distllm.core.learning_router import LearningRouter, RewardSignal
from distllm.core.load_balancer import (
    CoordinatorTarget,
    LBStats,
    LBStrategy,
    LoadBalancer,
    create_load_balancer,
)
from distllm.core.model_router import ModelRouter, RouteMatch, RouteRule, RoutingContext
from distllm.core.model_version_manager import (
    ABTestSplit,
    CanaryStage,
    ModelVersion,
    ModelVersionManager,
    VersionStatus,
)
from distllm.core.monitor import SystemMonitor
from distllm.core.notification_manager import (
    Notification,
    NotificationChannel,
    NotificationManager,
    NotificationSeverity,
)
from distllm.core.plugin_system import (
    PluginBase,
    PluginInstance,
    PluginMetadata,
    PluginState,
    PluginSystem,
)
from distllm.core.protocols import (
    ICacheBackend,
    ICacheManager,
    IMetricsExporter,
    IModelPartitioner,
    INodeClient,
    INodeFactory,
    IPipelineOrchestrator,
    IResourceManager,
    ITokenGenerator,
    ITokenizer,
)
from distllm.core.resource_manager import (
    CircuitBreakerConfig,
    NodeRegistration,
    ResourceManager,
)
from distllm.core.routing_extensions import (
    LRUModelCache,
    RoutingMetrics,
    SemanticRouter,
    SpeculativePreWarmer,
)
from distllm.core.token_generator import TokenGenerator
from distllm.core.usage_meter import (
    QuotaLimit,
    TenantUsage,
    UsageMeter,
    UsageRecord,
    UsageRecordStatus,
    create_usage_meter,
)
from distllm.core.webhook_manager import (
    WebhookDelivery,
    WebhookEvent,
    WebhookManager,
    WebhookTarget,
)
from distllm.core.memory_defragmenter import (
    DefragConfig,
    DefragPolicy,
    DefragResult,
    FragmentInfo,
    MemoryDefragmenter,
    TieredCompactionLevel,
)

__all__ = [
    "BackupManager", "BackupManifest", "BackupEntry",
    "BatchScheduler", "Sequence", "SequenceStatus", "ScheduledBatch",
    "CacheManager",
    "CertificateManager", "CertificateInfo",
    "ClusterStateStore", "ClusterState", "ClusterNodeState",
    "GPUResourceManager", "get_gpu_resource_manager", "GPUMemorySnapshot",
    "Allocation", "MemoryPriority",
    "KVCache", "KVCacheManager",
    "LoadBalancer", "LBStrategy", "LBStats", "CoordinatorTarget", "create_load_balancer",
    "ModelRouter", "RouteRule", "RouteMatch", "RoutingContext",
    "LearningRouter", "RewardSignal",
    "LRUModelCache", "SemanticRouter", "SpeculativePreWarmer", "RoutingMetrics",
    "ModelVersionManager", "ModelVersion", "VersionStatus", "ABTestSplit", "CanaryStage",
    "NotificationManager", "Notification", "NotificationSeverity", "NotificationChannel",
    "PluginSystem", "PluginBase", "PluginInstance", "PluginMetadata", "PluginState",
    "ResourceManager", "NodeRegistration", "CircuitBreakerConfig",
    "SystemMonitor",
    "TokenGenerator",
    "DistributedSpeculativeDecoder", "DraftTokenResult", "DraftLatencyStats",
    "RemoteDraftModel", "RemoteDraftConfig",
    "DraftModelFleet", "DraftModelRouter", "DraftModelSpec",
    "RoutingConstraints", "RoutingDecision",
    "UsageMeter", "UsageRecord", "QuotaLimit", "TenantUsage",
    "UsageRecordStatus", "create_usage_meter",
    "WebhookManager", "WebhookTarget", "WebhookEvent", "WebhookDelivery",
    "INodeClient", "ITokenizer", "IModelPartitioner", "ICacheBackend",
    "IMetricsExporter", "INodeFactory", "IResourceManager", "ICacheManager",
    "ITokenGenerator", "IPipelineOrchestrator",
    "ApiKeyStore", "get_api_key_store", "reset_api_key_store", "role_satisfies",
    "MemoryDefragmenter", "FragmentInfo", "DefragResult", "DefragConfig",
    "DefragPolicy", "TieredCompactionLevel",
]
