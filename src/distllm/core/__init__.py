"""Core components for distributed LLM inference.

Uses lazy imports to avoid loading all 24+ modules at import time.
Only loads a module when its symbols are actually accessed.
"""

# Lazy import map: symbol_name -> (module_path, symbol_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {}

def _register(module: str, *symbols: str) -> None:
    for sym in symbols:
        _LAZY_IMPORTS[sym] = (module, sym)

# Register all public symbols
_register("distllm.core.api_key_store", "ApiKeyStore", "get_api_key_store", "reset_api_key_store", "role_satisfies")
_register("distllm.core.backup_manager", "BackupEntry", "BackupManager", "BackupManifest")
_register("distllm.core.batch_scheduler", "BatchScheduler", "ScheduledBatch", "Sequence", "SequenceStatus")
_register("distllm.core.cache_manager", "CacheManager")
_register("distllm.core.certificate_manager", "CertificateInfo", "CertificateManager")
_register("distllm.core.cluster_state_store", "ClusterNodeState", "ClusterState", "ClusterStateStore")
_register("distllm.core.distributed_speculative", "DistributedSpeculativeDecoder", "DraftLatencyStats", "DraftTokenResult", "RemoteDraftConfig", "RemoteDraftModel")
_register("distllm.core.draft_model_router", "DraftModelFleet", "DraftModelRouter", "DraftModelSpec", "RoutingConstraints", "RoutingDecision")
_register("distllm.core.gpu_resource_manager", "Allocation", "GPUMemorySnapshot", "GPUResourceManager", "MemoryPriority", "get_gpu_resource_manager")
_register("distllm.core.kv_cache", "KVCache", "KVCacheManager")
_register("distllm.core.learning_router", "LearningRouter", "RewardSignal")
_register("distllm.core.load_balancer", "CoordinatorTarget", "LBStats", "LBStrategy", "LoadBalancer", "create_load_balancer")
_register("distllm.core.model_router", "ModelRouter", "RouteMatch", "RouteRule", "RoutingContext")
_register("distllm.core.model_version_manager", "ABTestSplit", "CanaryStage", "ModelVersion", "ModelVersionManager", "VersionStatus")
_register("distllm.core.monitor", "SystemMonitor")
_register("distllm.core.notification_manager", "Notification", "NotificationChannel", "NotificationManager", "NotificationSeverity")
_register("distllm.core.plugin_system", "PluginBase", "PluginInstance", "PluginMetadata", "PluginState", "PluginSystem")
_register("distllm.core.protocols", "ICacheBackend", "ICacheManager", "IMetricsExporter", "IModelPartitioner", "INodeClient", "INodeFactory", "IPipelineOrchestrator", "IResourceManager", "ITokenGenerator", "ITokenizer")
_register("distllm.core.resource_manager", "CircuitBreakerConfig", "NodeRegistration", "ResourceManager")
_register("distllm.core.routing_extensions", "LRUModelCache", "RoutingMetrics", "SemanticRouter", "SpeculativePreWarmer")
_register("distllm.core.token_generator", "TokenGenerator")
_register("distllm.core.usage_meter", "QuotaLimit", "TenantUsage", "UsageMeter", "UsageRecord", "UsageRecordStatus", "create_usage_meter")
_register("distllm.core.webhook_manager", "WebhookDelivery", "WebhookEvent", "WebhookManager", "WebhookTarget")
_register("distllm.core.event_bus", "EventBus", "MarketplaceEvent", "MarketplaceEventType")
_register("distllm.core.memory_defragmenter", "DefragConfig", "DefragPolicy", "DefragResult", "FragmentInfo", "MemoryDefragmenter", "TieredCompactionLevel")
_register("distllm.core.atlas_mesh",
    "AtlasMesh", "Cluster", "ClusterGraph", "ContextualBanditRewardModel",
    "ConstraintViolation", "LatencyCostReliabilityScorer",
    "LPSolverRouter", "MeshStats", "Observation", "RoutingAssignment",
    "RoutingRequest", "ScoringWeights",
)
_register("distllm.core.agentic_router", "AgenticRouter", "RouterJudge", "RoutingDecision")
_register("distllm.core.autonomous_healer", "AutonomousHealer", "FailurePredictor", "GPUResetManager", "GPUHeartbeat", "GPUHealthState")
_register("distllm.core.bargaining_engine", "SpotBidManager", "DQNAgent", "BudgetController", "MarketSnapshot", "SpotBid")
_register("distllm.core.compressed_speculative", "CompressedSpeculativeDecoder", "LightweightVerifier", "CompressionVerifierTrainer")
_register("distllm.core.persistence", "SQLiteBackend", "StorageBackend")
_register("distllm.core.prompt_library", "PromptRepository", "PromptVersion")
_register("distllm.core.evaluation_harness", "EvalRunner", "EvaluationReport")
_register("distllm.core.dp_inference", "DifferentialPrivacyInference", "PrivacyBudgetManager")
_register("distllm.core.media_pipeline", "AudioPipeline", "MediaStreamRouter", "SpeechRecognizer", "TextToSpeech", "VoiceActivityDetector", "LLMResponder")
_register("distllm.core.neural_partition_optimizer",
    # Data classes
    "LayerConfig", "HardwareSpec", "CostPrediction", "TrainingSample",
    "PartitionProposal", "OptimizationStats",
    # Classes
    "NeuralCostModel", "BayesianOptimizationLoop", "NeuralPartitionOptimizer",
)
_register("distllm.core.voyager_multimodal",
    # Enums
    "ModalityType", "RouteType",
    # Data classes
    "MultiModalRequest", "RoutingPlan", "EncodedOutput", "VoyagerResponse",
    # Core classes
    "ModalityEncoder", "MultiModalRouter", "ParallelEncoderPipeline", "Voyager",
)
_register("distllm.core.aether_federated",
    # Data classes
    "LoRAConfig", "FederatedConfig", "AetherState",
    # Core classes
    "LoRAAdapterManager", "GradientAggregator", "FederatedTrainer", "Aether",
)
_register("distllm.core.kraken_chaos",
    # Enums
    "ResilienceLevel", "ScenarioCategory", "ScenarioSeverity",
    # Data classes
    "ResilienceReport", "ScenarioConfig", "ScenarioResult",
    # Classes
    "AutomatedChaosPipeline", "FaultScenarioLibrary", "Kraken", "ResilienceScore",
)
_register("distllm.core.aegis_compliance",
    # Exceptions
    "ComplianceError", "WatermarkError",
    # Data classes
    "AuditEntry",
    # Core classes
    "AuditTrail", "ModelWatermark", "ComplianceRule", "Aegis",
    # Constants
    "HIPAA_RULES", "SOC2_RULES",
)
_register("distllm.core.vectorstore",
    "VectorDBInterface", "VectorDBFactory", "RAGPipeline",
)


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, symbol = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path)
        value = getattr(module, symbol)
        # Cache in module namespace for subsequent access
        globals()[name] = value
        return value
    raise AttributeError(f"module 'distllm.core' has no attribute {name!r}")


__all__ = list(_LAZY_IMPORTS.keys())
