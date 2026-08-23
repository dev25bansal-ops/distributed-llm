"""Distributed inference pipeline package.

Public API for the distributed inference engine.

Uses lazy __getattr__ imports to avoid circular import chains
(e.g., models.partitioner → dist.fsdp → dist.__init__ → dist.worker → models.partitioner).
"""

from __future__ import annotations

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {}

def _register(module: str, *symbols: str) -> None:
    for sym in symbols:
        _LAZY_IMPORTS[sym] = (module, sym)

# Core pipeline
_register("distllm.dist.pipeline", "PipelineOrchestrator", "TransportBackend", "TensorTransport")
_register("distllm.dist.worker", "WorkerNode")
_register("distllm.dist.node_registrar", "NodeRegistrar")
_register("distllm.dist.node_service", "NodeServer", "NodeServicer")
_register("distllm.dist.node_client", "NodeClient")

# Recovery & stragglers
_register("distllm.dist.recovery", "NodeRecoveryManager", "NodeRecoveryPlan", "LayerRedistribution", "SequenceCheckpoint")
_register("distllm.dist.straggler", "StragglerDetector", "DetectionMethod", "StragglerReport", "StragglerSeverity")

# Routing & topology
_register("distllm.dist.latency", "LatencyTracker")
_register("distllm.dist.rebalancer", "Rebalancer", "PartitionRecommendation")
_register("distllm.dist.redundant", "RedundantExecutor")
_register("distllm.dist.reputation", "ReputationSystem", "ReputationRecord")
_register("distllm.dist.topology_dynamic", "DynamicClusterTopology", "NodeInfo")

# P2P & federation
_register("distllm.dist.discovery", "DiscoveryService", "DiscoveryClient")
_register("distllm.dist.federation", "FederationConfig", "FederationCoordinator")
_register("distllm.dist.privacy", "PrivacySplitConfig", "PrivacyEnforcer")
_register("distllm.dist.async_pipeline", "AsyncPipelineEngine", "AsyncPipelineConfig")
_register("distllm.dist.config", "WideAreaConfig")
_register("distllm.dist.model_store", "ModelStore")
_register("distllm.dist.geo", "GeoRouter", "ClusterLoad", "LoadReporter")
_register("distllm.dist.cross_cluster", "CrossClusterForwarder")
_register("distllm.dist.merkle", "MerkleTree")
_register("distllm.dist.prefix_cache", "PrefixCache")
_register("distllm.dist.predictive_cache", "PredictiveCacheManager")
_register("distllm.dist.chunked_prefill", "ChunkState")
_register("distllm.dist.cache", "CacheIndex", "TTLPolicy")
_register("distllm.dist.attention", "PagedAttentionManager", "BlockPool")
_register("distllm.dist.preemption", "PreemptionPolicy", "GPUMemoryMonitor")
_register("distllm.dist.quality", "QualitySLA", "SLAPolicy")
_register("distllm.dist.nat", "StunClient", "TurnRelayServer", "TurnRelayClient")
_register("distllm.dist.wide_area", "WideAreaPipeline")
_register("distllm.dist.fsdp", "FSDPShard", "FSDPConfig")
_register("distllm.dist.parallel", "HybridParallelPlanner", "HybridParallelExecutor", "ParallelStrategy")
_register("distllm.dist.network", "Topology")
_register("distllm.dist.partition", "AutoPartitionConfig", "HardwareAwarePartitioner", "PartitionOptimizer", "PartitionSolution", "GPUProfiler", "TopologyGraph", "PartitionCostModel")
# Quantization tuning & coordination (registered in partition/__init__.py but
# exposed through the dist/ lazy-import facade so consumers can find them).
_register("distllm.dist.partition.quantization_tuner", "QuantizationAutoTuner", "QuantizationPlan", "QuantProfile", "QuantMethod", "SensitivityAnalyzer", "MixedPrecisionPlan", "LayerQuantPlan", "ScoreWeights", "ActivationQuantMethod", "KVCacheBits", "NodeInfo", "NodeQuantRecommendation", "QUANT_PROFILES", "select_for_node")
_register("distllm.dist.partition.quant_cost", "QuantizationAwareCostModel", "QuantNodeCost")
_register("distllm.dist.partition.quant_report", "ReportGenerator", "QuantizationReport", "NodeReport", "ConflictWarning")
_register("distllm.dist.partition.quant_calibrate", "QualityCalibrator", "CalibrationResult", "CalibrationReport")
_register("distllm.dist.partition.quant_coordinator", "QuantizationCoordinator", "NodeProfile", "NodeQuantAssignment", "CoordinatorState")

def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib
        module_path, symbol = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        value = getattr(module, symbol)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'distllm.dist' has no attribute {name!r}")

# __all__ derived from _LAZY_IMPORTS keys — keeps the public API in sync
# with _register() calls without manual drift.
__all__ = sorted(_LAZY_IMPORTS.keys())
