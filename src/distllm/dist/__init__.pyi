"""Type stubs for distllm.dist — enables static type checking of lazy imports.

This stub mirrors the _register() calls in __init__.py so that mypy/pyright
can resolve all 86 exported symbols through the __getattr__-based lazy import
facade. Each symbol is re-exported from its source module.
"""

# Core pipeline
from distllm.dist.pipeline import PipelineOrchestrator as PipelineOrchestrator
from distllm.dist.pipeline import TransportBackend as TransportBackend
from distllm.dist.pipeline import TensorTransport as TensorTransport
from distllm.dist.worker import WorkerNode as WorkerNode
from distllm.dist.node_registrar import NodeRegistrar as NodeRegistrar
from distllm.dist.node_service import NodeServer as NodeServer
from distllm.dist.node_service import NodeServicer as NodeServicer
from distllm.dist.node_client import NodeClient as NodeClient

# Recovery & stragglers
from distllm.dist.recovery import NodeRecoveryManager as NodeRecoveryManager
from distllm.dist.recovery import NodeRecoveryPlan as NodeRecoveryPlan
from distllm.dist.recovery import LayerRedistribution as LayerRedistribution
from distllm.dist.recovery import SequenceCheckpoint as SequenceCheckpoint
from distllm.dist.straggler import StragglerDetector as StragglerDetector
from distllm.dist.straggler import DetectionMethod as DetectionMethod
from distllm.dist.straggler import StragglerReport as StragglerReport
from distllm.dist.straggler import StragglerSeverity as StragglerSeverity

# Routing & topology
from distllm.dist.latency import LatencyTracker as LatencyTracker
from distllm.dist.rebalancer import Rebalancer as Rebalancer
from distllm.dist.rebalancer import PartitionRecommendation as PartitionRecommendation
from distllm.dist.redundant import RedundantExecutor as RedundantExecutor
from distllm.dist.reputation import ReputationSystem as ReputationSystem
from distllm.dist.reputation import ReputationRecord as ReputationRecord
from distllm.dist.topology_dynamic import DynamicClusterTopology as DynamicClusterTopology
from distllm.dist.topology_dynamic import NodeInfo as NodeInfo

# P2P & federation
from distllm.dist.discovery import DiscoveryService as DiscoveryService
from distllm.dist.discovery import DiscoveryClient as DiscoveryClient
from distllm.dist.federation import FederationConfig as FederationConfig
from distllm.dist.federation import FederationCoordinator as FederationCoordinator
from distllm.dist.privacy import PrivacySplitConfig as PrivacySplitConfig
from distllm.dist.privacy import PrivacyEnforcer as PrivacyEnforcer
from distllm.dist.async_pipeline import AsyncPipelineEngine as AsyncPipelineEngine
from distllm.dist.async_pipeline import AsyncPipelineConfig as AsyncPipelineConfig
from distllm.dist.config import WideAreaConfig as WideAreaConfig
from distllm.dist.model_store import ModelStore as ModelStore
from distllm.dist.geo import GeoRouter as GeoRouter
from distllm.dist.geo import ClusterLoad as ClusterLoad
from distllm.dist.geo import LoadReporter as LoadReporter
from distllm.dist.cross_cluster import CrossClusterForwarder as CrossClusterForwarder
from distllm.dist.merkle import MerkleTree as MerkleTree
from distllm.dist.prefix_cache import PrefixCache as PrefixCache
from distllm.dist.predictive_cache import PredictiveCacheManager as PredictiveCacheManager
from distllm.dist.chunked_prefill import ChunkState as ChunkState
from distllm.dist.cache import CacheIndex as CacheIndex
from distllm.dist.cache import TTLPolicy as TTLPolicy
from distllm.dist.attention import PagedAttentionManager as PagedAttentionManager
from distllm.dist.attention import BlockPool as BlockPool
from distllm.dist.preemption import PreemptionPolicy as PreemptionPolicy
from distllm.dist.preemption import GPUMemoryMonitor as GPUMemoryMonitor
from distllm.dist.quality import QualitySLA as QualitySLA
from distllm.dist.quality import SLAPolicy as SLAPolicy
from distllm.dist.nat import StunClient as StunClient
from distllm.dist.nat import TurnRelayServer as TurnRelayServer
from distllm.dist.nat import TurnRelayClient as TurnRelayClient
from distllm.dist.wide_area import WideAreaPipeline as WideAreaPipeline
from distllm.dist.fsdp import FSDPShard as FSDPShard
from distllm.dist.fsdp import FSDPConfig as FSDPConfig
from distllm.dist.parallel import HybridParallelPlanner as HybridParallelPlanner
from distllm.dist.parallel import HybridParallelExecutor as HybridParallelExecutor
from distllm.dist.parallel import ParallelStrategy as ParallelStrategy
from distllm.dist.network import Topology as Topology
from distllm.dist.partition import AutoPartitionConfig as AutoPartitionConfig
from distllm.dist.partition import HardwareAwarePartitioner as HardwareAwarePartitioner
from distllm.dist.partition import PartitionOptimizer as PartitionOptimizer
from distllm.dist.partition import PartitionSolution as PartitionSolution
from distllm.dist.partition import GPUProfiler as GPUProfiler
from distllm.dist.partition import TopologyGraph as TopologyGraph
from distllm.dist.partition import PartitionCostModel as PartitionCostModel

# P2P subpackage (registered via _register but living in p2p.*)
from distllm.dist.p2p.discovery import FederationPeerDiscovery as FederationPeerDiscovery
from distllm.dist.p2p.load_balancer import FederationLoadBalancer as FederationLoadBalancer
from distllm.dist.p2p.router import FederationRouter as FederationRouter
from distllm.dist.p2p.transport import P2PTransport as P2PTransport
from distllm.dist.p2p.gossip import GossipProtocol as GossipProtocol

# Cache digest & content routing
from distllm.dist.cache_digest import ContentRouter as ContentRouter
from distllm.dist.cache_digest import CacheDigestExchange as CacheDigestExchange
from distllm.dist.cache_digest import KVCacheDigest as KVCacheDigest

__all__ = [
    # Core pipeline
    "PipelineOrchestrator", "TransportBackend", "TensorTransport",
    "WorkerNode", "NodeRegistrar", "NodeServer", "NodeServicer", "NodeClient",
    # Recovery & stragglers
    "NodeRecoveryManager", "NodeRecoveryPlan", "LayerRedistribution", "SequenceCheckpoint",
    "StragglerDetector", "DetectionMethod", "StragglerReport", "StragglerSeverity",
    # Routing & topology
    "LatencyTracker", "Rebalancer", "PartitionRecommendation",
    "RedundantExecutor", "ReputationSystem", "ReputationRecord",
    "DynamicClusterTopology", "NodeInfo",
    # P2P & federation
    "DiscoveryService", "DiscoveryClient",
    "FederationConfig", "FederationCoordinator",
    "PrivacySplitConfig", "PrivacyEnforcer",
    "AsyncPipelineEngine", "AsyncPipelineConfig",
    "WideAreaConfig", "ModelStore",
    "GeoRouter", "ClusterLoad", "LoadReporter",
    "CrossClusterForwarder", "MerkleTree",
    "PrefixCache", "PredictiveCacheManager",
    "ChunkState", "CacheIndex", "TTLPolicy",
    "PagedAttentionManager", "BlockPool",
    "PreemptionPolicy", "GPUMemoryMonitor",
    "QualitySLA", "SLAPolicy",
    "StunClient", "TurnRelayServer", "TurnRelayClient",
    "WideAreaPipeline",
    "FSDPShard", "FSDPConfig",
    "HybridParallelPlanner", "HybridParallelExecutor", "ParallelStrategy",
    "Topology",
    "AutoPartitionConfig", "HardwareAwarePartitioner",
    "PartitionOptimizer", "PartitionSolution",
    "GPUProfiler", "TopologyGraph", "PartitionCostModel",
    # P2P subpackage
    "FederationPeerDiscovery", "FederationLoadBalancer",
    "FederationRouter", "P2PTransport", "GossipProtocol",
    # Cache digest
    "ContentRouter", "CacheDigestExchange", "KVCacheDigest",
]
