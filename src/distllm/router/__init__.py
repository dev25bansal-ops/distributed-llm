"""Request router for distributed-llm.

Supports single-cluster routing (``RouterService`` + ``ConsistentHashRing``)
and multi-cluster federation routing (``MultiClusterRouter`` +
``ClusterDiscovery``).
"""

from distllm.router.service import RouterService
from distllm.router.consistent_hash import ConsistentHashRing
from distllm.router.cluster_discovery import ClusterDiscovery, ClusterInfo, ClusterCoordinator
from distllm.router.multi_cluster_router import (
    MultiClusterRouter,
    ClusterAffinityRing,
    LatencyRouter,
)

__all__ = [
    "RouterService",
    "ConsistentHashRing",
    "ClusterDiscovery",
    "ClusterInfo",
    "ClusterCoordinator",
    "MultiClusterRouter",
    "ClusterAffinityRing",
    "LatencyRouter",
]
