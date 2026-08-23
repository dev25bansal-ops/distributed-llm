"""Cache subpackage for distributed KV-cache sharing.

Provides content-addressed prefix KV cache indexing, cross-cluster
gossip-based digest exchange, and cache lifecycle management.
"""

from __future__ import annotations

from distllm.cache.cross_cluster_prefix_index import (
    CacheDigest,
    CacheGossipProtocol,
    CrossClusterPrefixIndex,
)

__all__ = [
    "CacheDigest",
    "CacheGossipProtocol",
    "CrossClusterPrefixIndex",
]
