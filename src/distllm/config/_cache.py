"""Prefix cache, cache persistence, gossip, predictive cache, unified cache,
and defragmentation configuration classes."""

from pydantic import BaseModel, Field, field_validator

__all__ = [
    "PrefixCacheSettings",
    "CachePersistenceSettings",
    "GossipSettings",
    "PredictiveCacheSettings",
    "CacheSettings",
    "DefragmentationSettings",
]


class PrefixCacheSettings(BaseModel):
    """Prefix cache configuration."""
    enabled: bool = True
    max_entries: int = 1024
    min_prefix_len: int = 16
    radix_tree_enabled: bool = True  # Use RadixTree (trie) instead of hash-based LRU

    @field_validator("max_entries", "min_prefix_len")
    @classmethod
    def validate_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"Must be >= 1, got {v}")
        return v


class CachePersistenceSettings(BaseModel):
    """KV cache persistence to disk configuration."""
    enabled: bool = False
    storage_path: str = ".distllm_cache"
    max_disk_gb: float = 50.0
    ttl_hours: float = 24.0


class GossipSettings(BaseModel):
    """P2P KV cache gossip protocol configuration."""
    enabled: bool = False
    interval: float = 10.0
    max_peers: int = 16
    cache_ttl: float = 300.0


class PredictiveCacheSettings(BaseModel):
    """Predictive KV cache management configuration."""
    enabled: bool = False
    gpu_cache_mb: int = 512
    cpu_cache_mb: int = 4096
    pattern_decay_hours: float = 24.0
    min_prefix_len: int = 8
    background_compress_interval_s: int = 300


class CacheSettings(BaseModel):
    """E15: Unified cache configuration.

    Consolidates PrefixCacheSettings, CachePersistenceSettings,
    PredictiveCacheSettings, and GossipSettings into a single section.
    Individual sub-configs remain for backward compatibility.
    """
    # Prefix cache
    prefix_enabled: bool = True
    prefix_max_entries: int = 1024
    prefix_min_prefix_len: int = 16
    radix_tree_enabled: bool = True

    # Persistence
    persistence_enabled: bool = False
    persistence_storage_path: str = ".distllm_cache"
    persistence_max_disk_gb: float = 50.0
    persistence_ttl_hours: float = 24.0
    background_compaction_enabled: bool = False
    background_compaction_interval_s: float = 300.0

    # Predictive
    predictive_enabled: bool = False
    predictive_gpu_cache_mb: int = 512
    predictive_cpu_cache_mb: int = 4096
    predictive_pattern_decay_hours: float = 24.0
    predictive_min_prefix_len: int = 8

    # Gossip
    gossip_enabled: bool = False
    gossip_interval: float = 10.0
    gossip_max_peers: int = 16
    gossip_cache_ttl: float = 300.0

    # Eviction
    eviction_strategy: str = "hybrid"  # "lru", "lfu", "hybrid"
    size_aware_admission: bool = True
    memory_adaptive_budget: bool = True


class DefragmentationSettings(BaseModel):
    """GPU memory defragmentation configuration.

    Compacts fragmented KV cache blocks to prevent OOM errors during
    long-running inference sessions.
    """
    enabled: bool = False
    policy: str = Field(default="balanced", description="Compaction policy: lazy, balanced, or aggressive")
    interval_seconds: float = Field(default=60.0, ge=5.0, description="Seconds between background defrag checks")
    max_blocks_per_pass: int = Field(default=0, ge=0, description="Max blocks per pass (0 = unlimited)")
    threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="Override policy threshold (0 = use policy default)")
    tiered_compaction: bool = Field(default=False, description="Enable L2 (CPU swap) and L3 (NVMe) compaction")
    l2_cpu_swap_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    l3_nvme_swap_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    cuda_stream_priority: int = Field(default=-1, description="CUDA stream priority for copy ops")
    enable_predictive: bool = Field(default=False, description="Predictive (preemptive) defragmentation")
    enable_prometheus: bool = Field(default=False, description="Export Prometheus defrag metrics")

    @field_validator("policy")
    @classmethod
    def validate_policy(cls, v: str) -> str:
        allowed = {"lazy", "balanced", "aggressive"}
        if v not in allowed:
            raise ValueError(f"policy must be one of {allowed}, got '{v}'")
        return v
