from __future__ import annotations

from pydantic import BaseModel, Field


class DisaggPoolConfig(BaseModel):
    """Configuration for a single pool (prefill or decode)."""
    min_nodes: int = Field(default=1, ge=1, description="Minimum number of worker nodes")
    max_nodes: int = Field(default=16, ge=1, description="Maximum number of worker nodes")
    default_capacity: int = Field(default=4, ge=1, description="Default per-node request capacity")
    target_latency_ms: float = Field(default=500.0, ge=1.0, description="Target latency in ms")
    scale_up_threshold: float = Field(default=0.75, ge=0.0, le=1.0, description="Utilization threshold to trigger scale-up")
    scale_down_threshold: float = Field(default=0.30, ge=0.0, le=1.0, description="Utilization threshold to trigger scale-down")
    cooldown_seconds: float = Field(default=60.0, ge=1.0, description="Minimum time between scaling events")


class DisaggKVCacheConfig(BaseModel):
    """KV cache transfer and storage configuration."""
    transfer_backend: str = Field(default="grpc", pattern=r"^(grpc|rdma|nvlink|shared_memory)$", description="Backend for KV cache transfer between pools")
    default_ttl_secs: float = Field(default=300.0, ge=1.0, description="Default TTL for cached KV cache entries")
    enable_compression: bool = Field(default=False, description="Compress KV cache before transfer")
    compression_bits: int = Field(default=8, ge=4, le=16, description="Quantization bits for KV cache compression")


class DisaggScalingConfig(BaseModel):
    """Scaling behavior for each pool."""
    enabled: bool = Field(default=False, description="Enable autoscaling")
    prefill: DisaggPoolConfig = Field(default_factory=lambda: DisaggPoolConfig(max_nodes=16, default_capacity=4, target_latency_ms=500.0))
    decode: DisaggPoolConfig = Field(default_factory=lambda: DisaggPoolConfig(max_nodes=32, default_capacity=8, target_latency_ms=200.0))


class DisaggFullConfig(BaseModel):
    """Full configuration for disaggregated serving."""
    enabled: bool = Field(default=False, description="Enable disaggregated prefill/decode serving")
    scaling: DisaggScalingConfig = Field(default_factory=DisaggScalingConfig)
    kv_cache: DisaggKVCacheConfig = Field(default_factory=DisaggKVCacheConfig)
    prefill_nodes: list[dict] = Field(default_factory=list)
    decode_nodes: list[dict] = Field(default_factory=list)
