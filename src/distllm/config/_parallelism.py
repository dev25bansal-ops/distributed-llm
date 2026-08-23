"""Node roles, node settings, tensor/hybrid parallelism, partitioning,
batching, chunked prefill, priority, and disaggregation configuration classes."""

from enum import Enum
from pydantic import BaseModel, Field, field_validator

__all__ = [
    "NodeRole",
    "NodeSettings",
    "TensorParallelSettings",
    "HybridParallelSettings",
    "ZeroCopySettings",
    "PartitioningSettings",
    "RebalancerSettings",
    "BatchingSettings",
    "ChunkedPrefillSettings",
    "PrioritySettings",
    "DisaggSettings",
]


class NodeRole(str, Enum):
    """Node role for prefill-decode disaggregation."""
    AUTO = "auto"
    PREFILL = "prefill"
    DECODE = "decode"


class NodeSettings(BaseModel):
    """Worker node configuration."""
    node_id: str
    host: str = "localhost"
    port: int = 50051
    start_layer: int = Field(default=0, ge=0, description="First layer index in this node's pipeline partition.")
    end_layer: int = Field(default=3, ge=0, description="Last layer index in this node's pipeline partition.")
    device: str = "cuda"
    role: NodeRole = NodeRole.AUTO

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError(f"Port must be 1-65535, got {v}")
        return v

    @field_validator("end_layer")
    @classmethod
    def validate_end_layer(cls, v: int, info) -> int:
        values = info.data
        if "start_layer" in values and v < values["start_layer"]:
            raise ValueError(f"end_layer ({v}) must be >= start_layer ({values['start_layer']})")
        return v


class TensorParallelSettings(BaseModel):
    """Tensor parallelism configuration."""
    enabled: bool = False
    num_gpus: int = 2

    @field_validator("num_gpus")
    @classmethod
    def validate_num_gpus(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"num_gpus must be >= 1, got {v}")
        return v


class HybridParallelSettings(BaseModel):
    """Hybrid parallelism (TP + PP + EP) configuration."""
    enabled: bool = False
    auto_detect: bool = True
    tp_enabled: bool = True
    pp_overlap: bool = True
    ep_enabled: bool = True
    force_tp_world_size: int = 0
    force_pp_stages: int = 0
    shard_across_nodes: bool = False  # FSDP-style weight sharding across nodes


class ZeroCopySettings(BaseModel):
    """Zero-copy GPU tensor transfer configuration."""
    enabled: bool = False
    prefer_rdma: bool = True
    fallback_to_nccl: bool = True
    intranode_ipc: bool = True


class PartitioningSettings(BaseModel):
    """Layer partitioning strategy configuration."""
    strategy: str = "gpu_aware"  # "equal" | "gpu_aware"
    safety_margin: float = 0.1  # leave 10% VRAM free
    shard_across_nodes: bool = False  # FSDP-style weight sharding across nodes

    def to_auto_partition_config(self):
        """Convert to dict (legacy AutoPartitionConfig)."""
        return {
            "enabled": self.strategy != "equal",
            "strategy": self.strategy if self.strategy != "gpu_aware" else "auto",
            "safety_margin": self.safety_margin,
        }

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        allowed = {"equal", "gpu_aware"}
        if v not in allowed:
            raise ValueError(f"strategy must be one of {allowed}, got '{v}'")
        return v


class RebalancerSettings(BaseModel):
    """Dynamic pipeline rebalancing configuration."""
    enabled: bool = False
    check_interval: float = 30.0
    straggler_threshold: float = 1.5
    min_improvement_pct: float = 0.1
    cooldown_seconds: float = 300.0
    grace_period_steps: int = 3
    auto_mitigate: bool = False


class BatchingSettings(BaseModel):
    """Continuous batching configuration."""
    max_batch_size: int = 32
    max_tokens_per_batch: int = 4096

    @field_validator("max_batch_size", "max_tokens_per_batch")
    @classmethod
    def validate_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"Must be >= 1, got {v}")
        return v


class ChunkedPrefillSettings(BaseModel):
    """Chunked prefill configuration."""
    enabled: bool = True
    chunk_size: int = 512

    @field_validator("chunk_size")
    @classmethod
    def validate_chunk_size(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"chunk_size must be >= 1, got {v}")
        return v


class PrioritySettings(BaseModel):
    """Request priority queuing configuration."""
    enabled: bool = False
    num_levels: int = 4
    preemption_enabled: bool = False
    max_preempted: int = 10


class DisaggSettings(BaseModel):
    """Disaggregated prefill/decode serving configuration.

    Delegates to the full config model from the disagg package.
    """

    enabled: bool = False
    prefill_nodes: list[dict] = []
    decode_nodes: list[dict] = []

    def to_full_config(self):
        """Convert to the package-level DisaggFullConfig."""
        return {
            "enabled": self.enabled,
            "prefill_nodes": self.prefill_nodes,
            "decode_nodes": self.decode_nodes,
        }
