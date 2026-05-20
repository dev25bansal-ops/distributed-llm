"""Data models for edge deployment."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class QuantizationType(str, Enum):
    INT4 = "int4"
    INT8 = "int8"
    FP16 = "fp16"
    NF4 = "nf4"
    FP4 = "fp4"


@dataclass
class ModelShard:
    """A single shard of a quantized model."""
    shard_id: str
    model_name: str
    shard_index: int
    total_shards: int
    bytes_size: int
    quantization: QuantizationType = QuantizationType.INT4
    checksum: str = ""
    device: str = "cpu"


@dataclass
class EdgeHealth:
    """Health snapshot of an edge node."""
    node_id: str
    healthy: bool = True
    cpu_usage_pct: float = 0.0
    memory_usage_pct: float = 0.0
    gpu_memory_available_mb: float = 0.0
    active_requests: int = 0
    queue_depth: int = 0
    uptime_s: float = 0.0
    error: str = ""


@dataclass
class EdgeNodeInfo:
    """Static info about an edge deployment node."""
    node_id: str
    host: str = "127.0.0.1"
    port: int = 9100
    max_memory_mb: float = 4096
    max_requests: int = 8
    supported_quantizations: list[QuantizationType] = field(default_factory=lambda: list(QuantizationType))
    device: str = "cpu"
    models_deployed: list[str] = field(default_factory=list)


@dataclass
class EdgeConfig:
    """Configuration for an edge node."""
    enabled: bool = True
    node_id: str = "edge-1"
    host: str = "0.0.0.0"
    port: int = 9100
    device: str = "cpu"
    max_memory_mb: float = 4096
    max_concurrent_requests: int = 8
    quantization: QuantizationType = QuantizationType.INT4
    cloud_fallback_url: str = "http://cluster:8000"
    cloud_fallback_timeout_s: float = 30.0
    models: list[str] = field(default_factory=lambda: ["llama-3.2-1b", "qwen2.5-0.5b"])
    shard_dir: str = "/tmp/edge-shards"
    health_interval_s: float = 10.0
