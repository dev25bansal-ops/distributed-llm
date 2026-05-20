"""Data models for the model-as-a-service gateway."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class BackendType(str, Enum):
    NATIVE = "native"
    VLLM = "vllm"
    TGI = "tgi"
    OLLAMA = "ollama"


@dataclass
class BackendConfig:
    """Configuration for a single backend instance."""
    name: str
    backend_type: BackendType = BackendType.NATIVE
    base_url: str = "http://localhost:8000"
    api_key: str = ""
    timeout_s: float = 120.0
    max_concurrent: int = 10
    weight: int = 100
    health_interval_s: float = 10.0
    tags: dict[str, str] = field(default_factory=dict)

    def model_dump(self) -> dict:
        return {
            "name": self.name,
            "backend_type": self.backend_type.value,
            "base_url": self.base_url,
            "timeout_s": self.timeout_s,
            "max_concurrent": self.max_concurrent,
            "weight": self.weight,
            "health_interval_s": self.health_interval_s,
            "tags": self.tags,
        }


@dataclass
class ModelRoute:
    """Maps a model name to backend + optional fallback chain."""
    model_name: str
    primary_backend: str
    fallback_chain: list[str] = field(default_factory=list)
    min_healthy_backends: int = 1
    timeout_s: Optional[float] = None


@dataclass
class FallbackChain:
    """Ordered fallback chain: try primary, then alternatives."""
    model: str
    backends: list[str] = field(default_factory=list)


@dataclass
class BackendHealth:
    """Health status of a backend instance."""
    backend_name: str
    healthy: bool = True
    latency_ms: float = 0.0
    active_requests: int = 0
    last_check: float = 0.0
    error: str = ""
    models_available: list[str] = field(default_factory=list)
    gpu_utilization: float = 0.0


@dataclass
class GatewayConfig:
    """Top-level gateway configuration."""
    enabled: bool = True
    backends: list[BackendConfig] = field(default_factory=list)
    routes: list[ModelRoute] = field(default_factory=list)
    default_fallback: list[str] = field(default_factory=list)
    health_check_interval_s: float = 15.0
    redis_url: Optional[str] = None
