"""Model-as-a-Service Gateway: unified inference router to multiple backends.

Supports vLLM, TGI, Ollama, and native DistLLM backends with
model fallback chains, health-aware routing, and optional
multi-cluster federation via ``MultiClusterRouter``.
"""

from distllm.gateway.models import (
    GatewayConfig,
    BackendConfig,
    BackendType,
    ModelRoute,
    FallbackChain,
    BackendHealth,
)
from distllm.gateway.backend import (
    ModelBackend,
    NativeBackend,
    VLLMBackend,
    TGIBackend,
    OllamaBackend,
)
from distllm.gateway.router import GatewayRouter
from distllm.gateway.fallback import FallbackManager

__all__ = [
    "GatewayConfig",
    "BackendConfig",
    "BackendType",
    "ModelRoute",
    "FallbackChain",
    "BackendHealth",
    "ModelBackend",
    "NativeBackend",
    "VLLMBackend",
    "TGIBackend",
    "OllamaBackend",
    "GatewayRouter",
    "FallbackManager",
]
