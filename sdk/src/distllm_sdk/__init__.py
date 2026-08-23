"""DistLLM SDK — Python client for Distributed LLM API.

A standalone, typed Python package for connecting to any DistLLM cluster.
Supports both async and sync clients, REST and gRPC protocols.

Version 1.0.0 — Production stable release.
"""

__version__ = "1.0.0"

from distllm_sdk.client import DistLLMClient, DistLLMClientSync, RetryConfig, PoolConfig
from distllm_sdk.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerError, CircuitState
from distllm_sdk.types import (
    ChatCompletionResponse,
    ChatMessage,
    ChatChoice,
    CompletionResponse,
    CompletionChoice,
    ModelInfo,
    ModelList,
    EmbeddingResponse,
    EmbeddingObject,
    BatchJob,
    BatchList,
    TranscriptionResponse,
    SpeechResponse,
    ImageGenerationResponse,
    ImageObject,
    ModerationResponse,
    ModerationResult,
    FileInfo,
    FineTuningJob,
    UsageInfo,
    ClientStats,
    CallStats,
)
from distllm_sdk.errors import (
    ApiError,
    AuthenticationError,
    RateLimitError,
    TimeoutError,
    ModelNotFoundError,
    ServiceUnavailableError,
    InvalidRequestError,
)

__all__ = [
    "DistLLMClient",
    "DistLLMClientSync",
    "RetryConfig",
    "PoolConfig",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitState",
    "ChatCompletionResponse",
    "ChatMessage",
    "ChatChoice",
    "CompletionResponse",
    "CompletionChoice",
    "ModelInfo",
    "ModelList",
    "EmbeddingResponse",
    "EmbeddingObject",
    "BatchJob",
    "BatchList",
    "TranscriptionResponse",
    "SpeechResponse",
    "ImageGenerationResponse",
    "ImageObject",
    "ModerationResponse",
    "ModerationResult",
    "FileInfo",
    "FineTuningJob",
    "UsageInfo",
    "ClientStats",
    "CallStats",
    "ApiError",
    "AuthenticationError",
    "RateLimitError",
    "TimeoutError",
    "ModelNotFoundError",
    "ServiceUnavailableError",
    "InvalidRequestError",
    "CircuitBreakerError",
]

# Lazy import for gRPC (optional dependency)
def __getattr__(name: str):
    if name == "NodeGRPCClient":
        from distllm_sdk.grpc_client import NodeGRPCClient
        return NodeGRPCClient
    raise AttributeError(f"module 'distllm_sdk' has no attribute {name!r}")
