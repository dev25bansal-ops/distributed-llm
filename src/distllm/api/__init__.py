"""API layer for distributed LLM inference."""

from distllm.api.server import (
    app,
    create_coordinator,
    main,
    # Re-exported models for backward compatibility
    ChatMessage,
    ChatCompletionRequest,
    ChatChoice,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionChoice,
    CompletionResponse,
    EmbeddingRequest,
    EmbeddingObject,
    EmbeddingResponse,
    ModelInfo,
    ModelList,
    ParamUpdateRequest,
    AdapterLoadRequest,
)
from distllm.api.middleware import AuthMiddleware

__all__ = [
    "app",
    "create_coordinator",
    "main",
    "AuthMiddleware",
    # Models
    "ChatMessage",
    "ChatCompletionRequest",
    "ChatChoice",
    "ChatCompletionResponse",
    "CompletionRequest",
    "CompletionChoice",
    "CompletionResponse",
    "EmbeddingRequest",
    "EmbeddingObject",
    "EmbeddingResponse",
    "ModelInfo",
    "ModelList",
    "ParamUpdateRequest",
    "AdapterLoadRequest",
]
