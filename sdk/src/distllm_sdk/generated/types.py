"""Auto-generated type stubs from OpenAPI spec."""

from __future__ import annotations
from typing import Any


class ChatCompletionRequest:
    """Auto-generated from OpenAPI schema."""

    model: str | None
    messages: list["ChatMessage"]
    temperature: float | None
    top_p: float | None
    max_tokens: int | None
    stream: bool | None
    response_format: dict[str, Any] | None
    adapter: str | None
    tools: list["Tool"] | None
    stop: list[str] | None

class ChatMessage:
    """Auto-generated from OpenAPI schema."""

    role: str
    content: str
    name: str | None
    tool_call_id: str | None

class ChatCompletionResponse:
    """Auto-generated from OpenAPI schema."""

    id: str
    model: str
    created: int | None
    choices: list["ChatChoice"]
    usage: "UsageInfo" | None
    generation_time: float | None

class ChatChoice:
    """Auto-generated from OpenAPI schema."""

    index: int | None
    message: "ChatMessage" | None
    finish_reason: str | None

class CompletionRequest:
    """Auto-generated from OpenAPI schema."""

    model: str | None
    prompt: str
    temperature: float | None
    max_tokens: int | None
    stream: bool | None

class CompletionResponse:
    """Auto-generated from OpenAPI schema."""

    id: str
    model: str
    choices: list["CompletionChoice"]
    usage: "UsageInfo" | None

class CompletionChoice:
    """Auto-generated from OpenAPI schema."""

    index: int | None
    text: str | None
    finish_reason: str | None

class EmbeddingRequest:
    """Auto-generated from OpenAPI schema."""

    model: str | None
    input: str

class EmbeddingResponse:
    """Auto-generated from OpenAPI schema."""

    model: str | None
    data: list["EmbeddingObject"] | None
    usage: "UsageInfo" | None

class EmbeddingObject:
    """Auto-generated from OpenAPI schema."""

    index: int | None
    embedding: list[float] | None

class ModelList:
    """Auto-generated from OpenAPI schema."""

    data: list["ModelInfo"] | None

class ModelInfo:
    """Auto-generated from OpenAPI schema."""

    id: str | None
    owned_by: str | None
    created: int | None

class HealthResponse:
    """Auto-generated from OpenAPI schema."""

    status: str | None
    model: str | None
    nodes: int | None
    uptime: float | None

class BatchRequest:
    """Auto-generated from OpenAPI schema."""

    input_file_id: str
    endpoint: str
    metadata: dict[str, Any] | None

class BatchJob:
    """Auto-generated from OpenAPI schema."""

    id: str | None
    status: str | None
    input_file_id: str | None
    created_at: int | None

class BatchList:
    """Auto-generated from OpenAPI schema."""

    data: list["BatchJob"] | None

class UsageInfo:
    """Auto-generated from OpenAPI schema."""

    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    cost_usd: float | None
    tokens_per_second: float | None

class Tool:
    """Auto-generated from OpenAPI schema."""

    type: str | None
    function: dict[str, Any] | None

class ErrorResponse:
    """Auto-generated from OpenAPI schema."""

    error: dict[str, Any] | None
