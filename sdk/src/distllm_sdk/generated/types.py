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

class BatchSubmitItem:
    """Auto-generated from OpenAPI schema."""

    request_id: str
    prompt: str
    max_tokens: int | None
    temperature: float | None

class BatchSubmitRequest:
    """Auto-generated from OpenAPI schema."""

    items: list["BatchSubmitItem"]
    webhook_url: str | None
    webhook_token: str | None
    item_timeout: float | None

class BatchSubmitResponse:
    """Auto-generated from OpenAPI schema."""

    batch_id: str
    status: str
    total_items: int | None
    created_at: float | None
    message: str | None

class BatchItemResult:
    """Auto-generated from OpenAPI schema."""

    request_id: str | None
    prompt: str | None
    max_tokens: int | None
    temperature: float | None
    status: str | None
    result: Any | None
    error: str | None
    cost: float | None
    started_at: float | None
    completed_at: float | None

class BatchStatusResponse:
    """Auto-generated from OpenAPI schema."""

    batch_id: str
    status: str
    total_items: int | None
    pending: int | None
    running: int | None
    completed: int | None
    failed: int | None
    timed_out: int | None
    cancelled: int | None
    total_cost: float | None
    created_at: float | None
    completed_at: float | None
    items: list["BatchItemResult"] | None

class BatchCancelResponse:
    """Auto-generated from OpenAPI schema."""

    batch_id: str
    status: str
    cancelled_items: int | None
    message: str | None

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
