"""Type definitions for the DistLLM SDK."""

from dataclasses import dataclass, field
from typing import Any
from pydantic import BaseModel, Field


# --- Response dataclasses (typed, lightweight) ---

@dataclass
class UsageInfo:
    """Token usage and timing information."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tokens_per_second: float = 0.0
    estimated_cost: float = 0.0  # USD


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class ChatChoice:
    index: int
    message: ChatMessage | None = None
    delta: str | None = None
    finish_reason: str | None = None


@dataclass
class ChatCompletionResponse:
    id: str
    model: str
    choices: list[ChatChoice]
    created: int = 0
    object: str = "chat.completion"
    usage: UsageInfo | None = None
    generation_time: float | None = None


@dataclass
class CompletionChoice:
    index: int
    text: str = ""
    finish_reason: str | None = None


@dataclass
class CompletionResponse:
    id: str
    model: str
    choices: list[CompletionChoice]
    created: int = 0
    object: str = "text_completion"
    usage: UsageInfo | None = None
    generation_time: float | None = None


@dataclass
class EmbeddingObject:
    index: int
    embedding: list[float]


@dataclass
class EmbeddingResponse:
    model: str
    data: list[EmbeddingObject]
    usage: UsageInfo | None = None


@dataclass
class ModelInfo:
    id: str
    owned_by: str = "distributed-llm"
    created: int = 0
    object: str = "model"


@dataclass
class ModelList:
    data: list[ModelInfo]
    object: str = "list"


# --- Batch API types ---

@dataclass
class BatchJob:
    id: str
    status: str  # validating, in_progress, finalizing, completed, failed, cancelled
    input_file_id: str
    created_at: int
    completed_at: int | None = None
    output_file_id: str | None = None
    error_file_id: str | None = None
    request_counts: dict[str, int] | None = None


@dataclass
class BatchList:
    data: list[BatchJob]


# --- Audio types ---

@dataclass
class TranscriptionResponse:
    text: str
    language: str | None = None
    duration: float | None = None
    words: list[dict[str, Any]] | None = None


@dataclass
class SpeechResponse:
    content: bytes
    content_type: str = "audio/mpeg"


# --- Image types ---

@dataclass
class ImageObject:
    url: str | None = None
    b64_json: str | None = None
    revised_prompt: str | None = None


@dataclass
class ImageGenerationResponse:
    created: int
    data: list[ImageObject]


# --- Moderation types ---

@dataclass
class ModerationResult:
    flagged: bool
    categories: dict[str, bool]
    category_scores: dict[str, float]


@dataclass
class ModerationResponse:
    id: str
    model: str
    results: list[ModerationResult]


# --- File types ---

@dataclass
class FileInfo:
    id: str
    filename: str
    purpose: str
    bytes: int
    created_at: int


# --- Fine-tuning types ---

@dataclass
class FineTuningJob:
    id: str
    status: str
    model: str
    training_file: str
    created_at: int
    finished_at: int | None = None
    result_file: str | None = None
    error: str | None = None


# --- Usage tracking ---

@dataclass
class CallStats:
    """Per-call statistics."""
    endpoint: str
    latency: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    status_code: int = 200


@dataclass
class ClientStats:
    """Aggregate client statistics."""
    total_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_latency: float = 0.0
    errors: int = 0
    call_log: list[CallStats] = field(default_factory=list)

    @property
    def tokens_per_second(self) -> float:
        if self.total_latency > 0:
            return self.total_completion_tokens / self.total_latency
        return 0.0

    @property
    def avg_latency(self) -> float:
        if self.total_calls > 0:
            return self.total_latency / self.total_calls
        return 0.0

    def estimate_cost(
        self,
        price_per_million_input: float = 0.50,
        price_per_million_output: float = 1.50,
    ) -> float:
        """Estimate cost in USD based on token usage."""
        input_cost = (self.total_prompt_tokens / 1_000_000) * price_per_million_input
        output_cost = (self.total_completion_tokens / 1_000_000) * price_per_million_output
        return input_cost + output_cost
