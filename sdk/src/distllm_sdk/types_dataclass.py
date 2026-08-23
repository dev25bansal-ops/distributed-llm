"""Dataclass-based types — fallback when pydantic is not installed.

These are exact mirrors of the pydantic BaseModels in ``types.py``,
implemented as frozen dataclasses for zero-dependency operation.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class UsageInfo:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tokens_per_second: float = 0.0
    estimated_cost: float = 0.0
    cost_usd: float = 0.0
    gpu_time_seconds: float = 0.0
    savings_vs_cloud_usd: float = 0.0
    ttft_ms: float = 0.0


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class ChatCompletionRequest:
    messages: list[ChatMessage]
    model: str = "distributed-llm"
    temperature: float = 0.7
    top_p: float = 1.0
    n: int = 1
    max_tokens: int = 256
    stream: bool = False
    stop: list[str] | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logit_bias: dict[int, float] | None = None
    user: str | None = None
    response_format: dict | None = None
    adapter: str | None = None


@dataclass(frozen=True)
class CompletionRequest:
    prompt: str | list[str]
    model: str = "distributed-llm"
    temperature: float = 0.7
    top_p: float = 1.0
    n: int = 1
    max_tokens: int = 256
    stream: bool = False
    stop: list[str] | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logit_bias: dict[int, float] | None = None
    user: str | None = None


@dataclass(frozen=True)
class ChatChoice:
    index: int = 0
    message: ChatMessage | None = None
    delta: str | None = None
    finish_reason: str | None = None


@dataclass(frozen=True)
class ChatCompletionResponse:
    id: str
    model: str
    choices: list[ChatChoice]
    created: int = 0
    object: str = "chat.completion"
    usage: UsageInfo | None = None
    generation_time: float | None = None


@dataclass(frozen=True)
class CompletionChoice:
    index: int = 0
    text: str = ""
    finish_reason: str | None = None


@dataclass(frozen=True)
class CompletionResponse:
    id: str
    model: str
    choices: list[CompletionChoice]
    created: int = 0
    object: str = "text_completion"
    usage: UsageInfo | None = None
    generation_time: float | None = None


@dataclass(frozen=True)
class EmbeddingObject:
    index: int
    embedding: list[float]


@dataclass(frozen=True)
class EmbeddingResponse:
    model: str
    data: list[EmbeddingObject]
    usage: UsageInfo | None = None


@dataclass(frozen=True)
class ModelInfo:
    id: str
    owned_by: str = "distributed-llm"
    created: int = 0
    object: str = "model"


@dataclass(frozen=True)
class ModelList:
    data: list[ModelInfo]
    object: str = "list"


@dataclass(frozen=True)
class BatchJob:
    id: str
    status: str
    input_file_id: str
    created_at: int
    completed_at: int | None = None
    output_file_id: str | None = None
    error_file_id: str | None = None
    request_counts: dict[str, int] | None = None


@dataclass(frozen=True)
class BatchList:
    data: list[BatchJob]


@dataclass(frozen=True)
class TranscriptionResponse:
    text: str
    language: str | None = None
    duration: float | None = None
    words: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class SpeechResponse:
    content: bytes
    content_type: str = "audio/mpeg"


@dataclass(frozen=True)
class ImageObject:
    url: str | None = None
    b64_json: str | None = None
    revised_prompt: str | None = None


@dataclass(frozen=True)
class ImageGenerationResponse:
    created: int
    data: list[ImageObject]


@dataclass(frozen=True)
class ModerationResult:
    flagged: bool
    categories: dict[str, bool]
    category_scores: dict[str, float]


@dataclass(frozen=True)
class ModerationResponse:
    id: str
    model: str
    results: list[ModerationResult]


@dataclass(frozen=True)
class FileInfo:
    id: str
    filename: str
    purpose: str
    bytes: int
    created_at: int


@dataclass(frozen=True)
class FineTuningJob:
    id: str
    status: str
    model: str
    training_file: str
    created_at: int
    finished_at: int | None = None
    result_file: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class CallStats:
    endpoint: str
    latency: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    status_code: int = 200


@dataclass
class ClientStats:
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
        input_cost = (self.total_prompt_tokens / 1_000_000) * price_per_million_input
        output_cost = (self.total_completion_tokens / 1_000_000) * price_per_million_output
        return input_cost + output_cost
