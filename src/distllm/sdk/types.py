"""Type definitions for the DistLLM SDK."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Iterator, AsyncIterator
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
    message: Optional[ChatMessage] = None
    delta: Optional[str] = None
    finish_reason: Optional[str] = None


@dataclass
class ChatCompletionResponse:
    id: str
    model: str
    choices: List[ChatChoice]
    created: int = 0
    object: str = "chat.completion"
    usage: Optional[UsageInfo] = None
    generation_time: Optional[float] = None


@dataclass
class CompletionChoice:
    index: int
    text: str = ""
    finish_reason: Optional[str] = None


@dataclass
class CompletionResponse:
    id: str
    model: str
    choices: List[CompletionChoice]
    created: int = 0
    object: str = "text_completion"
    usage: Optional[UsageInfo] = None
    generation_time: Optional[float] = None


@dataclass
class EmbeddingObject:
    index: int
    embedding: List[float]


@dataclass
class EmbeddingResponse:
    model: str
    data: List[EmbeddingObject]
    usage: Optional[UsageInfo] = None


@dataclass
class ModelInfo:
    id: str
    owned_by: str = "distributed-llm"
    created: int = 0
    object: str = "model"


@dataclass
class ModelList:
    data: List[ModelInfo]
    object: str = "list"


# --- Batch API types ---

@dataclass
class BatchJob:
    id: str
    status: str  # validating, in_progress, finalizing, completed, failed, cancelled
    input_file_id: str
    created_at: int
    completed_at: Optional[int] = None
    output_file_id: Optional[str] = None
    error_file_id: Optional[str] = None
    request_counts: Optional[Dict[str, int]] = None


@dataclass
class BatchList:
    data: List[BatchJob]


# --- Audio types ---

@dataclass
class TranscriptionResponse:
    text: str
    language: Optional[str] = None
    duration: Optional[float] = None
    words: Optional[List[Dict[str, Any]]] = None


@dataclass
class SpeechResponse:
    content: bytes
    content_type: str = "audio/mpeg"


# --- Image types ---

@dataclass
class ImageObject:
    url: Optional[str] = None
    b64_json: Optional[str] = None
    revised_prompt: Optional[str] = None


@dataclass
class ImageGenerationResponse:
    created: int
    data: List[ImageObject]


# --- Moderation types ---

@dataclass
class ModerationResult:
    flagged: bool
    categories: Dict[str, bool]
    category_scores: Dict[str, float]


@dataclass
class ModerationResponse:
    id: str
    model: str
    results: List[ModerationResult]


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
    finished_at: Optional[int] = None
    result_file: Optional[str] = None
    error: Optional[str] = None


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
    call_log: List[CallStats] = field(default_factory=list)

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
