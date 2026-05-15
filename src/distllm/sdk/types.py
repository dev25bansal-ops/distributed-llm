"""Type definitions for the DistLLM SDK."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A message in a chat conversation."""

    role: str = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")


class ChatCompletionRequest(BaseModel):
    """Request body for a chat completion."""

    messages: List[ChatMessage]
    model: str = "distributed-llm"
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 256
    stream: bool = False
    response_format: Optional[Dict[str, str]] = None
    adapter: Optional[str] = None


class ChatChoice(BaseModel):
    """A choice in a chat completion response."""

    index: int = 0
    message: Optional[ChatMessage] = None
    delta: Optional[Dict[str, str]] = None
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    """Response from a chat completion request."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Optional[Dict[str, int]] = None
    generation_time: Optional[float] = None


class CompletionRequest(BaseModel):
    """Request body for a completion."""

    prompt: str
    model: str = "distributed-llm"
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 256
    stream: bool = False


class CompletionChoice(BaseModel):
    """A choice in a completion response."""

    index: int = 0
    text: str = ""
    delta: Optional[str] = None
    finish_reason: Optional[str] = None


class CompletionResponse(BaseModel):
    """Response from a completion request."""

    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    generation_time: Optional[float] = None


class ModelInfo(BaseModel):
    """Information about an available model."""

    id: str
    object: str = "model"
    created: int
    owned_by: str = "distributed-llm"


class ModelList(BaseModel):
    """List of available models."""

    object: str = "list"
    data: List[ModelInfo]
