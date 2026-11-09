"""OpenAI SDK compatibility layer — drop-in replacement for the openai package.

Allows users to switch from OpenAI to DistLLM by changing one import:

    # Before (OpenAI):
    import openai
    client = openai.OpenAI(api_key="sk-...")
    response = client.chat.completions.create(model="gpt-4", messages=[...])

    # After (DistLLM):
    from distllm_sdk.compat import openai_compat as openai
    client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="distllm-...")
    response = client.chat.completions.create(model="llama-3-70b", messages=[...])
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterator

import httpx


class OpenAI:
    """OpenAI-compatible client for DistLLM.

    Drop-in replacement for ``openai.OpenAI`` that routes requests
    to a DistLLM cluster.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "",
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = httpx.Client(timeout=httpx.Timeout(timeout), headers=self._base_headers())
        self.chat = ChatCompletions(self)
        self.completions = Completions(self)
        self.embeddings = Embeddings(self)
        self.models = Models(self)

    def _base_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def close(self):
        self._client.close()


class ChatCompletions:
    """Chat completions API."""

    def __init__(self, client: OpenAI):
        self._client = client

    def create(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 128,
        top_p: float = 0.9,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Create a chat completion.

        Args:
            model: Model name.
            messages: List of message dicts with 'role' and 'content'.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            top_p: Nucleus sampling threshold.
            stream: Whether to stream the response.

        Returns:
            ChatCompletion object or iterator of chunks.
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": stream,
            **kwargs,
        }

        if stream:
            return self._stream_create(payload)

        response = self._request_with_retry("POST", "/chat/completions", payload)
        return ChatCompletion.from_dict(response)

    def _stream_create(self, payload: dict) -> Iterator[ChatCompletionChunk]:
        """Stream chat completion chunks."""
        payload["stream"] = True
        url = f"{self._client.base_url}/chat/completions"
        headers = self._client._base_headers()

        with self._client._client.stream("POST", url, json=payload, headers=headers) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk_dict = json.loads(data)
                        yield ChatCompletionChunk.from_dict(chunk_dict)
                    except json.JSONDecodeError:
                        continue

    def _request_with_retry(self, method: str, path: str, payload: dict) -> dict:
        """POST with retry and exponential backoff."""
        url = f"{self._client.base_url}{path}"
        headers = self._client._base_headers()
        last_error: Exception | None = None
        for attempt in range(self._client.max_retries + 1):
            try:
                response = self._client._client.request(method, url, json=payload, headers=headers)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (429, 500, 502, 503, 504) and attempt < self._client.max_retries:
                    delay = min(1.0 * (2 ** attempt), 30.0)
                    time.sleep(delay)
                    last_error = e
                    continue
                raise
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                if attempt < self._client.max_retries:
                    delay = min(1.0 * (2 ** attempt), 30.0)
                    time.sleep(delay)
                    last_error = e
                    continue
                raise
        raise last_error  # type: ignore[misc]


class Completions:
    """Text completions API."""

    def __init__(self, client: OpenAI):
        self._client = client

    def create(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 128,
        **kwargs: Any,
    ) -> Any:
        """Create a text completion."""
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }
        url = f"{self._client.base_url}/completions"
        headers = self._client._base_headers()
        response = self._client._client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return Completion.from_dict(response.json())


class Embeddings:
    """Embeddings API."""

    def __init__(self, client: OpenAI):
        self._client = client

    def create(
        self,
        model: str,
        input: str | list[str],
        **kwargs: Any,
    ) -> Any:
        """Create embeddings."""
        payload = {"model": model, "input": input, **kwargs}
        url = f"{self._client.base_url}/embeddings"
        headers = self._client._base_headers()
        response = self._client._client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return EmbeddingResponse.from_dict(response.json())


class Models:
    """Models API."""

    def __init__(self, client: OpenAI):
        self._client = client

    def list(self) -> Any:
        """List available models."""
        url = f"{self._client.base_url}/models"
        headers = self._client._base_headers()
        response = self._client._client.get(url, headers=headers)
        response.raise_for_status()
        return ModelList.from_dict(response.json())


# ── Response Objects ──────────────────────────────────────────────────────
#
# Parsing contract (Wave-2 item 38): DistLLM extends the OpenAI schemas with
# native extras — top-level ``generation_time``, v2 ``system_fingerprint`` /
# ``api_version``, and extended ``usage`` objects (cost summaries,
# ``processing_time``).  Unknown fields are IGNORED (ignore-extra semantics);
# required structure (objects/lists/string content) is still validated with
# clear ValueErrors so callers get actionable errors on malformed payloads.


@dataclass
class Message:
    role: str = "assistant"
    content: str = ""


@dataclass
class Choice:
    index: int = 0
    message: Message = field(default_factory=Message)
    finish_reason: str = "stop"


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


def _require_dict(value: Any, what: str) -> dict:
    """Validate that ``value`` is a JSON object."""
    if not isinstance(value, dict):
        raise ValueError(f"{what} must be an object, got {type(value).__name__}")
    return value


def _require_list(value: Any, what: str) -> list:
    """Validate that ``value`` is a JSON array."""
    if not isinstance(value, list):
        raise ValueError(f"{what} must be a list, got {type(value).__name__}")
    return value


def _token_count(value: Any) -> int:
    """Coerce a usage counter to int; unknown/non-numeric values count as 0."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def _parse_usage(data: Any) -> Usage:
    """Parse a ``usage`` object with ignore-extra semantics.

    DistLLM merges cost summaries into the usage object (see
    ``core/streaming_cost.to_final_summary`` / ``api/streaming.py``) and adds
    ``processing_time`` on embeddings.  Unknown keys are ignored instead of
    crashing; the three standard counters are coerced to ints.
    """
    if data is None:
        return Usage()
    _require_dict(data, "usage")
    return Usage(
        prompt_tokens=_token_count(data.get("prompt_tokens")),
        completion_tokens=_token_count(data.get("completion_tokens")),
        total_tokens=_token_count(data.get("total_tokens")),
    )


@dataclass
class ChatCompletion:
    id: str = ""
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: list[Choice] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    @classmethod
    def from_dict(cls, data: dict) -> ChatCompletion:
        _require_dict(data, "chat completion")
        raw_choices = _require_list(data.get("choices", []), "choices")
        choices = []
        for c in raw_choices:
            _require_dict(c, "choice")
            msg = c.get("message")
            if msg is None:
                msg = {}
            _require_dict(msg, "message")
            role = msg.get("role", "assistant")
            if role is None:
                role = "assistant"
            if not isinstance(role, str):
                raise ValueError(
                    f"message.role must be a string, got {type(role).__name__}"
                )
            content = msg.get("content")
            # Server sends content=null when tool_calls are present.
            if content is None:
                content = ""
            if not isinstance(content, str):
                raise ValueError(
                    f"message.content must be a string, got "
                    f"{type(content).__name__}"
                )
            choices.append(Choice(
                index=c.get("index", 0),
                message=Message(role=role, content=content),
                finish_reason=c.get("finish_reason", "stop"),
            ))
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            created=data.get("created", int(time.time())),
            model=data.get("model", ""),
            choices=choices,
            usage=_parse_usage(data.get("usage")),
        )


@dataclass
class ChatCompletionChunk:
    id: str = ""
    object: str = "chat.completion.chunk"
    created: int = 0
    model: str = ""
    choices: list[dict] = field(default_factory=list)
    usage: Usage | None = None

    @classmethod
    def from_dict(cls, data: dict) -> ChatCompletionChunk:
        _require_dict(data, "chunk")
        choices = _require_list(data.get("choices", []), "choices")
        for c in choices:
            _require_dict(c, "chunk choice")
        usage_data = data.get("usage")
        return cls(
            id=data.get("id", ""),
            object=data.get("object", "chat.completion.chunk"),
            created=data.get("created", 0),
            model=data.get("model", ""),
            choices=choices,
            usage=_parse_usage(usage_data) if usage_data is not None else None,
        )


@dataclass
class Completion:
    id: str = ""
    object: str = "text_completion"
    choices: list[dict] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    @classmethod
    def from_dict(cls, data: dict) -> Completion:
        _require_dict(data, "completion")
        return cls(
            id=data.get("id", ""),
            object=data.get("object", "text_completion"),
            choices=_require_list(data.get("choices", []), "choices"),
            usage=_parse_usage(data.get("usage")),
        )


@dataclass
class EmbeddingResponse:
    data: list[dict] = field(default_factory=list)
    model: str = ""
    usage: Usage = field(default_factory=Usage)

    @classmethod
    def from_dict(cls, data: dict) -> EmbeddingResponse:
        _require_dict(data, "embedding response")
        return cls(
            data=_require_list(data.get("data", []), "data"),
            model=data.get("model", ""),
            usage=_parse_usage(data.get("usage")),
        )


@dataclass
class ModelList:
    object: str = "list"
    data: list[dict] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> ModelList:
        _require_dict(data, "model list")
        return cls(
            object=data.get("object", "list"),
            data=_require_list(data.get("data", []), "data"),
        )
