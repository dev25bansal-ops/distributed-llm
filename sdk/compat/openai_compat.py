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
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.chat = ChatCompletions(self)
        self.completions = Completions(self)
        self.embeddings = Embeddings(self)
        self.models = Models(self)


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

        response = self._post("/chat/completions", payload)
        return ChatCompletion.from_dict(response)

    def _stream_create(self, payload: dict) -> Iterator[ChatCompletionChunk]:
        """Stream chat completion chunks."""
        payload["stream"] = True
        url = f"{self._client.base_url}/chat/completions"
        headers = self._headers()

        with httpx.Client(timeout=self._client.timeout) as client:
            with client.stream("POST", url, json=payload, headers=headers) as response:
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

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self._client.base_url}{path}"
        headers = self._headers()
        with httpx.Client(timeout=self._client.timeout) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self._client.api_key:
            headers["Authorization"] = f"Bearer {self._client.api_key}"
        return headers


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
        headers = {"Content-Type": "application/json"}
        if self._client.api_key:
            headers["Authorization"] = f"Bearer {self._client.api_key}"
        with httpx.Client(timeout=self._client.timeout) as client:
            response = client.post(url, json=payload, headers=headers)
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
        headers = {"Content-Type": "application/json"}
        if self._client.api_key:
            headers["Authorization"] = f"Bearer {self._client.api_key}"
        with httpx.Client(timeout=self._client.timeout) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return EmbeddingResponse.from_dict(response.json())


class Models:
    """Models API."""

    def __init__(self, client: OpenAI):
        self._client = client

    def list(self) -> Any:
        """List available models."""
        url = f"{self._client.base_url}/models"
        headers = {"Content-Type": "application/json"}
        if self._client.api_key:
            headers["Authorization"] = f"Bearer {self._client.api_key}"
        with httpx.Client(timeout=self._client.timeout) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            return ModelList.from_dict(response.json())


# ── Response Objects ──────────────────────────────────────────────────────


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
        choices = []
        for c in data.get("choices", []):
            msg = c.get("message", {})
            choices.append(Choice(
                index=c.get("index", 0),
                message=Message(role=msg.get("role", "assistant"), content=msg.get("content", "")),
                finish_reason=c.get("finish_reason", "stop"),
            ))
        usage = data.get("usage", {})
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            created=data.get("created", int(time.time())),
            model=data.get("model", ""),
            choices=choices,
            usage=Usage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            ),
        )


@dataclass
class ChatCompletionChunk:
    id: str = ""
    object: str = "chat.completion.chunk"
    choices: list[dict] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> ChatCompletionChunk:
        return cls(
            id=data.get("id", ""),
            choices=data.get("choices", []),
        )


@dataclass
class Completion:
    id: str = ""
    object: str = "text_completion"
    choices: list[dict] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    @classmethod
    def from_dict(cls, data: dict) -> Completion:
        usage = data.get("usage", {})
        return cls(
            id=data.get("id", ""),
            choices=data.get("choices", []),
            usage=Usage(**usage) if usage else Usage(),
        )


@dataclass
class EmbeddingResponse:
    data: list[dict] = field(default_factory=list)
    model: str = ""
    usage: Usage = field(default_factory=Usage)

    @classmethod
    def from_dict(cls, data: dict) -> EmbeddingResponse:
        usage = data.get("usage", {})
        return cls(
            data=data.get("data", []),
            model=data.get("model", ""),
            usage=Usage(**usage) if usage else Usage(),
        )


@dataclass
class ModelList:
    object: str = "list"
    data: list[dict] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> ModelList:
        return cls(
            object=data.get("object", "list"),
            data=data.get("data", []),
        )
