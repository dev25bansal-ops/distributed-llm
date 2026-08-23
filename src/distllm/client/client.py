"""Async and sync HTTP client for DistLLM clusters.

Provides ``DistLLMClient`` — a typed, async-first SDK for interacting
with a DistLLM coordinator's REST API (OpenAI-compatible).

Usage::

    client = await DistLLMClient.connect(
        coordinator_url="http://10.0.0.1:8000",
        api_key="sk-...",
    )
    result = await client.generate("What is quantum computing?")
    print(result.text)
    await client.close()

Thread-safe: each client owns an ``httpx.AsyncClient`` with connection
pooling and automatic retry on transient failures.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx

# ── Response models ────────────────────────────────────────────────────


@dataclass
class CompletionChoice:
    """A single completion choice."""
    text: str = ""
    index: int = 0
    finish_reason: str = ""
    token_id: int = 0


@dataclass
class CompletionResponse:
    """Response from a completion or chat request."""
    text: str = ""
    choices: list[CompletionChoice] = field(default_factory=list)
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    request_id: str = ""


@dataclass
class ChatMessage:
    """A single chat message."""
    role: str = ""
    content: str = ""


@dataclass
class ChatChoice:
    """A single chat completion choice."""
    message: ChatMessage = field(default_factory=ChatMessage)
    index: int = 0
    finish_reason: str = ""


@dataclass
class ChatResponse:
    """Response from a chat completion request."""
    message: ChatMessage = field(default_factory=ChatMessage)
    choices: list[ChatChoice] = field(default_factory=list)
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    request_id: str = ""


@dataclass
class ModelInfo:
    """Model metadata."""
    id: str = ""
    object: str = "model"
    owned_by: str = ""
    created: int = 0


@dataclass
class NodeInfo:
    """Worker node information."""
    node_id: str = ""
    host: str = ""
    port: int = 0
    start_layer: int = 0
    end_layer: int = 0
    healthy: bool = True
    gpu_name: str = ""
    gpu_utilization: float = 0.0
    memory_free_mb: float = 0.0


@dataclass
class ClusterMetrics:
    """Cluster-wide metrics snapshot."""
    requests_total: int = 0
    tokens_generated: int = 0
    active_requests: int = 0
    pending_requests: int = 0
    node_count: int = 0
    p95_latency_ms: float = 0.0
    errors_total: int = 0
    cache_hit_rate: float = 0.0


# ── Main client ────────────────────────────────────────────────────────


class DistLLMClient:
    """Async HTTP client for a DistLLM coordinator.

    Wraps the OpenAI-compatible REST API (``/v1/chat/completions``,
    ``/v1/completions``, ``/v1/models``) plus DistLLM-specific
    endpoints (``/api/v1/nodes``, ``/api/v1/metrics``, etc.).

    Connection pooling, retry, and auth are handled automatically.
    """

    DEFAULT_TIMEOUT = 60.0
    MAX_RETRIES = 3

    def __init__(
        self,
        coordinator_url: str,
        api_key: str = "",
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        self._base_url = coordinator_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = max_retries

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=timeout,
        )
        self._connected = True

    # ── Factory ────────────────────────────────────────────────────────

    @classmethod
    async def connect(
        cls,
        coordinator_url: str,
        api_key: str = "",
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = MAX_RETRIES,
    ) -> DistLLMClient:
        """Create and return a connected client.

        Args:
            coordinator_url: Base URL of the coordinator HTTP API.
            api_key: Optional API key for authentication.
            timeout: HTTP request timeout in seconds.
            max_retries: Max retries for transient failures.

        Returns:
            A ready-to-use ``DistLLMClient`` instance.
        """
        client = cls(
            coordinator_url=coordinator_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )
        await client._check_connection()
        return client

    async def _check_connection(self) -> None:
        """Verify the coordinator is reachable."""
        try:
            await self._client.get("/health", timeout=5.0)
        except Exception as e:
            self._connected = False
            raise ConnectionError(
                f"Could not connect to DistLLM coordinator at {self._base_url}: {e}"
            ) from e

    # ── Generation ─────────────────────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        model: str = "",
        **kwargs: Any,
    ) -> CompletionResponse:
        """Generate text (non-streaming).

        Args:
            prompt: Input text.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            model: Model override (uses coordinator default if empty).

        Returns:
            ``CompletionResponse`` with generated text.
        """
        payload: dict[str, Any] = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if model:
            payload["model"] = model
        payload.update(kwargs)

        data = await self._request("POST", "/v1/completions", json=payload)
        return self._parse_completion(data)

    async def generate_chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        model: str = "",
        **kwargs: Any,
    ) -> ChatResponse:
        """Chat completion (non-streaming).

        Args:
            messages: List of ``{"role": "...", "content": "..."}`` dicts.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            model: Model override.

        Returns:
            ``ChatResponse`` with the assistant message.
        """
        payload: dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if model:
            payload["model"] = model
        payload.update(kwargs)

        data = await self._request("POST", "/v1/chat/completions", json=payload)
        return self._parse_chat(data)

    # ── Streaming ──────────────────────────────────────────────────────

    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream generated tokens.

        Yields text chunks as they arrive via SSE.

        Args:
            prompt: Input text.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Yields:
            Text chunks (strings) one at a time.
        """
        payload: dict[str, Any] = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        payload.update(kwargs)

        async with self._client.stream(
            "POST", "/v1/completions", json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data: "):
                    chunk = line[6:]
                    if chunk.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(chunk)
                        choices = data.get("choices", [])
                        if choices:
                            text = choices[0].get("text", "")
                            if text:
                                yield text
                    except json.JSONDecodeError:
                        continue

    # ── Cluster info ───────────────────────────────────────────────────

    async def list_models(self) -> list[ModelInfo]:
        """List available models."""
        data = await self._request("GET", "/v1/models")
        return [
            ModelInfo(id=m.get("id", ""), owned_by=m.get("owned_by", ""))
            for m in data.get("data", [])
        ]

    async def list_nodes(self) -> list[NodeInfo]:
        """List registered worker nodes."""
        data = await self._request("GET", "/api/v1/nodes")
        return [
            NodeInfo(
                node_id=n.get("node_id", ""),
                host=n.get("host", ""),
                port=n.get("port", 0),
                start_layer=n.get("start_layer", 0),
                end_layer=n.get("end_layer", 0),
                healthy=n.get("healthy", True),
                gpu_name=n.get("gpu_name", ""),
                gpu_utilization=n.get("gpu_utilization", 0.0),
                memory_free_mb=n.get("free_memory_bytes", 0) / (1024 * 1024),
            )
            for n in data if isinstance(n, dict)
        ]

    async def get_metrics(self) -> ClusterMetrics:
        """Get cluster-wide metrics."""
        data = await self._request("GET", "/api/v1/metrics")
        return ClusterMetrics(
            requests_total=data.get("requests_total", 0),
            tokens_generated=data.get("tokens_generated", 0),
            active_requests=data.get("active_requests", 0),
            pending_requests=data.get("pending_requests", 0),
            node_count=data.get("node_count", 0),
            p95_latency_ms=data.get("p95_latency_ms", 0.0),
            errors_total=data.get("errors_total", 0),
            cache_hit_rate=data.get("cache_hit_rate", 0.0),
        )

    # ── Lifecycle ──────────────────────────────────────────────────────

    async def close(self) -> None:
        """Close the underlying HTTP session."""
        await self._client.aclose()
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── Internal ───────────────────────────────────────────────────────

    async def _request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> Any:
        """Make an HTTP request with retry and error handling."""
        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = await self._client.request(method, path, **kwargs)
                response.raise_for_status()
                if response.headers.get("content-type", "").startswith("text/plain"):
                    return {"text": response.text}
                return response.json()
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if status in (429, 503, 502, 504) and attempt < self._max_retries - 1:
                    delay = 1.5 ** attempt
                    await asyncio.sleep(delay)
                    last_error = e
                    continue
                raise RuntimeError(
                    f"DistLLM API error {status}: {e.response.text}"
                ) from e
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    delay = 1.5 ** attempt
                    await asyncio.sleep(delay)
                    continue
                raise RuntimeError(
                    f"DistLLM request failed after {self._max_retries} retries: {e}"
                ) from e

    @staticmethod
    def _parse_completion(data: dict) -> CompletionResponse:
        choices = data.get("choices", [])
        # Support text/plain responses returned as {"text": "..."} by _request
        text = data.get("text", "")
        if not text and choices:
            text = choices[0].get("text", "")
        return CompletionResponse(
            text=text,
            choices=[
                CompletionChoice(
                    text=c.get("text", ""),
                    index=c.get("index", 0),
                    finish_reason=c.get("finish_reason", ""),
                )
                for c in choices
            ],
            model=data.get("model", ""),
            usage=data.get("usage", {}),
            request_id=data.get("id", ""),
        )

    @staticmethod
    def _parse_chat(data: dict) -> ChatResponse:
        choices = data.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            return ChatResponse(
                message=ChatMessage(role=msg.get("role", ""), content=msg.get("content", "")),
                choices=[
                    ChatChoice(
                        message=ChatMessage(
                            role=c.get("message", {}).get("role", ""),
                            content=c.get("message", {}).get("content", ""),
                        ),
                        index=c.get("index", 0),
                        finish_reason=c.get("finish_reason", ""),
                    )
                    for c in choices
                ],
                model=data.get("model", ""),
                usage=data.get("usage", {}),
                request_id=data.get("id", ""),
            )
        return ChatResponse()


# ── Sync convenience wrapper ───────────────────────────────────────────


class SyncDistLLMClient:
    """Synchronous wrapper around ``DistLLMClient``.

    Uses ``asyncio.run()`` under the hood so you don't need async
    code in simple scripts.

    Usage::

        client = SyncDistLLMClient("http://10.0.0.1:8000", api_key="sk-...")
        result = client.generate("Hello!")
        print(result.text)
        client.close()
    """

    def __init__(self, coordinator_url: str, api_key: str = ""):
        self._url = coordinator_url
        self._key = api_key
        self._client: DistLLMClient | None = None

    def _get(self) -> DistLLMClient:
        if self._client is None:
            import asyncio
            self._client = asyncio.run(
                DistLLMClient.connect(self._url, api_key=self._key)
            )
        return self._client

    def generate(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        import asyncio
        return asyncio.run(self._get().generate(prompt, **kwargs))

    def generate_chat(self, messages: list[dict[str, str]], **kwargs: Any) -> ChatResponse:
        import asyncio
        return asyncio.run(self._get().generate_chat(messages, **kwargs))

    def list_models(self) -> list[ModelInfo]:
        import asyncio
        return asyncio.run(self._get().list_models())

    def list_nodes(self) -> list[NodeInfo]:
        import asyncio
        return asyncio.run(self._get().list_nodes())

    def get_metrics(self) -> ClusterMetrics:
        import asyncio
        return asyncio.run(self._get().get_metrics())

    def close(self) -> None:
        if self._client:
            import asyncio
            asyncio.run(self._client.close())
            self._client = None
