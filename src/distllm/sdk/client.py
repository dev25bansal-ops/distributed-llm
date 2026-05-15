"""DistLLM SDK client for Distributed LLM API."""

import json
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, AsyncIterator, Iterator

import httpx

from distllm.sdk.types import (
    ChatCompletionResponse,
    CompletionResponse,
    ModelList,
)
from distllm.sdk.streaming import parse_sse_stream


class _BaseClient(ABC):
    """Shared implementation of the Distributed LLM API client.

    Subclasses implement _request() and _stream_request() for async/sync HTTP.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    def _build_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _build_chat_payload(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        stream: bool,
        response_format: Optional[dict],
        adapter: Optional[str],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if response_format:
            payload["response_format"] = response_format
        if adapter:
            payload["adapter"] = adapter
        return payload

    async def chat_completions(
        self,
        messages: List[Dict[str, str]],
        model: str = "distributed-llm",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 256,
        stream: bool = False,
        response_format: Optional[dict] = None,
        adapter: Optional[str] = None,
    ) -> ChatCompletionResponse:
        """Generate a chat completion."""
        payload = self._build_chat_payload(
            messages, model, temperature, top_p, max_tokens, stream, response_format, adapter,
        )
        data = await self._request("POST", "/v1/chat/completions", json=payload)
        return ChatCompletionResponse(**data)

    async def completions(
        self,
        prompt: str,
        model: str = "distributed-llm",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 256,
    ) -> CompletionResponse:
        """Generate a text completion."""
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        data = await self._request("POST", "/v1/completions", json=payload)
        return CompletionResponse(**data)

    async def list_models(self) -> ModelList:
        """List available models."""
        data = await self._request("GET", "/v1/models")
        return ModelList(**data)

    async def health_check(self) -> dict:
        """Check API server health."""
        return await self._request("GET", "/health")

    @abstractmethod
    async def _request(self, method: str, path: str, **kwargs) -> dict:
        """Make an HTTP request and return parsed JSON."""

    @abstractmethod
    def _stream_request(self, method: str, path: str, **kwargs) -> Iterator[dict]:
        """Make a streaming HTTP request and yield parsed SSE events."""


class DistLLMClient(_BaseClient):
    """Async client for the Distributed LLM API.

    Usage:
        async with DistLLMClient() as client:
            response = await client.chat_completions(
                messages=[{"role": "user", "content": "Hello"}]
            )
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: float = 120.0,
    ):
        super().__init__(base_url, api_key, timeout)
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=self._build_headers(),
            timeout=timeout,
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()

    async def chat_completions_stream(
        self,
        messages: List[Dict[str, str]],
        model: str = "distributed-llm",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 256,
        response_format: Optional[dict] = None,
        adapter: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream chat completions as an async generator.

        Yields:
            Generated text tokens
        """
        payload = self._build_chat_payload(
            messages, model, temperature, top_p, max_tokens, True, response_format, adapter,
        )
        async with self._client.stream(
            "POST", "/v1/chat/completions", json=payload
        ) as response:
            response.raise_for_status()
            async for event in parse_sse_stream(response):
                if "choices" in event and event["choices"]:
                    delta = event["choices"][0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield content

    async def _request(self, method: str, path: str, **kwargs) -> dict:
        response = await self._client.request(method, path, **kwargs)
        response.raise_for_status()
        return response.json()

    def _stream_request(self, method: str, path: str, **kwargs) -> AsyncIterator[dict]:
        raise NotImplementedError("Use chat_completions_stream for async streaming")


class DistLLMClientSync(_BaseClient):
    """Synchronous client for the Distributed LLM API.

    Usage:
        with DistLLMClientSync() as client:
            response = client.chat_completions(
                messages=[{"role": "user", "content": "Hello"}]
            )
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: float = 120.0,
    ):
        super().__init__(base_url, api_key, timeout)
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=self._build_headers(),
            timeout=timeout,
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def chat_completions(
        self,
        messages: List[Dict[str, str]],
        model: str = "distributed-llm",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 256,
        stream: bool = False,
        response_format: Optional[dict] = None,
        adapter: Optional[str] = None,
    ) -> ChatCompletionResponse:
        """Generate a chat completion."""
        payload = self._build_chat_payload(
            messages, model, temperature, top_p, max_tokens, stream, response_format, adapter,
        )
        data = self._request("POST", "/v1/chat/completions", json=payload)
        return ChatCompletionResponse(**data)

    def completions(
        self,
        prompt: str,
        model: str = "distributed-llm",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 256,
    ) -> CompletionResponse:
        """Generate a text completion."""
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        data = self._request("POST", "/v1/completions", json=payload)
        return CompletionResponse(**data)

    def list_models(self) -> ModelList:
        """List available models."""
        data = self._request("GET", "/v1/models")
        return ModelList(**data)

    def health_check(self) -> dict:
        """Check API server health."""
        return self._request("GET", "/health")

    def _request(self, method: str, path: str, **kwargs) -> dict:
        response = self._client.request(method, path, **kwargs)
        response.raise_for_status()
        return response.json()

    def _stream_request(self, method: str, path: str, **kwargs) -> Iterator[dict]:
        raise NotImplementedError("Streaming not supported in sync client")

    async def _request_async(self, method: str, path: str, **kwargs) -> dict:
        """Async shim for base class compatibility."""
        return self._request(method, path, **kwargs)
