"""Backend abstractions for the model-as-a-service gateway.

Each backend type (native, vLLM, TGI, Ollama) wraps an HTTP client
that communicates with the backend's OpenAI-compatible API.
"""

import time
from abc import ABC, abstractmethod
from typing import Optional

import httpx
from loguru import logger

from distllm.gateway.models import BackendConfig, BackendType


class ModelBackend(ABC):
    """Abstract base for a model serving backend."""

    def __init__(self, config: BackendConfig):
        self.config = config
        self.name = config.name
        self.base_url = config.base_url.rstrip("/")
        self._active_requests = 0
        self._healthy = True
        self._last_health_check = 0.0
        self._latency_ms = 0.0
        self._error = ""
        self._models_available: list[str] = []

    @abstractmethod
    async def chat_completion(self, body: dict, headers: dict | None = None) -> dict:
        """Send a chat completion request."""

    @abstractmethod
    async def chat_completion_stream(self, body: dict, headers: dict | None = None):
        """Stream a chat completion response."""

    @abstractmethod
    async def list_models(self) -> list[str]:
        """Return list of available model names."""

    @abstractmethod
    async def health(self) -> tuple[bool, float, str]:
        """Check backend health. Returns (healthy, latency_ms, error)."""

    @property
    def active_requests(self) -> int:
        return self._active_requests

    @property
    def healthy(self) -> bool:
        return self._healthy

    @property
    def models_available(self) -> list[str]:
        return self._models_available

    def _build_headers(self, extra: dict | None = None) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        if extra:
            headers.update(extra)
        return headers


class NativeBackend(ModelBackend):
    """Native DistLLM backend via OpenAI-compatible HTTP."""

    def __init__(self, config: BackendConfig, coordinator=None):
        super().__init__(config)
        self._coordinator = coordinator

    async def chat_completion(self, body: dict, headers: dict | None = None) -> dict:
        self._active_requests += 1
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout_s) as client:
                resp = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=body,
                    headers=self._build_headers(headers),
                )
                return resp.json()
        finally:
            self._active_requests -= 1

    async def chat_completion_stream(self, body: dict, headers: dict | None = None):
        self._active_requests += 1
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout_s) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/v1/chat/completions",
                    json=body,
                    headers=self._build_headers(headers),
                ) as resp:
                    async for chunk in resp.aiter_text():
                        yield chunk
        finally:
            self._active_requests -= 1

    async def list_models(self) -> list[str]:
        if self._coordinator:
            return self._coordinator.list_models()
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{self.base_url}/v1/models",
                    headers=self._build_headers(),
                )
                data = resp.json()
                return [m["id"] for m in data.get("data", [])]
        except Exception as e:
            logger.debug(f"Native list_models failed: {e}")
            return []

    async def health(self) -> tuple[bool, float, str]:
        start = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{self.base_url}/health",
                    headers=self._build_headers(),
                )
                elapsed = (time.monotonic() - start) * 1000
                if resp.status_code == 200:
                    self._latency_ms = elapsed
                    self._healthy = True
                    self._error = ""
                    return True, elapsed, ""
                self._healthy = False
                self._error = f"HTTP {resp.status_code}"
                return False, elapsed, self._error
        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            self._healthy = False
            self._error = str(e)
            return False, elapsed, str(e)


class VLLMBackend(ModelBackend):
    """vLLM backend via OpenAI-compatible API."""

    async def chat_completion(self, body: dict, headers: dict | None = None) -> dict:
        self._active_requests += 1
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout_s) as client:
                resp = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=body,
                    headers=self._build_headers(headers),
                )
                return resp.json()
        finally:
            self._active_requests -= 1

    async def chat_completion_stream(self, body: dict, headers: dict | None = None):
        self._active_requests += 1
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout_s) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/v1/chat/completions",
                    json=body,
                    headers=self._build_headers(headers),
                ) as resp:
                    async for chunk in resp.aiter_text():
                        yield chunk
        finally:
            self._active_requests -= 1

    async def list_models(self) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{self.base_url}/v1/models",
                    headers=self._build_headers(),
                )
                data = resp.json()
                return [m["id"] for m in data.get("data", [])]
        except Exception as e:
            logger.debug(f"vLLM list_models failed: {e}")
            return []

    async def health(self) -> tuple[bool, float, str]:
        start = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{self.base_url}/health",
                    headers=self._build_headers(),
                )
                elapsed = (time.monotonic() - start) * 1000
                ok = resp.status_code == 200
                self._healthy = ok
                self._latency_ms = elapsed
                self._error = "" if ok else f"HTTP {resp.status_code}"
                return ok, elapsed, self._error
        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            self._healthy = False
            self._error = str(e)
            return False, elapsed, str(e)


class TGIBackend(ModelBackend):
    """HuggingFace TGI backend via its API."""

    async def chat_completion(self, body: dict, headers: dict | None = None) -> dict:
        self._active_requests += 1
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout_s) as client:
                resp = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=body,
                    headers=self._build_headers(headers),
                )
                return resp.json()
        finally:
            self._active_requests -= 1

    async def chat_completion_stream(self, body: dict, headers: dict | None = None):
        self._active_requests += 1
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout_s) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/v1/chat/completions",
                    json=body,
                    headers=self._build_headers(headers),
                ) as resp:
                    async for chunk in resp.aiter_text():
                        yield chunk
        finally:
            self._active_requests -= 1

    async def list_models(self) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{self.base_url}/v1/models",
                    headers=self._build_headers(),
                )
                data = resp.json()
                return [m["id"] for m in data.get("data", [])]
        except Exception as e:
            logger.debug(f"TGI list_models failed: {e}")
            return []

    async def health(self) -> tuple[bool, float, str]:
        start = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{self.base_url}/health",
                    headers=self._build_headers(),
                )
                elapsed = (time.monotonic() - start) * 1000
                ok = resp.status_code == 200
                self._healthy = ok
                self._latency_ms = elapsed
                self._error = "" if ok else f"HTTP {resp.status_code}"
                return ok, elapsed, self._error
        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            self._healthy = False
            self._error = str(e)
            return False, elapsed, str(e)


class OllamaBackend(ModelBackend):
    """Ollama backend via its API."""

    async def chat_completion(self, body: dict, headers: dict | None = None) -> dict:
        self._active_requests += 1
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout_s) as client:
                ollama_body = {
                    "model": body.get("model", ""),
                    "messages": body.get("messages", []),
                    "stream": body.get("stream", False),
                    "options": {
                        "temperature": body.get("temperature", 0.7),
                        "top_p": body.get("top_p", 0.9),
                        "num_predict": body.get("max_tokens", 256),
                    },
                }
                resp = await client.post(
                    f"{self.base_url}/api/chat",
                    json=ollama_body,
                    headers=self._build_headers(headers),
                )
                raw = resp.json()
                return {
                    "id": raw.get("created_at", ""),
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": body.get("model", ""),
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": raw.get("message", {}).get("content", ""),
                        },
                        "finish_reason": raw.get("done_reason", "stop"),
                    }],
                    "usage": {
                        "prompt_tokens": raw.get("prompt_eval_count", 0),
                        "completion_tokens": raw.get("eval_count", 0),
                        "total_tokens": raw.get("prompt_eval_count", 0) + raw.get("eval_count", 0),
                    },
                }
        finally:
            self._active_requests -= 1

    async def chat_completion_stream(self, body: dict, headers: dict | None = None):
        self._active_requests += 1
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout_s) as client:
                ollama_body = {
                    "model": body.get("model", ""),
                    "messages": body.get("messages", []),
                    "stream": True,
                    "options": {
                        "temperature": body.get("temperature", 0.7),
                        "num_predict": body.get("max_tokens", 256),
                    },
                }
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/chat",
                    json=ollama_body,
                    headers=self._build_headers(headers),
                ) as resp:
                    async for line in resp.aiter_lines():
                        if line.strip():
                            yield f"data: {line}\n\n"
        finally:
            self._active_requests -= 1

    async def list_models(self) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{self.base_url}/api/tags",
                    headers=self._build_headers(),
                )
                data = resp.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.debug(f"Ollama list_models failed: {e}")
            return []

    async def health(self) -> tuple[bool, float, str]:
        start = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{self.base_url}/",
                    headers=self._build_headers(),
                )
                elapsed = (time.monotonic() - start) * 1000
                ok = resp.status_code == 200
                self._healthy = ok
                self._latency_ms = elapsed
                self._error = "" if ok else f"HTTP {resp.status_code}"
                return ok, elapsed, self._error
        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            self._healthy = False
            self._error = str(e)
            return False, elapsed, str(e)


def create_backend(config: BackendConfig, coordinator=None) -> ModelBackend:
    """Factory function to create a backend from config."""
    if config.backend_type == BackendType.NATIVE:
        return NativeBackend(config, coordinator=coordinator)
    elif config.backend_type == BackendType.VLLM:
        return VLLMBackend(config)
    elif config.backend_type == BackendType.TGI:
        return TGIBackend(config)
    elif config.backend_type == BackendType.OLLAMA:
        return OllamaBackend(config)
    else:
        raise ValueError(f"Unknown backend type: {config.backend_type}")
