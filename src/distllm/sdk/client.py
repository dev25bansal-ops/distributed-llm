"""DistLLM SDK client for Distributed LLM API.

Fully rewritten with:
- Native sync and async clients (no wrapping)
- Embeddings, batch, audio, image, moderation, files, fine-tuning methods
- Sync streaming via yield from httpx.stream()
- Automatic retry with configurable exponential backoff
- Connection pool config (max connections, keepalive)
- Per-call timeout override
- Typed dataclass responses
- Usage tracking (tokens/sec, cost estimation)
"""

import time
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, AsyncIterator

import httpx

from distllm.constants import DEFAULT_HTTP_TIMEOUT, MAX_RETRIES, RETRY_DELAY
from distllm.sdk.types import (
    ChatCompletionResponse,
    ChatMessage as DataChatMessage,
    ChatChoice,
    CompletionResponse,
    CompletionChoice,
    ModelList,
    ModelInfo,
    EmbeddingResponse,
    EmbeddingObject,
    BatchJob,
    BatchList,
    TranscriptionResponse,
    SpeechResponse,
    ImageGenerationResponse,
    ImageObject,
    ModerationResponse,
    ModerationResult,
    FileInfo,
    FineTuningJob,
    UsageInfo,
    ClientStats,
    CallStats,
)
from distllm.sdk.streaming import parse_sse_stream_async, parse_sse_stream_sync
from distllm.sdk.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerError


@dataclass
class RetryConfig:
    """Configuration for automatic retries."""
    max_retries: int = MAX_RETRIES
    initial_delay: float = RETRY_DELAY
    max_delay: float = 60.0
    exponential_base: float = 2.0
    retryable_status_codes: tuple = (429, 500, 502, 503, 504)


@dataclass
class PoolConfig:
    """Connection pool configuration."""
    max_connections: int = 100
    max_keepalive_connections: int = 20
    keepalive_expiry: float = 5.0


def _compute_delay(attempt: int, cfg: RetryConfig) -> float:
    """Compute exponential backoff delay with jitter."""
    import random
    delay = min(cfg.initial_delay * (cfg.exponential_base ** attempt), cfg.max_delay)
    return delay * (0.5 + random.random() * 0.5)  # noqa: S311 - retry jitter


def _parse_usage(data: dict) -> UsageInfo | None:
    """Parse usage dict from API response into UsageInfo."""
    raw = data.get("usage")
    if not raw:
        return None
    gen_time = data.get("generation_time")
    completion_tokens = raw.get("completion_tokens", raw.get("total_tokens", 0))
    tps = (completion_tokens / gen_time) if gen_time and gen_time > 0 else 0.0
    return UsageInfo(
        prompt_tokens=raw.get("prompt_tokens", 0),
        completion_tokens=completion_tokens,
        total_tokens=raw.get("total_tokens", 0),
        tokens_per_second=tps,
    )


class _BaseClient:
    """Shared implementation of the Distributed LLM API client."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str | None = None,
        timeout: float = DEFAULT_HTTP_TIMEOUT,
        retry: RetryConfig | None = None,
        pool: PoolConfig | None = None,
        circuit_breaker: CircuitBreakerConfig | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._retry = retry or RetryConfig()
        self._pool = pool or PoolConfig()
        self._stats = ClientStats()
        self._max_call_log_size = int(os.environ.get("DISTLLM_SDK_MAX_CALL_LOG", "1000"))
        self._circuit_breaker = CircuitBreaker(circuit_breaker) if circuit_breaker is not None else None

    @property
    def stats(self) -> ClientStats:
        """Return aggregate usage statistics."""
        return self._stats

    def reset_stats(self):
        """Reset usage statistics."""
        self._stats = ClientStats()

    @property
    def circuit_breaker(self) -> CircuitBreaker | None:
        """Return the circuit breaker instance, if configured."""
        return self._circuit_breaker

    def _build_headers(self, api_key: str | None = None) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = api_key or self._api_key
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    @staticmethod
    def _build_chat_payload(
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        stream: bool,
        response_format: dict | None = None,
        adapter: str | None = None,
        logprobs: dict | None = None,
        include_usage: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
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
        if logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = logprobs.get("top_n", 1)
        if include_usage:
            payload["stream_options"] = {"include_usage": True}
        return payload

    # ---- Public API methods (common to sync and async) ----

    async def _embeddings_async(self, input: str | list[str], model: str = "distributed-llm", **kwargs) -> EmbeddingResponse:
        payload = {"model": model, "input": input, **kwargs}
        data = await self._request("POST", "/v1/embeddings", json=payload)
        objects = [EmbeddingObject(index=e["index"], embedding=e["embedding"]) for e in data.get("data", [])]
        return EmbeddingResponse(model=data.get("model", model), data=objects, usage=_parse_usage(data))

    def _embeddings_sync(self, input: str | list[str], model: str = "distributed-llm", **kwargs) -> EmbeddingResponse:
        payload = {"model": model, "input": input, **kwargs}
        data = self._request_sync("POST", "/v1/embeddings", json=payload)
        objects = [EmbeddingObject(index=e["index"], embedding=e["embedding"]) for e in data.get("data", [])]
        return EmbeddingResponse(model=data.get("model", model), data=objects, usage=_parse_usage(data))

    async def _batch_submit_async(self, input_file_id: str, endpoint: str, metadata: dict | None = None) -> BatchJob:
        payload = {"input_file_id": input_file_id, "endpoint": endpoint}
        if metadata:
            payload["metadata"] = metadata
        data = await self._request("POST", "/v1/batches", json=payload)
        return self._parse_batch(data)

    def _batch_submit_sync(self, input_file_id: str, endpoint: str, metadata: dict | None = None) -> BatchJob:
        payload = {"input_file_id": input_file_id, "endpoint": endpoint}
        if metadata:
            payload["metadata"] = metadata
        data = self._request_sync("POST", "/v1/batches", json=payload)
        return self._parse_batch(data)

    async def _batch_get_async(
        self,
        batch_id: str | None = None,
    ) -> BatchJob | BatchList:
        if batch_id:
            data = await self._request("GET", f"/v1/batches/{batch_id}")
            return self._parse_batch(data)
        data = await self._request("GET", "/v1/batches")
        return BatchList(data=[self._parse_batch(d) for d in data.get("data", [])])

    def _batch_get_sync(
        self,
        batch_id: str | None = None,
    ) -> BatchJob | BatchList:
        if batch_id:
            data = self._request_sync("GET", f"/v1/batches/{batch_id}")
            return self._parse_batch(data)
        data = self._request_sync("GET", "/v1/batches")
        return BatchList(data=[self._parse_batch(d) for d in data.get("data", [])])

    async def _batch_cancel_async(self, batch_id: str) -> BatchJob:
        data = await self._request("POST", f"/v1/batches/{batch_id}/cancel")
        return self._parse_batch(data)

    def _batch_cancel_sync(self, batch_id: str) -> BatchJob:
        data = self._request_sync("POST", f"/v1/batches/{batch_id}/cancel")
        return self._parse_batch(data)

    async def _moderations_async(self, input: str | list[str], model: str = "distributed-llm") -> ModerationResponse:
        data = await self._request("POST", "/v1/moderations", json={"model": model, "input": input})
        results = [
            ModerationResult(
                flagged=r["flagged"],
                categories=r.get("categories", {}),
                category_scores=r.get("category_scores", {}),
            )
            for r in data.get("results", [])
        ]
        return ModerationResponse(id=data.get("id", ""), model=data.get("model", model), results=results)

    def _moderations_sync(self, input: str | list[str], model: str = "distributed-llm") -> ModerationResponse:
        data = self._request_sync("POST", "/v1/moderations", json={"model": model, "input": input})
        results = [
            ModerationResult(
                flagged=r["flagged"],
                categories=r.get("categories", {}),
                category_scores=r.get("category_scores", {}),
            )
            for r in data.get("results", [])
        ]
        return ModerationResponse(id=data.get("id", ""), model=data.get("model", model), results=results)

    async def _transcribe_async(
        self,
        file_path: str,
        model: str = "whisper-1",
        response_format: str = "json",
        language: str | None = None,
        temperature: float = 0.0,
    ) -> TranscriptionResponse:
        path = Path(file_path)
        with open(path, "rb") as f:
            files = {"file": (path.name, f, "audio/mpeg")}
            data_field = {"model": model, "response_format": response_format}
            if language:
                data_field["language"] = language
            data_field["temperature"] = temperature
            data = await self._request("POST", "/v1/audio/transcriptions", data=data_field, files=files)
        if response_format == "json":
            return TranscriptionResponse(text=data.get("text", ""))
        return TranscriptionResponse(text=data.get("text", str(data)))

    def _transcribe_sync(
        self,
        file_path: str,
        model: str = "whisper-1",
        response_format: str = "json",
        language: str | None = None,
        temperature: float = 0.0,
    ) -> TranscriptionResponse:
        path = Path(file_path)
        with open(path, "rb") as f:
            files = {"file": (path.name, f, "audio/mpeg")}
            data_field = {"model": model, "response_format": response_format}
            if language:
                data_field["language"] = language
            data_field["temperature"] = temperature
            data = self._request_sync("POST", "/v1/audio/transcriptions", data=data_field, files=files)
        if response_format == "json":
            return TranscriptionResponse(text=data.get("text", ""))
        return TranscriptionResponse(text=data.get("text", str(data)))

    async def _speech_async(
        self,
        input: str,
        model: str = "tts-1",
        voice: str = "alloy",
        response_format: str = "mp3",
        speed: float = 1.0,
    ) -> SpeechResponse:
        resp = await self._request_raw(
            "POST", "/v1/audio/speech",
            json={"model": model, "input": input, "voice": voice, "response_format": response_format, "speed": speed},
        )
        return SpeechResponse(content=resp.content, content_type=resp.headers.get("content-type", "audio/mpeg"))

    def _speech_sync(
        self,
        input: str,
        model: str = "tts-1",
        voice: str = "alloy",
        response_format: str = "mp3",
        speed: float = 1.0,
    ) -> SpeechResponse:
        resp = self._request_raw_sync(
            "POST", "/v1/audio/speech",
            json={"model": model, "input": input, "voice": voice, "response_format": response_format, "speed": speed},
        )
        return SpeechResponse(content=resp.content, content_type=resp.headers.get("content-type", "audio/mpeg"))

    async def _images_generate_async(
        self,
        prompt: str,
        model: str = "distributed-llm",
        n: int = 1,
        size: str = "1024x1024",
        response_format: str = "url",
        quality: str = "standard",
        style: str | None = None,
    ) -> ImageGenerationResponse:
        payload = {"model": model, "prompt": prompt, "n": n, "size": size, "response_format": response_format, "quality": quality}
        if style:
            payload["style"] = style
        data = await self._request("POST", "/v1/images/generations", json=payload)
        images = [
            ImageObject(url=img.get("url"), b64_json=img.get("b64_json"), revised_prompt=img.get("revised_prompt"))
            for img in data.get("data", [])
        ]
        return ImageGenerationResponse(created=data.get("created", 0), data=images)

    def _images_generate_sync(
        self,
        prompt: str,
        model: str = "distributed-llm",
        n: int = 1,
        size: str = "1024x1024",
        response_format: str = "url",
        quality: str = "standard",
        style: str | None = None,
    ) -> ImageGenerationResponse:
        payload = {"model": model, "prompt": prompt, "n": n, "size": size, "response_format": response_format, "quality": quality}
        if style:
            payload["style"] = style
        data = self._request_sync("POST", "/v1/images/generations", json=payload)
        images = [
            ImageObject(url=img.get("url"), b64_json=img.get("b64_json"), revised_prompt=img.get("revised_prompt"))
            for img in data.get("data", [])
        ]
        return ImageGenerationResponse(created=data.get("created", 0), data=images)

    async def _files_upload_async(self, file_path: str, purpose: str = "fine-tune") -> FileInfo:
        path = Path(file_path)
        with open(path, "rb") as f:
            files = {"file": (path.name, f)}
            data = await self._request("POST", "/v1/files", data={"purpose": purpose}, files=files)
        return FileInfo(id=data["id"], filename=data["filename"], purpose=data["purpose"], bytes=data["bytes"], created_at=data["created_at"])

    def _files_upload_sync(self, file_path: str, purpose: str = "fine-tune") -> FileInfo:
        path = Path(file_path)
        with open(path, "rb") as f:
            files = {"file": (path.name, f)}
            data = self._request_sync("POST", "/v1/files", data={"purpose": purpose}, files=files)
        return FileInfo(id=data["id"], filename=data["filename"], purpose=data["purpose"], bytes=data["bytes"], created_at=data["created_at"])

    async def _files_list_async(self) -> list[FileInfo]:
        data = await self._request("GET", "/v1/files")
        return [FileInfo(id=f["id"], filename=f["filename"], purpose=f["purpose"], bytes=f["bytes"], created_at=f["created_at"]) for f in data.get("data", [])]

    def _files_list_sync(self) -> list[FileInfo]:
        data = self._request_sync("GET", "/v1/files")
        return [FileInfo(id=f["id"], filename=f["filename"], purpose=f["purpose"], bytes=f["bytes"], created_at=f["created_at"]) for f in data.get("data", [])]

    async def _files_delete_async(self, file_id: str) -> bool:
        data = await self._request("DELETE", f"/v1/files/{file_id}")
        return data.get("deleted", False)

    def _files_delete_sync(self, file_id: str) -> bool:
        data = self._request_sync("DELETE", f"/v1/files/{file_id}")
        return data.get("deleted", False)

    async def _fine_tuning_create_async(
        self,
        training_file: str,
        model: str = "distributed-llm",
        validation_file: str | None = None,
        hyperparameters: dict | None = None,
        suffix: str | None = None,
    ) -> FineTuningJob:
        payload = {"training_file": training_file, "model": model}
        if validation_file:
            payload["validation_file"] = validation_file
        if hyperparameters:
            payload["hyperparameters"] = hyperparameters
        if suffix:
            payload["suffix"] = suffix
        data = await self._request("POST", "/v1/fine_tuning/jobs", json=payload)
        return self._parse_fine_tuning(data)

    def _fine_tuning_create_sync(
        self,
        training_file: str,
        model: str = "distributed-llm",
        validation_file: str | None = None,
        hyperparameters: dict | None = None,
        suffix: str | None = None,
    ) -> FineTuningJob:
        payload = {"training_file": training_file, "model": model}
        if validation_file:
            payload["validation_file"] = validation_file
        if hyperparameters:
            payload["hyperparameters"] = hyperparameters
        if suffix:
            payload["suffix"] = suffix
        data = self._request_sync("POST", "/v1/fine_tuning/jobs", json=payload)
        return self._parse_fine_tuning(data)

    async def _fine_tuning_list_async(self) -> list[FineTuningJob]:
        data = await self._request("GET", "/v1/fine_tuning/jobs")
        return [self._parse_fine_tuning(j) for j in data.get("data", [])]

    def _fine_tuning_list_sync(self) -> list[FineTuningJob]:
        data = self._request_sync("GET", "/v1/fine_tuning/jobs")
        return [self._parse_fine_tuning(j) for j in data.get("data", [])]

    async def _fine_tuning_cancel_async(self, job_id: str) -> FineTuningJob:
        data = await self._request("POST", f"/v1/fine_tuning/jobs/{job_id}/cancel")
        return self._parse_fine_tuning(data)

    def _fine_tuning_cancel_sync(self, job_id: str) -> FineTuningJob:
        data = self._request_sync("POST", f"/v1/fine_tuning/jobs/{job_id}/cancel")
        return self._parse_fine_tuning(data)

    @staticmethod
    def _parse_batch(data: dict) -> BatchJob:
        return BatchJob(
            id=data["id"],
            status=data["status"],
            input_file_id=data["input_file_id"],
            created_at=data.get("created_at", 0),
            completed_at=data.get("completed_at"),
            output_file_id=data.get("output_file_id"),
            error_file_id=data.get("error_file_id"),
            request_counts=data.get("request_counts"),
        )

    @staticmethod
    def _parse_fine_tuning(data: dict) -> FineTuningJob:
        return FineTuningJob(
            id=data["id"],
            status=data["status"],
            model=data.get("model", ""),
            training_file=data.get("training_file", ""),
            created_at=data.get("created_at", 0),
            finished_at=data.get("finished_at"),
            result_file=data.get("result_file"),
            error=data.get("error"),
        )


class DistLLMClient(_BaseClient):
    """Async client for the Distributed LLM API.

    Usage:
        async with DistLLMClient() as client:
            response = await client.chat_completions(
                messages=[{"role": "user", "content": "Hello"}]
            )

    Features:
        - Automatic retry with exponential backoff
        - Connection pool config
        - Per-call timeout override
        - Typed dataclass responses
        - Usage tracking
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str | None = None,
        timeout: float = DEFAULT_HTTP_TIMEOUT,
        retry: RetryConfig | None = None,
        pool: PoolConfig | None = None,
    ):
        super().__init__(base_url, api_key, timeout, retry, pool)
        limits = httpx.Limits(
            max_connections=self._pool.max_connections,
            max_keepalive_connections=self._pool.max_keepalive_connections,
            keepalive_expiry=self._pool.keepalive_expiry,
        )
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=self._build_headers(self._api_key),
            timeout=httpx.Timeout(self._timeout),
            limits=limits,
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def close(self):
        await self._client.aclose()

    # ---- Chat ----

    async def chat_completions(
        self,
        messages: list[dict[str, str]],
        model: str = "distributed-llm",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 256,
        stream: bool = False,
        response_format: dict | None = None,
        adapter: str | None = None,
        logprobs: dict | None = None,
        include_usage: bool = False,
        timeout: float | None = None,
    ) -> ChatCompletionResponse:
        """Generate a chat completion."""
        payload = self._build_chat_payload(
            messages, model, temperature, top_p, max_tokens, stream, response_format, adapter, logprobs, include_usage,
        )
        start = time.time()
        data = await self._request("POST", "/v1/chat/completions", json=payload, timeout=timeout)
        choices = [
            ChatChoice(
                index=c.get("index", 0),
                message=DataChatMessage(role=c["message"]["role"], content=c["message"]["content"]) if c.get("message") else None,
                finish_reason=c.get("finish_reason"),
            )
            for c in data.get("choices", [])
        ]
        elapsed = time.time() - start
        resp = ChatCompletionResponse(
            id=data["id"], model=data.get("model", model), choices=choices,
            created=data.get("created", 0), usage=_parse_usage(data), generation_time=elapsed,
        )
        self._record_call("chat_completions", elapsed, resp.usage)
        return resp

    async def chat_completions_stream(
        self,
        messages: list[dict[str, str]],
        model: str = "distributed-llm",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 256,
        response_format: dict | None = None,
        adapter: str | None = None,
        logprobs: dict | None = None,
        include_usage: bool = False,
        timeout: float | None = None,
    ) -> AsyncIterator[str]:
        """Stream chat completions as an async generator."""
        payload = self._build_chat_payload(
            messages, model, temperature, top_p, max_tokens, True, response_format, adapter, logprobs, include_usage,
        )
        kw = {}
        if timeout:
            kw["timeout"] = timeout
        async with self._client.stream("POST", "/v1/chat/completions", json=payload, **kw) as response:
            response.raise_for_status()
            async for event in parse_sse_stream_async(response):
                if "choices" in event and event["choices"]:
                    delta = event["choices"][0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield content

    # ---- Completions ----

    async def completions(
        self,
        prompt: str,
        model: str = "distributed-llm",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 256,
        timeout: float | None = None,
    ) -> CompletionResponse:
        """Generate a text completion."""
        payload = {
            "model": model, "prompt": prompt,
            "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens,
        }
        start = time.time()
        data = await self._request("POST", "/v1/completions", json=payload, timeout=timeout)
        choices = [CompletionChoice(index=c.get("index", 0), text=c.get("text", ""), finish_reason=c.get("finish_reason")) for c in data.get("choices", [])]
        elapsed = time.time() - start
        resp = CompletionResponse(
            id=data["id"], model=data.get("model", model), choices=choices,
            created=data.get("created", 0), usage=_parse_usage(data), generation_time=elapsed,
        )
        self._record_call("completions", elapsed, resp.usage)
        return resp

    # ---- Models ----

    async def list_models(self) -> ModelList:
        """List available models."""
        data = await self._request("GET", "/v1/models")
        return ModelList(data=[ModelInfo(id=m["id"], owned_by=m.get("owned_by", "distributed-llm")) for m in data.get("data", [])])

    async def health_check(self) -> dict:
        """Check API server health."""
        return await self._request("GET", "/health")

    # ---- Embeddings ----

    async def embeddings(
        self,
        input: str | list[str],
        model: str = "distributed-llm",
        encoding_format: str = "float",
        timeout: float | None = None,
    ) -> EmbeddingResponse:
        """Create embeddings for input text."""
        return await self._embeddings_async(input, model, encoding_format=encoding_format)

    # ---- Batch ----

    async def submit_batch(
        self,
        input_file_id: str,
        endpoint: str = "/v1/chat/completions",
        metadata: dict | None = None,
    ) -> BatchJob:
        """Submit a batch job."""
        return await self._batch_submit_async(input_file_id, endpoint, metadata)

    async def get_batch(self, batch_id: str | None = None) -> BatchJob | BatchList:
        """Get a batch job by ID, or list all batches."""
        return await self._batch_get_async(batch_id)

    async def cancel_batch(self, batch_id: str) -> BatchJob:
        """Cancel a batch job."""
        return await self._batch_cancel_async(batch_id)

    # ---- Moderations ----

    async def moderations(
        self,
        input: str | list[str],
        model: str = "distributed-llm",
    ) -> ModerationResponse:
        """Moderate input text."""
        return await self._moderations_async(input, model)

    # ---- Audio ----

    async def transcribe(
        self,
        file_path: str,
        model: str = "whisper-1",
        response_format: str = "json",
        language: str | None = None,
        temperature: float = 0.0,
    ) -> TranscriptionResponse:
        """Transcribe audio file."""
        return await self._transcribe_async(file_path, model, response_format, language, temperature)

    async def speech(
        self,
        input: str,
        model: str = "tts-1",
        voice: str = "alloy",
        response_format: str = "mp3",
        speed: float = 1.0,
    ) -> SpeechResponse:
        """Generate speech from text."""
        return await self._speech_async(input, model, voice, response_format, speed)

    # ---- Images ----

    async def generate_images(
        self,
        prompt: str,
        model: str = "distributed-llm",
        n: int = 1,
        size: str = "1024x1024",
        response_format: str = "url",
        quality: str = "standard",
        style: str | None = None,
    ) -> ImageGenerationResponse:
        """Generate images from prompt."""
        return await self._images_generate_async(prompt, model, n, size, response_format, quality, style)

    # ---- Files ----

    async def upload_file(self, file_path: str, purpose: str = "fine-tune") -> FileInfo:
        """Upload a file."""
        return await self._files_upload_async(file_path, purpose)

    async def list_files(self) -> list[FileInfo]:
        """List uploaded files."""
        return await self._files_list_async()

    async def delete_file(self, file_id: str) -> bool:
        """Delete a file."""
        return await self._files_delete_async(file_id)

    # ---- Fine-tuning ----

    async def create_fine_tuning(
        self,
        training_file: str,
        model: str = "distributed-llm",
        validation_file: str | None = None,
        hyperparameters: dict | None = None,
        suffix: str | None = None,
    ) -> FineTuningJob:
        """Create a fine-tuning job."""
        return await self._fine_tuning_create_async(training_file, model, validation_file, hyperparameters, suffix)

    async def list_fine_tuning(self) -> list[FineTuningJob]:
        """List fine-tuning jobs."""
        return await self._fine_tuning_list_async()

    async def cancel_fine_tuning(self, job_id: str) -> FineTuningJob:
        """Cancel a fine-tuning job."""
        return await self._fine_tuning_cancel_async(job_id)

    # ---- Internal ----

    def _record_call(self, endpoint: str, latency: float, usage: UsageInfo | None):
        self._stats.total_calls += 1
        self._stats.total_latency += latency
        if usage:
            self._stats.total_prompt_tokens += usage.prompt_tokens
            self._stats.total_completion_tokens += usage.completion_tokens
        self._stats.call_log.append(CallStats(
            endpoint=endpoint, latency=latency,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            status_code=200,
        ))
        if len(self._stats.call_log) > self._max_call_log_size:
            del self._stats.call_log[: len(self._stats.call_log) - self._max_call_log_size]

    async def _request(self, method: str, path: str, **kwargs) -> dict:
        """Make an HTTP request with automatic retry and circuit breaker."""
        if self._circuit_breaker and not self._circuit_breaker.can_execute():
            raise CircuitBreakerError("Request rejected: circuit breaker is open")

        last_exc = None
        for attempt in range(self._retry.max_retries + 1):
            try:
                response = await self._client.request(method, path, **kwargs)
                response.raise_for_status()
                if self._circuit_breaker:
                    self._circuit_breaker.record_success()
                return response.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code in self._retry.retryable_status_codes and attempt < self._retry.max_retries:
                    delay = _compute_delay(attempt, self._retry)
                    await self._sleep(delay)
                    last_exc = e
                    continue
                if self._circuit_breaker:
                    self._circuit_breaker.record_failure()
                raise
            except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as e:
                if attempt < self._retry.max_retries:
                    delay = _compute_delay(attempt, self._retry)
                    await self._sleep(delay)
                    last_exc = e
                    continue
                if self._circuit_breaker:
                    self._circuit_breaker.record_failure()
                raise
        if self._circuit_breaker:
            self._circuit_breaker.record_failure()
        raise last_exc  # type: ignore[misc]  # mypy: BaseException union narrowing

    async def _request_raw(self, method: str, path: str, **kwargs) -> httpx.Response:
        """Make a raw HTTP request (for binary responses) with circuit breaker."""
        if self._circuit_breaker and not self._circuit_breaker.can_execute():
            raise CircuitBreakerError("Request rejected: circuit breaker is open")

        last_exc = None
        for attempt in range(self._retry.max_retries + 1):
            try:
                response = await self._client.request(method, path, **kwargs)
                response.raise_for_status()
                if self._circuit_breaker:
                    self._circuit_breaker.record_success()
                return response
            except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as e:
                if attempt < self._retry.max_retries:
                    delay = _compute_delay(attempt, self._retry)
                    await self._sleep(delay)
                    last_exc = e
                    continue
                if self._circuit_breaker:
                    self._circuit_breaker.record_failure()
                raise
        if self._circuit_breaker:
            self._circuit_breaker.record_failure()
        raise last_exc  # type: ignore[misc]  # mypy: BaseException union narrowing

    @staticmethod
    async def _sleep(delay: float):
        await __import__("asyncio").sleep(delay)


class DistLLMClientSync(_BaseClient):
    """Synchronous client for the Distributed LLM API.

    Usage:
        with DistLLMClientSync() as client:
            response = client.chat_completions(
                messages=[{"role": "user", "content": "Hello"}]
            )

    Features:
        - Native sync HTTP (no asyncio.run wrapper)
        - Sync streaming via yield from httpx.stream()
        - Automatic retry with exponential backoff
        - Connection pool config
        - Per-call timeout override
        - Typed dataclass responses
        - Usage tracking
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str | None = None,
        timeout: float = DEFAULT_HTTP_TIMEOUT,
        retry: RetryConfig | None = None,
        pool: PoolConfig | None = None,
    ):
        super().__init__(base_url, api_key, timeout, retry, pool)
        limits = httpx.Limits(
            max_connections=self._pool.max_connections,
            max_keepalive_connections=self._pool.max_keepalive_connections,
            keepalive_expiry=self._pool.keepalive_expiry,
        )
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=self._build_headers(self._api_key),
            timeout=httpx.Timeout(self._timeout),
            limits=limits,
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self._client.close()

    # ---- Chat ----

    def chat_completions(
        self,
        messages: list[dict[str, str]],
        model: str = "distributed-llm",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 256,
        stream: bool = False,
        response_format: dict | None = None,
        adapter: str | None = None,
        logprobs: dict | None = None,
        include_usage: bool = False,
        timeout: float | None = None,
    ) -> ChatCompletionResponse:
        """Generate a chat completion."""
        payload = self._build_chat_payload(
            messages, model, temperature, top_p, max_tokens, stream, response_format, adapter, logprobs, include_usage,
        )
        start = time.time()
        data = self._request("POST", "/v1/chat/completions", json=payload, timeout=timeout)
        choices = [
            ChatChoice(
                index=c.get("index", 0),
                message=DataChatMessage(role=c["message"]["role"], content=c["message"]["content"]) if c.get("message") else None,
                finish_reason=c.get("finish_reason"),
            )
            for c in data.get("choices", [])
        ]
        elapsed = time.time() - start
        resp = ChatCompletionResponse(
            id=data["id"], model=data.get("model", model), choices=choices,
            created=data.get("created", 0), usage=_parse_usage(data), generation_time=elapsed,
        )
        self._record_call("chat_completions", elapsed, resp.usage)
        return resp

    def chat_completions_stream(
        self,
        messages: list[dict[str, str]],
        model: str = "distributed-llm",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 256,
        response_format: dict | None = None,
        adapter: str | None = None,
        logprobs: dict | None = None,
        include_usage: bool = False,
        timeout: float | None = None,
    ) -> Iterator[str]:
        """Stream chat completions as a sync generator (yield from httpx.stream())."""
        payload = self._build_chat_payload(
            messages, model, temperature, top_p, max_tokens, True, response_format, adapter, logprobs, include_usage,
        )
        kw = {}
        if timeout:
            kw["timeout"] = timeout
        with self._client.stream("POST", "/v1/chat/completions", json=payload, **kw) as response:
            response.raise_for_status()
            yield from parse_sse_stream_sync(response)

    # ---- Completions ----

    def completions(
        self,
        prompt: str,
        model: str = "distributed-llm",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 256,
        timeout: float | None = None,
    ) -> CompletionResponse:
        """Generate a text completion."""
        payload = {
            "model": model, "prompt": prompt,
            "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens,
        }
        start = time.time()
        data = self._request("POST", "/v1/completions", json=payload, timeout=timeout)
        choices = [CompletionChoice(index=c.get("index", 0), text=c.get("text", ""), finish_reason=c.get("finish_reason")) for c in data.get("choices", [])]
        elapsed = time.time() - start
        resp = CompletionResponse(
            id=data["id"], model=data.get("model", model), choices=choices,
            created=data.get("created", 0), usage=_parse_usage(data), generation_time=elapsed,
        )
        self._record_call("completions", elapsed, resp.usage)
        return resp

    # ---- Models ----

    def list_models(self) -> ModelList:
        """List available models."""
        data = self._request("GET", "/v1/models")
        return ModelList(data=[ModelInfo(id=m["id"], owned_by=m.get("owned_by", "distributed-llm")) for m in data.get("data", [])])

    def health_check(self) -> dict:
        """Check API server health."""
        return self._request("GET", "/health")

    # ---- Embeddings ----

    def embeddings(
        self,
        input: str | list[str],
        model: str = "distributed-llm",
        encoding_format: str = "float",
        timeout: float | None = None,
    ) -> EmbeddingResponse:
        """Create embeddings for input text."""
        return self._embeddings_sync(input, model, encoding_format=encoding_format)

    # ---- Batch ----

    def submit_batch(
        self,
        input_file_id: str,
        endpoint: str = "/v1/chat/completions",
        metadata: dict | None = None,
    ) -> BatchJob:
        """Submit a batch job."""
        return self._batch_submit_sync(input_file_id, endpoint, metadata)

    def get_batch(self, batch_id: str | None = None) -> BatchJob | BatchList:
        """Get a batch job by ID, or list all batches."""
        return self._batch_get_sync(batch_id)

    def cancel_batch(self, batch_id: str) -> BatchJob:
        """Cancel a batch job."""
        return self._batch_cancel_sync(batch_id)

    # ---- Moderations ----

    def moderations(
        self,
        input: str | list[str],
        model: str = "distributed-llm",
    ) -> ModerationResponse:
        """Moderate input text."""
        return self._moderations_sync(input, model)

    # ---- Audio ----

    def transcribe(
        self,
        file_path: str,
        model: str = "whisper-1",
        response_format: str = "json",
        language: str | None = None,
        temperature: float = 0.0,
    ) -> TranscriptionResponse:
        """Transcribe audio file."""
        return self._transcribe_sync(file_path, model, response_format, language, temperature)

    def speech(
        self,
        input: str,
        model: str = "tts-1",
        voice: str = "alloy",
        response_format: str = "mp3",
        speed: float = 1.0,
    ) -> SpeechResponse:
        """Generate speech from text."""
        return self._speech_sync(input, model, voice, response_format, speed)

    # ---- Images ----

    def generate_images(
        self,
        prompt: str,
        model: str = "distributed-llm",
        n: int = 1,
        size: str = "1024x1024",
        response_format: str = "url",
        quality: str = "standard",
        style: str | None = None,
    ) -> ImageGenerationResponse:
        """Generate images from prompt."""
        return self._images_generate_sync(prompt, model, n, size, response_format, quality, style)

    # ---- Files ----

    def upload_file(self, file_path: str, purpose: str = "fine-tune") -> FileInfo:
        """Upload a file."""
        return self._files_upload_sync(file_path, purpose)

    def list_files(self) -> list[FileInfo]:
        """List uploaded files."""
        return self._files_list_sync()

    def delete_file(self, file_id: str) -> bool:
        """Delete a file."""
        return self._files_delete_sync(file_id)

    # ---- Fine-tuning ----

    def create_fine_tuning(
        self,
        training_file: str,
        model: str = "distributed-llm",
        validation_file: str | None = None,
        hyperparameters: dict | None = None,
        suffix: str | None = None,
    ) -> FineTuningJob:
        """Create a fine-tuning job."""
        return self._fine_tuning_create_sync(training_file, model, validation_file, hyperparameters, suffix)

    def list_fine_tuning(self) -> list[FineTuningJob]:
        """List fine-tuning jobs."""
        return self._fine_tuning_list_sync()

    def cancel_fine_tuning(self, job_id: str) -> FineTuningJob:
        """Cancel a fine-tuning job."""
        return self._fine_tuning_cancel_sync(job_id)

    def marketplace_ads(self) -> list[dict]:
        """List active KV cache advertisements."""
        return self._request("GET", "/v1/marketplace/ads")

    def federated_peers(self) -> list[dict]:
        """List federated peer clusters."""
        return self._request("GET", "/v1/federated/peers")

    def webrtc_status(self) -> dict:
        """Get WebRTC signaling server status."""
        return self._request("GET", "/v1/webrtc/status")

    def defrag_status(self) -> dict:
        """Get defragmentation status."""
        return self._request("GET", "/v1/defrag/status")

    # ---- Internal ----

    def _record_call(self, endpoint: str, latency: float, usage: UsageInfo | None):
        self._stats.total_calls += 1
        self._stats.total_latency += latency
        if usage:
            self._stats.total_prompt_tokens += usage.prompt_tokens
            self._stats.total_completion_tokens += usage.completion_tokens
        self._stats.call_log.append(CallStats(
            endpoint=endpoint, latency=latency,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            status_code=200,
        ))
        if len(self._stats.call_log) > self._max_call_log_size:
            del self._stats.call_log[: len(self._stats.call_log) - self._max_call_log_size]

    def _request(self, method: str, path: str, **kwargs) -> dict:
        """Make an HTTP request with automatic retry and circuit breaker."""
        if self._circuit_breaker and not self._circuit_breaker.can_execute():
            raise CircuitBreakerError("Request rejected: circuit breaker is open")

        last_exc = None
        for attempt in range(self._retry.max_retries + 1):
            try:
                response = self._client.request(method, path, **kwargs)
                response.raise_for_status()
                if self._circuit_breaker:
                    self._circuit_breaker.record_success()
                return response.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code in self._retry.retryable_status_codes and attempt < self._retry.max_retries:
                    delay = _compute_delay(attempt, self._retry)
                    time.sleep(delay)
                    last_exc = e
                    continue
                if self._circuit_breaker:
                    self._circuit_breaker.record_failure()
                raise
            except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as e:
                if attempt < self._retry.max_retries:
                    delay = _compute_delay(attempt, self._retry)
                    time.sleep(delay)
                    last_exc = e
                    continue
                if self._circuit_breaker:
                    self._circuit_breaker.record_failure()
                raise
        if self._circuit_breaker:
            self._circuit_breaker.record_failure()
        raise last_exc  # type: ignore[misc]  # mypy: BaseException union narrowing

    def _request_raw(self, method: str, path: str, **kwargs) -> httpx.Response:
        """Make a raw HTTP request (for binary responses) with circuit breaker."""
        if self._circuit_breaker and not self._circuit_breaker.can_execute():
            raise CircuitBreakerError("Request rejected: circuit breaker is open")

        last_exc = None
        for attempt in range(self._retry.max_retries + 1):
            try:
                response = self._client.request(method, path, **kwargs)
                response.raise_for_status()
                if self._circuit_breaker:
                    self._circuit_breaker.record_success()
                return response
            except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as e:
                if attempt < self._retry.max_retries:
                    delay = _compute_delay(attempt, self._retry)
                    time.sleep(delay)
                    last_exc = e
                    continue
                if self._circuit_breaker:
                    self._circuit_breaker.record_failure()
                raise
        if self._circuit_breaker:
            self._circuit_breaker.record_failure()
        raise last_exc  # type: ignore[misc]  # mypy: BaseException union narrowing

