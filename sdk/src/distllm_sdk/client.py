import asyncio
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Iterator

import httpx

from distllm_sdk.constants import DEFAULT_HTTP_TIMEOUT, MAX_RETRIES, RETRY_DELAY
from distllm_sdk.types import (
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
    ApiError,
)
from distllm_sdk.streaming import parse_sse_stream_async, parse_sse_stream_sync
from distllm_sdk.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerError
from distllm_sdk.errors import AuthenticationError, RateLimitError, TimeoutError


def _compute_delay(attempt: int, cfg: "RetryConfig") -> float:
    delay = min(cfg.initial_delay * (cfg.exponential_base ** attempt), cfg.max_delay)
    return delay * (0.5 + random.random() * 0.5)


def _parse_usage(data: dict) -> UsageInfo | None:
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
        cost_usd=raw.get("cost_usd", 0.0),
        gpu_time_seconds=raw.get("gpu_time_seconds", 0.0),
        savings_vs_cloud_usd=raw.get("savings_vs_cloud_usd", 0.0),
        ttft_ms=raw.get("ttft_ms", 0.0),
    )


def _parse_cost_headers(headers: dict[str, str]) -> dict[str, float]:
    """Extract X-DistLLM-Cost headers from response."""
    cost = {}
    for key, header_key in [
        ("cost_usd", "x-distllm-cost"),
        ("gpu_time", "x-distllm-gpu-time"),
        ("tokens_per_second", "x-distllm-tokens-per-second"),
        ("savings_usd", "x-distllm-savings"),
        ("savings_pct", "x-distllm-savings-pct"),
        ("ttft_ms", "x-distllm-ttft"),
        ("latency_ms", "x-distllm-latency"),
    ]:
        val = headers.get(header_key)
        if val:
            try:
                cost[key] = float(val)
            except (ValueError, TypeError):
                pass
    return cost


def _map_http_error(status_code: int, body: dict, request_id: str | None = None) -> ApiError:
    msg = body.get("error", {}).get("message", "") if isinstance(body.get("error"), dict) else body.get("message", httpx.codes.get_reason_phrase(status_code))
    if status_code == 401:
        return AuthenticationError(msg or "Authentication failed", request_id=request_id)
    if status_code == 429:
        retry_after = body.get("retry_after") if isinstance(body.get("error"), dict) else None
        return RateLimitError(msg or "Rate limit exceeded", retry_after=retry_after, request_id=request_id)
    if status_code == 404:
        return ApiError(msg or "Not found", status_code=status_code, error_type="not_found", request_id=request_id)
    if status_code == 503:
        retry_after = body.get("retry_after") if isinstance(body.get("error"), dict) else None
        return ApiError(msg or "Service unavailable", status_code=status_code, error_type="service_unavailable", request_id=request_id)
    return ApiError(msg or "API error", status_code=status_code, error_type="api_error", request_id=request_id)


@dataclass
class RetryConfig:
    max_retries: int = MAX_RETRIES
    initial_delay: float = RETRY_DELAY
    max_delay: float = 60.0
    exponential_base: float = 2.0
    retryable_status_codes: tuple = (429, 500, 502, 503, 504)


@dataclass
class PoolConfig:
    max_connections: int = 100
    max_keepalive_connections: int = 20
    keepalive_expiry: float = 5.0


class _BaseClient:

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
        return self._stats

    def reset_stats(self):
        self._stats = ClientStats()

    @property
    def circuit_breaker(self) -> CircuitBreaker | None:
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

    # Shared API implementations

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

    async def _batch_get_async(self, batch_id: str | None = None) -> BatchJob | BatchList:
        if batch_id:
            data = await self._request("GET", f"/v1/batches/{batch_id}")
            return self._parse_batch(data)
        data = await self._request("GET", "/v1/batches")
        return BatchList(data=[self._parse_batch(d) for d in data.get("data", [])])

    def _batch_get_sync(self, batch_id: str | None = None) -> BatchJob | BatchList:
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
        results = [ModerationResult(flagged=r["flagged"], categories=r.get("categories", {}), category_scores=r.get("category_scores", {})) for r in data.get("results", [])]
        return ModerationResponse(id=data.get("id", ""), model=data.get("model", model), results=results)

    def _moderations_sync(self, input: str | list[str], model: str = "distributed-llm") -> ModerationResponse:
        data = self._request_sync("POST", "/v1/moderations", json={"model": model, "input": input})
        results = [ModerationResult(flagged=r["flagged"], categories=r.get("categories", {}), category_scores=r.get("category_scores", {})) for r in data.get("results", [])]
        return ModerationResponse(id=data.get("id", ""), model=data.get("model", model), results=results)

    async def _transcribe_async(self, file_path: str, model: str = "whisper-1", response_format: str = "json", language: str | None = None, temperature: float = 0.0) -> TranscriptionResponse:
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

    def _transcribe_sync(self, file_path: str, model: str = "whisper-1", response_format: str = "json", language: str | None = None, temperature: float = 0.0) -> TranscriptionResponse:
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

    async def _speech_async(self, input: str, model: str = "tts-1", voice: str = "alloy", response_format: str = "mp3", speed: float = 1.0) -> SpeechResponse:
        resp = await self._request_raw("POST", "/v1/audio/speech", json={"model": model, "input": input, "voice": voice, "response_format": response_format, "speed": speed})
        return SpeechResponse(content=resp.content, content_type=resp.headers.get("content-type", "audio/mpeg"))

    def _speech_sync(self, input: str, model: str = "tts-1", voice: str = "alloy", response_format: str = "mp3", speed: float = 1.0) -> SpeechResponse:
        resp = self._request_raw_sync("POST", "/v1/audio/speech", json={"model": model, "input": input, "voice": voice, "response_format": response_format, "speed": speed})
        return SpeechResponse(content=resp.content, content_type=resp.headers.get("content-type", "audio/mpeg"))

    async def _images_generate_async(self, prompt: str, model: str = "distributed-llm", n: int = 1, size: str = "1024x1024", response_format: str = "url", quality: str = "standard", style: str | None = None) -> ImageGenerationResponse:
        payload = {"model": model, "prompt": prompt, "n": n, "size": size, "response_format": response_format, "quality": quality}
        if style:
            payload["style"] = style
        data = await self._request("POST", "/v1/images/generations", json=payload)
        images = [ImageObject(url=img.get("url"), b64_json=img.get("b64_json"), revised_prompt=img.get("revised_prompt")) for img in data.get("data", [])]
        return ImageGenerationResponse(created=data.get("created", 0), data=images)

    def _images_generate_sync(self, prompt: str, model: str = "distributed-llm", n: int = 1, size: str = "1024x1024", response_format: str = "url", quality: str = "standard", style: str | None = None) -> ImageGenerationResponse:
        payload = {"model": model, "prompt": prompt, "n": n, "size": size, "response_format": response_format, "quality": quality}
        if style:
            payload["style"] = style
        data = self._request_sync("POST", "/v1/images/generations", json=payload)
        images = [ImageObject(url=img.get("url"), b64_json=img.get("b64_json"), revised_prompt=img.get("revised_prompt")) for img in data.get("data", [])]
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

    async def _fine_tuning_create_async(self, training_file: str, model: str = "distributed-llm", validation_file: str | None = None, hyperparameters: dict | None = None, suffix: str | None = None) -> FineTuningJob:
        payload = {"training_file": training_file, "model": model}
        if validation_file:
            payload["validation_file"] = validation_file
        if hyperparameters:
            payload["hyperparameters"] = hyperparameters
        if suffix:
            payload["suffix"] = suffix
        data = await self._request("POST", "/v1/fine_tuning/jobs", json=payload)
        return self._parse_fine_tuning(data)

    def _fine_tuning_create_sync(self, training_file: str, model: str = "distributed-llm", validation_file: str | None = None, hyperparameters: dict | None = None, suffix: str | None = None) -> FineTuningJob:
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
        return BatchJob(id=data["id"], status=data["status"], input_file_id=data["input_file_id"], created_at=data.get("created_at", 0), completed_at=data.get("completed_at"), output_file_id=data.get("output_file_id"), error_file_id=data.get("error_file_id"), request_counts=data.get("request_counts"))

    @staticmethod
    def _parse_fine_tuning(data: dict) -> FineTuningJob:
        return FineTuningJob(id=data["id"], status=data["status"], model=data.get("model", ""), training_file=data.get("training_file", ""), created_at=data.get("created_at", 0), finished_at=data.get("finished_at"), result_file=data.get("result_file"), error=data.get("error"))


class DistLLMClient(_BaseClient):

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str | None = None,
        timeout: float = DEFAULT_HTTP_TIMEOUT,
        retry: RetryConfig | None = None,
        pool: PoolConfig | None = None,
    ):
        super().__init__(base_url, api_key, timeout, retry, pool)
        limits = httpx.Limits(max_connections=self._pool.max_connections, max_keepalive_connections=self._pool.max_keepalive_connections, keepalive_expiry=self._pool.keepalive_expiry)
        self._client = httpx.AsyncClient(base_url=self.base_url, headers=self._build_headers(self._api_key), timeout=httpx.Timeout(self._timeout), limits=limits)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def close(self):
        await self._client.aclose()

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
        payload = self._build_chat_payload(messages, model, temperature, top_p, max_tokens, stream, response_format, adapter, logprobs, include_usage)
        start = time.time()
        data = await self._request("POST", "/v1/chat/completions", json=payload, timeout=timeout)
        choices = [ChatChoice(index=c.get("index", 0), message=DataChatMessage(role=c["message"]["role"], content=c["message"]["content"]) if c.get("message") else None, finish_reason=c.get("finish_reason")) for c in data.get("choices", [])]
        elapsed = time.time() - start
        resp = ChatCompletionResponse(id=data["id"], model=data.get("model", model), choices=choices, created=data.get("created", 0), usage=_parse_usage(data), generation_time=elapsed)
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
        payload = self._build_chat_payload(messages, model, temperature, top_p, max_tokens, True, response_format, adapter, logprobs, include_usage)
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

    async def completions(self, prompt: str, model: str = "distributed-llm", temperature: float = 0.7, top_p: float = 0.9, max_tokens: int = 256, timeout: float | None = None) -> CompletionResponse:
        payload = {"model": model, "prompt": prompt, "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}
        start = time.time()
        data = await self._request("POST", "/v1/completions", json=payload, timeout=timeout)
        choices = [CompletionChoice(index=c.get("index", 0), text=c.get("text", ""), finish_reason=c.get("finish_reason")) for c in data.get("choices", [])]
        elapsed = time.time() - start
        resp = CompletionResponse(id=data["id"], model=data.get("model", model), choices=choices, created=data.get("created", 0), usage=_parse_usage(data), generation_time=elapsed)
        self._record_call("completions", elapsed, resp.usage)
        return resp

    async def list_models(self) -> ModelList:
        data = await self._request("GET", "/v1/models")
        return ModelList(data=[ModelInfo(id=m["id"], owned_by=m.get("owned_by", "distributed-llm")) for m in data.get("data", [])])

    async def health_check(self) -> dict:
        return await self._request("GET", "/health")

    async def embeddings(self, input: str | list[str], model: str = "distributed-llm", encoding_format: str = "float", timeout: float | None = None) -> EmbeddingResponse:
        return await self._embeddings_async(input, model, encoding_format=encoding_format)

    async def submit_batch(self, input_file_id: str, endpoint: str = "/v1/chat/completions", metadata: dict | None = None) -> BatchJob:
        return await self._batch_submit_async(input_file_id, endpoint, metadata)

    async def get_batch(self, batch_id: str | None = None) -> BatchJob | BatchList:
        return await self._batch_get_async(batch_id)

    async def cancel_batch(self, batch_id: str) -> BatchJob:
        return await self._batch_cancel_async(batch_id)

    async def moderations(self, input: str | list[str], model: str = "distributed-llm") -> ModerationResponse:
        return await self._moderations_async(input, model)

    async def transcribe(self, file_path: str, model: str = "whisper-1", response_format: str = "json", language: str | None = None, temperature: float = 0.0) -> TranscriptionResponse:
        return await self._transcribe_async(file_path, model, response_format, language, temperature)

    async def speech(self, input: str, model: str = "tts-1", voice: str = "alloy", response_format: str = "mp3", speed: float = 1.0) -> SpeechResponse:
        return await self._speech_async(input, model, voice, response_format, speed)

    async def generate_images(self, prompt: str, model: str = "distributed-llm", n: int = 1, size: str = "1024x1024", response_format: str = "url", quality: str = "standard", style: str | None = None) -> ImageGenerationResponse:
        return await self._images_generate_async(prompt, model, n, size, response_format, quality, style)

    async def upload_file(self, file_path: str, purpose: str = "fine-tune") -> FileInfo:
        return await self._files_upload_async(file_path, purpose)

    async def list_files(self) -> list[FileInfo]:
        return await self._files_list_async()

    async def delete_file(self, file_id: str) -> bool:
        return await self._files_delete_async(file_id)

    async def create_fine_tuning(self, training_file: str, model: str = "distributed-llm", validation_file: str | None = None, hyperparameters: dict | None = None, suffix: str | None = None) -> FineTuningJob:
        return await self._fine_tuning_create_async(training_file, model, validation_file, hyperparameters, suffix)

    async def list_fine_tuning(self) -> list[FineTuningJob]:
        return await self._fine_tuning_list_async()

    async def cancel_fine_tuning(self, job_id: str) -> FineTuningJob:
        return await self._fine_tuning_cancel_async(job_id)

    def _record_call(self, endpoint: str, latency: float, usage: UsageInfo | None):
        self._stats.total_calls += 1
        self._stats.total_latency += latency
        if usage:
            self._stats.total_prompt_tokens += usage.prompt_tokens
            self._stats.total_completion_tokens += usage.completion_tokens
        self._stats.call_log.append(CallStats(endpoint=endpoint, latency=latency, prompt_tokens=usage.prompt_tokens if usage else 0, completion_tokens=usage.completion_tokens if usage else 0, status_code=200))
        if len(self._stats.call_log) > self._max_call_log_size:
            del self._stats.call_log[: len(self._stats.call_log) - self._max_call_log_size]

    async def _request(self, method: str, path: str, **kwargs) -> dict:
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
                    await asyncio.sleep(delay)
                    last_exc = e
                    continue
                if self._circuit_breaker:
                    self._circuit_breaker.record_failure()
                body = {}
                try:
                    body = e.response.json()
                except (json.JSONDecodeError, httpx.DecodingError):
                    pass
                raise _map_http_error(e.response.status_code, body) from e
            except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as e:
                if attempt < self._retry.max_retries:
                    delay = _compute_delay(attempt, self._retry)
                    await asyncio.sleep(delay)
                    last_exc = e
                    continue
                if self._circuit_breaker:
                    self._circuit_breaker.record_failure()
                if isinstance(e, httpx.TimeoutException):
                    raise TimeoutError(str(e)) from e
                raise
        if self._circuit_breaker:
            self._circuit_breaker.record_failure()
        raise last_exc  # type: ignore[misc]

    async def _request_raw(self, method: str, path: str, **kwargs) -> httpx.Response:
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
                    await asyncio.sleep(delay)
                    last_exc = e
                    continue
                if self._circuit_breaker:
                    self._circuit_breaker.record_failure()
                if isinstance(e, httpx.HTTPStatusError):
                    body = {}
                    try:
                        body = e.response.json()
                    except (json.JSONDecodeError, httpx.DecodingError):
                        pass
                    raise _map_http_error(e.response.status_code, body) from e
                if isinstance(e, httpx.TimeoutException):
                    raise TimeoutError(str(e)) from e
                raise
        if self._circuit_breaker:
            self._circuit_breaker.record_failure()
        raise last_exc  # type: ignore[misc]


class DistLLMClientSync(_BaseClient):

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str | None = None,
        timeout: float = DEFAULT_HTTP_TIMEOUT,
        retry: RetryConfig | None = None,
        pool: PoolConfig | None = None,
    ):
        super().__init__(base_url, api_key, timeout, retry, pool)
        limits = httpx.Limits(max_connections=self._pool.max_connections, max_keepalive_connections=self._pool.max_keepalive_connections, keepalive_expiry=self._pool.keepalive_expiry)
        self._client = httpx.Client(base_url=self.base_url, headers=self._build_headers(self._api_key), timeout=httpx.Timeout(self._timeout), limits=limits)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self._client.close()

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
        payload = self._build_chat_payload(messages, model, temperature, top_p, max_tokens, stream, response_format, adapter, logprobs, include_usage)
        start = time.time()
        data = self._request("POST", "/v1/chat/completions", json=payload, timeout=timeout)
        choices = [ChatChoice(index=c.get("index", 0), message=DataChatMessage(role=c["message"]["role"], content=c["message"]["content"]) if c.get("message") else None, finish_reason=c.get("finish_reason")) for c in data.get("choices", [])]
        elapsed = time.time() - start
        resp = ChatCompletionResponse(id=data["id"], model=data.get("model", model), choices=choices, created=data.get("created", 0), usage=_parse_usage(data), generation_time=elapsed)
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
        payload = self._build_chat_payload(messages, model, temperature, top_p, max_tokens, True, response_format, adapter, logprobs, include_usage)
        kw = {}
        if timeout:
            kw["timeout"] = timeout
        with self._client.stream("POST", "/v1/chat/completions", json=payload, **kw) as response:
            response.raise_for_status()
            yield from parse_sse_stream_sync(response)

    def completions(self, prompt: str, model: str = "distributed-llm", temperature: float = 0.7, top_p: float = 0.9, max_tokens: int = 256, timeout: float | None = None) -> CompletionResponse:
        payload = {"model": model, "prompt": prompt, "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}
        start = time.time()
        data = self._request("POST", "/v1/completions", json=payload, timeout=timeout)
        choices = [CompletionChoice(index=c.get("index", 0), text=c.get("text", ""), finish_reason=c.get("finish_reason")) for c in data.get("choices", [])]
        elapsed = time.time() - start
        resp = CompletionResponse(id=data["id"], model=data.get("model", model), choices=choices, created=data.get("created", 0), usage=_parse_usage(data), generation_time=elapsed)
        self._record_call("completions", elapsed, resp.usage)
        return resp

    def list_models(self) -> ModelList:
        data = self._request("GET", "/v1/models")
        return ModelList(data=[ModelInfo(id=m["id"], owned_by=m.get("owned_by", "distributed-llm")) for m in data.get("data", [])])

    def health_check(self) -> dict:
        return self._request("GET", "/health")

    def embeddings(self, input: str | list[str], model: str = "distributed-llm", encoding_format: str = "float", timeout: float | None = None) -> EmbeddingResponse:
        return self._embeddings_sync(input, model, encoding_format=encoding_format)

    def submit_batch(self, input_file_id: str, endpoint: str = "/v1/chat/completions", metadata: dict | None = None) -> BatchJob:
        return self._batch_submit_sync(input_file_id, endpoint, metadata)

    def get_batch(self, batch_id: str | None = None) -> BatchJob | BatchList:
        return self._batch_get_sync(batch_id)

    def cancel_batch(self, batch_id: str) -> BatchJob:
        return self._batch_cancel_sync(batch_id)

    def moderations(self, input: str | list[str], model: str = "distributed-llm") -> ModerationResponse:
        return self._moderations_sync(input, model)

    def transcribe(self, file_path: str, model: str = "whisper-1", response_format: str = "json", language: str | None = None, temperature: float = 0.0) -> TranscriptionResponse:
        return self._transcribe_sync(file_path, model, response_format, language, temperature)

    def speech(self, input: str, model: str = "tts-1", voice: str = "alloy", response_format: str = "mp3", speed: float = 1.0) -> SpeechResponse:
        return self._speech_sync(input, model, voice, response_format, speed)

    def generate_images(self, prompt: str, model: str = "distributed-llm", n: int = 1, size: str = "1024x1024", response_format: str = "url", quality: str = "standard", style: str | None = None) -> ImageGenerationResponse:
        return self._images_generate_sync(prompt, model, n, size, response_format, quality, style)

    def upload_file(self, file_path: str, purpose: str = "fine-tune") -> FileInfo:
        return self._files_upload_sync(file_path, purpose)

    def list_files(self) -> list[FileInfo]:
        return self._files_list_sync()

    def delete_file(self, file_id: str) -> bool:
        return self._files_delete_sync(file_id)

    def create_fine_tuning(self, training_file: str, model: str = "distributed-llm", validation_file: str | None = None, hyperparameters: dict | None = None, suffix: str | None = None) -> FineTuningJob:
        return self._fine_tuning_create_sync(training_file, model, validation_file, hyperparameters, suffix)

    def list_fine_tuning(self) -> list[FineTuningJob]:
        return self._fine_tuning_list_sync()

    def cancel_fine_tuning(self, job_id: str) -> FineTuningJob:
        return self._fine_tuning_cancel_sync(job_id)

    def _record_call(self, endpoint: str, latency: float, usage: UsageInfo | None):
        self._stats.total_calls += 1
        self._stats.total_latency += latency
        if usage:
            self._stats.total_prompt_tokens += usage.prompt_tokens
            self._stats.total_completion_tokens += usage.completion_tokens
        self._stats.call_log.append(CallStats(endpoint=endpoint, latency=latency, prompt_tokens=usage.prompt_tokens if usage else 0, completion_tokens=usage.completion_tokens if usage else 0, status_code=200))
        if len(self._stats.call_log) > self._max_call_log_size:
            del self._stats.call_log[: len(self._stats.call_log) - self._max_call_log_size]

    def _request(self, method: str, path: str, **kwargs) -> dict:
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
                body = {}
                try:
                    body = e.response.json()
                except (json.JSONDecodeError, httpx.DecodingError):
                    pass
                raise _map_http_error(e.response.status_code, body) from e
            except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as e:
                if attempt < self._retry.max_retries:
                    delay = _compute_delay(attempt, self._retry)
                    time.sleep(delay)
                    last_exc = e
                    continue
                if self._circuit_breaker:
                    self._circuit_breaker.record_failure()
                if isinstance(e, httpx.TimeoutException):
                    raise TimeoutError(str(e)) from e
                raise
        if self._circuit_breaker:
            self._circuit_breaker.record_failure()
        raise last_exc  # type: ignore[misc]

    def _request_raw(self, method: str, path: str, **kwargs) -> httpx.Response:
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
                if isinstance(e, httpx.HTTPStatusError):
                    body = {}
                    try:
                        body = e.response.json()
                    except (json.JSONDecodeError, httpx.DecodingError):
                        pass
                    raise _map_http_error(e.response.status_code, body) from e
                if isinstance(e, httpx.TimeoutException):
                    raise TimeoutError(str(e)) from e
                raise
        if self._circuit_breaker:
            self._circuit_breaker.record_failure()
        raise last_exc  # type: ignore[misc]
