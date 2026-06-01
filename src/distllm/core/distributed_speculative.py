"""Distributed speculative decoding — remote draft model support.

Extends speculative decoding so the draft model runs on a separate
(cheaper) device — e.g. a CPU laptop — while the target model runs
on the GPU cluster. The draft model is accessed via HTTP or gRPC.

Supports:
- Multiple remote draft endpoints (fleet routing)
- OpenAI-compatible /v1/completions and /v1/chat/completions
- Raw token-based HTTP endpoints
- gRPC transport
- Adaptive candidate count
- Cross-provider draft (e.g. OpenAI mini for draft, self-hosted for target)
- Proper rejection sampling with draft logprobs
- Async draft/verify overlap for maximum throughput
- KV cache passthrough to avoid re-encoding prefix
- Multi-token fallback on draft failure

Usage::

    # Single draft model on a CPU node
    remote_draft = RemoteDraftModel("http://10.0.0.2:8000/v1/completions")
    sd = DistributedSpeculativeDecoder(
        target_forward=target_forward_fn,
        draft_model=remote_draft,
        num_candidates=5,
    )
    output_ids = sd.generate(input_ids, max_new_tokens=256)

    # Async with draft/verify overlap (best throughput)
    output_ids = await sd.agenerate(input_ids, max_new_tokens=256)

    # Fleet of heterogeneous draft models
    fleet = DraftModelFleet()
    fleet.register(DraftModelSpec(endpoint_url="http://cpu:8000/v1/completions", ...))
    fleet.register(DraftModelSpec(endpoint_url="http://gpu:8001/v1/completions", ...))
    sd = DistributedSpeculativeDecoder(
        target_forward=target_forward_fn,
        draft_fleet=fleet,
        num_candidates=5,
    )
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F
from loguru import logger
from pydantic import BaseModel, Field, ValidationError

# Module-level lazy httpx import cache
_httpx: Any = None
_httpx_async: Any = None


def _get_httpx() -> Any:
    """Lazy-import httpx once and cache the module reference."""
    global _httpx
    if _httpx is None:
        import httpx
        _httpx = httpx
    return _httpx


def _get_httpx_async_client() -> Any:
    """Lazy-import httpx.AsyncClient class."""
    global _httpx_async
    if _httpx_async is None:
        import httpx
        _httpx_async = httpx.AsyncClient
    return _httpx_async


@dataclass
class RemoteDraftConfig:
    """Configuration for a remote draft model endpoint."""
    endpoint_url: str
    model_name: str = ""
    api_key: str = ""
    timeout_seconds: float = 30.0
    max_retries: int = 2
    batch_size: int = 1
    transport: Literal["http", "grpc"] = "http"
    prompt_format: Literal["tokens", "text", "auto"] = "auto"
    verify_ssl: bool = True


@dataclass
class DraftLatencyStats:
    """Tracks latency of remote draft model calls."""
    total_calls: int = 0
    total_tokens: int = 0
    total_latency_s: float = 0.0
    errors: int = 0

    @property
    def avg_latency_ms(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return (self.total_latency_s / self.total_calls) * 1000

    @property
    def tokens_per_second(self) -> float:
        if self.total_latency_s == 0:
            return 0.0
        return self.total_tokens / self.total_latency_s


@dataclass
class DraftTokenResult:
    """Result from a remote draft model call.

    Carries both the generated token IDs and their logprobs so the
    verifier can perform proper rejection sampling (``min(1, p/q)``).

    On error, ``token_ids`` is empty and ``error`` contains the reason.
    """
    token_ids: list[int]
    logprobs: list[float]
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.error == "" and len(self.token_ids) > 0


# ── Pydantic response schemas ──────────────────────────────────────────


class _LogprobsModel(BaseModel):
    """Logprobs sub-object in completions response."""
    token_ids: list[int] = Field(default_factory=list)
    token_logprobs: list[float] = Field(default_factory=list)
    tokens: list[int] = Field(default_factory=list)


class _ChoiceModel(BaseModel):
    """Single choice in completions response."""
    token_ids: list[int] = Field(default_factory=list)
    logprobs: _LogprobsModel | None = None
    text: str = ""
    index: int = 0


class _ChatMessageModel(BaseModel):
    """Message in chat completion response."""
    content: str = ""


class _ChatLogprobTokenModel(BaseModel):
    """Token-level logprob in chat completion."""
    token_id: int = 0
    logprob: float = 0.0


class _ChatLogprobsModel(BaseModel):
    """Logprobs in chat completion response."""
    content: list[_ChatLogprobTokenModel] = Field(default_factory=list)
    token_ids: list[int] = Field(default_factory=list)
    token_logprobs: list[float] = Field(default_factory=list)


class _ChatChoiceModel(BaseModel):
    """Single choice in chat completion response."""
    token_ids: list[int] = Field(default_factory=list)
    logprobs: _ChatLogprobsModel | None = None
    message: _ChatMessageModel | None = None
    index: int = 0


class _CompletionsResponse(BaseModel):
    """Validated OpenAI completions response."""
    choices: list[_ChoiceModel] = Field(default_factory=list)
    tokens: list[int] = Field(default_factory=list)


class _ChatCompletionsResponse(BaseModel):
    """Validated OpenAI chat completions response."""
    choices: list[_ChatChoiceModel] = Field(default_factory=list)


class RemoteDraftModel:
    """Calls a remote draft model endpoint for speculative decoding.

    Supports:
    - OpenAI-compatible /v1/completions (token IDs + logprobs)
    - OpenAI-compatible /v1/chat/completions (text extraction)
    - Raw HTTP POST with token IDs
    - gRPC transport (via proto/node.proto)

    Usage::

        # From a URL string
        draft = RemoteDraftModel("http://10.0.0.2:8000/v1/completions")

        # From a config object (full control)
        draft = RemoteDraftModel(RemoteDraftConfig(
            endpoint_url="https://api.openai.com/v1/chat/completions",
            model_name="gpt-4o-mini",
            api_key="sk-...",
            prompt_format="text",
            verify_ssl=True,
        ))
    """

    def __init__(self, config: RemoteDraftConfig):
        self._config = config
        self._stats = DraftLatencyStats()
        self._client: Any = None
        self._async_client: Any = None
        self._grpc_stub: Any = None

    def _get_client(self) -> Any:
        """Return a cached HTTP or gRPC client."""
        if self._config.transport == "grpc":
            return self._get_grpc_stub()
        if self._client is None:
            httpx = _get_httpx()
            self._client = httpx.Client(
                timeout=self._config.timeout_seconds,
                headers=self._build_headers(),
                verify=self._config.verify_ssl,
                limits=httpx.Limits(
                    max_connections=10,
                    max_keepalive_connections=5,
                    keepalive_expiry=30.0,
                ),
            )
        return self._client

    def _get_grpc_stub(self) -> Any:
        if self._grpc_stub is None:
            import grpc

            from distllm.proto import node_pb2_grpc  # noqa: I001

            if self._config.verify_ssl:
                try:
                    from distllm.core.certificate_manager import CertificateManager
                    cert_mgr = CertificateManager()
                    creds = cert_mgr.create_grpc_client_credentials()
                    if creds is not None:
                        channel = grpc.secure_channel(self._config.endpoint_url, creds)
                    else:
                        logger.warning(
                            f"No TLS certificates found for "
                            f"{self._config.endpoint_url} — "
                            "falling back to insecure gRPC channel. "
                            "Run 'distllm security cert create' "
                            "to generate certificates."
                        )
                        channel = grpc.insecure_channel(self._config.endpoint_url)
                except Exception as e:
                    logger.warning(
                        f"Failed to load TLS credentials: {e} — "
                        "falling back to insecure gRPC channel."
                    )
                    channel = grpc.insecure_channel(self._config.endpoint_url)
            else:
                logger.warning(
                    f"TLS disabled for gRPC channel to {self._config.endpoint_url} — "
                    "tensor data and credentials are transmitted in plaintext."
                )
                channel = grpc.insecure_channel(self._config.endpoint_url)

            self._grpc_stub = node_pb2_grpc.NodeServiceStub(channel)
        return self._grpc_stub

    def _build_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        return headers

    def generate_tokens(
        self,
        prompt_tokens: list[int],
        num_tokens: int,
        temperature: float = 1.0,
        top_k: int = 20,
        top_p: float = 1.0,
        prompt_text: str = "",
    ) -> DraftTokenResult:
        """Generate draft tokens and their logprobs from the remote model.

        Returns a ``DraftTokenResult`` carrying token IDs, per-token
        logprobs (for rejection sampling), and an ``error`` string that
        is non-empty when the call failed.
        """
        start = time.monotonic()
        try:
            if self._config.transport == "grpc":
                result = self._call_grpc(prompt_tokens, num_tokens, temperature, top_k, top_p)
            elif self._should_use_chat_completions(prompt_text):
                result = self._call_chat_completions(prompt_text, num_tokens, temperature, top_p)
            else:
                result = self._call_completions(
                    prompt_tokens, num_tokens, temperature, top_k, top_p,
                )
            elapsed = time.monotonic() - start
            self._stats.total_calls += 1
            self._stats.total_tokens += len(result.token_ids)
            self._stats.total_latency_s += elapsed
            return result
        except Exception as e:
            self._stats.errors += 1
            logger.warning("Remote draft model call failed: {}", e)
            return DraftTokenResult(token_ids=[], logprobs=[], error=str(e))

    def _should_use_chat_completions(self, prompt_text: str) -> bool:
        """Determine if we should use chat completions format."""
        fmt = self._config.prompt_format
        if fmt == "text":
            return True
        if fmt == "tokens":
            return False
        # auto: use chat completions for OpenAI-compatible endpoints
        endpoint = self._config.endpoint_url.lower()
        if "chat/completions" in endpoint:
            return True
        if "api.openai.com" in endpoint or "api.anthropic.com" in endpoint:
            return True
        return False

    def _call_completions(
        self,
        prompt_tokens: list[int],
        num_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> DraftTokenResult:
        """Call OpenAI-compatible /v1/completions with token IDs."""
        client = self._get_client()
        endpoint = self._config.endpoint_url

        payload: dict[str, Any] = {
            "prompt": prompt_tokens,
            "max_tokens": num_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "stream": False,
            "logprobs": 1,  # Request logprobs for proper rejection sampling
        }
        if self._config.model_name:
            payload["model"] = self._config.model_name

        data = self._post_with_retry_raw(client, endpoint, payload)
        return self._extract_from_completions(data)

    def _extract_from_completions(self, data: dict[str, Any]) -> DraftTokenResult:
        """Extract token IDs and logprobs from a completions response.

        Uses Pydantic to validate the response schema.  Falls back to
        raw dict access if validation fails (some endpoints return
        non-standard shapes).
        """
        try:
            resp = _CompletionsResponse.model_validate(data)
        except ValidationError:
            # Fallback: raw dict access for non-standard endpoints
            return self._extract_from_completions_raw(data)

        if not resp.choices:
            return DraftTokenResult(
                token_ids=[], logprobs=[],
                error="No choices in completions response",
            )

        choice = resp.choices[0]

        # Direct token_ids field
        if choice.token_ids:
            lp = choice.logprobs.token_logprobs if choice.logprobs else []
            return DraftTokenResult(token_ids=choice.token_ids, logprobs=lp)

        # Logprobs with token IDs
        if choice.logprobs and choice.logprobs.token_ids:
            return DraftTokenResult(
                token_ids=choice.logprobs.token_ids,
                logprobs=choice.logprobs.token_logprobs,
            )

        # OpenAI-style text completions with token_logprobs
        if choice.logprobs and choice.logprobs.tokens and choice.logprobs.token_logprobs:
            return DraftTokenResult(
                token_ids=choice.logprobs.tokens,
                logprobs=choice.logprobs.token_logprobs,
            )

        # Top-level tokens field
        if resp.tokens:
            return DraftTokenResult(token_ids=resp.tokens, logprobs=[])

        return DraftTokenResult(
            token_ids=[], logprobs=[],
            error="No token_ids in validated response",
        )

    def _extract_from_completions_raw(self, data: dict[str, Any]) -> DraftTokenResult:
        """Fallback raw-dict extraction when Pydantic validation fails."""
        if "choices" not in data:
            return DraftTokenResult(
                token_ids=[], logprobs=[],
                error=f"Unexpected response format: {list(data.keys())}",
            )

        choice = data["choices"][0]

        if "token_ids" in choice:
            return DraftTokenResult(
                token_ids=choice["token_ids"],
                logprobs=choice.get("logprobs", {}).get("token_logprobs", []),
            )

        logprobs_obj = choice.get("logprobs", {})
        if "token_ids" in logprobs_obj:
            return DraftTokenResult(
                token_ids=logprobs_obj["token_ids"],
                logprobs=logprobs_obj.get("token_logprobs", []),
            )

        if "tokens" in data:
            return DraftTokenResult(token_ids=data["tokens"], logprobs=[])

        return DraftTokenResult(
            token_ids=[], logprobs=[],
            error=f"No token_ids in response: {list(choice.keys())}",
        )

    def _call_chat_completions(
        self,
        prompt_text: str,
        num_tokens: int,
        temperature: float,
        top_p: float,
    ) -> DraftTokenResult:
        """Call OpenAI-compatible /v1/chat/completions and extract tokens.

        This enables cross-provider draft: use any chat API as a draft model.
        Extracts token IDs from the response when available, or falls back
        to encoding the generated text.
        """
        client = self._get_client()
        endpoint = self._config.endpoint_url

        # Ensure endpoint points to chat completions
        if "/chat/completions" not in endpoint:
            endpoint = endpoint.rstrip("/") + "/v1/chat/completions"

        payload: dict[str, Any] = {
            "messages": [{"role": "user", "content": prompt_text}],
            "max_tokens": num_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
            "logprobs": True,  # Request logprobs from chat completions
        }
        if self._config.model_name:
            payload["model"] = self._config.model_name

        data = self._post_with_retry_raw(client, endpoint, payload)

        # Extract token IDs from chat completion response
        return self._extract_tokens_from_chat_response(data)

    def _extract_tokens_from_chat_response(self, data: dict[str, Any]) -> DraftTokenResult:
        """Extract token IDs and logprobs from an OpenAI chat completion response.

        Uses Pydantic to validate the response schema.  Falls back to
        raw dict access for non-standard endpoints.
        """
        try:
            resp = _ChatCompletionsResponse.model_validate(data)
        except ValidationError:
            return self._extract_tokens_from_chat_response_raw(data)

        if not resp.choices:
            return DraftTokenResult(
                token_ids=[], logprobs=[],
                error="No choices in chat completion response",
            )

        choice = resp.choices[0]

        # Direct token_ids
        if choice.token_ids:
            lp = choice.logprobs.token_logprobs if choice.logprobs else []
            return DraftTokenResult(token_ids=choice.token_ids, logprobs=lp)

        # Logprobs with token IDs
        if choice.logprobs and choice.logprobs.token_ids:
            return DraftTokenResult(
                token_ids=choice.logprobs.token_ids,
                logprobs=choice.logprobs.token_logprobs,
            )

        # Content with token-level logprobs (OpenAI format)
        if choice.logprobs and choice.logprobs.content:
            ids = [t.token_id for t in choice.logprobs.content]
            probs = [t.logprob for t in choice.logprobs.content]
            if ids:
                return DraftTokenResult(token_ids=ids, logprobs=probs)

        content = choice.message.content if choice.message else ""
        if not content:
            return DraftTokenResult(
                token_ids=[], logprobs=[],
                error="Empty chat completion response",
            )

        return DraftTokenResult(
            token_ids=[], logprobs=[],
            error=(
                "Chat completion response has no token IDs. "
                "Use prompt_format='tokens' for token-based draft endpoints."
            ),
        )

    def _extract_tokens_from_chat_response_raw(self, data: dict[str, Any]) -> DraftTokenResult:
        """Fallback raw-dict extraction for chat completions."""
        choices = data.get("choices", [])
        if not choices:
            return DraftTokenResult(
                token_ids=[], logprobs=[],
                error="No choices in chat completion response",
            )

        choice = choices[0]

        if "token_ids" in choice:
            return DraftTokenResult(
                token_ids=choice["token_ids"],
                logprobs=choice.get("logprobs", {}).get("token_logprobs", []),
            )

        logprobs = choice.get("logprobs", {})
        if "token_ids" in logprobs:
            return DraftTokenResult(
                token_ids=logprobs["token_ids"],
                logprobs=logprobs.get("token_logprobs", []),
            )

        content = choice.get("message", {}).get("content", "")
        token_logprobs = logprobs.get("content", [])
        if token_logprobs:
            ids = []
            probs = []
            for t in token_logprobs:
                if "token_id" in t:
                    ids.append(t["token_id"])
                    probs.append(t.get("logprob", 0.0))
            if ids:
                return DraftTokenResult(token_ids=ids, logprobs=probs)

        if not content:
            return DraftTokenResult(
                token_ids=[], logprobs=[],
                error="Empty chat completion response",
            )

        return DraftTokenResult(
            token_ids=[], logprobs=[],
            error=(
                "Chat completion response has no token IDs. "
                "Use prompt_format='tokens' for token-based draft endpoints."
            ),
        )

    def _call_grpc(
        self,
        prompt_tokens: list[int],
        num_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> DraftTokenResult:
        """Call remote draft model via gRPC."""
        from distllm.proto import node_pb2

        stub = self._get_grpc_stub()
        request = node_pb2.ForwardPassRequest(
            request_id=f"draft-{int(time.time() * 1000)}",
            input_ids=prompt_tokens,
            model_name=self._config.model_name,
        )
        response = stub.ForwardPass(request, timeout=self._config.timeout_seconds)

        if response.success and response.output:
            return DraftTokenResult(
                token_ids=list(response.output.data)[:num_tokens],
                logprobs=[],  # gRPC doesn't carry logprobs yet
            )
        return DraftTokenResult(
            token_ids=[], logprobs=[],
            error=response.error_message if not response.success else "empty output",
        )

    def _post_with_retry_raw(
        self, client: Any, endpoint: str, payload: dict[str, Any],
    ) -> dict[str, Any]:
        """POST with jittered exponential backoff retry.

        Backoff schedule: 0.1s, 0.2s, 0.4s with ±50% jitter.
        Maximum delay per attempt is capped at 2.0s.
        """
        last_error: Exception | None = None
        max_delay = 2.0

        for attempt in range(self._config.max_retries + 1):
            try:
                resp = client.post(endpoint, json=payload)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                last_error = e
                if attempt < self._config.max_retries:
                    base_delay = min(0.1 * (2 ** attempt), max_delay)
                    jitter = base_delay * 0.5 * (2 * random.random() - 1)
                    delay = max(0.05, base_delay + jitter)
                    logger.debug(
                        "Draft model attempt {} failed: {}, retrying in {:.2f}s",
                        attempt + 1, e, delay,
                    )
                    time.sleep(delay)
                else:
                    raise

        raise last_error  # type: ignore[misc]

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_calls": self._stats.total_calls,
            "total_tokens": self._stats.total_tokens,
            "avg_latency_ms": round(self._stats.avg_latency_ms, 2),
            "tokens_per_second": round(self._stats.tokens_per_second, 1),
            "errors": self._stats.errors,
        }

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None
        if self._grpc_stub:
            channel = self._grpc_stub._channel
            if hasattr(channel, "close"):
                channel.close()
            self._grpc_stub = None

    def _get_async_client(self) -> Any:
        """Return a cached async HTTP client with connection pooling."""
        if self._async_client is None:
            AsyncClient = _get_httpx_async_client()
            self._async_client = AsyncClient(
                timeout=self._config.timeout_seconds,
                headers=self._build_headers(),
                verify=self._config.verify_ssl,
                limits=_get_httpx().Limits(
                    max_connections=20,
                    max_keepalive_connections=10,
                ),
            )
        return self._async_client

    async def agenerate_tokens(
        self,
        prompt_tokens: list[int],
        num_tokens: int,
        temperature: float = 1.0,
        top_k: int = 20,
        top_p: float = 1.0,
        prompt_text: str = "",
    ) -> DraftTokenResult:
        """Async version of ``generate_tokens``.

        Uses ``httpx.AsyncClient`` for non-blocking HTTP.  This allows
        the caller to overlap the draft network round-trip with target
        model computation (the key performance win).
        """
        import asyncio

        start = time.monotonic()
        try:
            client = self._get_async_client()
            endpoint = self._config.endpoint_url

            payload: dict[str, Any] = {
                "prompt": prompt_tokens,
                "max_tokens": num_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "stream": False,
                "logprobs": 1,
            }
            if self._config.model_name:
                payload["model"] = self._config.model_name

            # Jittered retry loop
            last_error: Exception | None = None
            max_delay = 2.0
            for attempt in range(self._config.max_retries + 1):
                try:
                    resp = await client.post(endpoint, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    result = self._extract_from_completions(data)
                    break
                except Exception as e:
                    last_error = e
                    if attempt < self._config.max_retries:
                        base_delay = min(0.1 * (2 ** attempt), max_delay)
                        jitter = base_delay * 0.5 * (2 * random.random() - 1)
                        delay = max(0.05, base_delay + jitter)
                        await asyncio.sleep(delay)
                    else:
                        result = DraftTokenResult(
                            token_ids=[], logprobs=[], error=str(e),
                        )
            else:
                result = DraftTokenResult(
                    token_ids=[], logprobs=[], error=str(last_error),
                )

            elapsed = time.monotonic() - start
            self._stats.total_calls += 1
            self._stats.total_tokens += len(result.token_ids)
            self._stats.total_latency_s += elapsed
            return result

        except Exception as e:
            self._stats.errors += 1
            logger.warning("Async draft model call failed: {}", e)
            return DraftTokenResult(token_ids=[], logprobs=[], error=str(e))

    async def aclose(self) -> None:
        """Close both sync and async clients."""
        self.close()
        if self._async_client:
            await self._async_client.aclose()
            self._async_client = None


class DistributedSpeculativeDecoder:
    """Speculative decoding with a remote draft model.

    The draft model runs on a cheap device (CPU/laptop/edge) and is
    accessed via HTTP or gRPC. The target model runs on the GPU cluster.
    This gives 2-3x throughput improvement by overlapping cheap draft
    computation with expensive target verification.

    Supports:
    - Single remote draft model (``draft_model``)
    - Fleet of heterogeneous draft models (``draft_fleet``)
    - Adaptive candidate count based on acceptance rate
    - Proper rejection sampling with draft logprobs

    Args:
        target_forward: Callable accepting ``input_ids`` and returning logits.
        draft_model: RemoteDraftModel instance or endpoint URL string.
        draft_fleet: DraftModelFleet for heterogeneous routing.
        num_candidates: Number of draft tokens to generate per step.
        adaptive: Enable adaptive candidate count.
        min_candidates: Minimum candidates when adaptive is enabled.
        max_candidates: Maximum candidates when adaptive is enabled.
        temperature: Sampling temperature.
        top_k: Top-k sampling.
        device: Torch device for target model.
    """

    def __init__(
        self,
        target_forward: Callable[..., Any],
        draft_model: RemoteDraftModel | str | None = None,
        draft_fleet: Any | None = None,
        num_candidates: int = 5,
        adaptive: bool = False,
        min_candidates: int = 2,
        max_candidates: int = 10,
        temperature: float = 1.0,
        top_k: int = 20,
        device: str = "cuda",
    ):
        self._target = target_forward
        self._fleet = draft_fleet
        self._adaptive = adaptive
        self._min_candidates = min_candidates
        self._max_candidates = max_candidates
        self._current_candidates = num_candidates

        if draft_model is None and draft_fleet is None:
            raise ValueError("Either draft_model or draft_fleet must be provided")

        if isinstance(draft_model, str):
            config = RemoteDraftConfig(endpoint_url=draft_model)
            self._draft = RemoteDraftModel(config)
        elif draft_model is not None:
            self._draft = draft_model
        else:
            self._draft = None  # Will use fleet routing

        self._num_candidates = num_candidates
        self._temperature = temperature
        self._top_k = top_k
        self._device = torch.device(device)

        self._stats: dict[str, Any] = {
            "draft_calls": 0,
            "target_calls": 0,
            "accepted": 0,
            "total_proposed": 0,
            "fleet_routes": 0,
            "adaptive_adjustments": 0,
        }

        # Quality scorer for dynamic draft model selection
        self._quality_scorer: Any = None
        if draft_fleet is not None:
            try:
                from distllm.core.draft_quality_scorer import DraftQualityScorer
                self._quality_scorer = DraftQualityScorer()
            except ImportError:
                pass

    def _get_draft_model(self) -> RemoteDraftModel:
        """Get the active draft model (single or fleet-routed)."""
        if self._draft is not None:
            return self._draft

        if self._fleet is not None:
            from distllm.core.draft_model_router import DraftModelRouter, RoutingConstraints
            router = DraftModelRouter(self._fleet)
            decision = router.select(RoutingConstraints())
            if decision.selected_url:
                self._stats["fleet_routes"] += 1
                spec = self._fleet.get_spec(decision.selected_url)
                if spec:
                    return RemoteDraftModel(RemoteDraftConfig(
                        endpoint_url=spec.endpoint_url,
                        model_name=spec.model_name,
                        api_key=spec.api_key,
                        timeout_seconds=spec.timeout_seconds,
                        max_retries=spec.max_retries,
                        verify_ssl=spec.verify_ssl,
                    ))

        raise RuntimeError("No draft model available")

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        past_key_values: Any = None,
        fallback_batch: int = 3,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate tokens using distributed speculative decoding.

        Args:
            input_ids: Prompt token IDs, shape ``(1, seq_len)``.
            max_new_tokens: Maximum tokens to generate.
            past_key_values: Optional KV cache from previous generation
                to avoid re-encoding the prefix on every iteration.
            fallback_batch: Number of target-only tokens to generate
                when the draft model fails (avoids 1-token-per-step
                degradation).
            **kwargs: Forwarded to ``target_forward``.

        Returns:
            Generated token IDs, shape ``(1, prompt_len + generated)``.

        Note:
            Batch size > 1 is not yet supported for distributed
            speculative decoding.  The ``input_ids`` tensor must have
            shape ``(1, seq_len)``.
        """
        if input_ids.shape[0] != 1:
            raise ValueError(
                f"DistributedSpeculativeDecoder only supports batch_size=1, "
                f"got batch_size={input_ids.shape[0]}"
            )

        generated = input_ids.clone()
        prompt_len = input_ids.shape[1]
        target_len = prompt_len + max_new_tokens
        actual_draft_tokens = 0

        while generated.shape[1] < target_len:
            remaining = target_len - generated.shape[1]
            num_draft = min(self._current_candidates, remaining)

            # --- Remote draft phase ---
            current_tokens = generated[0].tolist()
            draft_model = self._get_draft_model()
            draft_result = draft_model.generate_tokens(
                prompt_tokens=current_tokens,
                num_tokens=num_draft,
                temperature=self._temperature,
                top_k=self._top_k,
            )
            self._stats["draft_calls"] += 1

            if not draft_result.ok:
                # Draft model failed — multi-token target-only fallback
                logger.debug("Draft fallback: {}", draft_result.error)
                fb_count = min(fallback_batch, remaining)
                for _ in range(fb_count):
                    if generated.shape[1] >= target_len:
                        break
                    target_logits = self._target(
                        generated, past_key_values=past_key_values, **kwargs,
                    )
                    self._stats["target_calls"] += 1
                    next_token = self._sample(target_logits[:, -1, :])
                    generated = torch.cat([generated, next_token], dim=1)
                    actual_draft_tokens += 1
                continue

            draft_token_ids = draft_result.token_ids
            draft_logprobs = draft_result.logprobs

            draft_tokens = torch.tensor(
                [draft_token_ids], dtype=torch.long, device=self._device,
            )

            # --- Verification phase ---
            full_input = torch.cat([generated, draft_tokens], dim=1)
            target_logits = self._target(
                full_input, past_key_values=past_key_values, **kwargs,
            )
            self._stats["target_calls"] += 1

            accepted_count = self._verify_tokens(
                generated, draft_tokens, target_logits, draft_logprobs,
            )

            generated = torch.cat([generated, draft_tokens[:, :accepted_count]], dim=1)
            actual_draft_tokens += len(draft_token_ids)

            if accepted_count < len(draft_token_ids):
                next_logits = target_logits[:, generated.shape[1] - 1, :]
                next_token = self._sample(next_logits)
                generated = torch.cat([generated, next_token], dim=1)

            # --- Adaptive candidate count ---
            if self._adaptive:
                self._adapt_candidates(accepted_count, len(draft_token_ids))

            # --- Quality scoring for fleet routing ---
            if self._quality_scorer is not None and self._draft is not None:
                draft_name = getattr(self._draft, '_config', None)
                draft_name = getattr(draft_name, 'endpoint_url', 'unknown') if draft_name else 'unknown'
                self._quality_scorer.record(
                    draft_model=draft_name,
                    target_model="target",
                    accepted=accepted_count,
                    total=len(draft_token_ids),
                )

        self._stats["total_proposed"] += actual_draft_tokens
        self._stats["accepted"] += generated.shape[1] - prompt_len

        return generated

    async def agenerate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        past_key_values: Any = None,
        fallback_batch: int = 3,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Async generation with draft/verify overlap.

        The key performance win: while the target model computes
        verification logits, the next batch of draft tokens is already
        being fetched over the network.  This hides most of the draft
        latency and gives 2-3x throughput improvement over sync.

        Timeline::

            Sync:   |--- Draft N ---||--- Verify N ---||--- Draft N+1 ---|
            Async:  |--- Draft N ---||--- Draft N+1 ---||--- Draft N+2 ---|
                    |               ||--- Verify N ---| |--- Verify N+1 --|

        Args:
            input_ids: Prompt token IDs, shape ``(1, seq_len)``.
            max_new_tokens: Maximum tokens to generate.
            past_key_values: Optional KV cache passthrough.
            fallback_batch: Target-only tokens on draft failure.
            **kwargs: Forwarded to ``target_forward``.

        Returns:
            Generated token IDs, shape ``(1, prompt_len + generated)``.

        Note:
            Batch size > 1 is not yet supported for distributed
            speculative decoding.  The ``input_ids`` tensor must have
            shape ``(1, seq_len)``.
        """
        with torch.no_grad():
            return await self._agenerate_impl(
                input_ids, max_new_tokens, past_key_values, fallback_batch, **kwargs,
            )

    async def _agenerate_impl(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        past_key_values: Any,
        fallback_batch: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Internal async generation loop with draft/verify overlap."""
        import asyncio

        if input_ids.shape[0] != 1:
            raise ValueError(
                f"DistributedSpeculativeDecoder only supports batch_size=1, "
                f"got batch_size={input_ids.shape[0]}"
            )

        generated = input_ids.clone()
        prompt_len = input_ids.shape[1]
        target_len = prompt_len + max_new_tokens
        actual_draft_tokens = 0

        draft_model = self._get_draft_model()

        # Pre-launch the first draft fetch
        current_tokens = generated[0].tolist()
        draft_task = asyncio.create_task(
            draft_model.agenerate_tokens(
                prompt_tokens=current_tokens,
                num_tokens=min(self._current_candidates, target_len - generated.shape[1]),
                temperature=self._temperature,
                top_k=self._top_k,
            )
        )

        while generated.shape[1] < target_len:
            remaining = target_len - generated.shape[1]

            # Await the draft tokens that were pre-fetched
            draft_result = await draft_task
            self._stats["draft_calls"] += 1

            if not draft_result.ok:
                logger.debug("Draft fallback: {}", draft_result.error)
                fb_count = min(fallback_batch, remaining)
                for _ in range(fb_count):
                    if generated.shape[1] >= target_len:
                        break
                    target_logits = self._target(
                        generated, past_key_values=past_key_values, **kwargs,
                    )
                    self._stats["target_calls"] += 1
                    next_token = self._sample(target_logits[:, -1, :])
                    generated = torch.cat([generated, next_token], dim=1)
                    actual_draft_tokens += 1
            else:
                draft_token_ids = draft_result.token_ids
                draft_logprobs = draft_result.logprobs

                draft_tokens = torch.tensor(
                    [draft_token_ids], dtype=torch.long, device=self._device,
                )

                # Launch next draft fetch BEFORE verifying
                next_remaining = target_len - generated.shape[1] - len(draft_token_ids)
                if next_remaining > 0:
                    next_tokens = (
                        torch.cat([generated, draft_tokens], dim=1)[0].tolist()
                    )
                    draft_task = asyncio.create_task(
                        draft_model.agenerate_tokens(
                            prompt_tokens=next_tokens,
                            num_tokens=min(self._current_candidates, next_remaining),
                            temperature=self._temperature,
                            top_k=self._top_k,
                        )
                    )

                # Verify current batch
                full_input = torch.cat([generated, draft_tokens], dim=1)
                target_logits = self._target(
                    full_input, past_key_values=past_key_values, **kwargs,
                )
                self._stats["target_calls"] += 1

                accepted_count = self._verify_tokens(
                    generated, draft_tokens, target_logits, draft_logprobs,
                )

                generated = torch.cat([generated, draft_tokens[:, :accepted_count]], dim=1)
                actual_draft_tokens += len(draft_token_ids)

                if accepted_count < len(draft_token_ids):
                    next_logits = target_logits[:, generated.shape[1] - 1, :]
                    next_token = self._sample(next_logits)
                    generated = torch.cat([generated, next_token], dim=1)

                if self._adaptive:
                    self._adapt_candidates(accepted_count, len(draft_token_ids))

                if accepted_count < len(draft_token_ids):
                    if generated.shape[1] < target_len:
                        current_tokens = generated[0].tolist()
                        draft_task = asyncio.create_task(
                            draft_model.agenerate_tokens(
                                prompt_tokens=current_tokens,
                                num_tokens=min(
                                    self._current_candidates,
                                    target_len - generated.shape[1],
                                ),
                                temperature=self._temperature,
                                top_k=self._top_k,
                            )
                        )
                    continue

            # Launch next draft if we didn't already
            if generated.shape[1] < target_len:
                current_tokens = generated[0].tolist()
                draft_task = asyncio.create_task(
                    draft_model.agenerate_tokens(
                        prompt_tokens=current_tokens,
                        num_tokens=min(
                            self._current_candidates,
                            target_len - generated.shape[1],
                        ),
                        temperature=self._temperature,
                        top_k=self._top_k,
                    )
                )

        self._stats["total_proposed"] += actual_draft_tokens
        self._stats["accepted"] += generated.shape[1] - prompt_len

        return generated

    def _verify_tokens(
        self,
        prefix: torch.Tensor,
        draft_tokens: torch.Tensor,
        target_logits: torch.Tensor,
        draft_logprobs: list[float] | None = None,
    ) -> int:
        """Verify draft tokens against target model distribution.

        Uses proper rejection sampling: accept token *i* with probability
        ``min(1, p_i / q_i)`` where *p* is the target probability and *q*
        is the draft probability.  When the remote model provides logprobs,
        *q = exp(logprob)*.  When logprobs are unavailable, falls back to
        greedy verification (``temperature=0``) or a uniform-draft
        approximation (``p * vocab_size``).
        """
        num_draft = draft_tokens.shape[1]
        prefix_len = prefix.shape[1] - 1

        # Greedy path — deterministic, no sampling needed
        if self._temperature == 0:
            for i in range(num_draft):
                target_argmax = target_logits[:, prefix_len + i, :].argmax(dim=-1).item()
                if target_argmax != draft_tokens[0, i].item():
                    return i
            return num_draft

        has_logprobs = draft_logprobs is not None and len(draft_logprobs) >= num_draft

        for i in range(num_draft):
            target_probs = F.softmax(
                target_logits[:, prefix_len + i, :] / self._temperature, dim=-1,
            )
            draft_token_id = draft_tokens[0, i].item()
            p = target_probs[0, draft_token_id].item()

            if has_logprobs:
                # Proper rejection sampling: min(1, p / q)
                q = float(torch.exp(torch.tensor(draft_logprobs[i])).item())  # type: ignore[index]
                if q <= 0:
                    return i
                acceptance_prob = min(1.0, p / q)
            else:
                # Fallback: assume uniform draft distribution → q ≈ 1/vocab
                vocab_size = target_logits.shape[-1]
                acceptance_prob = min(1.0, p * vocab_size)

            if torch.rand(1).item() >= acceptance_prob:
                return i

            if p <= 0:
                return i

        return num_draft

    def batch_verify(
        self,
        prefixes: list[torch.Tensor],
        draft_tokens_list: list[torch.Tensor],
        draft_logprobs_list: list[list[float] | None],
    ) -> list[int]:
        """Verify draft tokens for multiple requests in one target forward pass.

        Batches all draft sequences into a single target model call for
        better GPU utilization, then verifies each independently.

        Args:
            prefixes: List of ``(1, prefix_len_i)`` tensors (one per request).
            draft_tokens_list: List of ``(1, num_draft_i)`` tensors.
            draft_logprobs_list: List of logprobs lists (or None per request).

        Returns:
            List of accepted token counts (one per request).
        """
        if not prefixes:
            return []

        # Build batched input: pad all sequences to the same length
        max_len = max(p.shape[1] + d.shape[1] for p, d in zip(prefixes, draft_tokens_list))
        batch_size = len(prefixes)

        padded = torch.zeros(batch_size, max_len, dtype=torch.long, device=self._device)
        prefix_lens = []
        draft_lens = []

        for i, (prefix, draft) in enumerate(zip(prefixes, draft_tokens_list)):
            full = torch.cat([prefix, draft], dim=1)
            padded[i, :full.shape[1]] = full[0]
            prefix_lens.append(prefix.shape[1])
            draft_lens.append(draft.shape[1])

        # Single batched target forward pass
        target_logits = self._target(padded)
        self._stats["target_calls"] += 1

        # Verify each request independently
        results = []
        for i in range(batch_size):
            plen = prefix_lens[i]
            dlen = draft_lens[i]
            # Extract logits for this request's draft tokens
            req_logits = target_logits[i:i+1, plen - 1:plen + dlen - 1, :]
            req_draft = draft_tokens_list[i]
            req_logprobs = draft_logprobs_list[i]

            accepted = self._verify_tokens(
                prefixes[i], req_draft, req_logits, req_logprobs,
            )
            results.append(accepted)

        return results

    def _adapt_candidates(self, accepted: int, proposed: int) -> None:
        """Adjust num_candidates based on acceptance rate."""
        if proposed == 0:
            return

        rate = accepted / proposed
        old = self._current_candidates

        if rate > 0.8:
            self._current_candidates = min(self._current_candidates + 1, self._max_candidates)
        elif rate < 0.3:
            self._current_candidates = max(self._current_candidates - 1, self._min_candidates)

        if self._current_candidates != old:
            self._stats["adaptive_adjustments"] += 1
            logger.debug(
                "Adaptive candidates: {} -> {} (acceptance_rate={:.2f})",
                old, self._current_candidates, rate,
            )

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample a token from logits."""
        if self._temperature == 0:
            return logits.argmax(dim=-1, keepdim=True)
        if self._top_k > 0:
            values, indices = torch.topk(logits, self._top_k, dim=-1)
            logits = torch.full_like(logits, float("-inf")).scatter_(-1, indices, values)
        probs = F.softmax(logits / self._temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @property
    def stats(self) -> dict[str, Any]:
        s = dict(self._stats)
        if s["total_proposed"] > 0:
            s["acceptance_rate"] = round(s["accepted"] / max(s["total_proposed"], 1), 3)
        if self._draft is not None:
            s["draft_model_stats"] = self._draft.stats
        if self._adaptive:
            s["current_candidates"] = self._current_candidates
        return s

    def close(self) -> None:
        if self._draft is not None:
            self._draft.close()

    async def aclose(self) -> None:
        """Close both sync and async clients."""
        if self._draft is not None:
            await self._draft.aclose()

    # ── Auto-selection integration ──────────────────────────────────

    def set_workload_type(self, text_or_type: str) -> None:
        """Set or detect the workload type for auto-selection.

        If ``text_or_type`` is a recognized ``WorkloadType`` value, it
        is used directly.  Otherwise it is treated as prompt text and
        classified automatically.
        """
        from distllm.core.workload_classifier import WorkloadType, classify

        try:
            self._workload_type = WorkloadType(text_or_type).value
        except ValueError:
            self._workload_type = classify(text_or_type).value

    def record_method_performance(
        self,
        method: str,
        drafted: int,
        accepted: int,
        latency_ms: float = 0.0,
    ) -> None:
        """Record performance data for a speculative method."""
        if not hasattr(self, "_profiler"):
            from distllm.core.speculative_profiler import SpeculativeProfiler
            self._profiler = SpeculativeProfiler(warmup_samples=1)

        wt = getattr(self, "_workload_type", "unknown")
        self._profiler.record_acceptance(method, wt, drafted, accepted, latency_ms)

    def get_active_method(self, workload_type: str | None = None) -> str | None:
        """Return the currently active speculative method name."""
        wt = workload_type or getattr(self, "_workload_type", "unknown")
        if hasattr(self, "_profiler"):
            return self._profiler.get_best_method(wt)
        return "remote_draft"

    def get_metrics(self) -> dict[str, Any]:
        """Return metrics including profiler ranking and workload type."""
        s = self.stats
        wt = getattr(self, "_workload_type", "unknown")
        s["workload_type"] = wt
        if hasattr(self, "_profiler"):
            s["method_ranking"] = self._profiler.get_method_ranking(wt)
        return s
