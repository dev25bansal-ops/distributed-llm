"""NVIDIA NIM backend adapter for distributed inference.

Connects to a NVIDIA NIM (NVIDIA Inference Microservice) endpoint via its
OpenAI-compatible HTTP API. Supports chat completions, streaming SSE, and
when running on the same host, optional CUDA Graph capture for forward-pass
optimization.

NIM provides optimized inference for NVIDIA GPUs through TensorRT-LLM
under the hood, exposing a familiar REST API that follows the OpenAI
chat/completions pattern.

Usage:
    backend = NimNodeAdapter(
        model_name="meta/llama3-8b-instruct",
        api_url="http://localhost:8000/v1",
        api_key="nvapi-...",
    )
    backend.load_model()
    logits, kv = backend.forward(input_ids=input_ids)
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, AsyncIterator, Iterator

import numpy as np
import torch
from loguru import logger

from distllm.backends.protocol import BackendAdapter
from distllm.errors import ModelLoadError


# ── Constants ────────────────────────────────────────────────────────

_NIM_DEFAULT_API_URL = "http://localhost:8000/v1"
"""Default base URL for a local NIM endpoint."""

_NIM_DEFAULT_TIMEOUT = 60.0
"""Default HTTP request timeout in seconds."""

_NIM_HEALTH_ENDPOINT = "/health"
"""NIM health check endpoint (relative to base URL)."""

_NIM_READY_ENDPOINT = "/health/ready"
"""NIM readiness endpoint."""

_SUPPORTED_ENDPOINTS = {
    "chat": "/chat/completions",
    "completions": "/completions",
    "embeddings": "/embeddings",
}
"""NIM API endpoint paths (OpenAI-compatible)."""


# ── CUDA Graph helpers (local optimization) ──────────────────────────


def _capture_cuda_graph(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    warmup_iters: int = 3,
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor] | None:
    """Capture a CUDA Graph for the forward pass of *model*.

    CUDA Graphs eliminate CPU launch overhead by capturing a sequence of
    GPU operations and replaying them directly. This is most beneficial
    for fixed-size inputs on local GPU deployments.

    Args:
        model: The PyTorch model whose forward pass to capture.
        sample_input: A representative input tensor.
        warmup_iters: Number of warmup iterations before capture.

    Returns:
        Tuple of ``(graph, static_input, static_output)`` or ``None`` if
        CUDA Graphs are not available or capture fails.
    """
    if not torch.cuda.is_available():
        return None

    device = next(model.parameters()).device
    sample_input = sample_input.to(device)

    try:
        # Warmup
        for _ in range(warmup_iters):
            _ = model(sample_input)

        # Allocate static buffers
        static_input = sample_input.clone()
        static_output = torch.zeros(
            1, 1, model.config.hidden_size
            if hasattr(model, "config") and hasattr(model.config, "hidden_size")
            else 4096,
            device=device,
        )

        # Capture graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output.copy_(model(static_input))

        logger.info(
            f"[NIM] CUDA Graph captured for {type(model).__name__} "
            f"on {device}"
        )
        return graph, static_input, static_output
    except Exception as e:
        logger.warning(f"[NIM] CUDA Graph capture failed: {e}")
        return None


def _replay_cuda_graph(
    graph: torch.cuda.CUDAGraph | None,
    static_input: torch.Tensor | None,
    static_output: torch.Tensor | None,
    input_tensor: torch.Tensor,
) -> torch.Tensor | None:
    """Replay a captured CUDA Graph with new input data.

    Copies *input_tensor* into the static input buffer, replays the
    graph, and returns the output. Falls back to ``None`` on mismatch
    or error.

    Args:
        graph: The captured CUDA Graph (or ``None``).
        static_input: Static input buffer from capture.
        static_output: Static output buffer from capture.
        input_tensor: The actual input for this inference call.

    Returns:
        Output tensor, or ``None`` if graph replay is not applicable.
    """
    if graph is None or static_input is None or static_output is None:
        return None

    if input_tensor.shape != static_input.shape:
        return None  # Shape mismatch; caller falls through to eager mode

    try:
        static_input.copy_(input_tensor)
        graph.replay()
        return static_output.clone()
    except Exception:
        return None


# ── Adapter ──────────────────────────────────────────────────────────


class NimNodeAdapter(BackendAdapter):
    """Wraps NVIDIA NIM as a per-node inference backend.

    Communicates with a NIM endpoint via OpenAI-compatible HTTP API.
    When running on the same GPU host, the adapter can optionally capture
    and replay CUDA Graphs to reduce forward-pass latency.

    Args:
        model_name: Model identifier used by the NIM deployment
            (e.g. ``"meta/llama3-8b-instruct"``).
        api_url: Base URL of the NIM API server.
        api_key: Optional API key for authenticated endpoints (NVIDIA
            API key or ``nvapi-...`` format).
        timeout: HTTP request timeout in seconds.
        max_retries: Number of retries for failed requests.
        enable_cuda_graph: Attempt CUDA Graph capture for local forward
            pass optimization (only effective when model is local).
        local_model: An optional local PyTorch model instance. When set,
            ``forward()`` runs locally using this model and ``generate()``
            uses the remote NIM API.
        extra_headers: Additional HTTP headers to include in every
            request (e.g. ``{"X-NIM-Sandbox": "1"}``).
    """

    def __init__(
        self,
        model_name: str,
        api_url: str = _NIM_DEFAULT_API_URL,
        api_key: str | None = None,
        timeout: float = _NIM_DEFAULT_TIMEOUT,
        max_retries: int = 3,
        enable_cuda_graph: bool = False,
        local_model: torch.nn.Module | None = None,
        extra_headers: dict[str, str] | None = None,
    ):
        self.model_name = model_name
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.enable_cuda_graph = enable_cuda_graph and torch.cuda.is_available()
        self._local_model = local_model
        self._extra_headers = dict(extra_headers or {})

        self._session = None
        self._tokenizer = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # CUDA Graph state (only used when local_model is set)
        self._cuda_graph: torch.cuda.CUDAGraph | None = None
        self._graph_static_input: torch.Tensor | None = None
        self._graph_static_output: torch.Tensor | None = None

    # ── Core interface ───────────────────────────────────────────────

    def load_model(self) -> None:
        """Verify the NIM endpoint is reachable and ready.

        Performs a GET to the NIM health endpoint to confirm the service
        is live. If a local model is provided, CUDA Graph capture is
        attempted when enabled.

        Raises:
            ModelLoadError: If the NIM endpoint is unreachable or
                returns a non-200 status.
        """
        logger.info(f"[NIM] Connecting to NIM endpoint at {self.api_url}")

        # Test connectivity
        try:
            import httpx

            self._session = httpx.Client(
                base_url=self.api_url,
                headers=self._build_headers(),
                timeout=self.timeout,
            )

            ready = self._check_endpoint("ready")
            if not ready:
                live = self._check_endpoint("live")
                if not live:
                    raise ModelLoadError(
                        self.model_name,
                        f"NIM endpoint at {self.api_url} is not reachable. "
                        f"Verify the NIM service is running.",
                    )
                logger.warning(
                    f"[NIM] NIM is live but not ready at {self.api_url}"
                )

            # Verify model is available via /v1/models
            if not self._verify_model_available():
                logger.warning(
                    f"[NIM] Model {self.model_name!r} not found in NIM model list. "
                    f"Requests may fail."
                )

            logger.info(f"[NIM] NIM endpoint ready: {self.api_url}")
        except ImportError as e:
            raise ModelLoadError(
                self.model_name,
                "httpx is required for the NIM backend. "
                "Install with: pip install httpx",
            ) from e
        except ModelLoadError:
            raise
        except Exception as e:
            self._session = None
            logger.error(f"[NIM] Failed to connect to {self.api_url}: {e}")
            raise ModelLoadError(self.model_name, str(e)) from e

        # Local model CUDA Graph setup
        if self._local_model is not None and self.enable_cuda_graph:
            self._local_model.to(self._device)
            self._local_model.eval()
            sample = torch.randint(0, 100, (1, 1), device=self._device)
            try:
                captured = _capture_cuda_graph(self._local_model, sample)
                if captured is not None:
                    self._cuda_graph, self._graph_static_input, self._graph_static_output = captured
            except Exception as e:
                logger.warning(f"[NIM] CUDA Graph capture failed for local model: {e}")

        # Extract tokenizer from local model if available
        if self._local_model is not None:
            try:
                from transformers import AutoTokenizer

                model_name_or_path = getattr(
                    self._local_model, "name_or_path", self.model_name
                )
                self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            except Exception:
                self._tokenizer = None

    def forward(
        self,
        hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Run a forward pass.

        When a local PyTorch model is configured, runs locally (with
        CUDA Graph optimization if enabled). Otherwise, dispatches via
        the NIM API.

        Args:
            input_ids: Token IDs for the first node.
            hidden_states: Activations from previous node.
            attention_mask: Causal + padding mask.
            position_ids: Position indices for RoPE.
            past_key_values: KV cache from previous iterations.

        Returns:
            Tuple of ``(output_tensor, kv_cache)``.
        """
        if input_ids is not None:
            return self._forward_input_ids(input_ids)
        if hidden_states is not None:
            return self._forward_hidden_states(
                hidden_states, attention_mask, position_ids, past_key_values
            )
        raise ValueError("Either input_ids or hidden_states must be provided")

    def _forward_input_ids(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass starting from token IDs.

        Prefers local model execution when available; falls back to NIM
        API call.
        """
        if self._local_model is not None:
            return self._forward_local(input_ids)
        return self._forward_via_api(input_ids)

    def _forward_local(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Local forward pass with optional CUDA Graph optimization."""
        input_ids = input_ids.to(self._device)

        # Try CUDA Graph replay first
        if self._cuda_graph is not None:
            output = _replay_cuda_graph(
                self._cuda_graph,
                self._graph_static_input,
                self._graph_static_output,
                input_ids,
            )
            if output is not None:
                return output, []

        # Eager fallback
        with torch.no_grad():
            output = self._local_model(
                input_ids,
                use_cache=True,
            )

        if isinstance(output, tuple):
            logits, new_kv = output
        elif hasattr(output, "logits"):
            logits = output.logits
            new_kv = getattr(output, "past_key_values", [])
        else:
            logits = output[0] if isinstance(output, (list, tuple)) else output
            new_kv = []

        return logits, new_kv

    def _forward_via_api(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass via NIM's completions API.

        Encodes token IDs as a prompt and extracts logits from the API
        response. This is a fallback for when no local model is
        available.
        """
        if self._tokenizer is not None:
            prompt = self._tokenizer.decode(
                input_ids[0].tolist(), skip_special_tokens=True
            )
        else:
            prompt = f"<input>{input_ids[0].tolist()}</input>"

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": 1,
        }

        try:
            response = self._request("POST", "/completions", json=payload)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            raise RuntimeError(
                f"NIM forward failed for {self.model_name}: {e}"
            ) from e

        # Extract logprobs if available, otherwise return placeholder
        choices = data.get("choices", [])
        if choices:
            logprobs = choices[0].get("logprobs")
            if logprobs and "top_logprobs" in logprobs:
                top = logprobs["top_logprobs"][0]
                vocab_size = len(top) * 4  # rough estimate if unknown
                logit_tensor = torch.zeros(
                    1, 1, max(vocab_size, 1), device=self._device
                )
                for token_str, prob in top.items():
                    idx = hash(token_str) % logit_tensor.shape[-1]
                    logit_tensor[0, 0, idx] = np.log(max(prob, 1e-10))
                return logit_tensor, []
            token_id = choices[0].get("token_id", 0)
            logit_tensor = torch.zeros(1, 1, 32000, device=self._device)
            logit_tensor[0, 0, token_id] = 1.0
            return logit_tensor, []

        raise RuntimeError(
            f"NIM forward returned empty choices for {self.model_name}"
        )

    def _forward_hidden_states(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Pipeline-mode forward via local model (NIM does not expose
        per-layer access through the API)."""
        if self._local_model is None:
            raise NotImplementedError(
                "Pipeline-mode forward via hidden_states requires a local "
                "model. NIM's HTTP API does not expose per-layer access. "
                "Set local_model= in the constructor."
            )

        self._local_model.to(self._device)
        self._local_model.eval()

        with torch.no_grad():
            output = self._local_model(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )

        if isinstance(output, tuple):
            logits, new_kv = output
        elif hasattr(output, "logits"):
            logits = output.logits
            new_kv = getattr(output, "past_key_values", [])
        else:
            logits = output[0] if isinstance(output, (list, tuple)) else output
            new_kv = []

        return logits, new_kv

    # ── Generation ───────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **sampling_kwargs: Any,
    ) -> str:
        """Single-shot text generation via NIM's chat/completions API.

        Args:
            prompt: Input text prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            Generated text content.
        """
        if self._session is None:
            raise ModelLoadError(
                self.model_name,
                "NIM not connected. Call load_model() first.",
            )

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            **sampling_kwargs,
        }

        try:
            response = self._request(
                "POST", _SUPPORTED_ENDPOINTS["chat"], json=payload
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            raise RuntimeError(
                f"NIM generation failed for {self.model_name}: {e}"
            ) from e

        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(
                f"NIM generation returned empty choices: {data}"
            )

        content = choices[0].get("message", {}).get("content", "")
        return content

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **sampling_kwargs: Any,
    ) -> Iterator[str]:
        """Streaming generation via NIM's SSE endpoint.

        Yields content tokens incrementally as they are generated by
        the NIM server, using server-sent events (SSE).

        Args:
            prompt: Input text prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Yields:
            Content tokens (strings) one chunk at a time.
        """
        if self._session is None:
            raise ModelLoadError(
                self.model_name,
                "NIM not connected. Call load_model() first.",
            )

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
            **sampling_kwargs,
        }

        try:
            with self._session.stream(
                "POST",
                _SUPPORTED_ENDPOINTS["chat"],
                json=payload,
                timeout=self.timeout,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
        except Exception as e:
            logger.error(
                f"[NIM] Stream generation failed for {self.model_name}: {e}"
            )

    async def async_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **sampling_kwargs: Any,
    ) -> str:
        """Async generation via NIM's chat API using httpx.

        Args:
            prompt: Input text prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            Generated text content.
        """
        import httpx

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            **sampling_kwargs,
        }

        async with httpx.AsyncClient(
            base_url=self.api_url,
            headers=self._build_headers(),
            timeout=self.timeout,
        ) as client:
            try:
                response = await client.post(
                    _SUPPORTED_ENDPOINTS["chat"], json=payload
                )
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                raise RuntimeError(
                    f"NIM async generation failed for {self.model_name}: {e}"
                ) from e

        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(
                f"NIM async generation returned empty choices: {data}"
            )
        return choices[0].get("message", {}).get("content", "")

    async def async_generate_stream(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **sampling_kwargs: Any,
    ) -> AsyncIterator[str]:
        """Async streaming generation via NIM's SSE endpoint.

        Args:
            prompt: Input text prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Yields:
            Content tokens asynchronously.
        """
        import httpx

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
            **sampling_kwargs,
        }

        async with httpx.AsyncClient(
            base_url=self.api_url,
            headers=self._build_headers(),
            timeout=self.timeout,
        ) as client:
            try:
                async with client.stream(
                    "POST",
                    _SUPPORTED_ENDPOINTS["chat"],
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue

                            choices = chunk.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
            except Exception as e:
                logger.error(
                    f"[NIM] Async stream failed for {self.model_name}: {e}"
                )

    # ── Health & load ────────────────────────────────────────────────

    def health_check(self) -> bool:
        """Check that the NIM endpoint is reachable and healthy.

        Returns:
            ``True`` if the NIM health endpoint returns 200.
        """
        if self._session is None:
            return False
        try:
            resp = self._request("GET", _NIM_HEALTH_ENDPOINT)
            return resp.status_code == 200
        except Exception:
            return False

    def current_load(self) -> float:
        """Return an estimated load factor.

        Since NIM does not expose a load endpoint, this returns ``0.0``
        (idle) by default. Override in subclasses that can query NIM
        metrics.
        """
        return 0.0

    # ── Shutdown ─────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Close the HTTP session and release resources."""
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None
            logger.info(f"[NIM] Session closed for {self.api_url}")

        # Release CUDA Graph resources
        self._cuda_graph = None
        self._graph_static_input = None
        self._graph_static_output = None

        if self._local_model is not None:
            self._local_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._tokenizer = None

    # ── Tokenizer ────────────────────────────────────────────────────

    def get_tokenizer(self) -> Any:
        return self._tokenizer

    # ── Metadata classmethods ────────────────────────────────────────

    @classmethod
    def display_name(cls) -> str:
        return "NVIDIA NIM"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import httpx  # noqa: F401
            return True
        except ImportError:
            return False

    @classmethod
    def priority_for(cls, device_type: str) -> int:
        # NIM is designed for NVIDIA GPUs; works well on CUDA.
        if device_type == "cuda":
            return 11
        if device_type == "rocm":
            return 5
        return 0

    # ── Internal helpers ─────────────────────────────────────────────

    def _build_headers(self) -> dict[str, str]:
        """Build the HTTP headers for NIM API requests."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            **self._extra_headers,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> Any:
        """Make an HTTP request with retries.

        Args:
            method: HTTP method (``"GET"``, ``"POST"``, etc.).
            path: URL path relative to the base URL.
            kwargs: Additional arguments for ``httpx.Client.request``.

        Returns:
            ``httpx.Response`` object.
        """
        if self._session is None:
            raise RuntimeError("NIM session not initialized. Call load_model() first.")

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._session.request(method, path, **kwargs)
                return response
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    delay = 2 ** attempt
                    logger.debug(
                        f"[NIM] Request failed (attempt {attempt}/{self.max_retries}), "
                        f"retrying in {delay}s: {e}"
                    )
                    time.sleep(delay)

        raise RuntimeError(
            f"NIM request failed after {self.max_retries} retries: {last_error}"
        ) from last_error

    def _check_endpoint(self, kind: str) -> bool:
        """Check a NIM health endpoint.

        Args:
            kind: ``"live"`` for liveness, ``"ready"`` for readiness.

        Returns:
            ``True`` if the endpoint returns a 2xx status.
        """
        path = _NIM_READY_ENDPOINT if kind == "ready" else _NIM_HEALTH_ENDPOINT
        try:
            resp = self._request("GET", path)
            return resp.is_success
        except Exception:
            return False

    def _verify_model_available(self) -> bool:
        """Check that *model_name* appears in NIM's model list.

        Returns:
            ``True`` if the model is listed or the check fails (fail-open).
        """
        try:
            resp = self._request("GET", "/models")
            if resp.is_success:
                data = resp.json()
                models = data.get("data", [])
                return any(
                    m.get("id") == self.model_name for m in models
                )
            return True  # fail-open if we cannot list models
        except Exception:
            return True  # fail-open on error


__all__ = ["NimNodeAdapter"]
