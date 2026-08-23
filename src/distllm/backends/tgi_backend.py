"""TGI BackendAdapter — connects DistLLM to HuggingFace TGI servers.

HuggingFace `Text Generation Inference` (TGI) exposes an HTTP API for
token generation. This adapter talks to a running TGI server over
``httpx``. It is *registered* unconditionally but reports
``is_available() == False`` unless the optional ``text_generation``
client library is installed, so registration never crashes when the
dependency is absent.
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from distllm.backends.protocol import BackendAdapter


@dataclass
class TGIBackendConfig:
    base_url: str = "http://localhost:8080"
    api_key: str = ""
    timeout_s: float = 60.0
    max_concurrent_requests: int = 64


class TGIBackendAdapter(BackendAdapter):
    """Wraps a HuggingFace Text Generation Inference (TGI) server."""

    def __init__(self, config: TGIBackendConfig | None = None):
        self.config = config or TGIBackendConfig()
        self._headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            self._headers["Authorization"] = f"Bearer {self.config.api_key}"

    # ── Required metadata classmethods ───────────────────────────────

    @classmethod
    def display_name(cls) -> str:
        return "TGI"

    @classmethod
    def version(cls) -> str:
        return "0.1.0"

    @classmethod
    def is_available(cls) -> bool:
        """True only if the optional ``text_generation`` client is present.

        The adapter uses ``httpx`` directly for its HTTP calls, but it is
        considered "available" only when the official TGI client library is
        installed, so it does not advertise itself as ready when the runtime
        dependency is missing. Import failures are caught so that merely
        importing / registering this backend can never crash.
        """
        try:
            import importlib.util as _util

            return _util.find_spec("text_generation") is not None
        except Exception:
            return False

    @classmethod
    def priority_for(cls, device_type: str) -> int:
        """TGI is a remote server, usable from any device with network access.

        It is not a local GPU engine, so it sits below the local engines
        (PyTorch/vLLM) but above pure-CPU fallbacks.
        """
        device_type = (device_type or "cpu").lower()
        if device_type in ("cuda", "rocm", "mps", "xpu"):
            return 4
        return 2

    # ── Required instance methods ────────────────────────────────────

    def load_model(self, model_name: str) -> bool:
        try:
            resp = httpx.get(f"{self.config.base_url}/info", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    def forward(self, hidden_states: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("TGI adapter does not support layer-level forward")

    def shutdown(self) -> None:
        """Nothing to release for a remote HTTP client."""
        self._headers = {}

    # ── Generation ───────────────────────────────────────────────────

    async def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7, top_p: float = 0.9, top_k: int = 40, repetition_penalty: float = 1.0, stop_sequences: list[str] | None = None, **kwargs: Any) -> dict[str, Any]:
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_new_tokens, "temperature": temperature, "top_p": top_p, "top_k": top_k, "repetition_penalty": repetition_penalty}}
        if stop_sequences:
            payload["parameters"]["stop"] = stop_sequences
        if kwargs:
            payload["parameters"].update(kwargs)
        async with httpx.AsyncClient(timeout=self.config.timeout_s) as client:
            resp = await client.post(f"{self.config.base_url}/generate", json=payload, headers=self._headers)
            resp.raise_for_status()
            data = resp.json()
            return {"text": data.get("generated_text", ""), "tokens": data.get("details", {}).get("generated_tokens", 0)}

    async def generate_stream(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7, **kwargs: Any) -> AsyncIterator[str]:
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_new_tokens, "temperature": temperature}}
        if kwargs:
            payload["parameters"].update(kwargs)
        async with httpx.AsyncClient(timeout=self.config.timeout_s) as client:
            async with client.stream("POST", f"{self.config.base_url}/generate_stream", json=payload, headers=self._headers) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.startswith("data:"):
                        token_data = json.loads(line[5:].strip())
                        token = token_data.get("token", {}).get("text", "")
                        if token:
                            yield token
