"""Ollama BackendAdapter - connects DistLLM to Ollama servers."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

from distllm.backends.protocol import GenerateOnly


@dataclass
class OllamaBackendConfig:
    base_url: str = "http://localhost:11434"
    timeout_s: float = 120.0
    keep_alive: str = "5m"

class OllamaBackendAdapter(GenerateOnly):
    """Remote-HTTP adapter for an Ollama inference server.

    Generate-only: ``forward()`` raises ``NotImplementedError`` (inherited
    from :class:`GenerateOnly`); ``supports_forward()`` is ``False`` so the
    registry never selects this adapter for pipeline ``forward()`` calls.
    """

    def __init__(self, config: OllamaBackendConfig | None = None):
        self.config = config or OllamaBackendConfig()

    # ── Required metadata classmethods ───────────────────────────────

    @classmethod
    def display_name(cls) -> str:
        return "Ollama"

    @classmethod
    def version(cls) -> str:
        return "0.1.0"

    @classmethod
    def is_available(cls) -> bool:
        """True only if the HTTP client dependency is importable.

        Server reachability is ``load_model()``/``health_check()``'s job;
        availability here just means the adapter's client stack is
        installed, mirroring the TGI adapter convention.
        """
        try:
            import importlib.util as _util

            return _util.find_spec("httpx") is not None
        except Exception:
            return False

    @classmethod
    def priority_for(cls, device_type: str) -> int:
        """Ollama is a remote server, usable from any device with network access.

        It is not a local GPU engine, so it sits below the local engines
        (PyTorch/vLLM) but above pure-CPU fallbacks.
        """
        device_type = (device_type or "cpu").lower()
        if device_type in ("cuda", "rocm", "mps", "xpu"):
            return 4
        return 2

    # ── Required instance methods ────────────────────────────────────

    def load_model(self, model_name: str | None = None) -> bool:
        """Probe the Ollama server (list installed models) before serving."""
        try:
            resp = httpx.get(f"{self.config.base_url}/api/tags", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    def shutdown(self) -> None:
        """Nothing to release for a remote HTTP client."""
        pass

    # ``forward()`` is inherited from GenerateOnly: it raises
    # NotImplementedError and supports_forward() returns False.
    # (The previous local stub raised the same error; the mixin now
    # centralises it so every generate-only adapter fails identically.)

    # ── Generation ───────────────────────────────────────────────────

    async def generate(self, prompt, max_new_tokens=256, temperature=0.7, **kwargs):
        payload = {"model": kwargs.get("model", "llama3.2"), "prompt": prompt, "stream": False,
                   "options": {"num_predict": max_new_tokens, "temperature": temperature},
                   "keep_alive": self.config.keep_alive}
        async with httpx.AsyncClient(timeout=self.config.timeout_s) as client:
            resp = await client.post(f"{self.config.base_url}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return {"text": data.get("response", ""), "tokens": data.get("eval_count", 0)}
    async def generate_stream(self, prompt, max_new_tokens=256, temperature=0.7, **kwargs):
        payload = {"model": kwargs.get("model", "llama3.2"), "prompt": prompt, "stream": True,
                   "options": {"num_predict": max_new_tokens, "temperature": temperature}}
        async with httpx.AsyncClient(timeout=self.config.timeout_s) as client:
            async with client.stream("POST", f"{self.config.base_url}/api/generate", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.strip():
                        data = json.loads(line)
                        token = data.get("response", "")
                        if token:
                            yield token
