"""Dify custom model provider plugin for DistLLM.

This plugin registers DistLLM as a custom OpenAI-compatible model
provider in Dify, enabling chat, completion, and embedding models.

Usage:
    1. Copy this file to your Dify plugins directory
    2. Configure via environment variables or Dify UI
    3. Restart Dify

Environment Variables:
    DISTLLM_API_BASE: DistLLM API URL (default: http://localhost:8000/v1)
    DISTLLM_API_KEY: API key (default: any non-empty string)
    DISTLLM_MODEL_NAME: Default model name (default: distributed-llm)
"""

from __future__ import annotations

import os
from typing import Any, Generator

import httpx


# Dify provider configuration
PROVIDER_CONFIG = {
    "provider": "distllm",
    "label": {"en_US": "DistLLM", "zh_CN": "DistLLM"},
    "description": {
        "en_US": "Distributed LLM inference framework with pipeline parallelism",
        "zh_CN": "分布式LLM推理框架，支持流水线并行",
    },
    "icon_large": None,
    "background": "#10a37f",
}

# Supported model definitions
MODELS = [
    {
        "model": "distributed-llm",
        "label": {"en_US": "DistLLM", "zh_CN": "DistLLM"},
        "model_type": "llm",
        "features": ["agent-thought", "streaming"],
        "model_properties": {
            "mode": "chat",
            "context_size": 4096,
            "max_chunks": 5,
        },
        "parameter_rules": [
            {
                "name": "temperature",
                "label": {"en_US": "Temperature"},
                "type": "float",
                "default": 0.7,
                "min": 0.0,
                "max": 2.0,
                "precision": 1,
            },
            {
                "name": "top_p",
                "label": {"en_US": "Top P"},
                "type": "float",
                "default": 0.9,
                "min": 0.0,
                "max": 1.0,
                "precision": 2,
            },
            {
                "name": "max_tokens",
                "label": {"en_US": "Max Tokens"},
                "type": "int",
                "default": 256,
                "min": 1,
                "max": 8192,
            },
        ],
    },
    {
        "model": "distributed-llm-completion",
        "label": {"en_US": "DistLLM Completion", "zh_CN": "DistLLM 补全"},
        "model_type": "llm",
        "features": ["streaming"],
        "model_properties": {
            "mode": "completion",
            "context_size": 4096,
        },
    },
]


class DistLLMProvider:
    """DistLLM model provider for Dify.

    Implements the Dify custom model provider interface using
    the DistLLM OpenAI-compatible API.
    """

    def __init__(self):
        # The DistLLM OpenAI-compatible API is served under /v1/... paths, so
        # the base URL must be the bare origin (no trailing /v1) — otherwise
        # combined with the /v1/... request paths it becomes /v1/v1/... (F-003).
        self._api_base = os.environ.get("DISTLLM_API_BASE", "http://localhost:8000").rstrip("/")
        self._api_key = os.environ.get("DISTLLM_API_KEY", "distllm")
        self._model = os.environ.get("DISTLLM_MODEL_NAME", "distributed-llm")

    def _get_client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self._api_base,
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=120.0,
        )

    def validate_credentials(self, model: str, credentials: dict) -> bool:
        """Validate that the DistLLM API is reachable."""
        try:
            with self._get_client() as client:
                resp = client.get("/health")
                return resp.status_code == 200
        except Exception:
            return False

    def invoke(
        self,
        model: str,
        credentials: dict,
        model_parameters: dict,
        prompt_messages: list[dict],
        model_kwargs: dict | None = None,
        stop: list[str] | None = None,
        stream: bool = False,
        user: str | None = None,
    ) -> dict | Generator:
        """Invoke the DistLLM API for chat or completion."""
        model_kwargs = model_kwargs or {}

        # Build request
        messages = []
        for msg in prompt_messages:
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            })

        request_body = {
            "model": model or self._model,
            "messages": messages,
            "temperature": model_parameters.get("temperature", 0.7),
            "top_p": model_parameters.get("top_p", 0.9),
            "max_tokens": model_parameters.get("max_tokens", 256),
            "stream": stream,
        }

        if stop:
            request_body["stop"] = stop
        if user:
            request_body["user"] = user

        if stream:
            return self._invoke_stream(request_body)
        return self._invoke_sync(request_body)

    def _invoke_sync(self, body: dict) -> dict:
        """Synchronous invocation."""
        with self._get_client() as client:
            resp = client.post("/v1/chat/completions", json=body)
            resp.raise_for_status()
            data = resp.json()

            return {
                "result": {
                    "message": {
                        "role": "assistant",
                        "content": data["choices"][0]["message"]["content"],
                    },
                    "finish_reason": data["choices"][0].get("finish_reason", "stop"),
                    "usage": {
                        "prompt_tokens": data.get("usage", {}).get("prompt_tokens", 0),
                        "completion_tokens": data.get("usage", {}).get("completion_tokens", 0),
                        "total_tokens": data.get("usage", {}).get("total_tokens", 0),
                    },
                },
            }

    def _invoke_stream(self, body: dict) -> Generator[dict, None, None]:
        """Streaming invocation."""
        with self._get_client() as client:
            with client.stream("POST", "/v1/chat/completions", json=body) as resp:
                resp.raise_for_status()
                buffer = ""
                for chunk in resp.iter_text():
                    buffer += chunk
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            import json
                            data = json.loads(data_str)
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield {
                                    "result": {
                                        "message": {
                                            "role": "assistant",
                                            "content": content,
                                        },
                                        "finish_reason": None,
                                    },
                                }
                        except (json.JSONDecodeError, KeyError):
                            continue

    def get_models(self) -> list[dict]:
        """Return available models."""
        try:
            with self._get_client() as client:
                resp = client.get("/v1/models")
                resp.raise_for_status()
                data = resp.json()
                return [
                    {"model": m["id"], "label": m["id"]}
                    for m in data.get("data", [])
                ]
        except Exception:
            return [{"model": self._model, "label": self._model}]


# Export for Dify plugin system
provider = DistLLMProvider()
