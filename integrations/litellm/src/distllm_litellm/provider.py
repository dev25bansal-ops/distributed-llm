"""LiteLLM custom provider for DistLLM.

Registers DistLLM as a custom OpenAI-compatible backend in LiteLLM.
Since DistLLM exposes an OpenAI-compatible API, this provider simply
wraps the base URL and forwards requests.

Usage::

    import litellm
    from distllm_litellm import get_distllm_custom_llm

    # Register the provider
    get_distllm_custom_llm()

    # Use it
    response = litellm.completion(
        model="distllm/distributed-llm",
        messages=[{"role": "user", "content": "Hello!"}],
        api_base="http://localhost:8000/v1",
    )
"""

from __future__ import annotations

import os
from typing import Any, Optional

_DEFAULT_BASE_URL = os.environ.get("DISTLLM_API_BASE", "http://localhost:8000/v1")


def get_distllm_custom_llm() -> Any:
    """Register and return the DistLLM LiteLLM provider.

    Call this once at startup to make ``distllm/`` model prefix available
    in all LiteLLM calls.
    """
    import litellm
    from litellm import CustomLLM
    from litellm.types.utils import ModelResponse

    class DistLLMLiteLLM(CustomLLM):
        """DistLLM custom LLM backend for LiteLLM."""

        def completion(
            self,
            model: str,
            messages: list[dict],
            api_base: Optional[str] = None,
            api_key: Optional[str] = None,
            **kwargs: Any,
        ) -> ModelResponse:
            """Synchronous completion via DistLLM's OpenAI-compatible API."""
            import httpx

            base = api_base or _DEFAULT_BASE_URL
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            # Strip the "distllm/" prefix if present
            model_name = model.removeprefix("distllm/")

            body: dict[str, Any] = {
                "model": model_name,
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "max_tokens": kwargs.get("max_tokens", 256),
                "stream": False,
            }
            if kwargs.get("stop"):
                body["stop"] = kwargs["stop"]

            with httpx.Client(timeout=kwargs.get("timeout", 120.0)) as client:
                resp = client.post(f"{base}/chat/completions", json=body, headers=headers)
                resp.raise_for_status()
                return ModelResponse(**resp.json())

        async def acompletion(
            self,
            model: str,
            messages: list[dict],
            api_base: Optional[str] = None,
            api_key: Optional[str] = None,
            **kwargs: Any,
        ) -> ModelResponse:
            """Async completion via DistLLM's OpenAI-compatible API."""
            import httpx

            base = api_base or _DEFAULT_BASE_URL
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            model_name = model.removeprefix("distllm/")

            body: dict[str, Any] = {
                "model": model_name,
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "max_tokens": kwargs.get("max_tokens", 256),
                "stream": False,
            }
            if kwargs.get("stop"):
                body["stop"] = kwargs["stop"]

            async with httpx.AsyncClient(timeout=kwargs.get("timeout", 120.0)) as client:
                resp = await client.post(f"{base}/chat/completions", json=body, headers=headers)
                resp.raise_for_status()
                return ModelResponse(**resp.json())

        def embedding(self, model: str, input: list[str], **kwargs: Any) -> Any:
            """Embedding via DistLLM."""
            import httpx

            base = kwargs.get("api_base") or _DEFAULT_BASE_URL
            model_name = model.removeprefix("distllm/")

            with httpx.Client(timeout=kwargs.get("timeout", 120.0)) as client:
                resp = client.post(
                    f"{base}/embeddings",
                    json={"model": model_name, "input": input},
                    headers={"Content-Type": "application/json"},
                )
                resp.raise_for_status()
                return resp.json()

    # Register with LiteLLM
    distllm_provider = DistLLMLiteLLM()
    litellm.custom_provider_map = [
        {"provider": "distllm", "custom_handler": distllm_provider}
    ]

    return distllm_provider
