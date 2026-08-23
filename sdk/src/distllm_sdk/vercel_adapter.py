"""Vercel AI SDK adapter for DistLLM.

Allows using DistLLM as a provider with the ``@ai-sdk/openai``
package or any OpenAI-compatible Vercel AI SDK integration.

Usage (JS/TS in your Next.js app)::

    import { createOpenAI } from '@ai-sdk/openai';

    const distllm = createOpenAI({
        baseURL: 'http://localhost:8000/v1',
        apiKey: 'optional-api-key',
    });

    const response = await generateText({
        model: distllm('distributed-llm'),
        prompt: 'Hello!',
    });

For Python users, this module provides a helper that matches
the Vercel AI SDK's OpenAI-compatible interface::

    from distllm_sdk.vercel_adapter import VercelAICompat

    client = VercelAICompat(base_url="http://localhost:8000/v1")
    response = client.generate_text(
        model="distributed-llm",
        prompt="Hello!",
    )
"""

from __future__ import annotations

from typing import Any


class VercelAICompat:
    """Python-side compatibility helper for Vercel AI SDK users.

    Provides a ``generate_text`` and ``stream_text`` interface
    that mirrors the Vercel AI SDK's ``generateText`` and ``streamText``
    functions, backed by a DistLLM cluster.

    Usage::

        from distllm_sdk.vercel_adapter import VercelAICompat

        client = VercelAICompat(base_url="http://localhost:8000")
        response = client.generate_text(
            model="llama-3-70b",
            prompt="What is the capital of France?",
        )
        print(response.text)  # "Paris"
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str | None = None,
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    def generate_text(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 256,
        **kwargs: Any,
    ) -> Any:
        """Generate text matching the Vercel AI SDK ``generateText`` interface.

        Returns a response object with ``.text``, ``.usage``, and ``.raw`` attributes.
        """
        import httpx

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }

        with httpx.Client(base_url=self.base_url, timeout=self._timeout) as client:
            resp = client.post("/v1/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        usage = data.get("usage", {})

        class _Response:
            text: str = msg.get("content", "")
            usage: dict = usage
            raw: dict = data

        return _Response()

    def stream_text(self, model: str, prompt: str, **kwargs: Any) -> Any:
        """Stream text matching the Vercel AI SDK ``streamText`` interface.

        Yields text chunks as they arrive from the API.
        """
        import httpx

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 256),
        }

        with httpx.Client(base_url=self.base_url, timeout=self._timeout) as client:
            with client.stream("POST", "/v1/chat/completions", json=payload, headers=headers) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data.strip() == "[DONE]":
                            break
                        import json as _json
                        try:
                            event = _json.loads(data)
                            delta = event.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content")
                            if content:
                                yield content
                        except _json.JSONDecodeError:
                            continue
