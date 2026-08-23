"""DSPy adapter for DistLLM — programmatic LLM programming.

Allows using DistLLM as a language model backend for DSPy
(Declarative Self-improving Python).  Enables DSPy's programming-
not-prompting workflow with a distributed inference backend.

Usage::

    import dspy
    from distllm_sdk.dspy_adapter import DistLLM

    # Use as the default LM in DSPy programs
    dspy.settings.configure(lm=DistLLM(model="llama-3-70b"))

    # Then use DSPy normally:
    class MyModule(dspy.Module):
        ...

Or use the adapter directly::

    lm = DistLLM(model="llama-3-70b", base_url="http://localhost:8000")
    response = lm("What is the capital of France?")
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("distllm_sdk")


class DistLLM:
    """DSPy-compatible language model backed by a DistLLM cluster.

    Implements the interface that DSPy's ``dspy.LM`` expects:
    ``__call__(prompt)`` returns a list of generated texts.
    """

    def __init__(
        self,
        model: str = "distributed-llm",
        base_url: str = "http://localhost:8000",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        top_p: float = 0.9,
        timeout: float = 120.0,
        **kwargs: Any,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self._timeout = timeout
        self._kwargs = kwargs

        import httpx
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
        )

        # DSPy integration
        self._dspy_available = False
        self.history: list[dict[str, Any]] = []

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def __call__(self, prompt: str, **kwargs: Any) -> list[str]:
        """Call the model with a prompt (DSPy-compatible interface).

        Args:
            prompt: The prompt string.

        Returns:
            List of generated text strings.
        """
        messages = self._parse_prompt(prompt)

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "top_p": kwargs.get("top_p", self.top_p),
            **self._kwargs,
        }

        resp = self._client.post("/v1/chat/completions", json=payload, headers=self._headers())
        resp.raise_for_status()
        data = resp.json()

        choices = data.get("choices", [])
        texts = [c.get("message", {}).get("content", "") for c in choices]

        # Record history for DSPy
        self.history.append({
            "prompt": prompt,
            "response": data,
            "timestamp": __import__("time").time(),
            "usage": data.get("usage", {}),
        })

        if not texts:
            return [""]
        return texts

    def _parse_prompt(self, prompt: str) -> list[dict[str, str]]:
        """Parse a prompt string into messages.

        Handles DSPy's conversation format where multiple turns
        are separated by ``\\n\\n`` with role prefixes.
        """
        messages: list[dict[str, str]] = []

        if "\n\n" not in prompt:
            messages.append({"role": "user", "content": prompt})
            return messages

        segments = prompt.split("\n\n")
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
            if segment.startswith("System:"):
                messages.append({"role": "system", "content": segment[7:].strip()})
            elif segment.startswith("User:"):
                messages.append({"role": "user", "content": segment[5:].strip()})
            elif segment.startswith("Assistant:"):
                messages.append({"role": "assistant", "content": segment[10:].strip()})
            else:
                messages.append({"role": "user", "content": segment})

        if not messages:
            messages.append({"role": "user", "content": prompt})

        return messages

    def inspect_history(self, n: int = 1) -> None:
        """Print the last *n* history entries (DSPy-compatible debug helper)."""
        for entry in self.history[-n:]:
            print(f"Prompt: {entry['prompt'][:200]}...")
            print(f"Response: {entry['response']}")
            print()

    # -- Context manager -----------------------------------------------------

    def __enter__(self) -> "DistLLM":
        return self

    def __exit__(self, *args: Any) -> None:
        self._client.close()
