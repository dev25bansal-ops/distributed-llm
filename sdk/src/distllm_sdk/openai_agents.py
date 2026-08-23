"""OpenAI Agents SDK adapter for DistLLM.

Drop-in replacement for the ``openai`` package that routes requests
to a DistLLM cluster while maintaining compatibility with the
OpenAI Agents SDK (``openai.agnts``).

Usage::

    from distllm_sdk.openai_agents import OpenAIAgentsCompat

    # Use with OpenAI Agents SDK
    from agents import Agent, Runner

    client = OpenAIAgentsCompat(base_url="http://localhost:8000")
    agent = Agent(name="Assistant", instructions="You are helpful.")
    result = Runner.run_sync(agent, "Hello!", client=client)
"""

from __future__ import annotations

from typing import Any


class OpenAIAgentsCompat:
    """OpenAI Agents SDK compatibility wrapper for DistLLM.

    Wraps the existing ``openai_compat`` module and adds agent-specific
    features like tool call iteration and structured outputs.

    Usage::

        from distllm_sdk.openai_agents import OpenAIAgentsCompat

        from agents import Agent, Runner
        from agents.models import OpenAIChatCompletionsModel

        client = OpenAIAgentsCompat(base_url="http://localhost:8000")
        model = OpenAIChatCompletionsModel(
            model="distributed-llm",
            openai_client=client.get_inner_client(),
        )
        agent = Agent(name="Assistant", instructions="You are helpful.", model=model)
        result = Runner.run_sync(agent, "Hello!")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        import httpx

        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = max_retries
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=self._headers(),
            timeout=httpx.Timeout(timeout),
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def get_inner_client(self) -> Any:
        """Return an OpenAI-compatible client for use with the Agents SDK.

        Returns an object that mimics ``openai.OpenAI()`` enough for
        the Agents SDK's ``OpenAIChatCompletionsModel``.
        """
        return _AgentsSDKClient(self)


class _AgentsSDKClient:
    """Minimal OpenAI client shape for the Agents SDK.

    Exposes ``client.chat.completions.create(...)`` and
    ``client.chat.completions.create(stream=True)`` compatible
    with what ``openai.agnts`` expects.
    """

    def __init__(self, parent: OpenAIAgentsCompat):
        self.chat = _ChatCompletions(parent)


class _ChatCompletions:
    def __init__(self, parent: OpenAIAgentsCompat):
        self._parent = parent
        self.completions = _Completions(parent)


class _Completions:
    def __init__(self, parent: OpenAIAgentsCompat):
        self._parent = parent

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 256,
        stream: bool = False,
        tools: list[dict] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Create a chat completion compatible with the Agents SDK."""
        import httpx

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools
        payload.update(kwargs)

        headers = self._parent._headers()
        url = f"{self._parent.base_url}/v1/chat/completions"

        if stream:
            return self._stream(url, headers, payload)

        with httpx.Client(timeout=self._parent._timeout) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        # Shape it like an OpenAI response for the Agents SDK
        return _to_openai_chunk(data)

    def _stream(self, url: str, headers: dict, payload: dict) -> Any:
        """Streaming response shaped for the Agents SDK."""
        import httpx
        import json as _json

        class _StreamIterator:
            def __init__(self, url: str, headers: dict, payload: dict, timeout: float):
                self._url = url
                self._headers = headers
                self._payload = payload
                self._timeout = timeout
                self._client = httpx.Client(timeout=timeout)

            def __iter__(self) -> Any:
                return self

            def __next__(self) -> Any:
                # First call sets up the stream
                if not hasattr(self, "_stream"):
                    self._stream = self._client.stream(
                        "POST", self._url, json=self._payload, headers=self._headers
                    )
                    self._iter = self._stream.__enter__()
                    self._line_iter = self._iter.iter_lines()
                try:
                    for line in self._line_iter:
                        if line.startswith("data: "):
                            data = line[6:]
                            if data.strip() == "[DONE]":
                                raise StopIteration
                            try:
                                event = _json.loads(data)
                                return _to_openai_chunk(event)
                            except _json.JSONDecodeError:
                                continue
                    raise StopIteration
                except StopIteration:
                    if hasattr(self, "_stream"):
                        self._stream.__exit__(None, None, None)
                    raise

        return _StreamIterator(url, headers, payload, self._parent._timeout)


def _to_openai_chunk(data: dict) -> Any:
    """Shape a DistLLM response dict to look like an OpenAI response chunk."""
    from types import SimpleNamespace

    choice_data = data.get("choices", [{}])[0]
    msg = choice_data.get("message", choice_data.get("delta", {}))
    usage = data.get("usage", {})

    return SimpleNamespace(
        id=data.get("id", ""),
        model=data.get("model", ""),
        created=data.get("created", 0),
        object=data.get("object", "chat.completion"),
        choices=[
            SimpleNamespace(
                index=choice_data.get("index", 0),
                message=SimpleNamespace(
                    role=msg.get("role", "assistant"),
                    content=msg.get("content", ""),
                    tool_calls=msg.get("tool_calls"),
                ) if not data.get("stream") else None,
                delta=SimpleNamespace(
                    role=msg.get("role"),
                    content=msg.get("content"),
                    tool_calls=msg.get("tool_calls"),
                ) if data.get("stream") else None,
                finish_reason=choice_data.get("finish_reason"),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        ),
    )
