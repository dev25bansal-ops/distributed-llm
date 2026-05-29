from typing import Any, Iterator, Optional

from distllm.sdk import DistLLMClient, DistLLMClientSync


class DistLLMCrewLLM:
    model: str = "distributed-llm"
    base_url: str = "http://localhost:8000"
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    timeout: float = 120.0

    def __init__(self, **kwargs: Any):
        self.model = kwargs.pop("model", "distributed-llm")
        self.base_url = kwargs.pop("base_url", "http://localhost:8000")
        self.api_key = kwargs.pop("api_key", None)
        self.temperature = kwargs.pop("temperature", 0.7)
        self.max_tokens = kwargs.pop("max_tokens", None)
        self.timeout = kwargs.pop("timeout", 120.0)
        self._client = DistLLMClientSync(
            base_url=self.base_url,
            api_key=self.api_key or None,
            timeout=self.timeout,
        )

    def generate_response(self, messages: list[dict], **kwargs: Any) -> str:
        resp = self._client.chat_completions(
            messages=messages,
            model=kwargs.pop("model", self.model),
            temperature=kwargs.pop("temperature", self.temperature),
            max_tokens=kwargs.pop("max_tokens", self.max_tokens) or 256,
        )
        if isinstance(resp, dict):
            return resp.get("choices", [{}])[0].get("message", {}).get("content", "")
        return resp.choices[0].message.content if resp.choices else ""

    def generate_stream(self, messages: list[dict], **kwargs: Any) -> Iterator[str]:
        for chunk in self._client.chat_completions_stream(
            messages=messages,
            model=kwargs.pop("model", self.model),
            temperature=kwargs.pop("temperature", self.temperature),
            max_tokens=kwargs.pop("max_tokens", self.max_tokens) or 256,
            stream=True,
        ):
            delta = chunk if isinstance(chunk, dict) else {}
            content = delta.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if content:
                yield content

    @property
    def model_name(self) -> str:
        return self.model
