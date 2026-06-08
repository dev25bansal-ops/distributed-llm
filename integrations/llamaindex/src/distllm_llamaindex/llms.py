from typing import Any, AsyncGenerator, Generator, Optional, Sequence

from pydantic import Field, PrivateAttr
from llama_index.core.llms import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    LLM,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.llms.llm import ChatResponseAsyncGen, CompletionResponseAsyncGen

from distllm.sdk import DistLLMClient, DistLLMClientSync
from distllm.sdk.types import ChatCompletionResponse, CompletionResponse as SDKCompletionResponse


_ROLE_MAP = {
    MessageRole.USER: "user",
    MessageRole.ASSISTANT: "assistant",
    MessageRole.SYSTEM: "system",
    MessageRole.TOOL: "tool",
    MessageRole.FUNCTION: "function",
}


def _resolve_max_tokens(value: Optional[int], default: Optional[int], fallback: int = 256) -> int:
    """Resolve max_tokens: explicit value wins, then instance default, then fallback."""
    if value is not None:
        return value
    if default is not None:
        return default
    return fallback


def _to_chat_message(msg: ChatMessage) -> dict[str, Any]:
    role = _ROLE_MAP.get(msg.role, str(msg.role))
    d: dict = {"role": role, "content": msg.content or ""}
    if msg.additional_kwargs:
        d.update(msg.additional_kwargs)
    return d


def _from_chat_response(data: Any) -> ChatMessage:
    if isinstance(data, dict):
        role_str = data.get("role", "assistant")
        content = data.get("content", "")
    else:
        role_str = getattr(data, "role", "assistant")
        content = getattr(data, "content", "")
    role_map = {
        "user": MessageRole.USER,
        "assistant": MessageRole.ASSISTANT,
        "system": MessageRole.SYSTEM,
        "tool": MessageRole.TOOL,
    }
    role = role_map.get(role_str, MessageRole.ASSISTANT)
    return ChatMessage(role=role, content=content or "")


def _from_completion_response(data: Any) -> str:
    if isinstance(data, SDKCompletionResponse):
        return data.choices[0].text if data.choices else ""
    if isinstance(data, dict):
        return data.get("choices", [{}])[0].get("text", "")
    return ""


class DistLLM(LLM):
    model: str = "distributed-llm"
    base_url: str = "http://localhost:8000"
    api_key: Optional[str] = None
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: Optional[int] = None
    timeout: float = 120.0
    context_window: int = 4096
    num_output: int = 256
    is_streaming: bool = True
    model_download_progress: Optional[float] = Field(
        default=None, description="Model download progress (0.0-1.0), None if already cached"
    )
    pipeline_info: Optional[dict[str, Any]] = Field(
        default=None, description="Pipeline parallelism metadata (layers, nodes, etc.)"
    )

    _client: DistLLMClientSync = PrivateAttr(default=None)
    _async_client: DistLLMClient = PrivateAttr(default=None)

    model_config = {"extra": "allow"}

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client = DistLLMClientSync(
            base_url=self.base_url,
            api_key=self.api_key or None,
            timeout=self.timeout,
        )
        self._async_client = DistLLMClient(
            base_url=self.base_url,
            api_key=self.api_key or None,
            timeout=self.timeout,
        )

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self.model,
            context_window=self._get_live_context_window(),
            num_output=self.num_output,
            is_chat_model=True,
            is_function_calling_model=True,
            is_streaming=self.is_streaming,
        )

    def _get_live_context_window(self) -> int:
        """Try to fetch the actual context window from the server."""
        try:
            import httpx

            resp = httpx.get(
                f"{self.base_url}/v1/models",
                headers={"Content-Type": "application/json"},
                timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                for m in data:
                    if m.get("id") == self.model:
                        return m.get("context_window", self.context_window)
        except Exception:
            pass
        return self.context_window

    def get_pipeline_info(self) -> Optional[dict[str, Any]]:
        """Fetch pipeline parallelism info from the server."""
        try:
            import httpx

            resp = httpx.get(
                f"{self.base_url}/health",
                headers={"Content-Type": "application/json"},
                timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("pipeline_info", data.get("pipeline", None))
        except Exception:
            pass
        return self.pipeline_info

    def _build_chat_payload(self, messages: Sequence[ChatMessage], **kwargs: Any) -> dict:
        raw_messages = [_to_chat_message(m) for m in messages]
        payload = {
            "messages": raw_messages,
            "model": kwargs.pop("model", self.model),
            "temperature": kwargs.pop("temperature", self.temperature),
            "top_p": kwargs.pop("top_p", self.top_p),
            "max_tokens": _resolve_max_tokens(kwargs.pop("max_tokens", None), self.max_tokens, self.num_output),
        }
        payload.update(kwargs)
        return payload

    def _build_completion_payload(self, prompt: str, **kwargs: Any) -> dict:
        return {
            "prompt": prompt,
            "model": kwargs.pop("model", self.model),
            "temperature": kwargs.pop("temperature", self.temperature),
            "top_p": kwargs.pop("top_p", self.top_p),
            "max_tokens": _resolve_max_tokens(kwargs.pop("max_tokens", None), self.max_tokens, self.num_output),
        }

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        payload = self._build_chat_payload(messages, **kwargs)
        resp = self._client.chat_completions(**payload)
        return self._to_chat_response(resp)

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        payload = self._build_chat_payload(messages, **kwargs)
        resp_obj = await self._async_client.chat_completions(**payload)
        return self._to_chat_response(resp_obj)

    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> Generator[ChatResponse, None, None]:
        payload = self._build_chat_payload(messages, **kwargs)
        for chunk in self._client.chat_completions_stream(**payload, stream=True):
            delta = chunk if isinstance(chunk, dict) else {}
            content = delta.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if not content:
                continue
            yield ChatResponse(
                message=ChatMessage(role=MessageRole.ASSISTANT, content=content),
                delta=content,
            )

    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseAsyncGen:
        payload = self._build_chat_payload(messages, **kwargs)
        async for chunk in self._async_client.chat_completions_stream(**payload, stream=True):
            delta = chunk if isinstance(chunk, dict) else {}
            content = delta.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if not content:
                continue
            yield ChatResponse(
                message=ChatMessage(role=MessageRole.ASSISTANT, content=content),
                delta=content,
            )

    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        payload = self._build_completion_payload(prompt, **kwargs)
        resp = self._client.completions(**payload)
        text = _from_completion_response(resp)
        return CompletionResponse(text=text, raw=resp)

    async def acomplete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        payload = self._build_completion_payload(prompt, **kwargs)
        resp = await self._async_client.completions(**payload)
        text = _from_completion_response(resp)
        return CompletionResponse(text=text, raw=resp)

    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> Generator[CompletionResponse, None, None]:
        payload = self._build_completion_payload(prompt, **kwargs)
        for chunk in self._client.completions_stream(**payload):
            delta = chunk if isinstance(chunk, dict) else {}
            text = delta.get("choices", [{}])[0].get("text", "")
            if not text:
                continue
            yield CompletionResponse(text=text, delta=text)

    async def astream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseAsyncGen:
        payload = self._build_completion_payload(prompt, **kwargs)
        async for chunk in self._async_client.completions_stream(**payload):
            delta = chunk if isinstance(chunk, dict) else {}
            text = delta.get("choices", [{}])[0].get("text", "")
            if not text:
                continue
            yield CompletionResponse(text=text, delta=text)

    @staticmethod
    def _to_chat_response(resp: Any) -> ChatResponse:
        if isinstance(resp, ChatCompletionResponse):
            choices = resp.choices
        elif isinstance(resp, dict):
            choices = resp.get("choices", [])
        else:
            choices = getattr(resp, "choices", [])
        if choices:
            choice = choices[0]
            if isinstance(choice, dict):
                msg_data = choice.get("message", choice)
            else:
                msg_data = getattr(choice, "message", choice)
            message = _from_chat_response(msg_data)
        else:
            message = ChatMessage(role=MessageRole.ASSISTANT, content="")
        return ChatResponse(message=message)
