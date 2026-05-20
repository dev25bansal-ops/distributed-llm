"""DistLLM LLM — LlamaIndex LLM implementation.

Wraps the DistLLM SDK client to provide a native LlamaIndex ``LLM``
with sync, async, streaming, and async-streaming chat/complete support.

Usage::

    from distllm_llamaindex import DistLLM

    llm = DistLLM(model="distributed-llm", base_url="http://localhost:8000")
    response = llm.complete("What is distributed inference?")
"""

from typing import Any, AsyncGenerator, Generator, List, Optional, Sequence

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
from distllm.sdk.types import ChatCompletionResponse


_ROLE_MAP = {
    MessageRole.USER: "user",
    MessageRole.ASSISTANT: "assistant",
    MessageRole.SYSTEM: "system",
    MessageRole.TOOL: "tool",
    MessageRole.FUNCTION: "function",
}


def _to_chat_message(msg: ChatMessage) -> dict[str, Any]:
    """Convert a LlamaIndex ChatMessage to the DistLLM API dict format."""
    role = _ROLE_MAP.get(msg.role, str(msg.role))
    d: dict = {"role": role, "content": msg.content or ""}
    if msg.additional_kwargs:
        d.update(msg.additional_kwargs)
    return d


def _from_chat_response(data: Any) -> ChatMessage:
    """Convert a DistLLM API response choice to a LlamaIndex ChatMessage."""
    if isinstance(data, dict):
        role_str = data.get("role", "assistant")
        content = data.get("content", "")
    else:
        role_str = getattr(data, "role", "assistant")
        content = getattr(data, "content", "")
    # Map string role back to MessageRole
    role_map = {
        "user": MessageRole.USER,
        "assistant": MessageRole.ASSISTANT,
        "system": MessageRole.SYSTEM,
        "tool": MessageRole.TOOL,
    }
    role = role_map.get(role_str, MessageRole.ASSISTANT)
    return ChatMessage(role=role, content=content or "")


def _from_completion_response(data: Any) -> str:
    """Extract text from a DistLLM completion response."""
    if isinstance(data, dict):
        return data.get("choices", [{}])[0].get("text", "")
    if data.choices:
        return data.choices[0].text
    return ""


class DistLLM(LLM):
    """LlamaIndex LLM backed by DistLLM's OpenAI-compatible API.

    Wraps the :class:`distllm.sdk.DistLLMClient` for sync/async chat
    completions, text completions, and streaming.

    .. rubric:: Example

    .. code-block:: python

        from distllm_llamaindex import DistLLM

        llm = DistLLM(
            model="distributed-llm",
            base_url="http://localhost:8000",
            temperature=0.7,
        )
        response = llm.complete("Explain distributed inference.")
    """

    model: str = "distributed-llm"
    base_url: str = "http://localhost:8000"
    api_key: Optional[str] = None
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: Optional[int] = None
    timeout: float = 120.0
    context_window: int = 4096
    num_output: int = 256

    _client: DistLLMClientSync = None
    _async_client: DistLLMClient = None

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
            context_window=self.context_window,
            num_output=self.num_output,
            is_chat_model=True,
            is_function_calling_model=True,
        )

    def _build_chat_payload(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> dict:
        """Build the request payload for the DistLLM chat API."""
        raw_messages = [_to_chat_message(m) for m in messages]
        payload = {
            "messages": raw_messages,
            "model": kwargs.pop("model", self.model),
            "temperature": kwargs.pop("temperature", self.temperature),
            "top_p": kwargs.pop("top_p", self.top_p),
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens) or self.num_output,
        }
        payload.update(kwargs)
        return payload

    def _build_completion_payload(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> dict:
        """Build the request payload for the DistLLM text completion API."""
        return {
            "prompt": prompt,
            "model": kwargs.pop("model", self.model),
            "temperature": kwargs.pop("temperature", self.temperature),
            "top_p": kwargs.pop("top_p", self.top_p),
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens) or self.num_output,
        }

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        payload = self._build_chat_payload(messages, **kwargs)
        resp = self._client.chat_completions(**payload)
        return self._to_chat_response(resp)

    async def achat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        payload = self._build_chat_payload(messages, **kwargs)
        resp_obj = await self._async_client.chat_completions(**payload)
        return self._to_chat_response(resp_obj)

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> Generator[ChatResponse, None, None]:
        payload = self._build_chat_payload(messages, **kwargs)
        payload["stream"] = True
        full_content = ""
        for chunk in self._client.chat_completions_stream(**payload):
            delta = chunk if isinstance(chunk, dict) else {}
            content = delta.get("choices", [{}])[0].get("delta", {}).get("content", "")
            finish_reason = delta.get("choices", [{}])[0].get("finish_reason")
            if not content and not finish_reason:
                continue
            full_content += content or ""
            yield ChatResponse(
                message=ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=content or "",
                ),
                delta=content or "",
            )

    async def astream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseAsyncGen:
        payload = self._build_chat_payload(messages, **kwargs)
        payload["stream"] = True
        async for chunk in self._async_client.chat_completions_stream(**payload):
            delta = chunk if isinstance(chunk, dict) else {}
            content = delta.get("choices", [{}])[0].get("delta", {}).get("content", "")
            finish_reason = delta.get("choices", [{}])[0].get("finish_reason")
            if not content and not finish_reason:
                continue
            yield ChatResponse(
                message=ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=content or "",
                ),
                delta=content or "",
            )

    # ------------------------------------------------------------------
    # Completion
    # ------------------------------------------------------------------

    def complete(
        self,
        prompt: str,
        formatted: bool = False,
        **kwargs: Any,
    ) -> CompletionResponse:
        payload = self._build_completion_payload(prompt, **kwargs)
        resp = self._client.completions(**payload)
        text = _from_completion_response(resp)
        return CompletionResponse(text=text, raw=resp)

    async def acomplete(
        self,
        prompt: str,
        formatted: bool = False,
        **kwargs: Any,
    ) -> CompletionResponse:
        payload = self._build_completion_payload(prompt, **kwargs)
        resp = await self._async_client.completions(**payload)
        text = _from_completion_response(resp)
        return CompletionResponse(text=text, raw=resp)

    def stream_complete(
        self,
        prompt: str,
        formatted: bool = False,
        **kwargs: Any,
    ) -> Generator[CompletionResponse, None, None]:
        payload = self._build_completion_payload(prompt, **kwargs)
        resp = self._client.completions(**payload)
        text = _from_completion_response(resp)
        yield CompletionResponse(text=text, raw=resp)

    async def astream_complete(
        self,
        prompt: str,
        formatted: bool = False,
        **kwargs: Any,
    ) -> CompletionResponseAsyncGen:
        payload = self._build_completion_payload(prompt, **kwargs)
        resp = await self._async_client.completions(**payload)
        text = _from_completion_response(resp)
        yield CompletionResponse(text=text, raw=resp)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_chat_response(resp: Any) -> ChatResponse:
        """Convert a DistLLM SDK response to a LlamaIndex ChatResponse."""
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
