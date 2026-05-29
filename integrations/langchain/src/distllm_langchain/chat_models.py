from typing import Any, AsyncIterator, Iterator, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from distllm.sdk import DistLLMClient, DistLLMClientSync
from distllm.sdk.types import ChatCompletionResponse


def _convert_message_to_dict(message: BaseMessage) -> dict[str, Any]:
    if isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content}
    if isinstance(message, AIMessage):
        d: dict = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            d["tool_calls"] = [
                {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": tc["args"]}}
                for tc in message.tool_calls
            ]
        return d
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    if isinstance(message, ToolMessage):
        return {"role": "tool", "content": message.content, "tool_call_id": message.tool_call_id}
    if isinstance(message, ChatMessage):
        return {"role": message.role, "content": message.content}
    return {"role": "user", "content": str(message.content)}


def _convert_dict_to_message(data: dict) -> BaseMessage:
    role = data.get("role", "assistant")
    content = data.get("content", "")
    if role == "assistant":
        additional_kwargs: dict = {}
        tool_calls = data.get("tool_calls")
        if tool_calls:
            additional_kwargs["tool_calls"] = tool_calls
        return AIMessage(content=content, additional_kwargs=additional_kwargs)
    return ChatMessage(role=role, content=content)


class DistLLMChat(BaseChatModel):
    model: str = "distributed-llm"
    base_url: str = "http://localhost:8000"
    api_key: Optional[str] = None
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: Optional[int] = None
    timeout: float = 120.0

    _client: DistLLMClientSync = None
    _async_client: DistLLMClient = None

    class Config:
        extra = "allow"

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
    def _llm_type(self) -> str:
        return "distllm-chat"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = self._build_payload(messages, stop, kwargs)
        resp = self._client.chat_completions(**payload)
        return self._to_chat_result(resp)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = self._build_payload(messages, stop, kwargs)
        resp_obj = await self._async_client.chat_completions(**payload)
        if isinstance(resp_obj, ChatCompletionResponse):
            data = {
                "id": resp_obj.id,
                "model": resp_obj.model,
                "choices": [
                    {
                        "index": c.index,
                        "message": {
                            "role": c.message.role if c.message else "assistant",
                            "content": c.message.content if c.message else "",
                        },
                        "finish_reason": c.finish_reason,
                    }
                    for c in resp_obj.choices
                ],
                "usage": {
                    "prompt_tokens": resp_obj.usage.prompt_tokens if resp_obj.usage else 0,
                    "completion_tokens": resp_obj.usage.completion_tokens if resp_obj.usage else 0,
                },
            }
            return self._to_chat_result(data)
        return self._to_chat_result(resp_obj)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        payload = self._build_payload(messages, stop, kwargs)
        for chunk in self._client.chat_completions_stream(**payload, stream=True):
            delta = chunk if isinstance(chunk, dict) else {}
            content = delta.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if not content:
                continue
            chunk_message = AIMessageChunk(content=content)
            gen_chunk = ChatGenerationChunk(message=chunk_message)
            if run_manager:
                run_manager.on_llm_new_token(content, chunk=gen_chunk)
            yield gen_chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        payload = self._build_payload(messages, stop, kwargs)
        async for chunk in self._async_client.chat_completions_stream(**payload, stream=True):
            delta = chunk if isinstance(chunk, dict) else {}
            content = delta.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if not content:
                continue
            chunk_message = AIMessageChunk(content=content)
            gen_chunk = ChatGenerationChunk(message=chunk_message)
            if run_manager:
                run_manager.on_llm_new_token(content, chunk=gen_chunk)
            yield gen_chunk

    def _build_payload(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]],
        kwargs: dict,
    ) -> dict:
        raw_messages = [_convert_message_to_dict(m) for m in messages]
        payload = {
            "messages": raw_messages,
            "model": kwargs.pop("model", self.model),
            "temperature": kwargs.pop("temperature", self.temperature),
            "top_p": kwargs.pop("top_p", self.top_p),
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens) or 256,
        }
        if stop:
            payload["stop"] = stop
        payload.update(kwargs)
        return payload

    @staticmethod
    def _to_chat_result(resp: Any) -> ChatResult:
        if isinstance(resp, dict):
            choices = resp.get("choices", [])
        else:
            choices = getattr(resp, "choices", [])
        generations = []
        for choice in choices:
            if isinstance(choice, dict):
                msg_data = choice.get("message", choice)
            else:
                msg_data = choice
            message = _convert_dict_to_message(msg_data if isinstance(msg_data, dict) else {"role": "assistant", "content": msg_data.content if msg_data else ""})
            generations.append(ChatGeneration(message=message))
        return ChatResult(generations=generations)
