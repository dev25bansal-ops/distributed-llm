from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator, Iterator, Optional, Type

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
from pydantic import BaseModel, Field, PrivateAttr

from distllm.sdk import DistLLMClient, DistLLMClientSync
from distllm.sdk.types import ChatCompletionResponse


# ---------------------------------------------------------------------------
# Message conversion helpers
# ---------------------------------------------------------------------------


def _convert_message_to_dict(message: BaseMessage) -> dict[str, Any]:
    if isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content}
    if isinstance(message, AIMessage):
        d: dict = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["args"]},
                }
                for tc in message.tool_calls
            ]
        return d
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    if isinstance(message, ToolMessage):
        return {
            "role": "tool",
            "content": message.content,
            "tool_call_id": message.tool_call_id,
        }
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
    if role == "user":
        return HumanMessage(content=content)
    if role == "system":
        return SystemMessage(content=content)
    if role == "tool":
        return ToolMessage(content=content, tool_call_id=data.get("tool_call_id", ""))
    return ChatMessage(role=role, content=content)


def _resolve_max_tokens(
    value: Optional[int], default: Optional[int], fallback: int = 256
) -> int:
    """Resolve max_tokens: explicit value wins, then instance default, then fallback."""
    if value is not None:
        return value
    if default is not None:
        return default
    return fallback


def _extract_usage(resp: Any) -> dict[str, int]:
    """Extract token usage from a response dict or typed object."""
    if isinstance(resp, dict):
        usage = resp.get("usage") or {}
        return {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
    usage = getattr(resp, "usage", None)
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0),
        "completion_tokens": getattr(usage, "completion_tokens", 0),
        "total_tokens": getattr(usage, "total_tokens", 0),
    }


def _extract_model_name(resp: Any) -> str:
    if isinstance(resp, dict):
        return resp.get("model", "")
    return getattr(resp, "model", "")


# ---------------------------------------------------------------------------
# DistLLMChat
# ---------------------------------------------------------------------------


class DistLLMChat(BaseChatModel):
    """LangChain ChatModel backed by a DistLLM cluster."""

    model: str = "distributed-llm"
    base_url: str = "http://localhost:8000"
    api_key: Optional[str] = None
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: Optional[int] = None
    timeout: float = 120.0
    cache: Optional[bool] = None

    # 4.1 — Federation-aware routing
    federation_strategy: Optional[str] = Field(
        default=None,
        description="Federation routing strategy: 'latency', 'cost', or 'gpu_utilization'",
    )
    preferred_regions: list[str] = Field(
        default_factory=list,
        description="Preferred regions for federation routing",
    )
    spillover_enabled: bool = Field(
        default=True,
        description="Allow request spillover to federated peers when local cluster is busy",
    )

    _client: DistLLMClientSync = PrivateAttr(default=None)
    _async_client: DistLLMClient = PrivateAttr(default=None)
    _cost_tracker: Any = PrivateAttr(default=None)

    class Config:
        extra = "allow"

    def __init__(self, **kwargs: Any) -> None:
        cost_tracker = kwargs.pop("cost_tracker", None)
        super().__init__(**kwargs)
        self._cost_tracker = cost_tracker
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

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def _llm_type(self) -> str:
        return "distllm-chat"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

    # ------------------------------------------------------------------
    # Generate (sync)
    # ------------------------------------------------------------------

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = self._build_payload(messages, stop, kwargs)
        t0 = time.monotonic()
        resp = self._client.chat_completions(**payload)
        elapsed_ms = (time.monotonic() - t0) * 1000

        result = self._to_chat_result(resp)

        # 3.1.1 — Token usage tracking
        usage = _extract_usage(resp)
        result.llm_output = {
            "token_usage": usage,
            "model_name": _extract_model_name(resp),
            "distllm_latency_ms": round(elapsed_ms, 1),
        }

        # 4.4 — Cost tracking
        if self._cost_tracker is not None:
            self._cost_tracker.record(result.llm_output, model=self.model)

        # 3.1.4 — Callback enrichment
        if run_manager and result.generations:
            run_manager.on_llm_end(
                result,
                response=result.generations[0].message,
            )

        return result

    # ------------------------------------------------------------------
    # Generate (async)
    # ------------------------------------------------------------------

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = self._build_payload(messages, stop, kwargs)
        t0 = time.monotonic()
        resp_obj = await self._async_client.chat_completions(**payload)
        elapsed_ms = (time.monotonic() - t0) * 1000

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
            result = self._to_chat_result(data)
            result.llm_output = {
                "token_usage": _extract_usage(data),
                "model_name": data.get("model", ""),
                "distllm_latency_ms": round(elapsed_ms, 1),
            }
        else:
            result = self._to_chat_result(resp_obj)
            result.llm_output = {
                "token_usage": _extract_usage(resp_obj),
                "model_name": _extract_model_name(resp_obj),
                "distllm_latency_ms": round(elapsed_ms, 1),
            }

        return result

    # ------------------------------------------------------------------
    # Stream (sync)
    # ------------------------------------------------------------------

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
            content = (
                delta.get("choices", [{}])[0].get("delta", {}).get("content", "")
            )
            if not content:
                continue
            # 4.2 — Pipeline-aware streaming metadata
            additional: dict[str, Any] = {}
            if "pipeline_stage" in delta:
                additional["pipeline_stage"] = delta["pipeline_stage"]
            if "node_id" in delta:
                additional["node_id"] = delta["node_id"]
            if "latency_ms" in delta:
                additional["latency_ms"] = delta["latency_ms"]
            chunk_message = AIMessageChunk(content=content, additional_kwargs=additional)
            gen_chunk = ChatGenerationChunk(message=chunk_message)
            if run_manager:
                run_manager.on_llm_new_token(content, chunk=gen_chunk)
            yield gen_chunk

    # ------------------------------------------------------------------
    # Stream (async)
    # ------------------------------------------------------------------

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        payload = self._build_payload(messages, stop, kwargs)
        async for chunk in self._async_client.chat_completions_stream(
            **payload, stream=True
        ):
            delta = chunk if isinstance(chunk, dict) else {}
            content = (
                delta.get("choices", [{}])[0].get("delta", {}).get("content", "")
            )
            if not content:
                continue
            # 4.2 — Pipeline-aware streaming metadata
            additional: dict[str, Any] = {}
            if "pipeline_stage" in delta:
                additional["pipeline_stage"] = delta["pipeline_stage"]
            if "node_id" in delta:
                additional["node_id"] = delta["node_id"]
            if "latency_ms" in delta:
                additional["latency_ms"] = delta["latency_ms"]
            chunk_message = AIMessageChunk(content=content, additional_kwargs=additional)
            gen_chunk = ChatGenerationChunk(message=chunk_message)
            if run_manager:
                run_manager.on_llm_new_token(content, chunk=gen_chunk)
            yield gen_chunk

    # ------------------------------------------------------------------
    # 3.1.2 — Structured output
    # ------------------------------------------------------------------

    def with_structured_output(
        self,
        schema: dict[str, Any] | Type[BaseModel],
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Return a wrapper that forces the model to respond with valid JSON
        matching *schema*.

        Accepts either a JSON-Schema dict or a Pydantic model class.
        """
        from langchain_core.output_parsers import JsonOutputParser
        from langchain_core.runnables import RunnableLambda

        _parser = JsonOutputParser()

        def _enforce_schema(messages: list[BaseMessage], **kw: Any) -> AIMessage:
            schema_str = (
                json.dumps(schema) if isinstance(schema, dict) else schema.model_json_schema()
            )
            instruction = (
                f"\n\nYou MUST respond with a single JSON object that conforms "
                f"to this schema:\n```json\n{schema_str}\n```"
            )
            # Append instruction to the last user message
            enriched = list(messages)
            if enriched and isinstance(enriched[-1], HumanMessage):
                enriched[-1] = HumanMessage(
                    content=enriched[-1].content + instruction
                )
            else:
                enriched.append(HumanMessage(content=instruction))

            result = self.invoke(enriched, **kw)
            if include_raw:
                return {"raw": result, "parsed": _parser.invoke(result)}
            return result

        return RunnableLambda(lambda msgs: _enforce_schema(msgs, **kwargs))

    # ------------------------------------------------------------------
    # 3.1.5 — bind_tools
    # ------------------------------------------------------------------

    def bind_tools(
        self,
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        """Bind OpenAI-style tool definitions to the model.

        Each tool should be a dict with ``type``, ``function.name``,
        ``function.description``, and ``function.parameters``.
        """
        from langchain_core.runnables import RunnableBinding

        return RunnableBinding(
            bound=self,
            kwargs={"tools": tools, **kwargs},
        )

    # ------------------------------------------------------------------
    # Payload helpers
    # ------------------------------------------------------------------

    def _build_payload(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]],
        kwargs: dict,
    ) -> dict:
        raw_messages = [_convert_message_to_dict(m) for m in messages]
        payload: dict[str, Any] = {
            "messages": raw_messages,
            "model": kwargs.pop("model", self.model),
            "temperature": kwargs.pop("temperature", self.temperature),
            "top_p": kwargs.pop("top_p", self.top_p),
            "max_tokens": _resolve_max_tokens(
                kwargs.pop("max_tokens", self.max_tokens), self.max_tokens
            ),
        }
        if stop:
            payload["stop"] = stop
        # Forward bound tools if present
        bound_tools = kwargs.pop("tools", None)
        if bound_tools:
            payload["tools"] = bound_tools
        # 4.1 — Federation routing hints
        if self.federation_strategy:
            payload["federation_strategy"] = self.federation_strategy
        if self.preferred_regions:
            payload["preferred_regions"] = self.preferred_regions
        if not self.spillover_enabled:
            payload["spillover_enabled"] = False
        payload.update(kwargs)
        return payload

    # ------------------------------------------------------------------
    # 4.3 — Model sharding visualization
    # ------------------------------------------------------------------

    def get_model_layout(self) -> Optional[dict[str, Any]]:
        """Fetch which layers are on which nodes.

        Returns a dict like::

            {"node1": {"layers": [0,1,2,3], "gpu_memory_used": "12Gi"}, ...}

        Returns ``None`` if the API doesn't expose this information.
        """
        try:
            import httpx

            resp = httpx.get(
                f"{self.base_url}/v1/models/{self.model}/layout",
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return None

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
            message = _convert_dict_to_message(
                msg_data
                if isinstance(msg_data, dict)
                else {
                    "role": "assistant",
                    "content": msg_data.content if msg_data else "",
                }
            )
            generations.append(ChatGeneration(message=message))
        return ChatResult(generations=generations)
