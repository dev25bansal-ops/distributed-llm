"""Semantic Kernel chat completion service backed by DistLLM."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Optional

import httpx
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.streaming_chat_message_content import StreamingChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole


class DistLLMChatCompletion(ChatCompletionClientBase):
    """Semantic Kernel chat completion backed by DistLLM's OpenAI-compatible API.

    Usage::

        import semantic_kernel as sk
        from distllm_sk import DistLLMChatCompletion

        kernel = sk.Kernel()
        chat = DistLLMChatCompletion(
            model_id="distributed-llm",
            base_url="http://localhost:8000/v1",
        )
        kernel.add_service(chat)
    """

    def __init__(
        self,
        model_id: str = "distributed-llm",
        base_url: str = "http://localhost:8000/v1",
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        **kwargs: Any,
    ):
        super().__init__(ai_model_id=model_id, **kwargs)
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key or "distllm"
        self._timeout = timeout

    def _to_openai_messages(self, chat_history: ChatHistory) -> list[dict]:
        messages = []
        for msg in chat_history.messages:
            role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            messages.append({"role": role, "content": msg.content or ""})
        return messages

    async def get_chat_message_contents(
        self,
        chat_history: ChatHistory,
        settings: Any,
        **kwargs: Any,
    ) -> list[ChatMessageContent]:
        messages = self._to_openai_messages(chat_history)
        body = {
            "model": self.ai_model_id,
            "messages": messages,
            "temperature": getattr(settings, "temperature", 0.7),
            "top_p": getattr(settings, "top_p", 0.9),
            "max_tokens": getattr(settings, "max_tokens", 256),
            "stream": False,
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions",
                json=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        content = data["choices"][0]["message"]["content"]
        return [
            ChatMessageContent(
                role=AuthorRole.ASSISTANT,
                content=content,
                ai_model_id=self.ai_model_id,
            )
        ]

    async def get_streaming_chat_message_contents(
        self,
        chat_history: ChatHistory,
        settings: Any,
        **kwargs: Any,
    ) -> AsyncIterator[StreamingChatMessageContent]:
        messages = self._to_openai_messages(chat_history)
        body = {
            "model": self.ai_model_id,
            "messages": messages,
            "temperature": getattr(settings, "temperature", 0.7),
            "top_p": getattr(settings, "top_p", 0.9),
            "max_tokens": getattr(settings, "max_tokens", 256),
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                json=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                },
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield StreamingChatMessageContent(
                                role=AuthorRole.ASSISTANT,
                                content=content,
                                ai_model_id=self.ai_model_id,
                            )
                    except (json.JSONDecodeError, KeyError):
                        continue
