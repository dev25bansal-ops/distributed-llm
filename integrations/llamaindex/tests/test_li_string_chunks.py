"""String-chunk streaming tests: the SDK's completions_stream now yields text
STRINGS (matching chat_completions_stream's contract). These tests mirror the
existing dict-mocked streaming tests but feed string chunks, proving both item
types work through every adapter stream method."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ── LangChain ────────────────────────────────────────────────────────────────


class TestLlamaindexStringChunks:
    def _llm(self):
        from distllm_langchain import DistLLMLangchain  # type: ignore[attr-defined]

        return DistLLMLangchain(model="test", base_url="http://localhost:8000")

    def test_stream_string_chunks(self):
        llm = self._llm()
        chunks = ["Hello", " world"]
        with patch.object(llm._client, "completions_stream", return_value=iter(chunks)):
            out = list(llm._stream("hi"))
        assert [c.text for c in out] == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_astream_string_chunks(self):
        llm = self._llm()
        chunks = ["Hello", " world"]

        async def _gen():
            for c in chunks:
                yield c

        with patch.object(llm._async_client, "completions_stream", return_value=_gen()):
            out = [c async for c in llm._astream("hi")]
        assert [c.text for c in out] == ["Hello", " world"]

    def test_stream_dict_chunks_still_work(self):
        """Backward compat: dict chunks (older SDKs/mocks) still parse."""
        llm = self._llm()
        chunks = [{"choices": [{"text": "A"}]}, {"choices": [{"text": "B"}]}]
        with patch.object(llm._client, "completions_stream", return_value=iter(chunks)):
            out = list(llm._stream("hi"))
        assert [c.text for c in out] == ["A", "B"]


# ── LlamaIndex ───────────────────────────────────────────────────────────────


class TestLlamaindexStringChunks:
    def _llm(self):
        from distllm_llamaindex.llms import DistLLM

        return DistLLM(model="test", base_url="http://localhost:8000")

    def test_stream_complete_string_chunks(self):
        llm = self._llm()
        chunks = ["Once", " upon"]
        with patch.object(llm._client, "completions_stream", return_value=iter(chunks)):
            out = list(llm.stream_complete("hi"))
        assert [c.delta for c in out] == ["Once", " upon"]
        assert "".join(c.text for c in out) == "Once upon"

    @pytest.mark.asyncio
    async def test_astream_complete_string_chunks(self):
        llm = self._llm()
        chunks = ["Once", " upon"]

        async def _gen():
            for c in chunks:
                yield c

        with patch.object(llm._async_client, "completions_stream", return_value=_gen()):
            out = []
            async for c in llm.astream_complete("hi"):
                out.append(c)
        assert [c.delta for c in out] == ["Once", " upon"]

    def test_stream_complete_dict_chunks_still_work(self):
        llm = self._llm()
        chunks = [{"choices": [{"text": "A"}]}, {"choices": [{"text": "B"}]}]
        with patch.object(llm._client, "completions_stream", return_value=iter(chunks)):
            out = list(llm.stream_complete("hi"))
        assert "".join(c.text for c in out) == "AB"

    def test_stream_chat_string_chunks(self):
        """chat_completions_stream also yields strings post-fix."""
        from llama_index.core.llms import ChatMessage, MessageRole

        llm = self._llm()
        chunks = ["Hi", " there"]
        with patch.object(
            llm._client, "chat_completions_stream", return_value=iter(chunks)
        ):
            out = list(llm.stream_chat([ChatMessage(role=MessageRole.USER, content="q")]))
        assert [c.delta for c in out] == ["Hi", " there"]

    @pytest.mark.asyncio
    async def test_astream_chat_string_chunks(self):
        from llama_index.core.llms import ChatMessage, MessageRole

        llm = self._llm()
        chunks = ["Hi", " there"]

        async def _gen():
            for c in chunks:
                yield c

        with patch.object(
            llm._async_client, "chat_completions_stream", return_value=_gen()
        ):
            out = []
            async for c in llm.astream_chat([ChatMessage(role=MessageRole.USER, content="q")]):
                out.append(c)
        assert [c.delta for c in out] == ["Hi", " there"]

    def test_stream_chat_dict_chunks_still_work(self):
        from llama_index.core.llms import ChatMessage, MessageRole

        llm = self._llm()
        chunks = [
            {"choices": [{"delta": {"content": "old"}, "finish_reason": None}]},
        ]
        with patch.object(
            llm._client, "chat_completions_stream", return_value=iter(chunks)
        ):
            out = list(llm.stream_chat([ChatMessage(role=MessageRole.USER, content="q")]))
        assert [c.delta for c in out] == ["old"]
