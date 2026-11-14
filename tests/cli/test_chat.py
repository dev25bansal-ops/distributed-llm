"""Tests: CLI chat command — message handling, conversation state."""

from unittest.mock import MagicMock, patch, call

import httpx
import pytest

from rich.console import Console


class TestRunChat:
    def test_chat_quit_command(self):
        """'quit' exits the chat loop."""
        from distllm.cli.chat import run_chat

        console = Console(quiet=True)
        with patch("builtins.input", side_effect=["quit"]):
            run_chat("model", "localhost", 8000, 128, 0.7, console)

    def test_chat_exit_command(self):
        """'exit' exits the chat loop."""
        from distllm.cli.chat import run_chat

        console = Console(quiet=True)
        with patch("builtins.input", side_effect=["exit"]):
            run_chat("model", "localhost", 8000, 128, 0.7, console)

    def test_chat_keyboard_interrupt(self):
        """Ctrl+C exits gracefully."""
        from distllm.cli.chat import run_chat

        console = Console(quiet=True)
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            run_chat("model", "localhost", 8000, 128, 0.7, console)

    def test_chat_eof_error(self):
        """EOF exits gracefully."""
        from distllm.cli.chat import run_chat

        console = Console(quiet=True)
        with patch("builtins.input", side_effect=EOFError):
            run_chat("model", "localhost", 8000, 128, 0.7, console)

    def test_chat_clear_resets_conversation(self):
        """'clear' resets the conversation history."""
        from distllm.cli.chat import run_chat

        console = Console(quiet=True)
        inputs = iter(["Hello", "clear", "quit"])
        with patch("builtins.input", side_effect=lambda _: next(inputs)):
            with patch("httpx.Client") as mock_client_cls:
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.json.return_value = {
                    "choices": [{"message": {"content": "Hi!"}}],
                    "usage": {"completion_tokens": 2},
                }
                mock_resp.raise_for_status = MagicMock()

                mock_client = MagicMock()
                mock_client.post.return_value = mock_resp
                mock_client.__enter__ = MagicMock(return_value=mock_client)
                mock_client.__exit__ = MagicMock(return_value=False)
                mock_client_cls.return_value = mock_client

                run_chat("model", "localhost", 8000, 128, 0.7, console)

    def test_chat_empty_prompt_skipped(self):
        """Empty prompts are skipped."""
        from distllm.cli.chat import run_chat

        console = Console(quiet=True)
        inputs = iter(["", "  ", "quit"])
        with patch("builtins.input", side_effect=lambda _: next(inputs)):
            run_chat("model", "localhost", 8000, 128, 0.7, console)

    def test_chat_http_error_handled(self):
        """HTTP errors are displayed, not crashed."""
        from distllm.cli.chat import run_chat

        console = Console(quiet=True)
        inputs = iter(["Hello", "quit"])
        with patch("builtins.input", side_effect=lambda _: next(inputs)):
            with patch("httpx.Client") as mock_client_cls:
                mock_client = MagicMock()
                mock_client.post.side_effect = httpx.HTTPStatusError(
                    "bad", request=MagicMock(), response=MagicMock(status_code=500)
                )
                mock_client.__enter__ = MagicMock(return_value=mock_client)
                mock_client.__exit__ = MagicMock(return_value=False)
                mock_client_cls.return_value = mock_client

                run_chat("model", "localhost", 8000, 128, 0.7, console)

    def test_chat_displays_usage_stats(self):
        """Usage stats are displayed when available."""
        from distllm.cli.chat import run_chat

        console = Console(quiet=True)
        inputs = iter(["Hello", "quit"])
        with patch("builtins.input", side_effect=lambda _: next(inputs)):
            with patch("httpx.Client") as mock_client_cls:
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.json.return_value = {
                    "choices": [{"message": {"content": "Hi there!"}}],
                    "usage": {"completion_tokens": 3},
                    "generation_time": 0.5,
                }
                mock_resp.raise_for_status = MagicMock()

                mock_client = MagicMock()
                mock_client.post.return_value = mock_resp
                mock_client.__enter__ = MagicMock(return_value=mock_client)
                mock_client.__exit__ = MagicMock(return_value=False)
                mock_client_cls.return_value = mock_client

                run_chat("model", "localhost", 8000, 128, 0.7, console)


# ---------------------------------------------------------------------------
# chat --stream: the streaming path must authenticate like every other command
# ---------------------------------------------------------------------------


class TestChatStreamAuth:
    """Regression: ``distllm chat --stream`` sent NO Authorization header.

    Audit finding: ``_stream_chat`` built a bare ``httpx.AsyncClient`` and the
    module never referenced ``auth_headers``/``resolve_api_key`` anywhere, so
    streaming chat 401'd against every server (auth is always required) while
    the non-streaming path worked because it uses ``client_for()``.
    """

    @pytest.fixture(autouse=True)
    def _isolated_key_env(self, tmp_path, monkeypatch):
        for var in ("API_KEY", "DISTLLM_API_KEY", "DISTLLM_KEY_FILE"):
            monkeypatch.delenv(var, raising=False)
        self.key_file = tmp_path / "api_key"
        monkeypatch.setenv("DISTLLM_KEY_FILE", str(self.key_file))

    def _capture_client_headers(self, monkeypatch) -> dict:
        """Run one streaming turn against a recording client; return headers."""
        import asyncio

        from distllm.cli import chat_stream

        captured: dict = {}

        class _NullResponse:
            def raise_for_status(self):
                pass

            async def aiter_lines(self):
                return
                yield  # pragma: no cover — makes the method an async generator

        class _NullStream:
            async def __aenter__(self):
                return _NullResponse()

            async def __aexit__(self, *exc_info):
                return False

        class _RecordingAsyncClient:
            def __init__(self, **kwargs):
                captured["headers"] = kwargs.get("headers")

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc_info):
                return False

            def stream(self, method, url, **kwargs):
                return _NullStream()

        monkeypatch.setattr(chat_stream.httpx, "AsyncClient", _RecordingAsyncClient)

        state = chat_stream.ChatState(messages=[{"role": "user", "content": "hi"}])

        async def _drive():
            async for _ in chat_stream._stream_chat(state):
                pass

        asyncio.run(_drive())
        return captured

    def test_streaming_sends_the_key_file_bearer(self, monkeypatch):
        """A cluster started by `distllm cluster start` keeps its key only in
        the key file; the streaming client must send it."""
        self.key_file.write_text("sk-from-file\n", encoding="utf-8")

        captured = self._capture_client_headers(monkeypatch)

        assert captured["headers"].get("Authorization") == "Bearer sk-from-file"

    def test_streaming_prefers_env_over_file(self, monkeypatch):
        self.key_file.write_text("sk-from-file\n", encoding="utf-8")
        monkeypatch.setenv("API_KEY", "sk-from-env")

        captured = self._capture_client_headers(monkeypatch)

        assert captured["headers"].get("Authorization") == "Bearer sk-from-env"

    def test_streaming_without_any_key_sends_no_header(self, monkeypatch):
        captured = self._capture_client_headers(monkeypatch)

        assert captured["headers"] == {}
