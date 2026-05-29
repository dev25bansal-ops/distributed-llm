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
