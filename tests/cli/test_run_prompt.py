"""Tests for ``system run --prompt`` single-shot mode (defect A-3/5).

Before the fix, ``run_inference`` accepted ``prompt`` but never used it: it
always started uvicorn and never called ``coordinator.generate``. These tests
fail on that old code because they assert generate() is called, the completion
is printed, and uvicorn is NOT started.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console


def _captured_console() -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=200)
    return console, buf


class TestRunPromptSingleShot:
    def test_prompt_generates_prints_and_skips_server(self):
        from distllm.cli.run import run_inference

        console, buf = _captured_console()
        coordinator = MagicMock()
        coordinator.generate.return_value = "Hello from the model!"

        with patch(
            "distllm.cli.run.create_coordinator", return_value=coordinator
        ) as mock_create, patch("distllm.cli.run.uvicorn") as mock_uvicorn:
            run_inference(
                "test-model",
                True,
                "",
                8000,
                "float16",
                64,
                0.3,
                console,
                prompt="hello",
            )

        # generate() was called with the exact prompt and generation params
        coordinator.generate.assert_called_once_with(
            "hello", max_new_tokens=64, temperature=0.3
        )
        # coordinator was created for this model
        mock_create.assert_called_once_with(
            model_name="test-model", dtype="float16", local=True
        )
        # uvicorn was NOT started
        mock_uvicorn.run.assert_not_called()
        # completion text appears in console output
        output = buf.getvalue()
        assert "Hello from the model!" in output
        # and it is labelled so the raw completion is distinguishable
        assert "Completion" in output

    def test_prompt_generate_failure_exits_nonzero(self):
        from distllm.cli.run import run_inference

        console, buf = _captured_console()
        coordinator = MagicMock()
        coordinator.generate.side_effect = RuntimeError("model exploded")

        with patch(
            "distllm.cli.run.create_coordinator", return_value=coordinator
        ), patch("distllm.cli.run.uvicorn") as mock_uvicorn:
            with pytest.raises(SystemExit) as excinfo:
                run_inference(
                    "test-model",
                    True,
                    "",
                    8000,
                    "float16",
                    64,
                    0.3,
                    console,
                    prompt="hello",
                )

        assert excinfo.value.code == 1
        mock_uvicorn.run.assert_not_called()
        output = buf.getvalue()
        assert "generation failed" in output
        assert "model exploded" in output

    def test_prompt_coordinator_load_failure_exits_nonzero(self):
        from distllm.cli.run import run_inference

        console, buf = _captured_console()

        with patch(
            "distllm.cli.run.create_coordinator",
            side_effect=RuntimeError("cannot load weights"),
        ), patch("distllm.cli.run.uvicorn") as mock_uvicorn:
            with pytest.raises(SystemExit) as excinfo:
                run_inference(
                    "test-model",
                    True,
                    "",
                    8000,
                    "float16",
                    64,
                    0.3,
                    console,
                    prompt="hello",
                )

        assert excinfo.value.code == 1
        mock_uvicorn.run.assert_not_called()
        output = buf.getvalue()
        assert "failed to load model/coordinator" in output
        assert "cannot load weights" in output

    def test_prompt_completion_printed_as_literal_text(self):
        """Model output must not be interpreted as Rich markup."""
        from distllm.cli.run import run_inference

        console, buf = _captured_console()
        coordinator = MagicMock()
        coordinator.generate.return_value = "[bold red]literal[/bold red]"

        with patch(
            "distllm.cli.run.create_coordinator", return_value=coordinator
        ), patch("distllm.cli.run.uvicorn"):
            run_inference(
                "test-model",
                True,
                "",
                8000,
                "float16",
                64,
                0.3,
                console,
                prompt="hello",
            )

        assert "[bold red]literal[/bold red]" in buf.getvalue()

    def test_no_prompt_still_starts_server(self):
        """Interactive/server mode is unchanged when --prompt is absent."""
        from distllm.cli.run import run_inference

        console, buf = _captured_console()
        coordinator = MagicMock()

        with patch(
            "distllm.cli.run.create_coordinator", return_value=coordinator
        ), patch("distllm.cli.run.uvicorn") as mock_uvicorn:
            # simulate server shutdown so run_inference returns
            mock_uvicorn.run.side_effect = KeyboardInterrupt()
            run_inference(
                "test-model", True, "", 8000, "float16", 64, 0.3, console
            )

        mock_uvicorn.run.assert_called_once()
        coordinator.generate.assert_not_called()
