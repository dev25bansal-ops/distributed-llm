"""Regression: batch chat must honour the whole conversation, not just the
last user message.

Defect: ``_process_chat`` scanned ``messages`` in reverse for the first
``user`` entry and used only that string as the prompt.  The system prompt and
all prior turns were dropped, and only ``max_tokens``/``temperature`` were
forwarded — ``top_p``, ``top_k`` and ``stop`` were silently ignored.  A batch
of multi-turn conversations therefore produced answers with no context.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from distllm.api.routes import batch as batch_mod


class _RecordingCoordinator:
    """Captures every ``generate`` call so the prompt can be inspected."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.tokenizer = None

    def generate(self, prompt, max_new_tokens=128, temperature=0.7, top_p=0.9,
                 top_k=0, **kwargs):
        self.calls.append(
            {
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                **kwargs,
            }
        )
        return "answer"


def _run(coro):
    return asyncio.run(coro)


def test_system_prompt_is_preserved():
    coord = _RecordingCoordinator()
    body = {
        "model": "m",
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "What is 2+2?"},
        ],
        "max_tokens": 32,
    }
    _run(batch_mod._process_chat(body, coord))

    assert coord.calls, "coordinator.generate was never called"
    prompt = coord.calls[0]["prompt"]
    assert "You are a terse assistant." in prompt, prompt
    assert "What is 2+2?" in prompt, prompt


def test_multi_turn_history_is_preserved():
    coord = _RecordingCoordinator()
    body = {
        "model": "m",
        "messages": [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "FIRST-USER-TURN"},
            {"role": "assistant", "content": "FIRST-ASSISTANT-REPLY"},
            {"role": "user", "content": "SECOND-USER-TURN"},
        ],
        "max_tokens": 16,
    }
    _run(batch_mod._process_chat(body, coord))

    prompt = coord.calls[0]["prompt"]
    for marker in ("SYS", "FIRST-USER-TURN", "FIRST-ASSISTANT-REPLY", "SECOND-USER-TURN"):
        assert marker in prompt, (marker, prompt)
    # Order must be preserved.
    assert prompt.index("FIRST-USER-TURN") < prompt.index("SECOND-USER-TURN"), prompt


def test_sampling_params_are_forwarded():
    coord = _RecordingCoordinator()
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 24,
        "temperature": 0.2,
        "top_p": 0.5,
        "top_k": 7,
        "stop": ["\n\n"],
    }
    _run(batch_mod._process_chat(body, coord))

    call = coord.calls[0]
    assert call["max_new_tokens"] == 24
    assert call["temperature"] == 0.2
    assert call["top_p"] == 0.5
    assert call["top_k"] == 7
    # ``stop`` has no corresponding ``Coordinator.generate`` parameter, so it
    # is deliberately not asserted here: forwarding it would be a no-op the
    # backend cannot honour, and a test claiming otherwise would encode a
    # contract that does not exist.


def test_multimodal_content_is_stringified_not_dropped():
    coord = _RecordingCoordinator()
    body = {
        "model": "m",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            }
        ],
        "max_tokens": 8,
    }
    _run(batch_mod._process_chat(body, coord))

    prompt = coord.calls[0]["prompt"]
    assert "describe this" in prompt, prompt


def test_no_user_message_still_errors():
    coord = _RecordingCoordinator()
    body = {
        "model": "m",
        "messages": [{"role": "system", "content": "only a system prompt"}],
        "max_tokens": 8,
    }
    with pytest.raises(ValueError):
        _run(batch_mod._process_chat(body, coord))
    assert not coord.calls