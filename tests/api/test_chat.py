"""Chat completion tests — backward-compatible re-export shim.

All test classes have been split into focused files:

- test_chat_basic.py      — Basic completion, multi-turn, params, response format
- test_chat_streaming.py  — SSE streaming tests
- test_chat_tools.py      — Tool/function calling tests
- test_chat_ssr.py        — SSRF protection tests
- test_chat_adapters.py   — LoRA adapter tests
- test_chat_multimodal.py — Multi-modal (image) input tests

This file re-exports every test class so existing ``pytest test_chat.py``
invocations continue to work unchanged.
"""

# Re-export all test classes for backward compatibility
from tests.api.test_chat_basic import (  # noqa: F401
    TestChatBasic,
    TestChatEmptyPrompt,
    TestChatHybridRouting,
    TestChatLogprobs,
    TestChatMaxTokens,
    TestChatMultiTurn,
    TestChatPriority,
    TestChatResponseFormat,
    TestChatSeed,
    TestChatStopSequences,
    TestChatTemperatureBounds,
    TestChatTopPBounds,
    make_mock_coordinator,
)
from tests.api.test_chat_streaming import TestChatStreaming  # noqa: F401
from tests.api.test_chat_tools import TestChatToolCalling  # noqa: F401
from tests.api.test_chat_ssr import TestChatSSRF  # noqa: F401
from tests.api.test_chat_adapters import TestChatAdapter  # noqa: F401
from tests.api.test_chat_multimodal import TestChatMultiModal  # noqa: F401
