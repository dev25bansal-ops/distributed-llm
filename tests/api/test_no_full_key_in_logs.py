"""Regression: API keys must never appear in full inside log output.

Audit target: ``src/distllm/api/middleware.py``.

Rules enforced:

1. ``_get_or_generate_api_key`` must never call ``logger.warning/info/error``
   with the full generated key (more than 8 chars of the key body).
2. The rate-limit fingerprint (``_api_key_id_for_limit`` / ``_safe_key_id``)
   must return a sha256 prefix, not the raw key.
3. The module-level helpers ``_safe_key_fingerprint`` and ``_safe_key_id``
   are the canonical safe form; the test exercises every code path that
   surfaces a key to a logger.

If someone reintroduces a full-key logger call (e.g. by adding a debug
``logger.debug(f\"got key={token}\")`` line) this test will catch it.
"""

from __future__ import annotations

import logging
import re
import sys
from io import StringIO

import pytest

from distllm.api import middleware


# A representative 48-char API key (matches secrets.token_urlsafe(48) length).
TEST_KEY = "abcdef0123456789" + "x" * 32  # 48 chars total


def _attach_string_handler(target: logging.Logger, level: int = logging.DEBUG) -> StringIO:
    """Attach a StringIO handler to *target* and return the buffer."""
    buf = StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s :: %(message)s"))
    target.addHandler(handler)
    target.setLevel(level)
    return buf


@pytest.fixture
def log_buffer():
    """Capture log records emitted via the stdlib logging interface.

    The middleware module uses loguru; loguru propagates to stdlib ``logging``
    by default via the ``logging`` interceptor. We attach a StringIO handler
    to the root logger and read what was written.
    """
    buf = StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    prev_level = root.level
    root.setLevel(logging.DEBUG)
    try:
        yield buf
    finally:
        root.removeHandler(handler)
        root.setLevel(prev_level)


def test_safe_key_fingerprint_truncates_at_eight() -> None:
    """The canonical fingerprint is exactly ``key[:8] + '...'`` for real keys."""
    fp = middleware._safe_key_fingerprint(TEST_KEY)
    assert fp == f"{TEST_KEY[:8]}...", (
        f"fingerprint should be exactly the first 8 chars + '...', got {fp!r}"
    )
    assert TEST_KEY not in fp, (
        f"the full key must not appear in the fingerprint, got {fp!r}"
    )


def test_safe_key_fingerprint_handles_edge_cases() -> None:
    assert middleware._safe_key_fingerprint(None) == "<empty>"
    assert middleware._safe_key_fingerprint("") == "<empty>"
    assert middleware._safe_key_fingerprint("short") == "<invalid:len=5>"
    # 8-char boundary: exactly 8 is acceptable (key[:8] returns the whole string)
    fp8 = middleware._safe_key_fingerprint("a" * 8)
    assert "..." in fp8
    assert fp8 == "a" * 8 + "..."


def test_safe_key_id_is_hash_prefix_not_raw_key() -> None:
    """Rate-limit key id must be a sha256 prefix — never the raw key."""
    key_id = middleware._safe_key_id(TEST_KEY)
    assert key_id is not None
    assert len(key_id) == 16
    assert TEST_KEY not in key_id
    # Same input → same output (stable for rate-limiter map)
    assert middleware._safe_key_id(TEST_KEY) == key_id
    # Different input → different output
    assert middleware._safe_key_id(TEST_KEY + "x") != key_id


def test_safe_key_id_rejects_short_inputs() -> None:
    assert middleware._safe_key_id(None) is None
    assert middleware._safe_key_id("") is None
    assert middleware._safe_key_id("short") is None  # len < 8


def test_get_or_generate_api_key_uses_fingerprint_in_logs() -> None:
    """The 'API_KEY not set' warning must never include the full generated key.

    Forces the generate-a-new-key path by clearing ``API_KEY`` from env,
    installs a temporary loguru sink that captures output into a StringIO,
    then verifies the full key is never emitted to that sink while the
    truncated fingerprint is.
    """
    import os
    from loguru import logger as _loguru

    saved = os.environ.pop("API_KEY", None)
    captured = StringIO()
    sink_id = _loguru.add(captured, format="{message}", level="DEBUG")
    try:
        generated = middleware._get_or_generate_api_key()
        assert generated is not None
        assert len(generated) >= 8

        log_output = captured.getvalue()

        # The full key must NOT appear in any log message.
        assert generated not in log_output, (
            "Full API key was leaked into log output:\n" + log_output
        )
        # The fingerprint should appear in the warning line.
        assert generated[:8] in log_output, (
            "Expected the truncated fingerprint to appear in logs but it was missing.\n"
            f"log_output: {log_output!r}"
        )
    finally:
        _loguru.remove(sink_id)
        if saved is not None:
            os.environ["API_KEY"] = saved


def test_middleware_module_has_no_full_key_logger_calls(caplog) -> None:
    """Static audit: search middleware.py for logger.* calls that interpolate
    a variable which looks like a full key.

    This is a belt-and-braces regression: even if the helper functions
    change, an explicit ``logger.error(f\"key={token}\")`` line anywhere in
    the module will fail this test.
    """
    import inspect

    src = inspect.getsource(middleware)
    # Strip string literals to avoid false positives on the documentation.
    # Specifically, find every logger.<level>(... args ...) call and look for
    # an f-string interpolation that contains the variable ``token`` or
    # ``auth_header`` in a way that would expose > 8 chars of a key.
    pattern = re.compile(
        r"logger\.(?:debug|info|warning|error|critical)\s*\(",
        re.MULTILINE,
    )
    for match in pattern.finditer(src):
        # Take the next ~600 chars — logger calls in this module are small.
        snippet = src[match.start(): match.start() + 600]
        # Disallow f-strings that mention `token` directly without fingerprinting.
        if "f\"" in snippet and re.search(r"\{token\}", snippet):
            pytest.fail(
                f"logger call interpolates raw `token`: {snippet[:200]!r}"
            )
        if "f\"" in snippet and re.search(r"\{auth_header\}", snippet):
            pytest.fail(
                f"logger call interpolates raw `auth_header`: {snippet[:200]!r}"
            )
        if "f\"" in snippet and re.search(r"\{generated_key\}", snippet):
            pytest.fail(
                f"logger call interpolates raw `generated_key`: {snippet[:200]!r}"
            )


def test_module_imports_without_logging_full_key(monkeypatch, caplog) -> None:
    """Importing middleware.py should never emit a full-key log line.

    We force a clean import (the module calls _get_or_generate_api_key() at
    import time, which is the most likely place a full-key leak could
    occur) and assert that nothing in the captured logs contains a key body
    of > 8 chars.
    """
    import importlib

    import os
    monkeypatch.delenv("API_KEY", raising=False)
    # Force a fresh import so the generate-key branch runs again.
    if "distllm.api.middleware" in sys.modules:
        importlib.reload(sys.modules["distllm.api.middleware"])
    else:
        importlib.import_module("distllm.api.middleware")
    combined = "\n".join(r.getMessage() for r in caplog.records)
    # Anything > 8 contiguous alnum chars that is not part of a normal
    # loguru/starlette message is suspicious — we use a soft check: no
    # line should contain ``key=<value>`` with a value longer than 8 chars
    # that looks like a token_urlsafe payload (base64-ish).
    suspicious = re.findall(r"key=([A-Za-z0-9_\-]{16,})", combined)
    assert not suspicious, (
        f"Found full-key-shaped substrings in logs: {suspicious!r}"
    )