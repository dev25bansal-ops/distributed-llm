"""Regression tests for HIGH fix E10: global log-redaction sink.

Previously, redaction was applied PER-CALL via ``LogRedactor.redact(...)`` at
specific instrumented points.  A stray ``logger.info(f"token={raw_token}")``
elsewhere in the codebase (or in a third-party library logging through the
root logger) would still leak the secret.

The fix installs a ``RedactingFilter`` (a ``logging.Filter``) on the root
logger's handlers so that *every* emitted record -- from any logger, including
child and third-party loggers that propagate to root -- is redacted before it
reaches any handler/output.  It is configurable (``DISTLLM_REDACT_PATTERNS``)
and can be disabled for debugging (``DISTLLM_LOG_REDACTION=off``).

NOTE: No real secrets are used in this test -- only obviously-fake tokens.
"""

from __future__ import annotations

import logging
import os
from io import StringIO

import pytest

import distllm.security.log_redaction as lr
from distllm.security.log_redaction import (
    ENV_REDACTION_ENABLED,
    ENV_REDACTION_PATTERNS,
    RedactingFilter,
    install_global_redaction,
)


# Fake (non-real) tokens used to exercise the redaction patterns.
FAKE_OPENAI_KEY = "sk-" + "abcdefghijklmnopqrstuvwx"  # sk- + 24 chars
FAKE_BEARER = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
FAKE_PASSWORD = "supersecretpassword123"
FAKE_LONG_HEX = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4"  # 56 hex


def _configure_with_filter(monkeypatch, *, enabled_env: str | None = None):
    """Reset root logging and attach a capturing handler + RedactingFilter.

    Returns a tuple ``(stream, formatter_used)`` where ``stream.getvalue()``
    is the fully-formatted log output.
    """
    root = logging.getLogger()
    # Clean slate so previous tests do not leak handlers/filters.
    for h in list(root.handlers):
        root.removeHandler(h)
    root.filters = []
    root.setLevel(logging.DEBUG)
    root.propagate = True

    if enabled_env is not None:
        monkeypatch.setenv(ENV_REDACTION_ENABLED, enabled_env)
    else:
        monkeypatch.delenv(ENV_REDACTION_ENABLED, raising=False)

    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(name)s %(levelname)s %(message)s"))

    # Add the handler FIRST, then install the global sink so that the filter is
    # attached to the handler too.  (Child-propagated records are only redacted
    # by handler filters, while root-emitted records use the root-logger filter.)
    root.addHandler(handler)
    install_global_redaction(logger=root, force=True)
    return stream


def test_global_filter_masks_secret_on_plain_logger(monkeypatch):
    """A plain ``logger.info`` with a fake secret must be masked in output."""
    stream = _configure_with_filter(monkeypatch)
    log = logging.getLogger("distllm.test.e10")
    log.info(
        "Fetched credentials key=%s token=%s password=%s",
        FAKE_OPENAI_KEY,
        FAKE_BEARER,
        FAKE_PASSWORD,
    )
    out = stream.getvalue()
    assert FAKE_OPENAI_KEY not in out, f"OpenAI key leaked: {out!r}"
    assert FAKE_PASSWORD not in out, f"Password leaked: {out!r}"
    assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in out, f"Bearer leaked: {out!r}"
    assert "[REDACTED]" in out, "Expected [REDACTED] token in output"


def test_global_filter_covers_child_and_third_party_loggers(monkeypatch):
    """A non-distllm (third-party-style) child logger must also be masked."""
    stream = _configure_with_filter(monkeypatch)
    # Simulate a third-party library logger (no "distllm" prefix) that simply
    # logs through the root logger -- the classic leak path.
    third_party = logging.getLogger("some_external_library.http_client")
    third_party.info("Request signed with Authorization client_secret=%s", FAKE_LONG_HEX)

    other_child = logging.getLogger("requests_oauthlib")
    other_child.warning("refresh_token=%s", FAKE_BEARER)

    out = stream.getvalue()
    assert FAKE_LONG_HEX not in out, f"third-party secret leaked: {out!r}"
    assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in out, f"bearer leaked: {out!r}"
    assert "[REDACTED]" in out, "Expected [REDACTED] token in output"


def test_opt_out_leaves_secret_intact(monkeypatch):
    """With DISTLLM_LOG_REDACTION=off the secret must remain in the output."""
    stream = _configure_with_filter(monkeypatch, enabled_env="off")
    log = logging.getLogger("distllm.test.e10.optout")
    log.info("debug token=%s", FAKE_OPENAI_KEY)
    out = stream.getvalue()
    # Opt-out: redaction disabled, raw secret preserved for debugging.
    assert FAKE_OPENAI_KEY in out, f"Opt-out should keep secret, got: {out!r}"
    assert "[REDACTED]" not in out, "Opt-out should not redact"


def test_filter_is_idempotent_and_enabled_by_default(monkeypatch):
    """install_global_redaction adds exactly one filter and is on by default."""
    monkeypatch.delenv(ENV_REDACTION_ENABLED, raising=False)
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.filters = []

    handler = logging.StreamHandler(StringIO())
    root.addHandler(handler)

    f1 = install_global_redaction(logger=root)
    f2 = install_global_redaction(logger=root)  # idempotent
    assert f1 is f2 or isinstance(f1, RedactingFilter)
    assert isinstance(f1, RedactingFilter)
    assert f1.enabled is True
    # Only one RedactingFilter on the handler.
    redactors = [f for f in handler.filters if isinstance(f, RedactingFilter)]
    assert len(redactors) == 1


def test_custom_patterns_via_env(monkeypatch):
    """DISTLLM_REDACT_PATTERNS adds extra patterns that get applied."""
    monkeypatch.delenv(ENV_REDACTION_ENABLED, raising=False)
    monkeypatch.setenv(ENV_REDACTION_PATTERNS, r"TOPSECRET-\d+")
    f = RedactingFilter.from_env()
    redacted = f.redact_message("config=TOPSECRET-42 and key=public")
    assert "TOPSECRET-42" not in redacted
    assert "public" in redacted
    assert "[REDACTED]" in redacted


def test_record_structure_preserved(monkeypatch):
    """Redaction masks the secret but keeps the message readable/structured."""
    stream = _configure_with_filter(monkeypatch)
    log = logging.getLogger("distllm.test.e10.struct")
    log.info("User %s authenticated with key %s", "alice", FAKE_OPENAI_KEY)
    out = stream.getvalue()
    assert "alice" in out, "non-secret context should survive redaction"
    assert FAKE_OPENAI_KEY not in out
    assert "[REDACTED]" in out
