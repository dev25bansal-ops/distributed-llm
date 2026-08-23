"""Regression test: LogRedactor must redact the ApiKeyStore's own token_urlsafe
keys (which have no 'sk-' prefix and no api_key= context), so they cannot leak
into logs."""

from __future__ import annotations

import secrets

from distllm.security.log_redaction import LogRedactor


class TestLogRedactionLongTokens:
    def test_token_urlsafe_key_is_redacted(self):
        """A secrets.token_urlsafe(48) token (the store's auto-gen format)
        must be redacted even without an 'sk-' prefix / api_key= context."""
        key = secrets.token_urlsafe(48)
        result = LogRedactor.redact(f"issued key={key} for tenant A")
        assert key not in result
        assert "[REDACTED]" in result

    def test_short_random_string_left_alone(self):
        """Short strings and prose are not over-redacted."""
        result = LogRedactor.redact("hello world this is a normal sentence")
        assert result == "hello world this is a normal sentence"

    def test_sk_prefix_key_still_redacted(self):
        result = LogRedactor.redact("sk-abcdef1234567890abcdef1234567890abcdef12")
        assert result == "[REDACTED]" or "sk-" not in result