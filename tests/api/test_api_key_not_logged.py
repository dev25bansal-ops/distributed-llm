"""Regression: a generated API key must never be written to the logger.

P0 finding: ``api/middleware.py`` `_get_or_generate_api_key` logged the FULL
generated key via ``logger.warning`` at every startup, persisting the credential
to disk logs/aggregators/CI.  The fix logs only a fingerprint and prints the
full key once to stdout.
"""

from __future__ import annotations

import pytest


class _FakeLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def warning(self, msg, *args, **kwargs) -> None:
        self.messages.append(str(msg))

    def info(self, *args, **kwargs) -> None:  # noqa: ARG002
        pass

    def debug(self, *args, **kwargs) -> None:  # noqa: ARG002
        pass

    def error(self, *args, **kwargs) -> None:  # noqa: ARG002
        pass


def test_generated_key_not_logged_in_cleartext(monkeypatch, capsys):
    from distllm.api import middleware

    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("API_KEY_WAS_SET", raising=False)

    fake = _FakeLogger()
    monkeypatch.setattr(middleware, "logger", fake)

    key = middleware._get_or_generate_api_key()
    assert key, "a key must be generated"

    joined_log = "\n".join(fake.messages)
    # The full key must never reach the logger (logs/aggregators/CI).
    assert key not in joined_log
    # A fingerprint IS logged so operators can recognise the key.
    assert "fingerprint" in joined_log

    # The full key is surfaced once on stdout for the operator.
    assert key in capsys.readouterr().out


def test_configured_key_not_regenerated_or_logged(monkeypatch):
    from distllm.api import middleware

    monkeypatch.setenv("API_KEY", "preconfigured-test-key")
    monkeypatch.delenv("API_KEY_WAS_SET", raising=False)

    fake = _FakeLogger()
    monkeypatch.setattr(middleware, "logger", fake)

    key = middleware._get_or_generate_api_key()
    assert key == "preconfigured-test-key"
    assert fake.messages == [], "configured keys should not log anything"
