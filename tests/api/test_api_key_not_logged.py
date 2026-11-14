"""Regression: a generated API key must never be written to the logger.

P0 finding: ``api/middleware.py`` `_get_or_generate_api_key` logged the FULL
generated key via ``logger.warning`` at every startup, persisting the credential
to disk logs/aggregators/CI.  The fix logs only a fingerprint and persists the
full key to an owner-only (0600) file (``DISTLLM_KEY_FILE`` or
``.distllm/api_key``) — never to stdout, never to the logger.
"""

from __future__ import annotations

import os
import stat

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

    def error(self, msg, *args, **kwargs) -> None:
        self.messages.append(str(msg))


def test_generated_key_not_logged_in_cleartext(monkeypatch, capsys, tmp_path):
    from distllm.api import middleware

    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("API_KEY_WAS_SET", raising=False)
    monkeypatch.setenv("DISTLLM_KEY_FILE", str(tmp_path / "api_key"))

    fake = _FakeLogger()
    monkeypatch.setattr(middleware, "logger", fake)

    key = middleware._get_or_generate_api_key()
    assert key, "a key must be generated"

    joined_log = "\n".join(fake.messages)
    # The full key must never reach the logger (logs/aggregators/CI).
    assert key not in joined_log
    # A fingerprint IS logged so operators can recognise the key.
    assert "fingerprint" in joined_log

    # SEC-API-06: the full key must never hit stdout/stderr either — CI and
    # job captures archive both. Only a fingerprint may surface, and it goes
    # to the logger, not the console.
    captured = capsys.readouterr()
    assert key not in captured.out, "full API key leaked to stdout"
    assert key not in captured.err, "full API key leaked to stderr"
    assert key[:8] not in captured.out, "key body leaked to stdout"
    # Fingerprint pattern (prefix + '...') is logged, not printed raw.
    assert f"{key[:8]}..." in joined_log
    # The key was persisted for the operator instead of being printed.
    assert tmp_path.joinpath("api_key").read_text().strip() == key


def test_generated_key_written_to_owner_only_file(monkeypatch, tmp_path):
    """The generated key lands in DISTLLM_KEY_FILE with 0600 permissions."""
    from distllm.api import middleware

    key_file = tmp_path / "nested" / "api_key"
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("API_KEY_WAS_SET", raising=False)
    monkeypatch.setenv("DISTLLM_KEY_FILE", str(key_file))

    fake = _FakeLogger()
    monkeypatch.setattr(middleware, "logger", fake)

    key = middleware._get_or_generate_api_key()
    assert key
    assert key_file.exists(), "key file must be created (incl. parent dirs)"
    assert key_file.read_text().strip() == key

    if os.name == "posix":
        mode = stat.S_IMODE(key_file.stat().st_mode)
        assert mode == 0o600, f"key file must be owner-only, got {oct(mode)}"


def test_key_file_permissions_tightened_when_pre_existing(monkeypatch, tmp_path):
    """A pre-existing too-open key file is re-tightened to 0600 (POSIX)."""
    if os.name != "posix":
        pytest.skip("chmod semantics are POSIX-only")

    from distllm.api import middleware

    key_file = tmp_path / "api_key"
    key_file.write_text("old-secret\n")
    key_file.chmod(0o644)

    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("API_KEY_WAS_SET", raising=False)
    monkeypatch.setenv("DISTLLM_KEY_FILE", str(key_file))

    monkeypatch.setattr(middleware, "logger", _FakeLogger())
    key = middleware._get_or_generate_api_key()
    assert key
    assert key_file.read_text().strip() == key
    assert stat.S_IMODE(key_file.stat().st_mode) == 0o600


def test_persistence_failure_does_not_raise_or_print_key(monkeypatch, capsys, tmp_path):
    """An unwritable key file is logged as an error — without the key."""
    from distllm.api import middleware

    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("API_KEY_WAS_SET", raising=False)
    # Point at a path whose parent is a regular file -> makedirs fails
    # deterministically on every platform.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory\n")
    monkeypatch.setenv("DISTLLM_KEY_FILE", str(blocker / "api_key"))

    fake = _FakeLogger()
    monkeypatch.setattr(middleware, "logger", fake)

    key = middleware._get_or_generate_api_key()
    assert key, "generation must survive a persistence failure"

    captured = capsys.readouterr()
    assert key not in captured.out and key not in captured.err
    assert key not in "\n".join(fake.messages)
    assert any("Could not persist" in m for m in fake.messages)


def test_configured_key_not_regenerated_or_logged(monkeypatch):
    from distllm.api import middleware

    monkeypatch.setenv("API_KEY", "preconfigured-test-key")
    monkeypatch.delenv("API_KEY_WAS_SET", raising=False)

    fake = _FakeLogger()
    monkeypatch.setattr(middleware, "logger", fake)

    key = middleware._get_or_generate_api_key()
    assert key == "preconfigured-test-key"
    assert fake.messages == [], "configured keys should not log anything"
