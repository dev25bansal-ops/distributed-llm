"""Regression: an existing auto-generated key file must be ADOPTED, never overwritten.

P0 finding: with no ``API_KEYS`` / ``API_KEYS_FILE`` / ``API_KEY`` configured,
``ApiKeyStore._load`` generated a fresh admin key and rewrote ``DISTLLM_KEY_FILE``
(default ``.distllm/api_key``) unconditionally with ``O_TRUNC``.  A second
process — e.g. another ``distllm cluster start`` run while a server was already
live — therefore clobbered the file the running server had minted: every client
now presented a key the server had never seen, and every request 401'd until
the server was restarted.

Behaviour pinned here:
* an existing non-empty key file is REUSED (hashed into the store, file left
  byte-for-byte untouched);
* fresh generation creates the file atomically with ``O_EXCL`` and 0600;
* a lost creation race adopts the winner's key (hashed into the store);
* adoption is logged with the file path and never the raw key;
* a persistence failure leaves the in-memory key working and unlogged.
"""

from __future__ import annotations

import os
import stat

import pytest

import distllm.core.api_key_store as aks
from distllm.core.api_key_store import ApiKeyStore, reset_api_key_store

KEY_A = "sk-existing-key-A-0123456789"
KEY_WINNER = "sk-race-winner-abcdef"


class _FakeLogger:
    """Captures every level the key store might use (loguru-style call shape)."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def warning(self, msg, *args, **kwargs) -> None:  # noqa: ARG002
        self.messages.append(str(msg))

    def info(self, msg, *args, **kwargs) -> None:  # noqa: ARG002
        self.messages.append(str(msg))

    def error(self, msg, *args, **kwargs) -> None:  # noqa: ARG002
        self.messages.append(str(msg))

    def debug(self, msg, *args, **kwargs) -> None:  # noqa: ARG002
        self.messages.append(str(msg))


@pytest.fixture(autouse=True)
def _isolated_key_env(monkeypatch, tmp_path):
    """Clear every key-source env var and point DISTLLM_KEY_FILE at tmp_path."""
    for var in ("API_KEYS", "API_KEYS_FILE", "API_KEY", "DISTLLM_KEY_FILE"):
        monkeypatch.delenv(var, raising=False)
    key_file = tmp_path / "api_key"
    monkeypatch.setenv("DISTLLM_KEY_FILE", str(key_file))
    reset_api_key_store()
    yield key_file
    reset_api_key_store()


# ── The reported regression: a second store must not rotate the key ────────


def test_second_store_adopts_existing_file_key(_isolated_key_env):
    """Two stores in the same process/env both authenticate the file's key.

    This is the exact regression that 401'd running servers: store 2 used to
    generate a new key and O_TRUNC the file, invalidating store 1's key.
    """
    key_file = _isolated_key_env
    key_file.write_text(KEY_A + "\n", encoding="utf-8")

    store1 = ApiKeyStore()
    assert store1.authenticate(KEY_A) is not None, "first store must adopt the file key"
    assert key_file.read_text(encoding="utf-8").strip() == KEY_A

    store2 = ApiKeyStore()
    assert store2.authenticate(KEY_A) is not None, (
        "second store must adopt the persisted key, not mint a fresh one"
    )
    assert key_file.read_text(encoding="utf-8").strip() == KEY_A, (
        "an existing non-empty key file must never be rewritten"
    )
    assert store2.get_display_key() == KEY_A
    # Adoption looks like a normal auto key: key_id "auto", admin role.
    assert any(k["key_id"] == "auto" and k["role"] == "admin" for k in store2.list_keys())


def test_existing_file_is_never_truncated(_isolated_key_env):
    """Constructing a store over an existing key file leaves its bytes intact."""
    key_file = _isolated_key_env
    key_file.write_text(KEY_A + "\n", encoding="utf-8")
    before = key_file.read_bytes()

    ApiKeyStore()

    assert key_file.read_bytes() == before


# ── Fresh generation ────────────────────────────────────────────────────────


def test_fresh_generation_creates_owner_only_file_matching_store(_isolated_key_env):
    """No file -> generate, persist with 0600, and authenticate what was written."""
    key_file = _isolated_key_env
    assert not key_file.exists()

    store = ApiKeyStore()
    raw = store.get_display_key()
    assert raw, "a key must be generated"

    assert key_file.exists(), "the generated key must be persisted for clients"
    assert key_file.read_text(encoding="utf-8").strip() == raw
    assert store.authenticate(raw) is not None

    if os.name == "posix":
        mode = stat.S_IMODE(key_file.stat().st_mode)
        assert mode == 0o600, f"key file must be owner-only, got {oct(mode)}"


def test_empty_file_is_refilled_with_a_generated_key(_isolated_key_env):
    """An existing but EMPTY file may be replaced (nothing to preserve)."""
    key_file = _isolated_key_env
    key_file.write_text("", encoding="utf-8")

    store = ApiKeyStore()
    raw = store.get_display_key()
    assert raw
    assert key_file.read_text(encoding="utf-8").strip() == raw
    assert store.authenticate(raw) is not None


def test_whitespace_only_file_is_refilled_not_treated_as_a_key(_isolated_key_env):
    """A whitespace-only file holds no key and must be refilled, not adopted.

    ``echo > .distllm/api_key`` leaves a newline on disk. If that read as
    "an existing key", every process would mint a DIFFERENT ephemeral key
    that never reaches the file — clients reading it get whitespace and
    401 forever, the same user-visible failure the adoption logic fixes.
    """
    key_file = _isolated_key_env
    key_file.write_text(" \n\t\n", encoding="utf-8")

    store = ApiKeyStore()
    raw = store.get_display_key()
    assert raw, "a usable key must exist"
    assert key_file.read_text(encoding="utf-8").strip() == raw
    assert store.authenticate(raw) is not None

    # And the refilled key is stable: a second store adopts it.  (Before the
    # fix this failed -- the whitespace file was left untouched and store 2
    # minted an unrelated in-memory key.)
    assert " \n\t\n" not in key_file.read_text(encoding="utf-8")
    store2 = ApiKeyStore()
    assert store2.authenticate(raw) is not None


# ── Creation race ───────────────────────────────────────────────────────────


def test_creation_race_adopts_winner_key(_isolated_key_env, monkeypatch):
    """A FileExistsError from the exclusive create adopts the winner's key."""
    key_file = _isolated_key_env
    real_open = os.open

    def racing_open(path, flags, *args, **kwargs):
        if flags & os.O_EXCL:
            # Another process won the race between our read and our create.
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(KEY_WINNER + "\n")
            raise FileExistsError(path)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(aks.os, "open", racing_open)

    store = ApiKeyStore()
    assert store.authenticate(KEY_WINNER) is not None, (
        "the winner's key — not the discarded generated one — must be hashed into the store"
    )
    assert store.get_display_key() == KEY_WINNER
    assert key_file.read_text(encoding="utf-8").strip() == KEY_WINNER


# ── Logging ─────────────────────────────────────────────────────────────────


def test_adoption_logs_the_file_path_but_never_the_key(_isolated_key_env, monkeypatch):
    key_file = _isolated_key_env
    key_file.write_text(KEY_A + "\n", encoding="utf-8")

    fake = _FakeLogger()
    monkeypatch.setattr(aks, "logger", fake)
    ApiKeyStore()

    joined = "\n".join(fake.messages)
    assert KEY_A not in joined, "the raw key must never reach the logger"
    assert str(key_file) in joined, "adoption must name the file it reused"
    assert "reus" in joined.lower() or "existing" in joined.lower(), (
        "the log must say an existing key was reused, so operators know how to rotate"
    )


def test_persistence_failure_keeps_in_memory_key_and_does_not_log_it(
    _isolated_key_env, monkeypatch, tmp_path
):
    """A write failure must not raise, and the key must not leak to logs."""
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory\n", encoding="utf-8")
    monkeypatch.setenv("DISTLLM_KEY_FILE", str(blocker / "api_key"))

    fake = _FakeLogger()
    monkeypatch.setattr(aks, "logger", fake)

    store = ApiKeyStore()
    raw = store.get_display_key()
    assert raw, "generation must survive a persistence failure"
    assert store.authenticate(raw) is not None
    assert raw not in "\n".join(fake.messages)
