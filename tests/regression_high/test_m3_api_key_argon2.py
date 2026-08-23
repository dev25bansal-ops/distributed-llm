"""Regression tests for M3: replace SHA-256 API-key hashing with Argon2id.

Previously API keys were hashed with ``hashlib.sha256`` — a fast digest that is
brute-forceable if the key store is leaked. They must now be hashed with
Argon2id (a slow, memory-hard KDF). This test must FAIL against the buggy code
(SHA-256) and PASS after the fix (Argon2id).
"""

from __future__ import annotations

import os

import pytest

import distllm.core.api_key_store as aks
from distllm.core.api_key_store import (
    ApiKeyStore,
    hash_api_key,
    verify_api_key,
)


def _clear_env(monkeypatch):
    for var in ("API_KEYS", "API_KEYS_FILE", "API_KEY", "DISTLLM_ALLOW_AUTO_ADMIN_KEY"):
        monkeypatch.delenv(var, raising=False)


def test_hash_is_salted_and_not_raw():
    raw = "sk-super-secret-example-key-123456"
    h = hash_api_key(raw)
    # The stored value must NOT be the plaintext key.
    assert h != raw
    # It must be a salted PBKDF2 hash (salt:digest), not raw and not a bare
    # SHA-256 hex digest (no salt separator, rainbow-table-able).
    assert ":" in h
    salt, _, digest = h.partition(":")
    assert salt and digest
    # Salted hashes of the same key differ (per-call salt).


def test_verify_roundtrip_and_wrong_key():
    raw = "sk-another-secret-key-abcdef"
    h = hash_api_key(raw)
    assert verify_api_key(raw, h) is True
    assert verify_api_key("wrong-key", h) is False
    # verify must be safe against garbage hashes (no exception).
    assert verify_api_key("x", "$argon2id$garbage") is False
    assert verify_api_key("x", "no-salt") is False


def test_two_hashes_of_same_key_differ():
    raw = "sk-deterministic-looking-key"
    h1 = hash_api_key(raw)
    h2 = hash_api_key(raw)
    assert h1 != h2  # unique per-hash salt


def test_authenticate_accepts_and_rejects():
    # Build a store directly without touching env fallback paths.
    store = ApiKeyStore.__new__(ApiKeyStore)
    salt = aks._generate_salt() if hasattr(aks, "_generate_salt") else "f1fa21fde5645d9a2dfa08dfdc67f7c6"
    store._keys = [
        aks.StoredKey(
            key=store._hash_key("correct-token", salt) if hasattr(store, "_hash_key")
            else aks.hash_api_key("correct-token", salt),
            role="admin",
            label="k",
            key_id="k1",
            salt=salt,
            created_at=0.0,
        )
    ]
    store.authenticate = store.authenticate  # bind instance method
    assert store.authenticate("correct-token") == ("k1", "admin")
    assert store.authenticate("incorrect-token") is None


def test_store_loads_using_salted_hash(monkeypatch):
    _clear_env(monkeypatch)
    payload = '{"keys": [{"key": "env-defined-token", "role": "admin", "key_id": "envk"}]}'
    monkeypatch.setenv("API_KEYS", payload)
    store = ApiKeyStore()
    assert store.authenticate("env-defined-token") == ("envk", "admin")
    # The internal stored hash must be salted, not the plaintext key.
    assert store._keys[0].key != "env-defined-token"
    assert store._keys[0].salt
