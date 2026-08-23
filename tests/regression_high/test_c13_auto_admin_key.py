"""Regression tests for HIGH fix C13: auto plaintext admin key.

Previously, when no API keys were configured, the store silently generated a
random in-memory admin key with no way for the operator to retrieve it — a
silent auth hole. Now auto-generation is opt-in via
``DISTLLM_ALLOW_AUTO_ADMIN_KEY=1``; otherwise ``load_keys`` fails loudly.
"""

from __future__ import annotations

import importlib

import pytest

import distllm.core.api_key_store as aks


def _clear_env(monkeypatch):
    for var in ("API_KEYS", "API_KEYS_FILE", "API_KEY", "DISTLLM_ALLOW_AUTO_ADMIN_KEY"):
        monkeypatch.delenv(var, raising=False)


def test_no_keys_and_auto_disabled_raises(monkeypatch):
    _clear_env(monkeypatch)
    store = aks.ApiKeyStore.__new__(aks.ApiKeyStore)
    store._keys = []
    store._auto_generated_key = None
    with pytest.raises(RuntimeError):
        store._load()


def test_auto_key_generated_when_opt_in(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("DISTLLM_ALLOW_AUTO_ADMIN_KEY", "1")
    store = aks.ApiKeyStore.__new__(aks.ApiKeyStore)
    store._keys = []
    store._auto_generated_key = None
    store._load()
    assert store._auto_generated_key is not None
    assert any(k.role == "admin" for k in store._keys)
