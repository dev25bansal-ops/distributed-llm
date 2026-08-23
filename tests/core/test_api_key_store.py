"""Tests for the role-based API key authentication store."""

import os
import json

import pytest

from distllm.core.api_key_store import (
    ApiKeyStore,
    get_api_key_store,
    reset_api_key_store,
    role_satisfies,
    VALID_ROLES,
)


# ── role_satisfies ──────────────────────────────────────────────────────────


class TestRoleSatisfies:
    def test_admin_satisfies_admin(self):
        assert role_satisfies("admin", "admin") is True

    def test_admin_satisfies_read_only(self):
        assert role_satisfies("admin", "read-only") is True

    def test_admin_satisfies_inference(self):
        assert role_satisfies("admin", "inference-only") is True

    def test_read_only_satisfies_read_only(self):
        assert role_satisfies("read-only", "read-only") is True

    def test_read_only_does_not_satisfy_admin(self):
        assert role_satisfies("read-only", "admin") is False

    def test_read_only_does_not_satisfy_inference(self):
        assert role_satisfies("read-only", "inference-only") is False

    def test_inference_satisfies_inference(self):
        assert role_satisfies("inference-only", "inference-only") is True

    def test_inference_satisfies_read_only(self):
        assert role_satisfies("inference-only", "read-only") is True

    def test_inference_does_not_satisfy_admin(self):
        assert role_satisfies("inference-only", "admin") is False


# ── API_KEYS env var ────────────────────────────────────────────────────────


class TestApiKeysEnvVar:
    def test_load_multiple_keys(self):
        os.environ["API_KEYS"] = json.dumps({
            "keys": [
                {"key": "sk-admin-1", "role": "admin", "label": "admin-key"},
                {"key": "sk-read-1", "role": "read-only", "label": "reader"},
                {"key": "sk-inf-1", "role": "inference-only", "label": "inference"},
            ]
        })
        reset_api_key_store()
        store = get_api_key_store()
        assert store.get_key_count() == 3

    def test_authenticate_valid_key(self):
        os.environ["API_KEYS"] = json.dumps({
            "keys": [
                {"key": "sk-admin-1", "role": "admin", "label": "admin-key"},
            ]
        })
        reset_api_key_store()
        store = get_api_key_store()
        result = store.authenticate("sk-admin-1")
        assert result is not None
        key_id, role = result
        assert role == "admin"

    def test_authenticate_invalid_key(self):
        os.environ["API_KEYS"] = json.dumps({
            "keys": [
                {"key": "sk-admin-1", "role": "admin", "label": "admin-key"},
            ]
        })
        reset_api_key_store()
        store = get_api_key_store()
        result = store.authenticate("wrong-key")
        assert result is None

    def test_authenticate_constant_time(self):
        """Verify authenticate works for different keys (not timing)."""
        os.environ["API_KEYS"] = json.dumps({
            "keys": [
                {"key": "key-a", "role": "admin", "label": "a"},
                {"key": "key-b", "role": "read-only", "label": "b"},
            ]
        })
        reset_api_key_store()
        store = get_api_key_store()
        assert store.authenticate("key-a")[1] == "admin"
        assert store.authenticate("key-b")[1] == "read-only"
        assert store.authenticate("key-c") is None

    def test_invalid_role_raises_value_error(self):
        os.environ["API_KEYS"] = json.dumps({
            "keys": [
                {"key": "sk-bad", "role": "superadmin", "label": "bad"},
            ]
        })
        reset_api_key_store()
        with pytest.raises(ValueError, match="Invalid API key role"):
            get_api_key_store()

    def test_has_role(self):
        os.environ["API_KEYS"] = json.dumps({
            "keys": [
                {"key": "k1", "role": "admin"},
                {"key": "k2", "role": "read-only"},
            ]
        })
        reset_api_key_store()
        store = get_api_key_store()
        assert store.has_role("admin") is True
        assert store.has_role("read-only") is True
        assert store.has_role("inference-only") is False

    def test_list_keys_excludes_secrets(self):
        os.environ["API_KEYS"] = json.dumps({
            "keys": [
                {"key": "sk-secret", "role": "admin", "label": "my-key"},
            ]
        })
        reset_api_key_store()
        store = get_api_key_store()
        listed = store.list_keys()
        assert len(listed) == 1
        assert listed[0]["role"] == "admin"
        assert listed[0]["label"] == "my-key"
        assert "key" not in listed[0]
        assert "sk-secret" not in str(listed[0])


# ── API_KEY fallback (legacy) ───────────────────────────────────────────────


class TestApiKeyFallback:
    def test_api_key_env_var_admin_role(self):
        os.environ.pop("API_KEYS", None)
        os.environ["API_KEY"] = "legacy-key-123"
        reset_api_key_store()
        store = get_api_key_store()
        assert store.get_key_count() == 1
        result = store.authenticate("legacy-key-123")
        assert result is not None
        assert result[1] == "admin"

    def test_auto_generated_key(self):
        os.environ.pop("API_KEYS", None)
        os.environ.pop("API_KEY", None)
        reset_api_key_store()
        store = get_api_key_store()
        assert store.get_key_count() == 1
        # Can't predict the generated key, but it should exist
        assert store.list_keys()[0]["role"] == "admin"


# ── API_KEYS_FILE ───────────────────────────────────────────────────────────


class TestApiKeysFile:
    def test_load_from_json_file(self, tmp_path):
        f = tmp_path / "api_keys.json"
        f.write_text(json.dumps({
            "keys": [
                {"key": "file-key", "role": "read-only", "label": "file"},
            ]
        }))
        os.environ.pop("API_KEYS", None)
        os.environ["API_KEYS_FILE"] = str(f)
        reset_api_key_store()
        store = get_api_key_store()
        assert store.get_key_count() == 1
        assert store.authenticate("file-key") is not None

    def test_missing_file_falls_back(self, tmp_path):
        os.environ.pop("API_KEYS", None)
        os.environ["API_KEYS_FILE"] = str(tmp_path / "nonexistent.json")
        os.environ["API_KEY"] = "fallback-key"
        reset_api_key_store()
        store = get_api_key_store()
        assert store.get_key_count() == 1
        assert store.authenticate("fallback-key") is not None


# ── Cleanup ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _cleanup():
    """Reset store between tests and ensure env is clean."""
    reset_api_key_store()
    yield
    reset_api_key_store()
    # Don't leak test keys into other tests
    for var in ("API_KEYS", "API_KEY", "API_KEYS_FILE"):
        os.environ.pop(var, None)
