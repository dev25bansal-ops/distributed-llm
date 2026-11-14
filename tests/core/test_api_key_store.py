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
        # DISTLLM_API_KEY is also a single-key source; the "nothing
        # configured" premise requires it absent too.
        os.environ.pop("DISTLLM_API_KEY", None)
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


# ── DISTLLM_API_KEY: the documented v0.4 name ─────────────────────────────


class TestDistllmApiKeyEnvVar:
    """The server must accept the env var the docs tell operators to set.

    Audit finding: the store and the auth middleware read only
    ``API_KEYS`` / ``API_KEYS_FILE`` / ``API_KEY`` while
    docs/SECURITY_HARDENING.md says ``export DISTLLM_API_KEY=sk-...`` —
    following the docs started a server that authenticated a *different*
    auto-generated key, and every client sending the documented value 401'd.
    """

    def _isolated_key_file(self, tmp_path, monkeypatch):
        for var in ("API_KEYS", "API_KEYS_FILE", "API_KEY", "DISTLLM_API_KEY", "DISTLLM_KEY_FILE"):
            monkeypatch.delenv(var, raising=False)
        key_path = tmp_path / "api_key"
        monkeypatch.setenv("DISTLLM_KEY_FILE", str(key_path))
        return key_path

    def test_distllm_api_key_authenticates_as_admin(self, tmp_path, monkeypatch):
        self._isolated_key_file(tmp_path, monkeypatch)
        monkeypatch.setenv("DISTLLM_API_KEY", "sk-documented")

        store = ApiKeyStore()

        assert store.authenticate("sk-documented") == ("distllm-api-key", "admin")

    def test_distllm_api_key_is_not_shadowed_by_auto_generation(self, tmp_path, monkeypatch):
        """The documented env var must win over the auto-generated fallback."""
        key_path = self._isolated_key_file(tmp_path, monkeypatch)
        monkeypatch.setenv("DISTLLM_API_KEY", "sk-documented")

        store = ApiKeyStore()

        assert not key_path.exists(), "an explicitly configured key must not mint a key file"
        assert store.get_key_count() == 1
        assert store.get_display_key() == "sk-documented"

    def test_both_single_key_env_vars_authenticate(self, tmp_path, monkeypatch):
        """A half-migrated deployment must not lock out either client generation.

        Before the fix the store honored only API_KEY=A while every CLI client
        preferred DISTLLM_API_KEY=B, so each command sent a key the server had
        never seen.
        """
        self._isolated_key_file(tmp_path, monkeypatch)
        monkeypatch.setenv("API_KEY", "sk-a")
        monkeypatch.setenv("DISTLLM_API_KEY", "sk-b")

        store = ApiKeyStore()

        assert store.authenticate("sk-a") is not None
        assert store.authenticate("sk-b") is not None
        assert store.authenticate("sk-other") is None

    def test_same_value_in_both_vars_loads_one_key(self, tmp_path, monkeypatch):
        self._isolated_key_file(tmp_path, monkeypatch)
        monkeypatch.setenv("API_KEY", "sk-same")
        monkeypatch.setenv("DISTLLM_API_KEY", "sk-same")

        store = ApiKeyStore()

        assert store.get_key_count() == 1


# ── resolve_api_key: one client-side precedence ────────────────────────────


class TestResolveApiKey:
    """The single client-side resolution rule the CLI/worker/adapters share."""

    @pytest.fixture(autouse=True)
    def _env(self, tmp_path, monkeypatch):
        for var in ("API_KEY", "DISTLLM_API_KEY", "DISTLLM_KEY_FILE"):
            monkeypatch.delenv(var, raising=False)
        self.key_file = tmp_path / "api_key"
        monkeypatch.setenv("DISTLLM_KEY_FILE", str(self.key_file))

    def test_api_key_wins_over_distllm_api_key(self):
        from distllm.core.api_key_store import resolve_api_key

        os.environ["API_KEY"] = "sk-api-key"
        os.environ["DISTLLM_API_KEY"] = "sk-distllm"
        assert resolve_api_key() == "sk-api-key"

    def test_distllm_api_key_wins_over_file(self):
        from distllm.core.api_key_store import resolve_api_key

        self.key_file.write_text("sk-file\n", encoding="utf-8")
        os.environ["DISTLLM_API_KEY"] = "sk-distllm"
        assert resolve_api_key() == "sk-distllm"

    def test_file_is_the_last_resort(self):
        from distllm.core.api_key_store import resolve_api_key

        self.key_file.write_text("sk-file\n", encoding="utf-8")
        assert resolve_api_key() == "sk-file"

    def test_nothing_configured_returns_none(self):
        from distllm.core.api_key_store import resolve_api_key

        assert resolve_api_key() is None

    def test_whitespace_only_values_are_ignored(self):
        from distllm.core.api_key_store import resolve_api_key

        os.environ["API_KEY"] = "   "
        assert resolve_api_key() is None


# ── Cleanup ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _cleanup():
    """Reset store between tests and ensure env is clean."""
    reset_api_key_store()
    yield
    reset_api_key_store()
    # Don't leak test keys into other tests
    for var in ("API_KEYS", "API_KEY", "API_KEYS_FILE", "DISTLLM_API_KEY"):
        os.environ.pop(var, None)
