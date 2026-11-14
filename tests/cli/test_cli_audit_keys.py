"""Regression: ``distllm audit`` / cli-doctor read the same key locations as the server.

Audit finding: ``check_api_key_store`` looked for ``~/.distllm/api_key`` while
the server and clients read the cwd-relative ``.distllm/api_key`` (or
``DISTLLM_KEY_FILE``), so it reported "generated" for a deployment running off
the real key file; ``check_auth`` read only ``API_KEY`` and missed
``DISTLLM_API_KEY`` and the persisted file.
"""

from __future__ import annotations

import pytest

from distllm.cli.audit import check_api_key_store, check_auth


@pytest.fixture(autouse=True)
def _isolated_key_env(monkeypatch, tmp_path):
    for var in ("API_KEYS", "API_KEYS_FILE", "API_KEY", "DISTLLM_API_KEY", "DISTLLM_KEY_FILE"):
        monkeypatch.delenv(var, raising=False)
    key_file = tmp_path / "api_key"
    monkeypatch.setenv("DISTLLM_KEY_FILE", str(key_file))
    return key_file


class TestCheckApiKeyStore:
    def test_reports_env_when_distllm_api_key_set(self, monkeypatch):
        monkeypatch.setenv("DISTLLM_API_KEY", "sk-doc")

        check = check_api_key_store()

        assert "env" in check.detail
        assert "DISTLLM_API_KEY" in check.detail

    def test_reports_env_when_api_key_set(self, monkeypatch):
        monkeypatch.setenv("API_KEY", "sk-legacy")

        check = check_api_key_store()

        assert "env" in check.detail

    def test_reports_file_at_the_path_the_server_reads(self, _isolated_key_env):
        """The bug: this used to look at ~/.distllm/api_key, not DISTLLM_KEY_FILE."""
        key_file = _isolated_key_env
        key_file.write_text("sk-on-disk\n", encoding="utf-8")

        check = check_api_key_store()

        assert "file" in check.detail
        assert str(key_file) in check.detail

    def test_reports_generated_only_when_nothing_is_configured(self):
        check = check_api_key_store()

        assert "generated" in check.detail

    def test_uses_distllm_key_file_override(self, tmp_path, monkeypatch):
        custom = tmp_path / "custom_key"
        custom.write_text("sk-custom\n", encoding="utf-8")
        monkeypatch.setenv("DISTLLM_KEY_FILE", str(custom))

        check = check_api_key_store()

        assert "file" in check.detail
        assert str(custom) in check.detail


class TestCheckAuth:
    def test_passes_with_distllm_api_key(self, monkeypatch):
        monkeypatch.setenv("DISTLLM_API_KEY", "sk-doc")

        check = check_auth()

        assert check.status == "PASS"
        assert "DISTLLM_API_KEY" in check.detail

    def test_passes_with_persisted_key_file(self, _isolated_key_env):
        _isolated_key_env.write_text("sk-on-disk\n", encoding="utf-8")

        check = check_auth()

        assert check.status == "PASS"
        assert str(_isolated_key_env) in check.detail

    def test_warns_when_nothing_configured(self):
        check = check_auth()

        assert check.status == "WARN"
        assert "DISTLLM_API_KEY" in check.remediation
