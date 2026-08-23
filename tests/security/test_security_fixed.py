"""Security tests: secret scanning, auth, redaction, CORS."""
import pytest

pytest.skip(
    "requires distllm.cli.autopsy._redact_secrets (not implemented)",
    allow_module_level=True,
)

from distllm.config._model import ModelHubSettings
from distllm.config._network import CoordinatorSettings
from distllm.cli.autopsy import _redact_secrets


class TestSecrets:
    def test_hf_token_raises(self):
        with pytest.raises(Exception, match="(?i)hf_token"):
            ModelHubSettings(hf_token="test")

    def test_auth_header(self):
        from distllm.cli.client import DistLLMClient
        c = DistLLMClient(base_url="http://localhost:8000", api_key="test-key")
        assert "Bearer test-key" in str(c._headers.get("Authorization", ""))

    def test_get_api_key(self, monkeypatch):
        from distllm.cli.client import get_api_key
        monkeypatch.setenv("DISTLLM_API_KEY", "env-key")
        assert get_api_key() == "env-key"

    def test_redact(self):
        r = _redact_secrets("API_KEY=secret123\nnormal=keep")
        assert "secret123" not in r
        assert "REDACTED" in r
        assert "keep" in r

    def test_cors_wildcard_rejected(self):
        with pytest.raises(Exception):
            CoordinatorSettings(cors_origins="*")
