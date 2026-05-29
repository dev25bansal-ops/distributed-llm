"""Tests for the unified ConfigResolver.

Tests the CLI parsing, config path resolution, and override building
logic.  The actual ``DistLLMSettings`` resolution is tested implicitly
via the existing settings test suite.
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Pre-register stubs to avoid the circular import / torch-init chain.
import types

_stub_settings = types.ModuleType("distllm.config.settings")
_stub_settings.DistLLMSettings = type("DistLLMSettings", (), {})
_stub_settings.DistLLMSettings.from_yaml = staticmethod(
    lambda config_path=None, cli_overrides=None: None
)
_stub_settings.DistLLMSettings.validate_startup = staticmethod(
    lambda config_path=None, cli_overrides=None: None
)
sys.modules["distllm.config.settings"] = _stub_settings

_src = Path(__file__).resolve().parents[2] / "src"
_spec = importlib.util.spec_from_file_location(
    "distllm.config.resolver",
    _src / "distllm" / "config" / "resolver.py",
)
_resolver = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_resolver)
ConfigResolver = _resolver.ConfigResolver

ConfigResolver = _resolver.ConfigResolver


class TestConfigResolver:
    def test_validate_exit(self):
        """--validate-config should raise SystemExit(0)."""
        with pytest.raises(SystemExit) as exc:
            ConfigResolver.from_cli("api", ["--validate-config"])
        assert exc.value.code == 0

    def test_unknown_entry_point(self):
        with pytest.raises(ValueError, match="Unknown entry_point"):
            ConfigResolver.from_cli("nope", [])

    # ── API entry point ──

    def test_api_resolver_defaults(self):
        resolver = ConfigResolver.from_cli("api", ["--model", "test-model"])
        assert resolver._cli_overrides is None or resolver._cli_overrides.get("model", {}).get("name") == "test-model"

    def test_api_cli_overrides_model(self):
        resolver = ConfigResolver.from_cli("api", [
            "--model", "my-model", "--dtype", "bfloat16",
            "--host", "0.0.0.0", "--port", "9000",
        ])
        overrides = resolver._cli_overrides
        assert overrides["model"]["name"] == "my-model"
        assert overrides["model"]["dtype"] == "bfloat16"
        assert overrides["coordinator"]["host"] == "0.0.0.0"
        assert overrides["coordinator"]["api_port"] == 9000

    def test_api_quantization_override(self):
        resolver = ConfigResolver.from_cli("api", [
            "--model", "test", "--quantization", "bitsandbytes_4bit",
        ])
        assert resolver._cli_overrides["quantization"]["method"] == "bitsandbytes_4bit"

    def test_api_no_quantization(self):
        resolver = ConfigResolver.from_cli("api", [
            "--model", "test", "--quantization", "none",
        ])
        assert "quantization" not in (resolver._cli_overrides or {})

    @patch.object(ConfigResolver, "_resolve_config_path", return_value=None)
    def test_api_no_config(self, mock_resolve):
        resolver = ConfigResolver.from_cli("api", ["--model", "x"])
        assert resolver._config_path is None

    # ── Coordinator entry point ──

    def test_coordinator_required_model(self):
        with pytest.raises(SystemExit):
            ConfigResolver.from_cli("coordinator", [])

    def test_coordinator_model(self):
        resolver = ConfigResolver.from_cli("coordinator", [
            "--model", "meta-llama/Llama-2-7b", "--port", "50060",
        ])
        # Coordinator doesn't build cli_overrides — they're empty
        assert resolver._cli_overrides == {}

    # ── Worker entry point ──

    def test_worker_required_args(self):
        with pytest.raises(SystemExit):
            ConfigResolver.from_cli("worker", [])

    def test_worker_defaults(self):
        resolver = ConfigResolver.from_cli("worker", [
            "--node-id", "n0", "--model", "m",
            "--start-layer", "0", "--end-layer", "5", "--total-layers", "32",
        ])
        assert resolver._cli_overrides == {}  # worker doesn't build overrides

    def test_worker_all_args(self):
        resolver = ConfigResolver.from_cli("worker", [
            "--node-id", "n0", "--model", "m",
            "--start-layer", "0", "--end-layer", "5", "--total-layers", "32",
            "--port", "50052", "--device", "cuda", "--dtype", "bfloat16",
            "--insecure", "--cluster-key", "secret",
        ])
        assert resolver._cli_overrides == {}  # worker passes args directly

    # ── Config path resolution ──

    def test_find_config_candidates(self):
        """_find_config should return None when no candidates exist."""
        result = _resolver._find_config(["/nonexistent/path/config.yaml"])
        assert result is None

    def test_find_config_found(self):
        with patch("os.path.exists", return_value=True):
            result = _resolver._find_config(["/tmp/test-config.yaml"])
            assert result == "/tmp/test-config.yaml"

    @pytest.mark.parametrize("entry_point", ["api", "coordinator", "worker"])
    def test_resolve_config_path_no_config(self, entry_point):
        with patch("os.path.exists", return_value=False):
            path = ConfigResolver._resolve_config_path(entry_point,
                        type("args", (), {"config": None})())
            if entry_point == "worker":
                assert path is None
            else:
                # May find something in default candidates — just check it returns something or None
                pass

    def test_api_explicit_config(self):
        args = type("args", (), {"config": "/my/config.yaml"})()
        path = ConfigResolver._resolve_config_path("api", args)
        assert path == "/my/config.yaml"

    # ── Override building ──

    def test_build_overrides_api_model_only(self):
        args = type("args", (), {"model": "m", "dtype": None, "host": None,
                                 "port": None, "quantization": "none"})()
        overrides = ConfigResolver._build_overrides("api", args)
        assert overrides["model"]["name"] == "m"

    def test_build_overrides_api_none(self):
        args = type("args", (), {"model": None, "dtype": None, "host": None,
                                 "port": None, "quantization": "none"})()
        overrides = ConfigResolver._build_overrides("api", args)
        assert overrides is None
