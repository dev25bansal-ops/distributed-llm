"""Integration tests: CLI + config + API client + profile merging + config validation."""
from __future__ import annotations
import os
import tempfile
import pytest
from typer.testing import CliRunner
from distllm.cli.main import app
from distllm.config.settings import DistLLMSettings
from distllm.config._model import ModelSettings
from distllm.config._network import CoordinatorSettings, TLSSettings
from distllm.config._generation import GenerationSettings
from distllm.config.profiles import ProfileConfig

runner = CliRunner()


# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------


class _Stub:
    """Lightweight stub: callable, with configurable return values and call tracking."""
    def __init__(self, **kwargs):
        self._rv = None
        self.call_count = 0
        self._call_args_list = []
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __call__(self, *args, **kwargs):
        self.call_count += 1
        self._call_args_list.append((args, kwargs))
        if self._rv is not None:
            if callable(self._rv):
                return self._rv(*args, **kwargs)
            return self._rv
        return self

    @property
    def return_value(self):
        return self._rv

    @return_value.setter
    def return_value(self, val):
        self._rv = val


class TestConfigCLIIntegration:
    def test_config_validate_valid(self):
        orig = DistLLMSettings.validate_startup
        try:
            DistLLMSettings.validate_startup = lambda *a, **kw: DistLLMSettings(model=ModelSettings(name="test"))
            result = runner.invoke(app, ["config", "validate"])
            assert result.exit_code == 0
        finally:
            DistLLMSettings.validate_startup = orig

    def test_config_validate_invalid(self):
        from pydantic import ValidationError
        orig = DistLLMSettings.validate_startup
        try:
            try:
                ModelSettings(name="")
            except ValidationError as e:
                def _raise(*a, **kw):
                    raise e
                DistLLMSettings.validate_startup = _raise
            result = runner.invoke(app, ["config", "validate"])
            assert result.exit_code == 1
        finally:
            DistLLMSettings.validate_startup = orig

    def test_config_reference(self):
        result = runner.invoke(app, ["config", "reference"])
        assert result.exit_code == 0

    def test_config_profile_list(self):
        result = runner.invoke(app, ["config", "profile", "list"])
        assert result.exit_code == 0
        assert "gpu-single" in result.stdout

    def test_config_profile_show(self):
        result = runner.invoke(app, ["config", "profile", "show", "gpu-single"])
        assert result.exit_code == 0
        assert "dtype" in result.stdout

    def test_config_validate_help(self):
        result = runner.invoke(app, ["config", "validate", "--help"])
        assert result.exit_code == 0


class TestConfigPrecedence:
    def test_yaml_only(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("model:\n  name: yaml-model\n  dtype: bfloat16\n")
            path = f.name
        try:
            s = DistLLMSettings.from_yaml(config_path=path)
            assert s.model.name == "yaml-model"
            assert s.model.dtype == "bfloat16"
        finally:
            os.unlink(path)

    def test_cli_overrides_yaml(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("model:\n  name: yaml-model\n  dtype: float16\n")
            path = f.name
        try:
            s = DistLLMSettings.from_yaml(config_path=path, cli_overrides={"model": {"name": "cli-model"}})
            assert s.model.name == "cli-model"
            assert s.model.dtype == "float16"
        finally:
            os.unlink(path)

    def test_missing_yaml_defaults(self):
        s = DistLLMSettings.from_yaml(config_path="/nonexistent/path/config.yaml")
        assert s.model.dtype == "float16"

    def test_extra_yaml_keys_rejected(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("nonexistent_key: true\nmodel:\n  name: test\n")
            path = f.name
        try:
            with pytest.raises(Exception):
                DistLLMSettings.from_yaml(config_path=path)
        finally:
            os.unlink(path)


class TestProfileMerging:
    def test_deep_merge_dict(self):
        base = {"model": {"dtype": "float16"}, "batching": {"max_batch_size": 8}}
        override = {"batching": {"max_batch_size": 32}}
        merged = ProfileConfig._deep_merge(base, override)
        assert merged["batching"]["max_batch_size"] == 32
        assert merged["model"]["dtype"] == "float16"

    def test_deep_merge_new_key(self):
        base = {"model": {"name": "test"}}
        override = {"quantization": {"method": "int4"}}
        merged = ProfileConfig._deep_merge(base, override)
        assert merged["quantization"]["method"] == "int4"

    def test_all_presets_valid(self):
        for preset_name in ["gpu-single", "gpu-multi", "multi-node", "cpu-only", "edge", "high-throughput", "low-latency"]:
            preset_data = ProfileConfig.get_preset(preset_name)
            assert preset_data is not None, f"Preset '{preset_name}' is None"
            assert "description" in preset_data
            assert isinstance(preset_data, dict)


class TestAPIClientMocked:
    def test_get_models(self):
        from distllm.cli.client import DistLLMClient

        orig_session_getter = DistLLMClient._get_session

        class _ClientStub:
            def __init__(self):
                self.call_count = 0
            def request(self, method, url, **kw):
                self.call_count += 1
                return _RespStub(200, {"data": [{"id": "model-1"}]})

        class _RespStub:
            def __init__(self, status_code, json_data):
                self.status_code = status_code
                self._json = json_data
            def json(self):
                return self._json

        try:
            DistLLMClient._get_session = lambda s: _ClientStub()
            client = DistLLMClient(base_url="http://localhost:8000")
            result = client.get("/v1/models")
            assert result["data"][0]["id"] == "model-1"
        finally:
            DistLLMClient._get_session = orig_session_getter

    def test_401_auth_failure(self):
        from distllm.cli.client import DistLLMClient, DistLLMError

        orig_session_getter = DistLLMClient._get_session

        class _ClientStub:
            def request(self, method, url, **kw):
                return _RespStub(401, {"error": "unauthorized"})

        class _RespStub:
            def __init__(self, status_code, json_data):
                self.status_code = status_code
                self._json = json_data
            def json(self):
                return self._json

        try:
            DistLLMClient._get_session = lambda s: _ClientStub()
            client = DistLLMClient(base_url="http://localhost:8000")
            with pytest.raises(DistLLMError) as exc:
                client.get("/v1/models")
            assert exc.value.status_code == 401
        finally:
            DistLLMClient._get_session = orig_session_getter

    def test_503_retry_then_success(self):
        from distllm.cli.client import DistLLMClient, DistLLMError

        orig_session_getter = DistLLMClient._get_session

        class _ClientStub:
            def __init__(self):
                self._call_count = 0
            def request(self, method, url, **kw):
                self._call_count += 1
                if self._call_count <= 2:
                    return _RespStub(503, {"error": "overloaded"})
                return _RespStub(200, {"status": "ok"})

        class _RespStub:
            def __init__(self, status_code, json_data):
                self.status_code = status_code
                self._json = json_data
            def json(self):
                return self._json

        try:
            DistLLMClient._get_session = lambda s: _ClientStub()
            client = DistLLMClient(config=type("Cfg", (), {"base_url": "http://localhost:8000", "max_retries": 2, "retry_delay": 0.01, "timeout": 5.0})())
            result = client.get("/health")
            assert result == {"status": "ok"}
        finally:
            DistLLMClient._get_session = orig_session_getter


class TestConfigValidationErrors:
    def test_invalid_model_name(self):
        with pytest.raises(Exception):
            ModelSettings(name="")

    def test_port_out_of_range(self):
        with pytest.raises(Exception):
            CoordinatorSettings(port=99999)

    def test_temperature_out_of_range(self):
        with pytest.raises(Exception):
            GenerationSettings(temperature=5.0)

    def test_invalid_dtype(self):
        with pytest.raises(Exception):
            ModelSettings(name="test", dtype="int4")
