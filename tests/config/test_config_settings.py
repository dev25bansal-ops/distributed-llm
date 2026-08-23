"""Tests for config settings classes — defaults, validation, cross-field, from_yaml, from_profile, diff, overrides."""
from __future__ import annotations
import os
import tempfile
import pytest
from pydantic import ValidationError
from distllm.config.settings import DistLLMSettings
from distllm.config._model import ModelSettings, QuantizationSettings, SpeculativeSettings
from distllm.config._network import CoordinatorSettings, NetworkSettings, TLSSettings, RateLimitSettings
from distllm.config._cache import PrefixCacheSettings, CachePersistenceSettings, CacheSettings
from distllm.config._parallelism import BatchingSettings, ChunkedPrefillSettings, TensorParallelSettings
from distllm.config._performance import CudaGraphSettings, CompileSettings
from distllm.config._backends import VLLMSettings, LlamacppSettings
from distllm.config._generation import GenerationSettings
from distllm.config._observability import MonitoringSettings, AlertingSettings
from distllm.config._application import AgentSettings, RAGSettings, PluginSettings
from distllm.config._deployment import CanarySettings, CostSettings


class TestDefaults:
    def test_model_defaults(self):
        m = ModelSettings()
        assert m.dtype == "float16"

    def test_generation_defaults(self):
        g = GenerationSettings()
        assert g.max_new_tokens == 256
        assert g.temperature == 0.7

    def test_batching_defaults(self):
        b = BatchingSettings()
        assert b.max_batch_size >= 1

    def test_chunked_prefill_defaults(self):
        c = ChunkedPrefillSettings()
        assert c.enabled is True

    def test_prefix_cache_defaults(self):
        p = PrefixCacheSettings()
        assert p.enabled is True

    def test_tls_defaults(self):
        t = TLSSettings()
        assert t.enabled is False

    def test_cache_settings_nested(self):
        # CacheSettings uses flat prefix_* fields (nested submodel was flattened).
        c = CacheSettings()
        assert c.prefix_enabled is True
        assert c.prefix_max_entries >= 1

    def test_distllm_settings_defaults(self):
        s = DistLLMSettings()
        assert s.model.dtype == "float16"

    def test_model_dump_contains_model_section(self):
        s = DistLLMSettings()
        dumped = s.model_dump()
        assert "model" in dumped


class TestValidValues:
    def test_model_name_valid(self):
        m = ModelSettings(name="meta-llama/Llama-2-70b")
        assert m.name == "meta-llama/Llama-2-70b"

    def test_valid_dtype(self):
        for dtype in ("float16", "float32", "bfloat16"):
            m = ModelSettings(name="test", dtype=dtype)
            assert m.dtype == dtype

    def test_valid_temperature(self):
        g = GenerationSettings(temperature=1.5)
        assert g.temperature == 1.5

    def test_valid_port(self):
        c = CoordinatorSettings(port=50050)
        assert c.port == 50050

    def test_valid_batch_size(self):
        b = BatchingSettings(max_batch_size=32)
        assert b.max_batch_size == 32

    def test_valid_tensor_parallel(self):
        t = TensorParallelSettings(enabled=True, num_gpus=4)
        assert t.num_gpus == 4

    def test_valid_rate_limit(self):
        r = RateLimitSettings(enabled=True, default_rpm=60.0)
        assert r.default_rpm == 60.0

    def test_alerting_with_url(self):
        a = AlertingSettings(enabled=True, prometheus_url="http://localhost:9090")
        assert a.prometheus_url == "http://localhost:9090"


class TestInvalidValues:
    def test_empty_model_name_raises(self):
        with pytest.raises(ValidationError):
            ModelSettings(name="")

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValidationError):
            ModelSettings(name="test", dtype="int4")

    def test_temperature_too_high_raises(self):
        with pytest.raises(ValidationError):
            GenerationSettings(temperature=3.0)

    def test_port_too_low_raises(self):
        with pytest.raises(ValidationError):
            CoordinatorSettings(port=0)

    def test_batch_size_zero_raises(self):
        with pytest.raises(ValidationError):
            BatchingSettings(max_batch_size=0)

    def test_prefix_entries_zero_raises(self):
        with pytest.raises(ValidationError):
            PrefixCacheSettings(max_entries=0)


class TestCrossFieldValidation:
    def test_vllm_dtype_mismatch_raises(self):
        with pytest.raises(ValueError):
            DistLLMSettings(model=ModelSettings(name="test", dtype="float16"), vllm=VLLMSettings(enabled=True, dtype="bfloat16"))

    def test_chunked_prefill_needs_token_budget(self):
        with pytest.raises(ValueError):
            DistLLMSettings(model=ModelSettings(name="test"), chunked_prefill=ChunkedPrefillSettings(enabled=True), batching=BatchingSettings(max_tokens_per_batch=0))

    def test_tls_needs_cert_file(self):
        with pytest.raises(ValueError):
            DistLLMSettings(model=ModelSettings(name="test"), tls=TLSSettings(enabled=True))

    def test_valid_cross_field_passes(self):
        s = DistLLMSettings(model=ModelSettings(name="test"), tls=TLSSettings(enabled=False))
        assert s.model.name == "test"


class TestFromYAML:
    def test_from_yaml(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("model:\n  name: yaml-model\n")
            path = f.name
        try:
            s = DistLLMSettings.from_yaml(config_path=path)
            assert s.model.name == "yaml-model"
        finally:
            os.unlink(path)

    def test_with_cli_overrides(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("model:\n  name: yaml-model\n")
            path = f.name
        try:
            s = DistLLMSettings.from_yaml(config_path=path, cli_overrides={"model": {"name": "cli-model"}})
            assert s.model.name == "cli-model"
        finally:
            os.unlink(path)

    def test_nonexistent_path(self):
        s = DistLLMSettings.from_yaml(config_path="/nonexistent/config.yaml")
        assert s.model.dtype == "float16"


class TestDiff:
    def test_diff_no_changes(self):
        a = DistLLMSettings()
        b = DistLLMSettings()
        assert a.diff(b) == {}

    def test_diff_model_name(self):
        a = DistLLMSettings(model=ModelSettings(name="model-a"))
        b = DistLLMSettings(model=ModelSettings(name="model-b"))
        diff = a.diff(b)
        assert "model.name" in diff


class TestConftestFixtures:
    """Test the fixtures from conftest."""
    def test_config_fixture(self, config_fixture):
        assert config_fixture.model.name == "test-model"

    def test_mock_env_vars(self, mock_env_vars):
        import os
        assert os.environ.get("DISTLLM__MODEL__NAME") == "env-test-model"
        assert os.environ.get("DISTLLM__MODEL__DTYPE") == "bfloat16"
