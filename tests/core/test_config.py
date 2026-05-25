"""Tests: Configuration system — validators, from_yaml, precedence, profiles, converters.

Covers every field_validator boundary value, YAML loading, env var precedence,
profile resolution, deep_merge, startup validation, and converter methods.

Run: pytest tests/core/test_config.py -v
"""

import os
import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError, SecretStr

from distllm.config.settings import (
    DistLLMSettings,
    ModelSettings,
    CoordinatorSettings,
    NodeSettings,
    GenerationSettings,
    NetworkSettings,
    BatchingSettings,
    PrefixCacheSettings,
    ChunkedPrefillSettings,
    QuantizationSettings,
    SpeculativeSettings,
    PartitioningSettings,
    CompressionSettings,
    AlertingSettings,
    ChaosSettings,
    CanarySettings,
    CostSettings,
    RateLimitSettings,
    ModelHubSettings,
    PromptTemplateSettings,
    PluginSettings,
    WideAreaSettings,
    VLLMSettings,
    LlamacppSettings,
    HardwareSettings,
    NodeRole,
)
from distllm.config.profiles import ProfileConfig
from distllm.config.settings import TensorParallelSettings


# ===========================================================================
# 1. Every field_validator — Boundary Values
# ===========================================================================


class TestModelSettingsValidators:
    def test_valid_name(self):
        s = ModelSettings(name="my_model")
        assert s.name == "my_model"

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError):
            ModelSettings(name="")

    def test_whitespace_name_raises(self):
        with pytest.raises(ValidationError):
            ModelSettings(name="   ")

    def test_valid_dtype(self):
        for dt in ("float16", "float32", "bfloat16"):
            s = ModelSettings(dtype=dt)
            assert s.dtype == dt

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValidationError):
            ModelSettings(dtype="int8")


class TestCoordinatorSettingsValidators:
    def test_port_valid(self):
        s = CoordinatorSettings(port=50050)
        assert s.port == 50050

    def test_port_min_boundary(self):
        s = CoordinatorSettings(port=1)
        assert s.port == 1

    def test_port_max_boundary(self):
        s = CoordinatorSettings(port=65535)
        assert s.port == 65535

    def test_port_zero_raises(self):
        with pytest.raises(ValidationError):
            CoordinatorSettings(port=0)

    def test_port_over_max_raises(self):
        with pytest.raises(ValidationError):
            CoordinatorSettings(port=65536)

    def test_api_port_valid(self):
        s = CoordinatorSettings(api_port=8000)
        assert s.api_port == 8000

    def test_valid_cors_origin_http(self):
        s = CoordinatorSettings(cors_origins="http://example.com")
        assert "example.com" in s.cors_origins

    def test_valid_cors_origin_https(self):
        s = CoordinatorSettings(cors_origins="https://example.com")
        assert "example.com" in s.cors_origins

    def test_valid_cors_wildcard(self):
        s = CoordinatorSettings(cors_origins="*")
        assert s.cors_origins == "*"

    def test_invalid_cors_origin_raises(self):
        with pytest.raises(ValidationError):
            CoordinatorSettings(cors_origins="ftp://bad.com")

    def test_multi_cors_origins(self):
        s = CoordinatorSettings(cors_origins="http://a.com,https://b.org")
        assert "http://a.com" in s.cors_origins

    def test_cors_chrome_extension_accepted(self):
        s = CoordinatorSettings(cors_origins="chrome-extension://abc123")
        assert "chrome-extension" in s.cors_origins

    def test_cors_moz_extension_accepted(self):
        s = CoordinatorSettings(cors_origins="moz-extension://def456")
        assert "moz-extension" in s.cors_origins

    def test_cors_empty_string_raises(self):
        with pytest.raises(ValidationError):
            CoordinatorSettings(cors_origins="")

    def test_cors_multiple_mixed_valid_schemes(self):
        s = CoordinatorSettings(cors_origins="http://a.com,chrome-extension://x,moz-extension://y,https://b.org")
        assert s.cors_origins.count(",") == 3


class TestNodeSettingsValidators:
    def test_port_valid(self):
        s = NodeSettings(node_id="n1", port=50051)
        assert s.port == 50051

    def test_port_boundary(self):
        with pytest.raises(ValidationError):
            NodeSettings(node_id="n1", port=0)

    def test_end_layer_gte_start(self):
        s = NodeSettings(node_id="n1", start_layer=2, end_layer=5)
        assert s.end_layer >= s.start_layer

    def test_end_layer_lt_start_raises(self):
        with pytest.raises(ValidationError):
            NodeSettings(node_id="n1", start_layer=5, end_layer=2)


class TestGenerationSettingsValidators:
    def test_temperature_zero(self):
        s = GenerationSettings(temperature=0.0)
        assert s.temperature == 0.0

    def test_temperature_mid(self):
        s = GenerationSettings(temperature=1.0)
        assert s.temperature == 1.0

    def test_temperature_max(self):
        s = GenerationSettings(temperature=2.0)
        assert s.temperature == 2.0

    def test_temperature_over_max_raises(self):
        with pytest.raises(ValidationError):
            GenerationSettings(temperature=2.1)

    def test_temperature_negative_raises(self):
        with pytest.raises(ValidationError):
            GenerationSettings(temperature=-0.1)

    def test_top_p_max(self):
        s = GenerationSettings(top_p=1.0)
        assert s.top_p == 1.0

    def test_top_p_zero_raises(self):
        with pytest.raises(ValidationError):
            GenerationSettings(top_p=0.0)

    def test_top_p_over_max_raises(self):
        with pytest.raises(ValidationError):
            GenerationSettings(top_p=1.1)

    def test_top_k_zero(self):
        s = GenerationSettings(top_k=0)
        assert s.top_k == 0

    def test_top_k_positive(self):
        s = GenerationSettings(top_k=50)
        assert s.top_k == 50

    def test_top_k_negative_raises(self):
        with pytest.raises(ValidationError):
            GenerationSettings(top_k=-1)


class TestNetworkSettingsValidators:
    def test_grpc_timeout_min(self):
        s = NetworkSettings(grpc_timeout=1)
        assert s.grpc_timeout == 1

    def test_grpc_timeout_zero_raises(self):
        with pytest.raises(ValidationError):
            NetworkSettings(grpc_timeout=0)

    def test_max_retries_min(self):
        s = NetworkSettings(max_retries=1)
        assert s.max_retries == 1

    def test_max_retries_zero_raises(self):
        with pytest.raises(ValidationError):
            NetworkSettings(max_retries=0)


class TestBatchingSettingsValidators:
    def test_max_batch_size_min(self):
        s = BatchingSettings(max_batch_size=1)
        assert s.max_batch_size == 1

    def test_max_batch_size_zero_raises(self):
        with pytest.raises(ValidationError):
            BatchingSettings(max_batch_size=0)

    def test_max_tokens_per_batch_min(self):
        s = BatchingSettings(max_tokens_per_batch=1)
        assert s.max_tokens_per_batch == 1

    def test_max_tokens_per_batch_zero_raises(self):
        with pytest.raises(ValidationError):
            BatchingSettings(max_tokens_per_batch=0)


class TestPrefixCacheValidators:
    def test_max_entries_min(self):
        s = PrefixCacheSettings(max_entries=1)
        assert s.max_entries == 1

    def test_min_prefix_len_min(self):
        s = PrefixCacheSettings(min_prefix_len=1)
        assert s.min_prefix_len == 1


class TestQuantizationSettingsValidators:
    def test_valid_methods(self):
        for m in ("none", "bnb_4bit", "bnb_8bit", "gptq", "awq", "fp8"):
            s = QuantizationSettings(method=m)
            assert s.method == m

    def test_invalid_method_raises(self):
        with pytest.raises(ValidationError):
            QuantizationSettings(method="invalid")

    def test_gptq_bits_valid(self):
        for b in (4, 8):
            s = QuantizationSettings(gptq_bits=b)
            assert s.gptq_bits == b

    def test_gptq_bits_invalid_raises(self):
        with pytest.raises(ValidationError):
            QuantizationSettings(gptq_bits=16)

    def test_awq_bits_valid(self):
        s = QuantizationSettings(awq_bits=4)
        assert s.awq_bits == 4

    def test_kv_cache_bits_valid(self):
        for b in (4, 8):
            s = QuantizationSettings(kv_cache_bits=b)
            assert s.kv_cache_bits == b


class TestSpeculativeValidators:
    def test_valid_methods(self):
        for m in ("draft_model", "medusa", "eagle", "ngram", "auto"):
            s = SpeculativeSettings(method=m)
            assert s.method == m

    def test_invalid_method_raises(self):
        with pytest.raises(ValidationError):
            SpeculativeSettings(method="unknown")

    def test_num_assistant_tokens_min(self):
        s = SpeculativeSettings(num_assistant_tokens=1)
        assert s.num_assistant_tokens == 1

    def test_num_assistant_tokens_zero_raises(self):
        with pytest.raises(ValidationError):
            SpeculativeSettings(num_assistant_tokens=0)


class TestPartitioningValidators:
    def test_valid_strategies(self):
        for strat in ("equal", "gpu_aware"):
            s = PartitioningSettings(strategy=strat)
            assert s.strategy == strat

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValidationError):
            PartitioningSettings(strategy="unknown")


class TestCompressionValidators:
    def test_target_bits_valid(self):
        for b in (4, 8):
            s = CompressionSettings(target_bits=b)
            assert s.target_bits == b

    def test_target_bits_invalid_raises(self):
        with pytest.raises(ValidationError):
            CompressionSettings(target_bits=16)

    def test_pruning_ratio_zero(self):
        s = CompressionSettings(pruning_ratio=0.0)
        assert s.pruning_ratio == 0.0

    def test_pruning_ratio_one(self):
        s = CompressionSettings(pruning_ratio=1.0)
        assert s.pruning_ratio == 1.0

    def test_pruning_ratio_over_one_raises(self):
        with pytest.raises(ValidationError):
            CompressionSettings(pruning_ratio=1.1)

    def test_pruning_ratio_negative_raises(self):
        with pytest.raises(ValidationError):
            CompressionSettings(pruning_ratio=-0.1)


class TestAlertingValidators:
    def test_valid_url_http(self):
        s = AlertingSettings(prometheus_url="http://prometheus:9090")
        assert s.prometheus_url == "http://prometheus:9090"

    def test_valid_url_https(self):
        s = AlertingSettings(prometheus_url="https://prometheus.example.com")
        assert "prometheus" in s.prometheus_url

    def test_invalid_url_raises(self):
        with pytest.raises(ValidationError):
            AlertingSettings(prometheus_url="ftp://prometheus")

    def test_empty_url_raises(self):
        with pytest.raises(ValidationError):
            AlertingSettings(prometheus_url="")


class TestChaosValidators:
    def test_max_latency_ms_min(self):
        s = ChaosSettings(max_latency_ms=1)
        assert s.max_latency_ms == 1

    def test_max_latency_ms_zero_raises(self):
        with pytest.raises(ValidationError):
            ChaosSettings(max_latency_ms=0)


class TestCanaryValidators:
    def test_rollback_threshold_valid(self):
        s = CanarySettings(rollback_threshold=0.05)
        assert s.rollback_threshold == 0.05

    def test_rollback_threshold_max(self):
        s = CanarySettings(rollback_threshold=1.0)
        assert s.rollback_threshold == 1.0

    def test_rollback_threshold_zero_raises(self):
        with pytest.raises(ValidationError):
            CanarySettings(rollback_threshold=0.0)

    def test_rollback_threshold_negative_raises(self):
        with pytest.raises(ValidationError):
            CanarySettings(rollback_threshold=-0.1)


class TestCostValidators:
    def test_budget_zero(self):
        s = CostSettings(budget_per_hour=0.0)
        assert s.budget_per_hour == 0.0

    def test_budget_positive(self):
        s = CostSettings(budget_per_hour=10.0)
        assert s.budget_per_hour == 10.0

    def test_budget_negative_raises(self):
        with pytest.raises(ValidationError):
            CostSettings(budget_per_hour=-1.0)

    def test_spot_preference_zero(self):
        s = CostSettings(spot_preference=0.0)
        assert s.spot_preference == 0.0

    def test_spot_preference_one(self):
        s = CostSettings(spot_preference=1.0)
        assert s.spot_preference == 1.0

    def test_spot_preference_over_one_raises(self):
        with pytest.raises(ValidationError):
            CostSettings(spot_preference=1.1)


class TestRateLimitValidators:
    def test_default_rpm_valid(self):
        s = RateLimitSettings(default_rpm=60.0)
        assert s.default_rpm == 60.0

    def test_default_rpm_zero_raises(self):
        with pytest.raises(ValidationError):
            RateLimitSettings(default_rpm=0.0)

    def test_burst_multiplier_positive(self):
        s = RateLimitSettings(burst_multiplier=2.0)
        assert s.burst_multiplier == 2.0


class TestModelHubValidators:
    def test_cache_size_positive(self):
        s = ModelHubSettings(max_cache_size_gb=10.0)
        assert s.max_cache_size_gb == 10.0

    def test_cache_size_zero_raises(self):
        with pytest.raises(ValidationError):
            ModelHubSettings(max_cache_size_gb=0.0)

    def test_cache_size_negative_raises(self):
        with pytest.raises(ValidationError):
            ModelHubSettings(max_cache_size_gb=-5.0)

    def test_download_timeout_min(self):
        s = ModelHubSettings(download_timeout_s=1)
        assert s.download_timeout_s == 1

    def test_download_timeout_zero_raises(self):
        with pytest.raises(ValidationError):
            ModelHubSettings(download_timeout_s=0)

    def test_offline_mode_default_false(self):
        s = ModelHubSettings()
        assert s.offline_mode is False

    def test_offline_mode_enabled(self):
        s = ModelHubSettings(offline_mode=True)
        assert s.offline_mode is True
        assert s.enabled is True


class TestHardwareValidators:
    def test_valid_device_types(self):
        for dt in ("auto", "cuda", "rocm", "mps", "xpu", "cpu"):
            s = HardwareSettings(device_type=dt)
            assert s.device_type == dt

    def test_invalid_device_type_raises(self):
        with pytest.raises(ValidationError):
            HardwareSettings(device_type="npu")

    def test_valid_backends(self):
        for b in ("auto", "vllm", "pytorch", "llamacpp"):
            s = HardwareSettings(preferred_backend=b)
            assert s.preferred_backend == b

    def test_invalid_backend_raises(self):
        with pytest.raises(ValidationError):
            HardwareSettings(preferred_backend="tensorrt")


class TestVLLMSettingsValidators:
    def test_gpu_memory_utilization_max(self):
        s = VLLMSettings(gpu_memory_utilization=1.0)
        assert s.gpu_memory_utilization == 1.0

    def test_gpu_memory_utilization_zero_raises(self):
        with pytest.raises(ValidationError):
            VLLMSettings(gpu_memory_utilization=0.0)

    def test_tp_size_min(self):
        s = VLLMSettings(tensor_parallel_size=1)
        assert s.tensor_parallel_size == 1

    def test_tp_size_zero_raises(self):
        with pytest.raises(ValidationError):
            VLLMSettings(tensor_parallel_size=0)

    def test_vllm_dtype_valid(self):
        s = VLLMSettings(dtype="auto")
        assert s.dtype == "auto"

    def test_vllm_dtype_invalid_raises(self):
        with pytest.raises(ValidationError):
            VLLMSettings(dtype="int8")


class TestLlamacppValidators:
    def test_n_gpu_layers_zero(self):
        s = LlamacppSettings(n_gpu_layers=0)
        assert s.n_gpu_layers == 0

    def test_n_gpu_layers_negative_raises(self):
        with pytest.raises(ValidationError):
            LlamacppSettings(n_gpu_layers=-1)

    def test_n_ctx_min(self):
        s = LlamacppSettings(n_ctx=128)
        assert s.n_ctx == 128

    def test_n_ctx_below_min_raises(self):
        with pytest.raises(ValidationError):
            LlamacppSettings(n_ctx=127)


class TestPluginValidators:
    def test_valid_plugin_module(self):
        s = PluginSettings(plugins=[{"module": "my.module.path"}])
        assert len(s.plugins) == 1

    def test_plugin_without_dot_raises(self):
        with pytest.raises(ValidationError):
            PluginSettings(plugins=[{"module": "badmodule"}])


# ===========================================================================
# 2. DistLLMSettings.from_yaml()
# ===========================================================================


class TestFromYaml:
    def test_basic_from_yaml(self):
        yaml_content = """
model:
  name: test-model
  dtype: float16
coordinator:
  port: 50050
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name
        try:
            settings = DistLLMSettings.from_yaml(config_path=path)
            assert settings.model.name == "test-model"
            assert settings.model.dtype == "float16"
            assert settings.coordinator.port == 50050
        finally:
            os.unlink(path)

    def test_from_yaml_with_cli_overrides(self):
        yaml_content = """
model:
  name: base-model
generation:
  temperature: 0.7
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name
        try:
            settings = DistLLMSettings.from_yaml(
                config_path=path,
                cli_overrides={"model": {"name": "cli-model"}},
            )
            assert settings.model.name == "cli-model"
            assert settings.generation.temperature == 0.7
        finally:
            os.unlink(path)

    def test_from_yaml_precedence_cli_overrides_yaml(self):
        yaml_content = """
model:
  name: test-model
generation:
  temperature: 0.7
  top_p: 0.9
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name
        try:
            settings = DistLLMSettings.from_yaml(
                config_path=path,
                cli_overrides={"generation": {"temperature": 0.1}},
            )
            assert settings.generation.temperature == 0.1
            assert settings.generation.top_p == 0.9
        finally:
            os.unlink(path)

    def test_from_yaml_no_path_uses_defaults(self):
        settings = DistLLMSettings.from_yaml(config_path=None)
        assert settings.model.name == ""
        assert settings.coordinator.port == 50050

    def test_from_yaml_missing_file_uses_defaults(self):
        settings = DistLLMSettings.from_yaml(config_path="/nonexistent/path/config.yaml")
        assert settings.model.dtype == "float16"
        assert settings.coordinator.port == 50050

    def test_from_yaml_partial_yaml_uses_defaults_for_rest(self):
        yaml_content = "model:\n  name: partial-model\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name
        try:
            settings = DistLLMSettings.from_yaml(config_path=path)
            assert settings.model.name == "partial-model"
            assert settings.coordinator.port == 50050
            assert settings.generation.temperature == 0.7
            assert settings.network.grpc_timeout == 30
        finally:
            os.unlink(path)

    def test_from_yaml_invalid_yaml_raises(self):
        yaml_content = "model:\n  name: test\n  dtype: invalid_dtype\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name
        try:
            with pytest.raises((ValidationError, Exception)):
                DistLLMSettings.from_yaml(config_path=path)
        finally:
            os.unlink(path)

    def test_from_yaml_parse_error(self):
        yaml_content = "model:\n  name: test\n  dtype: float16\n  invalid_yaml: [\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name
        try:
            with pytest.raises(Exception):
                DistLLMSettings.from_yaml(config_path=path)
        finally:
            os.unlink(path)


# ===========================================================================
# 3. from_yaml() — Precedence Chain (CLI > env > YAML > defaults)
# ===========================================================================


class TestPrecedenceChain:
    def test_defaults_when_no_yaml_no_env(self):
        settings = DistLLMSettings()
        assert settings.model.dtype == "float16"

    def test_env_overrides_defaults(self):
        s = DistLLMSettings(**{"model": {"name": "env-model", "dtype": "bfloat16"}})
        assert s.model.name == "env-model"
        assert s.model.dtype == "bfloat16"

    def test_env_overrides_yaml(self):
        yaml_content = """
model:
  name: yaml-model
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name
        try:
            settings = DistLLMSettings.from_yaml(
                config_path=path,
                cli_overrides={"model": {"name": "env-wins"}},
            )
            assert settings.model.name == "env-wins"
        finally:
            os.unlink(path)

    def test_env_override_all_major_keys(self):
        s = DistLLMSettings(**{
            "model": {"name": "m", "dtype": "bfloat16"},
            "coordinator": {"host": "0.0.0.0", "port": 50051, "api_port": 8001},
            "generation": {"max_new_tokens": 512, "temperature": 0.5, "top_p": 0.9, "top_k": 50},
            "network": {"grpc_timeout": 60, "max_retries": 5, "retry_delay": 2.0},
            "batching": {"max_batch_size": 64, "max_tokens_per_batch": 8192},
            "prefix_cache": {"enabled": False, "max_entries": 512},
            "chunked_prefill": {"enabled": False, "chunk_size": 256},
            "monitoring": {"enabled": False},
            "quantization": {"method": "fp8"},
            "speculative": {"method": "eagle", "num_assistant_tokens": 3},
            "tensor_parallel": {"enabled": True, "num_gpus": 4},
            "hardware": {"device_type": "cuda", "preferred_backend": "pytorch"},
        })
        assert s.model.name == "m"
        assert s.model.dtype == "bfloat16"
        assert s.coordinator.host == "0.0.0.0"
        assert s.coordinator.port == 50051
        assert s.generation.max_new_tokens == 512
        assert s.generation.temperature == 0.5
        assert s.network.grpc_timeout == 60
        assert s.network.max_retries == 5
        assert s.batching.max_batch_size == 64
        assert s.batching.max_tokens_per_batch == 8192
        assert s.prefix_cache.enabled is False
        assert s.prefix_cache.max_entries == 512
        assert s.chunked_prefill.enabled is False
        assert s.chunked_prefill.chunk_size == 256
        assert s.monitoring.enabled is False
        assert s.quantization.method == "fp8"
        assert s.speculative.method == "eagle"
        assert s.speculative.num_assistant_tokens == 3
        assert s.tensor_parallel.enabled is True
        assert s.tensor_parallel.num_gpus == 4
        assert s.hardware.device_type == "cuda"
        assert s.hardware.preferred_backend == "pytorch"

    def test_cli_overrides_all_keys(self):
        yaml_content = """
model:
  name: base-model
  dtype: float16
generation:
  temperature: 0.7
batching:
  max_batch_size: 16
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name
        try:
            settings = DistLLMSettings.from_yaml(
                config_path=path,
                cli_overrides={
                    "model": {"name": "cli-model", "dtype": "bfloat16"},
                    "generation": {"temperature": 0.1, "top_p": 0.5, "top_k": 10},
                    "batching": {"max_batch_size": 4},
                    "network": {"grpc_timeout": 120},
                    "tensor_parallel": {"enabled": True},
                    "hardware": {"device_type": "cpu"},
                },
            )
            assert settings.model.name == "cli-model"
            assert settings.model.dtype == "bfloat16"
            assert settings.generation.temperature == 0.1
            assert settings.generation.top_p == 0.5
            assert settings.batching.max_batch_size == 4
            assert settings.network.grpc_timeout == 120
            assert settings.tensor_parallel.enabled is True
            assert settings.hardware.device_type == "cpu"
        finally:
            os.unlink(path)


# ===========================================================================
# 4. Profiles
# ===========================================================================


class TestProfiles:
    def test_profile_config_load_dev(self):
        yaml_content = """
model:
  name: base-model
dev:
  generation:
    temperature: 0.8
  batching:
    max_batch_size: 16
staging:
  generation:
    temperature: 0.5
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name
        try:
            merged = ProfileConfig.load(config_path=path, profile="dev")
            assert merged["model"]["name"] == "base-model"
            assert merged["generation"]["temperature"] == 0.8
            assert merged["batching"]["max_batch_size"] == 16
        finally:
            os.unlink(path)

    def test_profile_config_load_production(self):
        yaml_content = """
model:
  name: base-model
production:
  generation:
    temperature: 0.3
    top_k: 40
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name
        try:
            merged = ProfileConfig.load(config_path=path, profile="production")
            assert merged["model"]["name"] == "base-model"
            assert merged["generation"]["temperature"] == 0.3
            assert merged["generation"]["top_k"] == 40
        finally:
            os.unlink(path)

    def test_profile_unknown_raises(self):
        yaml_content = "model:\n  name: test\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name
        try:
            with pytest.raises(ValueError):
                ProfileConfig.load(config_path=path, profile="unknown")
        finally:
            os.unlink(path)

    def test_deep_merge(self):
        base = {"a": 1, "b": {"c": 2, "d": 3}, "e": [1]}
        override = {"b": {"c": 99}, "f": 4}
        merged = ProfileConfig._deep_merge(base, override)
        assert merged["a"] == 1
        assert merged["b"]["c"] == 99
        assert merged["b"]["d"] == 3
        assert merged["f"] == 4

    def test_deep_merge_non_dict_override(self):
        base = {"a": {"b": 1}}
        override = {"a": 2}
        merged = ProfileConfig._deep_merge(base, override)
        assert merged["a"] == 2

    def test_from_profile_dev_override(self):
        yaml_content = """
model:
  name: base-model
dev:
  generation:
    temperature: 0.9
  batching:
    max_batch_size: 8
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name
        try:
            settings = DistLLMSettings.from_profile(config_path=path, profile="dev")
            assert settings.model.name == "base-model"
            assert settings.generation.temperature == 0.9
            assert settings.batching.max_batch_size == 8
        finally:
            os.unlink(path)

    def test_from_profile_production_all_settings(self):
        yaml_content = """
model:
  name: prod-model
  dtype: bfloat16
production:
  coordinator:
    port: 50051
    cors_origins: "https://prod.example.com"
  generation:
    temperature: 0.3
    top_p: 0.95
    top_k: 40
  batching:
    max_batch_size: 64
    max_tokens_per_batch: 8192
  network:
    grpc_timeout: 60
    max_retries: 5
  monitoring:
    enabled: true
  prefix_cache:
    enabled: true
    max_entries: 2048
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name
        try:
            settings = DistLLMSettings.from_profile(config_path=path, profile="production")
            assert settings.model.name == "prod-model"
            assert settings.model.dtype == "bfloat16"
            assert settings.coordinator.port == 50051
            assert "prod.example.com" in settings.coordinator.cors_origins
            assert settings.generation.temperature == 0.3
            assert settings.generation.top_k == 40
            assert settings.batching.max_batch_size == 64
            assert settings.network.grpc_timeout == 60
            assert settings.monitoring.enabled is True
            assert settings.prefix_cache.max_entries == 2048
        finally:
            os.unlink(path)

    def test_from_profile_unknown_profile_raises(self):
        yaml_content = "model:\n  name: test\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name
        try:
            with pytest.raises(ValueError):
                DistLLMSettings.from_profile(config_path=path, profile="nonexistent")
        finally:
            os.unlink(path)


# ===========================================================================
# 5. validate_startup()
# ===========================================================================


class TestValidateStartup:
    def test_validate_startup_valid(self):
        settings = DistLLMSettings.validate_startup()
        assert isinstance(settings, DistLLMSettings)

    def test_validate_startup_valid_yaml(self):
        yaml_content = "model:\n  name: test\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name
        try:
            settings = DistLLMSettings.validate_startup(config_path=path)
            assert settings.model.name == "test"
        finally:
            os.unlink(path)

    def test_validate_startup_invalid_yaml(self):
        yaml_content = "model:\n  dtype: invalid_dtype\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name
        try:
            with pytest.raises(SystemExit):
                DistLLMSettings.validate_startup(config_path=path)
        finally:
            os.unlink(path)


# ===========================================================================
# 6. hf_token handling
# ===========================================================================


class TestHfToken:
    def test_hf_token_secret(self):
        s = ModelHubSettings(hf_token=SecretStr("my-token"))
        assert s.hf_token is not None
        assert s.hf_token_value is not None

    def test_hf_token_from_env(self):
        s = ModelHubSettings(hf_token=SecretStr("direct-token"))
        assert s.hf_token_value == "direct-token"

    def test_hf_token_prefers_distllm_env(self):
        s = ModelHubSettings(hf_token=SecretStr("config-token"))
        assert s.hf_token_value is not None

    def test_hf_token_env_var_wins(self):
        s = ModelHubSettings(hf_token=SecretStr("config-value"))
        assert s.hf_token.get_secret_value() == "config-value"

    def test_hf_token_warn_if_plain_text(self, recwarn):
        import os as _os
        was_set_distllm = "DISTLLM__MODEL_HUB__HF_TOKEN" in _os.environ
        was_set_hf = "HF_TOKEN" in _os.environ
        distllm_val = _os.environ.pop("DISTLLM__MODEL_HUB__HF_TOKEN", None)
        hf_val = _os.environ.pop("HF_TOKEN", None)
        try:
            _ = ModelHubSettings(hf_token=SecretStr("plain-text-token"))
            token_warnings = [w for w in recwarn if "hf_token" in str(w.message).lower()]
            assert len(token_warnings) >= 1
        finally:
            if was_set_distllm:
                _os.environ["DISTLLM__MODEL_HUB__HF_TOKEN"] = distllm_val
            if was_set_hf:
                _os.environ["HF_TOKEN"] = hf_val


# ===========================================================================
# 7. Config Converters
# ===========================================================================


class TestConfigConverters:
    def test_self_optimizing_to_optimization_config(self):
        from distllm.config.settings import SelfOptimizingSettings
        s = SelfOptimizingSettings(enabled=True, tune_interval_seconds=120.0, warmup_seconds=60.0)
        opt = s.to_optimization_config()
        assert opt.enabled is True
        assert opt.runner.warmup_seconds == 60.0

    def test_partitioning_to_auto_partition_config(self):
        s = PartitioningSettings(strategy="gpu_aware")
        ap = s.to_auto_partition_config()
        assert ap is not None

    def test_partitioning_equal_disables_auto(self):
        s = PartitioningSettings(strategy="equal")
        ap = s.to_auto_partition_config()
        assert ap.enabled is False

    def test_disagg_to_full_config(self):
        from distllm.config.settings import DisaggSettings
        s = DisaggSettings(enabled=True, prefill_nodes=[{"host": "n1", "port": 50051}])
        full = s.to_full_config()
        assert full is not None


# ===========================================================================
# 8. DistLLMSettings — Full construction
# ===========================================================================


class TestDistLLMSettings:
    def test_defaults(self):
        s = DistLLMSettings()
        assert s.model.dtype == "float16"
        assert s.coordinator.port == 50050
        assert s.generation.temperature == 0.7
        assert s.network.grpc_timeout == 30
        assert s.batching.max_batch_size == 32

    def test_nested_settings(self):
        s = DistLLMSettings(
            model={"name": "test", "dtype": "bfloat16"},
            coordinator={"port": 50051, "api_port": 8001},
            generation={"temperature": 0.5, "top_p": 0.95},
        )
        assert s.model.name == "test"
        assert s.model.dtype == "bfloat16"
        assert s.coordinator.port == 50051
        assert s.generation.temperature == 0.5

    def test_node_role_enum(self):
        s = DistLLMSettings(nodes=[{"node_id": "n1", "role": "prefill"}])
        assert s.nodes[0].role == NodeRole.PREFILL

    def test_tensor_parallel_defaults(self):
        s = TensorParallelSettings()
        assert s.enabled is False
        assert s.num_gpus == 2


# ===========================================================================
# 9. Additional validator edge cases
# ===========================================================================


class TestSecretStrHandling:
    def test_secret_not_in_repr(self):
        s = ModelHubSettings(hf_token=SecretStr("sensitive-token"))
        r = repr(s)
        assert "sensitive-token" not in r

    def test_secret_value_accessible(self):
        s = ModelHubSettings(hf_token=SecretStr("my-token"))
        assert s.hf_token.get_secret_value() == "my-token"


class TestLlamacppModelPathRequired:
    def test_model_path_required_when_enabled(self):
        with pytest.raises(ValidationError):
            LlamacppSettings(enabled=True, model_path="")


class TestPromptTemplate:
    def test_empty_template_raises(self):
        with pytest.raises(ValidationError):
            PromptTemplateSettings(template="")


class TestWideAreaSettings:
    def test_tokens_before_forward_min(self):
        s = WideAreaSettings(tokens_before_forward=1)
        assert s.tokens_before_forward == 1

    def test_tokens_before_forward_zero_raises(self):
        with pytest.raises(ValidationError):
            WideAreaSettings(tokens_before_forward=0)

    def test_wan_timeout_min(self):
        s = WideAreaSettings(wan_timeout_seconds=1)
        assert s.wan_timeout_seconds == 1


class TestMultiModelSettings:
    def test_max_models_at_least_one(self):
        from distllm.config.settings import MultiModelSettings
        s = MultiModelSettings(max_models=1)
        assert s.max_models == 1

    def test_max_models_zero_raises(self):
        from distllm.config.settings import MultiModelSettings
        with pytest.raises(ValidationError):
            MultiModelSettings(max_models=0)
