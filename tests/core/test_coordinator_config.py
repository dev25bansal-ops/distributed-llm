"""Tests for CoordinatorConfig -- configuration schema for the distributed coordinator.

Covers:
- Default values and field validation
- dtype validation
- Port bounds
- from_settings factory method
"""

from __future__ import annotations

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_coord_cfg = load_module("distllm/core/coordinator_config.py")
CoordinatorConfig = _coord_cfg.CoordinatorConfig


class _FakeSettings:
    """Deterministic settings stub for CoordinatorConfig.from_settings."""

    class _WideArea:
        def __init__(self, enabled: bool = False) -> None:
            self.enabled = enabled

    class _Model:
        def __init__(self, name: str = "test-model", dtype: str = "float32",
                     trust_remote_code: bool = False) -> None:
            self.name = name
            self.dtype = dtype
            self.trust_remote_code = trust_remote_code

    class _Coordinator:
        def __init__(self, port: int = 50050) -> None:
            self.port = port

    class _Batching:
        def __init__(self, max_batch_size: int = 4, max_tokens_per_batch: int = 1024) -> None:
            self.max_batch_size = max_batch_size
            self.max_tokens_per_batch = max_tokens_per_batch

    class _Network:
        def __init__(self, grpc_timeout: float = 30.0) -> None:
            self.grpc_timeout = grpc_timeout

    class _ModelHub:
        def __init__(self, cache_dir: str = "/tmp/cache") -> None:
            self.cache_dir = cache_dir

    def __init__(self, **overrides: object) -> None:
        self.wide_area = self._WideArea(enabled=overrides.get("wa_enabled", False))
        self.model = self._Model(
            name=overrides.get("model_name", "test-model"),
            dtype=overrides.get("dtype", "float32"),
            trust_remote_code=overrides.get("trust_remote_code", False),
        )
        self.coordinator = self._Coordinator(port=overrides.get("port", 50050))
        self.batching = self._Batching(
            max_batch_size=overrides.get("max_batch_size", 4),
            max_tokens_per_batch=overrides.get("max_tokens_per_batch", 1024),
        )
        self.network = self._Network(grpc_timeout=overrides.get("pipeline_timeout", 30.0))
        self.model_hub = self._ModelHub(cache_dir=overrides.get("cache_dir", "/tmp/cache"))


class TestCoordinatorConfigDefaults:
    """Default values for all fields."""

    def test_default_values(self):
        cfg = CoordinatorConfig(model_name="test-model")
        assert cfg.model_name == "test-model"
        assert cfg.port == 50050
        assert cfg.dtype == "float16"
        assert cfg.trust_remote_code is None
        assert cfg.max_batch_size == 4
        assert cfg.max_tokens_per_batch == 1024
        assert cfg.pipeline_timeout == 30.0
        assert cfg.cluster_key is None
        assert cfg.model_cache_dir is None
        assert cfg.metrics_exporter is None
        assert cfg.discovery_mode is None
        assert cfg.wide_area_config is None
        assert cfg.redundancy == 1
        assert cfg.federation_config is None
        assert cfg.plugin_system is None
        assert cfg.min_reputation == 0.0
        assert cfg.prefix_cache_enabled is False
        assert cfg.prefix_cache_max_entries == 256
        assert cfg.prefix_cache_min_prefix_len == 4
        assert cfg.radix_tree_cache_enabled is False
        assert cfg.chunked_prefill_enabled is False
        assert cfg.chunked_prefill_chunk_size == 512
        assert cfg.enable_pipeline_overlap is False


class TestCoordinatorConfigValidation:
    """Field validation via pydantic validators and constraints."""

    def test_valid_dtype_values(self):
        for dt in ("float16", "float32", "bfloat16"):
            cfg = CoordinatorConfig(model_name="m", dtype=dt)
            assert cfg.dtype == dt

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValueError, match="dtype must be one of"):
            CoordinatorConfig(model_name="m", dtype="float64")

    def test_port_ge_1(self):
        with pytest.raises((ValueError, AssertionError)):
            CoordinatorConfig(model_name="m", port=0)

    def test_port_le_65535(self):
        with pytest.raises((ValueError, AssertionError)):
            CoordinatorConfig(model_name="m", port=70000)

    def test_redundancy_ge_1(self):
        with pytest.raises((ValueError, AssertionError)):
            CoordinatorConfig(model_name="m", redundancy=0)

    def test_min_reputation_bounds(self):
        with pytest.raises((ValueError, AssertionError)):
            CoordinatorConfig(model_name="m", min_reputation=-0.1)
        with pytest.raises((ValueError, AssertionError)):
            CoordinatorConfig(model_name="m", min_reputation=1.1)

    def test_min_reputation_valid(self):
        cfg = CoordinatorConfig(model_name="m", min_reputation=0.5)
        assert cfg.min_reputation == 0.5

    def test_max_batch_size_ge_1(self):
        with pytest.raises((ValueError, AssertionError)):
            CoordinatorConfig(model_name="m", max_batch_size=0)

    def test_pipeline_timeout_gt_0(self):
        with pytest.raises((ValueError, AssertionError)):
            CoordinatorConfig(model_name="m", pipeline_timeout=0)

    def test_chunked_prefill_chunk_size_ge_1(self):
        with pytest.raises((ValueError, AssertionError)):
            CoordinatorConfig(model_name="m", chunked_prefill_chunk_size=0)


class TestCoordinatorConfigCustom:
    """Explicit field setting and edge values."""

    def test_custom_fields(self):
        cfg = CoordinatorConfig(
            model_name="custom-model",
            port=50051,
            dtype="bfloat16",
            trust_remote_code=True,
            max_batch_size=16,
            max_tokens_per_batch=4096,
            pipeline_timeout=60.0,
            cluster_key="secret",
            model_cache_dir="/models/cache",
            redundancy=3,
            min_reputation=0.5,
            prefix_cache_enabled=True,
            prefix_cache_max_entries=512,
            prefix_cache_min_prefix_len=8,
            radix_tree_cache_enabled=True,
            chunked_prefill_enabled=True,
            chunked_prefill_chunk_size=1024,
            enable_pipeline_overlap=True,
        )
        assert cfg.model_name == "custom-model"
        assert cfg.port == 50051
        assert cfg.dtype == "bfloat16"
        assert cfg.trust_remote_code is True
        assert cfg.max_batch_size == 16
        assert cfg.max_tokens_per_batch == 4096
        assert cfg.pipeline_timeout == 60.0
        assert cfg.cluster_key == "secret"
        assert cfg.model_cache_dir == "/models/cache"
        assert cfg.redundancy == 3
        assert cfg.min_reputation == 0.5
        assert cfg.prefix_cache_enabled is True
        assert cfg.prefix_cache_max_entries == 512
        assert cfg.prefix_cache_min_prefix_len == 8
        assert cfg.chunked_prefill_chunk_size == 1024
        assert cfg.enable_pipeline_overlap is True


class TestCoordinatorConfigFromSettings:
    """from_settings classmethod."""

    def _make_settings(self, **overrides) -> _FakeSettings:
        return _FakeSettings(**overrides)

    def test_from_settings_basic(self):
        settings = self._make_settings()
        cfg = CoordinatorConfig.from_settings(settings)

        assert cfg.model_name == "test-model"
        assert cfg.port == 50050
        assert cfg.dtype == "float32"
        assert cfg.max_batch_size == 4
        assert cfg.max_tokens_per_batch == 1024
        assert cfg.pipeline_timeout == 30.0
        assert cfg.model_cache_dir == "/tmp/cache"
        assert cfg.wide_area_config is None

    def test_from_settings_with_overrides(self):
        settings = self._make_settings()
        cfg = CoordinatorConfig.from_settings(settings, port=50051, max_batch_size=16)

        assert cfg.port == 50051
        assert cfg.max_batch_size == 16
        # Other values should remain from settings
        assert cfg.model_name == "test-model"

    def test_from_settings_override_unknown_key(self):
        """Override for a key that doesn't exist raises (was silently ignored).

        Silently dropping a typo'd override (e.g. ``max_batchsize`` instead
        of ``max_batch_size``) meant the intended change never took effect
        and nothing signalled it — the constructor path is the strict one.
        """
        settings = self._make_settings()
        with pytest.raises(TypeError, match="max_batchsize"):
            CoordinatorConfig.from_settings(settings, max_batchsize=999)

    def test_from_settings_copies_prefix_cache_and_chunked_prefill(self):
        """Configured prefix-cache/chunked-prefill settings must reach the
        CoordinatorConfig instead of being replaced by its defaults.

        Repro of the audit finding: real DistLLMSettings with
        ``prefix_cache={'enabled': True, 'max_entries': 5000}`` used to come
        out of from_settings as enabled=False / max_entries=256, silently.
        """

        class _Pc:
            enabled = True
            max_entries = 5000
            min_prefix_len = 8
            radix_tree_enabled = False

        class _Cp:
            enabled = True
            chunk_size = 2048

        settings = self._make_settings()
        settings.prefix_cache = _Pc()
        settings.chunked_prefill = _Cp()

        cfg = CoordinatorConfig.from_settings(settings)
        assert cfg.prefix_cache_enabled is True
        assert cfg.prefix_cache_max_entries == 5000
        assert cfg.prefix_cache_min_prefix_len == 8
        assert cfg.radix_tree_cache_enabled is False
        assert cfg.chunked_prefill_enabled is True
        assert cfg.chunked_prefill_chunk_size == 2048

    def test_from_settings_override_is_validated(self):
        """Overrides pass through pydantic: out-of-range/invalid values raise
        exactly like the direct constructor, instead of being setattr'd in."""
        settings = self._make_settings()
        with pytest.raises(ValueError):
            CoordinatorConfig.from_settings(settings, port=99999)
        with pytest.raises(ValueError):
            CoordinatorConfig.from_settings(settings, dtype="int4")
        # A valid override still applies.
        cfg = CoordinatorConfig.from_settings(settings, prefix_cache_enabled=True)
        assert cfg.prefix_cache_enabled is True
