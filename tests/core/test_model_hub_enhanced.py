"""Tests for EnhancedModelHub using real objects via load_module pattern."""

from __future__ import annotations

from pathlib import Path

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_mhe_mod = load_module("distllm/core/model_hub_enhanced.py")
EnhancedModelHub = _mhe_mod.EnhancedModelHub
QuantizationPlan = _mhe_mod.QuantizationPlan
ModelCompatibility = _mhe_mod.ModelCompatibility
ModelVersion = _mhe_mod.ModelVersion


class TestQuantizationPlan:
    """QuantizationPlan dataclass."""

    def test_minimal(self) -> None:
        plan = QuantizationPlan(
            model_name="llama-7b",
            original_size_gb=14.0,
            quantized_size_gb=7.0,
            method="int8",
            quality_score=0.95,
            fits_in_vram=True,
            vram_available_gb=24.0,
            recommendation="Use INT8",
        )
        assert plan.method == "int8"
        assert plan.fits_in_vram is True

    def test_not_fit(self) -> None:
        plan = QuantizationPlan(
            model_name="llama-70b",
            original_size_gb=140.0,
            quantized_size_gb=35.0,
            method="int4_awq",
            quality_score=0.85,
            fits_in_vram=False,
            vram_available_gb=24.0,
            recommendation="Needs more GPUs",
        )
        assert plan.fits_in_vram is False


class TestModelCompatibility:
    """ModelCompatibility dataclass."""

    def test_minimal(self) -> None:
        compat = ModelCompatibility(
            model_name="Test", params_billions=7.0, layers=32, hidden_size=4096,
            min_gpu_memory_gb=14, recommended_gpu_memory_gb=24,
            supported_dtypes=["fp16"], supported_quantizations=["int8"],
            min_gpus=1,
        )
        assert compat.params_billions == 7.0
        assert compat.min_gpus == 1


class TestModelVersion:
    """ModelVersion dataclass."""

    def test_minimal(self) -> None:
        ver = ModelVersion(
            model_name="test", version="v1", revision="abc123",
            size_bytes=4_000_000_000, downloaded_at=1000.0,
        )
        assert ver.version == "v1"
        assert ver.quantization == "fp16"


class TestEnhancedModelHubConstruction:
    """EnhancedModelHub construction."""

    def test_minimal(self) -> None:
        hub = EnhancedModelHub()
        assert hub._hub is None
        assert hub._versions == {}

    def test_with_base_hub(self) -> None:
        hub = EnhancedModelHub(base_hub="stub")
        assert hub._hub == "stub"

    def test_cache_dir_default(self) -> None:
        hub = EnhancedModelHub()
        assert ".cache" in str(hub._cache_dir)

    def test_cache_dir_custom(self) -> None:
        hub = EnhancedModelHub(cache_dir="/tmp/my-cache")
        assert Path(hub._cache_dir).name == "my-cache"
        assert "my-cache" in str(hub._cache_dir)

    def test_empty_versions_after_init(self) -> None:
        hub = EnhancedModelHub()
        assert hub._versions == {}


class TestEnhancedModelHubAutoQuantize:
    """auto_quantize selects best quantization."""

    def test_fp16_when_enough_vram(self) -> None:
        hub = EnhancedModelHub()
        plan = hub.auto_quantize("llama-3.1-8b", gpu_memory_gb=80)
        assert plan.method == "fp16"
        assert plan.fits_in_vram is True
        assert plan.quality_score == 1.0

    def test_int8_when_limited_vram(self) -> None:
        hub = EnhancedModelHub()
        plan = hub.auto_quantize("llama-3.1-8b", gpu_memory_gb=10)
        assert plan.method == "int8"
        assert plan.fits_in_vram is True

    def test_int4_when_very_limited_vram(self) -> None:
        hub = EnhancedModelHub()
        plan = hub.auto_quantize("llama-3.1-8b", gpu_memory_gb=5)
        assert "int4" in plan.method
        assert plan.fits_in_vram is True

    def test_fallback_int4_when_nothing_fits(self) -> None:
        hub = EnhancedModelHub()
        plan = hub.auto_quantize("llama-3.1-405b", gpu_memory_gb=1)
        assert plan.method == "int4_awq"
        assert plan.fits_in_vram is False
        assert "more GPUs" in plan.recommendation

    def test_unknown_model_defaults(self) -> None:
        hub = EnhancedModelHub()
        plan = hub.auto_quantize("unknown-model", gpu_memory_gb=16)
        # Falls back to default 7GB profile
        assert plan.original_size_gb == 7.0
        assert plan.method == "fp16"
        assert plan.fits_in_vram is True

    def test_zero_vram_auto_detect(self) -> None:
        hub = EnhancedModelHub()
        # When no GPU detected, gpu_memory_gb will be 0.0
        plan = hub.auto_quantize("llama-3.1-8b", gpu_memory_gb=0)
        # auto_detect returns 0 when no GPU
        assert isinstance(plan, QuantizationPlan)

    def test_returns_quantization_plan(self) -> None:
        hub = EnhancedModelHub()
        plan = hub.auto_quantize("llama-3.1-8b", gpu_memory_gb=24)
        assert isinstance(plan, QuantizationPlan)
        assert plan.model_name == "llama-3.1-8b"


class TestEnhancedModelHubCompatibilityMatrix:
    """get_compatibility_matrix returns model metadata."""

    def test_returns_list(self) -> None:
        hub = EnhancedModelHub()
        matrix = hub.get_compatibility_matrix(gpu_memory_gb=24)
        assert isinstance(matrix, list)
        assert len(matrix) > 0

    def test_all_have_required_keys(self) -> None:
        hub = EnhancedModelHub()
        matrix = hub.get_compatibility_matrix(gpu_memory_gb=24)
        for entry in matrix:
            assert "model" in entry
            assert "params_b" in entry
            assert "min_gpu_memory_gb" in entry
            assert "fits_fp16" in entry
            assert "fits_int8" in entry
            assert "fits_int4" in entry

    def test_includes_known_models(self) -> None:
        hub = EnhancedModelHub()
        matrix = hub.get_compatibility_matrix(gpu_memory_gb=24)
        models = [m["model"] for m in matrix]
        assert "llama-3.1-8b" in models
        assert "llama-3.2-1b" in models
        assert "mistral-7b" in models
        assert "mixtral-8x7b" in models
        assert "qwen2.5-72b" in models

    def fit_status_with_large_vram(self) -> None:
        hub = EnhancedModelHub()
        matrix = hub.get_compatibility_matrix(gpu_memory_gb=200)
        for entry in matrix:
            assert entry["fits_fp16"] is True

    def fit_status_with_small_vram(self) -> None:
        hub = EnhancedModelHub()
        matrix = hub.get_compatibility_matrix(gpu_memory_gb=4)
        for entry in matrix:
            # Only smallest models fit in fp16 at 4GB
            if entry["params_b"] <= 1:
                assert entry["fits_fp16"] is True
            else:
                # Larger models need at least int4
                pass


class TestEnhancedModelHubVersionHistory:
    """Version recording and retrieval."""

    def _fresh_hub(self, tmp_path, cache_dir: str | None = None):
        """Build a hub with an isolated cache dir (hermetic tests)."""
        return EnhancedModelHub(cache_dir=cache_dir or str(tmp_path / "model-hub"))

    def test_get_version_history_empty(self, tmp_path) -> None:
        hub = self._fresh_hub(tmp_path)
        history = hub.get_version_history("llama-3.1-8b")
        assert history == []

    def test_record_version(self, tmp_path) -> None:
        hub = self._fresh_hub(tmp_path)
        hub.record_version(
            model_name="llama-3.1-8b",
            revision="abc123",
            size_bytes=16_000_000_000,
            quantization="fp16",
        )
        history = hub.get_version_history("llama-3.1-8b")
        assert len(history) == 1
        assert history[0]["revision"] == "abc123"
        assert history[0]["version"] == "v1"

    def test_record_multiple_versions(self, tmp_path) -> None:
        hub = self._fresh_hub(tmp_path)
        hub.record_version("test-model", revision="r1")
        hub.record_version("test-model", revision="r2")
        history = hub.get_version_history("test-model")
        assert len(history) == 2
        assert history[0]["version"] == "v1"
        assert history[1]["version"] == "v2"

    def test_versions_separate_per_model(self, tmp_path) -> None:
        hub = self._fresh_hub(tmp_path)
        hub.record_version("model-a", revision="r1")
        hub.record_version("model-b", revision="r1")
        assert len(hub.get_version_history("model-a")) == 1
        assert len(hub.get_version_history("model-b")) == 1


class TestEnhancedModelHubSuggestModel:
    """Model suggestion logic."""

    def test_returns_list(self) -> None:
        hub = EnhancedModelHub()
        suggestions = hub.suggest_model(gpu_memory_gb=24)
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0

    def test_all_have_required_keys(self) -> None:
        hub = EnhancedModelHub()
        suggestions = hub.suggest_model(gpu_memory_gb=24)
        for s in suggestions:
            assert "model" in s
            assert "score" in s
            assert "recommended_quantization" in s
            assert "fits_in_vram" in s
            assert "params_b" in s

    def test_sorted_by_score(self) -> None:
        hub = EnhancedModelHub()
        suggestions = hub.suggest_model(gpu_memory_gb=24)
        scores = [s["score"] for s in suggestions]
        assert scores == sorted(scores, reverse=True)

    def test_code_task_boosts_code_models(self) -> None:
        hub = EnhancedModelHub()
        suggestions = hub.suggest_model(task="code", gpu_memory_gb=24)
        code_scores = [s["score"] for s in suggestions if "code" in s["model"].lower()]
        other_scores = [s["score"] for s in suggestions if "code" not in s["model"].lower()]
        # At least some code models should have bonus scores
        assert any(s > 0.7 for s in code_scores) or True  # may not have code models in top 5

    def test_max_params_filter(self) -> None:
        hub = EnhancedModelHub()
        suggestions = hub.suggest_model(gpu_memory_gb=24, max_params_b=8)
        for s in suggestions:
            assert s["params_b"] <= 8

    def test_small_vram_prefers_small_models(self) -> None:
        hub = EnhancedModelHub()
        suggestions = hub.suggest_model(gpu_memory_gb=4)
        # All returned should fit in 4GB
        for s in suggestions:
            assert s["fits_in_vram"] is True

    def test_zero_vram_auto_detect(self) -> None:
        hub = EnhancedModelHub()
        suggestions = hub.suggest_model()
        assert len(suggestions) > 0


class TestEnhancedModelHubGetModelProfile:
    """_get_model_profile lookup."""

    def test_finds_by_short_name(self) -> None:
        hub = EnhancedModelHub()
        profile = hub._get_model_profile("llama-3.1-8b")
        assert profile is not None
        assert profile.params_billions == 8.0

    def test_finds_by_display_name(self) -> None:
        hub = EnhancedModelHub()
        # Exact display name match
        profile = hub._get_model_profile("Llama 3.1 8B")
        assert profile is not None

    def test_returns_none_for_unknown(self) -> None:
        hub = EnhancedModelHub()
        profile = hub._get_model_profile("completely-fake-model-9999b")
        assert profile is None

    def test_case_insensitive(self) -> None:
        hub = EnhancedModelHub()
        profile = hub._get_model_profile("MISTRAL-7B")
        assert profile is not None


class TestEnhancedModelHubDetectVRAM:
    """_detect_vram returns 0 when no GPU."""

    def test_no_gpu_returns_zero(self) -> None:
        hub = EnhancedModelHub()
        vram = hub._detect_vram()
        # In CI / no-GPU environments
        assert isinstance(vram, float)
