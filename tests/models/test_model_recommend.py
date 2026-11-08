"""Hardware-aware model recommendation and catalog search.

The onboarding wizard used to pick models from a hardcoded VRAM-tier table
containing 14 entries, two of which were not real repository ids, and which
recommended 70B models to 16 GB cards. These tests pin the properties that
replaced it: sizing responds to parameters/dtype/context, recommendations
respect actual capacity, and MoE names are sized by total parameters.
"""

from __future__ import annotations

import pytest

from distllm.models.catalog import ModelCandidate, _moe_note, _params_from_name
from distllm.models.recommend import (
    BYTES_PER_PARAM,
    HardwareProfile,
    bytes_per_param,
    estimate_memory_gb,
    evaluate,
    rank,
)


class TestMemoryEstimate:
    def test_weights_scale_with_parameter_count(self):
        small_w, _ = estimate_memory_gb(1_000_000_000, "float16")
        big_w, _ = estimate_memory_gb(8_000_000_000, "float16")
        assert big_w > small_w * 7

    def test_int4_is_about_a_quarter_of_fp16(self):
        fp16, _ = estimate_memory_gb(7_000_000_000, "float16")
        int4, _ = estimate_memory_gb(7_000_000_000, "int4")
        assert int4 == pytest.approx(fp16 / 4, rel=0.05)

    def test_kv_cache_grows_with_context(self):
        _, kv_4k = estimate_memory_gb(7_000_000_000, "float16", context_tokens=4096)
        _, kv_32k = estimate_memory_gb(7_000_000_000, "float16", context_tokens=32768)
        assert kv_32k == pytest.approx(kv_4k * 8, rel=0.01)

    def test_kv_cache_grows_with_batch(self):
        _, kv1 = estimate_memory_gb(7_000_000_000, "float16", batch_size=1)
        _, kv8 = estimate_memory_gb(7_000_000_000, "float16", batch_size=8)
        assert kv8 == pytest.approx(kv1 * 8, rel=0.01)

    def test_unknown_dtype_defaults_to_two_bytes(self):
        assert bytes_per_param("mystery") == BYTES_PER_PARAM["float16"]

    def test_zero_params_is_zero(self):
        assert estimate_memory_gb(0) == (0.0, 0.0)


class TestEvaluate:
    def test_small_model_fits_small_gpu(self):
        hw = HardwareProfile(gpu_memory_gb=8.0, gpu_count=1)
        fit = evaluate("tiny", 1_000_000_000, hw)
        assert fit.fits
        assert fit.headroom_gb > 0

    def test_large_model_does_not_fit_small_gpu(self):
        """An 8B model needs ~18 GB in fp16; an 8 GB card cannot hold it."""
        hw = HardwareProfile(gpu_memory_gb=8.0, gpu_count=1)
        fit = evaluate("meta-llama/Llama-3.1-8B", 8_030_000_000, hw)
        assert not fit.fits
        assert any("quantization" in n for n in fit.notes)

    def test_quantization_hint_names_a_dtype_that_fits(self):
        hw = HardwareProfile(gpu_memory_gb=8.0, gpu_count=1)
        fit = evaluate("m", 8_000_000_000, hw)
        hint = next(n for n in fit.notes if "quantization" in n)
        assert "int4" in hint or "int8" in hint

    def test_multi_gpu_reports_sharding(self):
        hw = HardwareProfile(gpu_memory_gb=8.0, gpu_count=2)
        fit = evaluate("m", 4_000_000_000, hw)
        if fit.fits and fit.total_gb > hw.gpu_memory_gb:
            assert any("sharding" in n for n in fit.notes)

    def test_cpu_only_uses_system_ram(self):
        hw = HardwareProfile(ram_gb=64.0, cpu_cores=16)
        assert hw.is_cpu_only
        fit = evaluate("m", 1_000_000_000, hw)
        assert fit.fits
        assert any("CPU" in n for n in fit.notes)

    def test_cpu_only_rejects_models_larger_than_ram(self):
        hw = HardwareProfile(ram_gb=8.0, cpu_cores=4)
        fit = evaluate("m", 70_000_000_000, hw)
        assert not fit.fits


class TestRank:
    def test_prefers_largest_model_that_fits(self):
        hw = HardwareProfile(gpu_memory_gb=8.0, gpu_count=1)
        candidates = [
            ("a/1B", 1_000_000_000),
            ("a/3B", 3_000_000_000),
            ("a/70B", 70_000_000_000),
        ]
        fits, too_big = rank(candidates, hw)
        assert [f.model_id for f in fits] == ["a/3B", "a/1B"]
        assert [f.model_id for f in too_big] == ["a/70B"]

    def test_collapses_quantized_variants(self):
        hw = HardwareProfile(gpu_memory_gb=8.0, gpu_count=1)
        candidates = [
            ("meta-llama/Llama-3.2-3B-Instruct", 3_210_000_000),
            ("unsloth/Llama-3.2-3B-Instruct-GGUF", 3_210_000_000),
            ("RedHatAI/Llama-3.2-3B-Instruct-FP8-dynamic", 3_210_000_000),
        ]
        fits, _ = rank(candidates, hw)
        assert len(fits) == 1
        assert "GGUF" not in fits[0].model_id
        assert "FP8" not in fits[0].model_id

    def test_skips_unknown_sizes(self):
        hw = HardwareProfile(gpu_memory_gb=8.0, gpu_count=1)
        fits, too_big = rank([("mystery/model", 0)], hw)
        assert fits == [] and too_big == []


class TestCatalogParsing:
    def test_moe_name_uses_total_parameters(self):
        """35B-A3B must be sized by 35B: the weights all have to be resident."""
        assert _params_from_name("Qwen/Qwen3-35B-A3B") == 35_000_000_000

    def test_moe_note_describes_both_counts(self):
        note = _moe_note("nvidia/Qwen3.6-35B-A3B-NVFP4")
        assert "35B total" in note
        assert "3B active" in note

    def test_plain_name_parses(self):
        assert _params_from_name("meta-llama/Llama-3.1-8B-Instruct") == 8_000_000_000

    def test_name_without_size_returns_zero(self):
        assert _params_from_name("some-org/custom-model") == 0

    def test_candidate_size_label(self):
        assert ModelCandidate("a/b", "hf", params=8_030_000_000).size_label == "8.0B"
        assert ModelCandidate("a/b", "hf").size_label == "?"