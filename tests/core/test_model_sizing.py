"""Tests for ModelSizing -- model parameter count and VRAM estimation.

Covers:
- estimate_model_size with known models
- estimate_model_size fallback regex parsing
- estimate_model_size returns default for unknown
- estimate_num_layers
- estimate_hidden_dim
- estimate_num_kv_heads
- estimate_head_dim
- estimate_vram_gb
- estimate_vram_per_layer
- model_info returns complete dict

No MagicMock -- pure math and dict lookup.
"""

from __future__ import annotations

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_mod = load_module("distllm/core/model_sizing.py")
estimate_model_size = _mod.estimate_model_size
estimate_num_layers = _mod.estimate_num_layers
estimate_hidden_dim = _mod.estimate_hidden_dim
estimate_num_kv_heads = _mod.estimate_num_kv_heads
estimate_head_dim = _mod.estimate_head_dim
estimate_vram_gb = _mod.estimate_vram_gb
estimate_vram_per_layer = _mod.estimate_vram_per_layer
model_info = _mod.model_info


class TestModelSizingEstimateModelSize:
    """Parameter count estimation."""

    def test_known_exact_match(self) -> None:
        assert estimate_model_size("meta-llama/Llama-3.1-8B") == 8.0

    def test_known_exact_lowercase(self) -> None:
        assert estimate_model_size("llama3.1-8b") == 8.0

    def test_fallback_regex_parse(self) -> None:
        assert estimate_model_size("custom-model-70b") == 70.0

    def test_fallback_millions(self) -> None:
        result = estimate_model_size("tiny-model-350m")
        assert result == 0.35

    def test_default_for_unknown(self) -> None:
        result = estimate_model_size("completely-unknown-model-name")
        assert result == 7.0

    def test_normalize_prefix(self) -> None:
        result = estimate_model_size("hf-meta-llama/Llama-3.1-8B")
        assert result == 8.0


class TestModelSizingNumLayers:
    """Layer count estimation."""

    def test_known_model(self) -> None:
        assert estimate_num_layers("llama3.1-8b") == 32

    def test_fallback_large(self) -> None:
        assert estimate_num_layers("unknown-100b") == 80

    def test_fallback_medium(self) -> None:
        assert estimate_num_layers("unknown-7b") == 32

    def test_fallback_small(self) -> None:
        assert estimate_num_layers("unknown-1b") == 24


class TestModelSizingHiddenDim:
    """Hidden dimension estimation."""

    def test_known_model(self) -> None:
        assert estimate_hidden_dim("llama3.1-8b") == 4096

    def test_fallback_large(self) -> None:
        assert estimate_hidden_dim("unknown-100b") == 10240


class TestModelSizingKvHeads:
    """KV head estimation."""

    def test_known_model(self) -> None:
        assert estimate_num_kv_heads("llama3.1-8b") == 8

    def test_unknown_returns_32(self) -> None:
        assert estimate_num_kv_heads("unknown-1b") == 32


class TestModelSizingHeadDim:
    """Head dimension estimation."""

    def test_known_model(self) -> None:
        hd = estimate_head_dim("llama3.1-8b")
        assert hd > 0


class TestModelSizingVRAM:
    """VRAM estimation."""

    def test_vram_fp16(self) -> None:
        vram = estimate_vram_gb(8.0, dtype="fp16")
        # 8B * 2 bytes * 1.2 overhead / 1 GPU / 1e9
        assert vram > 0
        assert vram < 50  # reasonable

    def test_vram_int8(self) -> None:
        vram_fp16 = estimate_vram_gb(8.0, dtype="fp16")
        vram_int8 = estimate_vram_gb(8.0, dtype="int8")
        assert vram_int8 < vram_fp16

    def test_vram_multi_gpu(self) -> None:
        vram_1 = estimate_vram_gb(8.0, dtype="fp16", num_gpus=1)
        vram_2 = estimate_vram_gb(8.0, dtype="fp16", num_gpus=2)
        assert vram_2 < vram_1


class TestModelSizingVramPerLayer:
    """Per-layer VRAM estimation."""

    def test_vram_per_layer(self) -> None:
        vram = estimate_vram_per_layer(8.0, dtype="fp16")
        assert vram > 0

    def test_vram_per_layer_quantized(self) -> None:
        vram_fp16 = estimate_vram_per_layer(8.0, dtype="fp16")
        vram_4bit = estimate_vram_per_layer(8.0, dtype="fp16", quantization="4bit")
        assert vram_4bit < vram_fp16


class TestModelSizingInfo:
    """Complete model_info dict."""

    def test_model_info_contains_all_keys(self) -> None:
        info = model_info("llama3.1-8b")
        expected_keys = {
            "params_b", "num_layers", "hidden_dim", "num_kv_heads",
            "head_dim", "vram_fp16_gb", "vram_int8_gb", "vram_int4_gb",
        }
        assert set(info.keys()) == expected_keys
        assert info["params_b"] == 8.0
        assert info["num_layers"] == 32
        assert info["hidden_dim"] == 4096
