"""Tests for VRAM-aware quantization method selection."""

import pytest
from distllm.core.quantization_selector import (
    NodeVRAMInfo,
    select_for_node,
    estimate_model_size_bytes,
    build_quantization_config,
)


class TestNodeVRAMInfo:
    """Test NodeVRAMInfo dataclass."""

    def test_default_values(self):
        info = NodeVRAMInfo()
        assert info.total_memory == 0
        assert info.available_memory == 0
        assert info.device_type == "cpu"

    def test_custom_values(self):
        info = NodeVRAMInfo(total_memory=16e9, available_memory=12e9, device_type="cuda")
        assert info.total_memory == 16e9
        assert info.available_memory == 12e9
        assert info.device_type == "cuda"


class TestSelectForNode:
    """Test quantization method selection based on VRAM."""

    def test_cpu_node_returns_none(self):
        info = NodeVRAMInfo(device_type="cpu")
        assert select_for_node(info, 1e9) == "none"

    def test_zero_vram_returns_none(self):
        info = NodeVRAMInfo(device_type="cuda", available_memory=0)
        assert select_for_node(info, 1e9) == "none"

    def test_sufficient_vram_no_quantization(self):
        # VRAM > model * 1.8 → no quantization
        info = NodeVRAMInfo(device_type="cuda", available_memory=10e9)
        assert select_for_node(info, 4e9) == "none"

    def test_moderate_vram_8bit(self):
        # VRAM < model * 1.8 but >= model * 1.2 → 8-bit
        info = NodeVRAMInfo(device_type="cuda", available_memory=6e9)
        assert select_for_node(info, 4e9) == "bnb_8bit"

    def test_low_vram_4bit(self):
        # VRAM < model * 1.2 → 4-bit
        info = NodeVRAMInfo(device_type="cuda", available_memory=4e9)
        assert select_for_node(info, 4e9) == "bnb_4bit"

    def test_very_low_vram_4bit(self):
        # VRAM much less than model → still 4-bit (most aggressive)
        info = NodeVRAMInfo(device_type="cuda", available_memory=1e9)
        assert select_for_node(info, 4e9) == "bnb_4bit"

    def test_boundary_1_2x(self):
        # Exactly at 1.2x boundary → should be 8-bit (strictly less triggers 4-bit)
        info = NodeVRAMInfo(device_type="cuda", available_memory=6e9)
        # 6e9 == 5e9 * 1.2, so it's not < 1.2x
        # 6e9 < 5e9 * 1.8 = 9e9, so it's 8-bit
        assert select_for_node(info, 5e9) == "bnb_8bit"

    def test_target_latency_unused(self):
        # target_latency_ms is reserved for future use
        info = NodeVRAMInfo(device_type="cuda", available_memory=10e9)
        assert select_for_node(info, 4e9, target_latency_ms=100.0) == "none"


class TestEstimateModelSize:
    """Test model size estimation."""

    def test_small_model(self):
        # TinyStories-like: hidden=64, layers=4, vocab=5000
        size = estimate_model_size_bytes(64, 4, 5000)
        assert size > 0
        # Roughly: 2 * (5000*64) + 4 * 4 * 64^2 = 640000 + 65536 = ~705K params
        assert size < 10e6  # < 10MB in fp16

    def test_medium_model(self):
        # GPT-2 small: hidden=768, layers=12, vocab=50257
        size = estimate_model_size_bytes(768, 12, 50257)
        assert size > 100e6  # > 100MB

    def test_fp32_larger_than_fp16(self):
        size_32 = estimate_model_size_bytes(768, 12, 50257, dtype_bytes=4)
        size_16 = estimate_model_size_bytes(768, 12, 50257, dtype_bytes=2)
        assert size_32 == size_16 * 2

    def test_more_layers_increases_size(self):
        size_12 = estimate_model_size_bytes(768, 12, 50257)
        size_24 = estimate_model_size_bytes(768, 24, 50257)
        assert size_24 > size_12


class TestBuildQuantizationConfig:
    """Test BitsAndBytesConfig creation."""

    def test_none_method(self):
        assert build_quantization_config("none") is None

    def test_bnb_8bit(self):
        config = build_quantization_config("bnb_8bit")
        assert config is not None
        assert config.load_in_8bit is True
        assert config.llm_int8_threshold == 6.0

    def test_bnb_8bit_custom_threshold(self):
        config = build_quantization_config("bnb_8bit", llm_int8_threshold=8.0)
        assert config.llm_int8_threshold == 8.0

    def test_bnb_4bit(self):
        config = build_quantization_config("bnb_4bit")
        assert config is not None
        assert config.load_in_4bit is True
        assert config.bnb_4bit_quant_type == "nf4"
        assert config.bnb_4bit_use_double_quant is True

    def test_unknown_method(self):
        config = build_quantization_config("gptq")
        assert config is None
