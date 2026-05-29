"""Tests for VRAM-aware quantization method selection.

Updated to import from distllm.dist.partition.quantization_tuner
(the Adaptive Precision Optimizer) instead of the non-existent
distllm.core.quantization_selector module.
"""

import pytest
from distllm.dist.partition.quantization_tuner import (
    NodeInfo,
    QuantMethod,
    QuantizationAutoTuner,
    select_for_node,
)


class TestNodeInfo:
    """Test NodeInfo model (replaces NodeVRAMInfo)."""

    def test_default_values(self):
        info = NodeInfo(node_id="test")
        assert info.total_memory_bytes == 8 * 1024**3
        assert info.device_type == "cuda"

    def test_custom_values(self):
        info = NodeInfo(
            node_id="test",
            total_memory_bytes=16_000_000_000,
            device_type="cuda",
        )
        assert info.total_memory_bytes == 16_000_000_000
        assert info.device_type == "cuda"


class TestSelectForNode:
    """Test quantization method selection based on VRAM."""

    def test_cpu_node_returns_none(self):
        info = NodeInfo(node_id="cpu", device_type="cpu", total_memory_bytes=16 * 1024**3)
        assert select_for_node(info, 1_000_000_000) == QuantMethod.NONE

    def test_sufficient_vram_no_quantization(self):
        # 80GB >> 4GB model -> no quantization needed
        info = NodeInfo(node_id="big", total_memory_bytes=80 * 1024**3)
        assert select_for_node(info, 4_000_000_000) == QuantMethod.NONE

    def test_moderate_vram_needs_quant(self):
        # 8GB VRAM, 14GB model -> needs quantization
        info = NodeInfo(node_id="tight", total_memory_bytes=8 * 1024**3)
        result = select_for_node(info, 14_000_000_000)
        assert result != QuantMethod.NONE

    def test_very_low_vram_returns_method(self):
        info = NodeInfo(node_id="tiny", total_memory_bytes=4 * 1024**3)
        result = select_for_node(info, 70_000_000_000)
        assert isinstance(result, QuantMethod)

    def test_returns_quant_method_type(self):
        info = NodeInfo(node_id="n0", total_memory_bytes=10 * 1024**3)
        result = select_for_node(info, 4_000_000_000)
        assert isinstance(result, QuantMethod)
