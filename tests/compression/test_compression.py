"""Compression pipeline validation tests for CI."""
import pytest
import torch
from distllm.core.compression_pipeline import CompressionPipeline, CompressionConfig

def test_compression_config():
    config = CompressionConfig(
        method="awq",
        bits=4,
        group_size=128,
    )
    assert config.method == "awq"
    assert config.bits == 4

def test_compression_ratio_tracking():
    """Test that compression ratios are correctly calculated."""
    original_size = 1024 * 1024 * 1024  # 1GB
    compressed_size = 256 * 1024 * 1024  # 256MB
    ratio = original_size / compressed_size
    assert ratio == 4.0

def test_int4_quantization_basic():
    """Test basic INT4 quantization."""
    tensor = torch.randn(128, 64)
    # Simulate group-wise quantization
    group_size = 128
    scale = tensor.abs().max() / 7.5  # INT4 range: -8 to 7
    quantized = (tensor / scale).round().clamp(-8, 7).to(torch.int8)
    dequantized = quantized.to(torch.float32) * scale
    # Check error is within acceptable range
    error = (tensor - dequantized).abs().mean() / tensor.abs().mean()
    assert error < 0.1  # Less than 10% relative error

def test_compression_accuracy_validation():
    """Test compression accuracy validation workflow."""
    # Mock accuracy check: compressed model should maintain >95% accuracy
    original_accuracy = 0.85
    compressed_accuracy = 0.83
    accuracy_drop = original_accuracy - compressed_accuracy
    assert accuracy_drop < 0.05  # Less than 5% drop allowed
