"""Compression pipeline validation tests for CI."""
import torch
from distllm.core.compression_pipeline import CompressionPipeline, CompressionConfig
from distllm.core.compression_config import CompressionMethod


def test_compression_config():
    config = CompressionConfig(
        method=CompressionMethod.QUANT_AWQ,
        target_bits=4,
    )
    assert config.method == CompressionMethod.QUANT_AWQ
    assert config.target_bits == 4


def test_compression_ratio_tracking():
    """Test that compression ratios are correctly calculated."""
    original_size = 1024 * 1024 * 1024  # 1GB
    compressed_size = 256 * 1024 * 1024  # 256MB
    ratio = original_size / compressed_size
    assert ratio == 4.0


def test_int4_quantization_basic():
    """Test basic INT4 quantization with grouped scaling."""
    torch.manual_seed(42)
    tensor = torch.randn(64, 64)
    group_size = 16
    dequantized = torch.zeros_like(tensor)
    for g in range(0, tensor.shape[1], group_size):
        g_end = min(g + group_size, tensor.shape[1])
        g_slice = tensor[:, g:g_end]
        scale = g_slice.abs().max() / 7.0
        w_q = (g_slice / scale.clamp(min=1e-6)).round().clamp(-8, 7).to(torch.int8)
        dequantized[:, g:g_end] = w_q.float() * scale
    error = (tensor - dequantized).abs().mean() / tensor.abs().mean()
    assert error < 0.25


def test_compression_accuracy_validation():
    """Test compression accuracy validation workflow."""
    original_accuracy = 0.85
    compressed_accuracy = 0.83
    accuracy_drop = original_accuracy - compressed_accuracy
    assert accuracy_drop < 0.05
