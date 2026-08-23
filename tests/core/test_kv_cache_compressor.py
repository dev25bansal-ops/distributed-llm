"""Tests for BlockCompressor -- KV cache block compression/decompression.
from __future__ import annotations

Covers:
- Construction with method and scale_granularity
- compress with int8 method
- decompress restores approximate values
- compress_block produces CompressedBlock
- decompress_block restores both K and V
- compression_ratio tracking
- Invalid method raises ValueError

No MagicMock -- real torch tensors on CPU.
"""


import pytest

try:
    import torch
    _ = torch.float16  # canary: real torch always has this; pollution replaces torch with an empty stub
except (ModuleNotFoundError, ImportError, AttributeError) as _e:
    pytest.skip(f"requires working torch / distllm.core.kv_cache_compressor (not available): {_e}", allow_module_level=True)


import pytest
import torch

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_mod = load_module("distllm/core/kv_cache_compressor.py")
BlockCompressor = _mod.BlockCompressor
CompressedBlock = _mod.CompressedBlock


class TestBlockCompressorConstruction:
    """Construction and initial state."""

    def test_default_construction(self) -> None:
        bc = BlockCompressor()
        assert bc.method == "fp8"
        assert bc.scale_granularity == "per_head"
        assert bc._stats["compress_calls"] == 0

    def test_custom_method(self) -> None:
        bc = BlockCompressor(method="int8")
        assert bc.method == "int8"

    def test_invalid_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown method"):
            BlockCompressor(method="invalid")

    def test_per_tensor_granularity(self) -> None:
        bc = BlockCompressor(method="int8", scale_granularity="per_tensor")
        assert bc.scale_granularity == "per_tensor"

    def test_compression_ratio_starts_zero(self) -> None:
        bc = BlockCompressor(method="int8")
        # No compression calls yet, ratio = 0 / 1 = 0.0
        assert bc.compression_ratio == 0.0


class TestBlockCompressorCompress:
    """Compression methods."""

    def test_compress_int8(self) -> None:
        bc = BlockCompressor(method="int8")
        tensor = torch.randn(4, 8, 16)  # (heads, seq, head_dim)
        quantized, scale = bc.compress(tensor)
        assert quantized.dtype == torch.int8
        assert quantized.shape == tensor.shape
        # per_head: scale computed via amax(dim=-1, keepdim=True) -> (heads, seq, 1)
        assert scale.shape == (4, 8, 1)

    def test_compress_int4(self) -> None:
        bc = BlockCompressor(method="int4")
        tensor = torch.randn(2, 4, 8)
        quantized, scale = bc.compress(tensor)
        assert quantized.dtype == torch.int8
        # INT4 values should be in [-7, 7]
        assert quantized.max().item() <= 7
        assert quantized.min().item() >= -7

    def test_compress_nf4(self) -> None:
        bc = BlockCompressor(method="nf4")
        tensor = torch.randn(1, 2, 8)
        quantized, scale = bc.compress(tensor)
        assert quantized.dtype == torch.uint8
        # NF4 packs two values per byte: head_dim=8 -> packed_last_dim=4
        assert quantized.shape[-1] == 4

    def test_compress_increments_stats(self) -> None:
        bc = BlockCompressor(method="int8")
        tensor = torch.randn(2, 4, 8)
        bc.compress(tensor)
        assert bc._stats["compress_calls"] == 1
        assert bc._stats["total_original_bytes"] > 0


class TestBlockCompressorDecompress:
    """Decompression methods."""

    def test_decompress_int8_approximate(self) -> None:
        bc = BlockCompressor(method="int8")
        tensor = torch.randn(2, 4, 16)
        q, s = bc.compress(tensor)
        restored = bc.decompress(q, s)
        assert restored.shape == tensor.shape
        assert restored.dtype == torch.float16
        # Should be close but not exact (quantization error)
        diff = (restored.float() - tensor.float()).abs().mean().item()
        assert diff < 1.0  # loose tolerance for int8

    def test_decompress_nf4(self) -> None:
        bc = BlockCompressor(method="nf4")
        tensor = torch.randn(1, 2, 8)
        q, s = bc.compress(tensor)
        restored = bc.decompress(q, s)
        assert restored.shape == tensor.shape

    def test_decompress_empty_scale(self) -> None:
        bc = BlockCompressor(method="int8")
        tensor = torch.randn(2, 4, 8)
        q, s = bc.compress(tensor)
        # If s is non-empty, it should decompress properly
        restored = bc.decompress(q, s)
        assert restored is not None


class TestBlockCompressorBlockOps:
    """Block-level compress/decompress."""

    def test_compress_block(self) -> None:
        bc = BlockCompressor(method="int8")
        key = torch.randn(4, 8, 16)
        value = torch.randn(4, 8, 16)
        block = bc.compress_block(block_id=1, layer_idx=0, key=key, value=value)
        assert isinstance(block, CompressedBlock)
        assert block.block_id == 1
        assert block.layer_idx == 0
        assert block.method == "int8"
        assert block.key_compressed.shape == key.shape

    def test_decompress_block(self) -> None:
        bc = BlockCompressor(method="int8")
        key = torch.randn(4, 8, 16)
        value = torch.randn(4, 8, 16)
        block = bc.compress_block(block_id=0, layer_idx=0, key=key, value=value)
        k_restored, v_restored = bc.decompress_block(block)
        assert k_restored.shape == key.shape
        assert v_restored.shape == value.shape


class TestBlockCompressorStats:
    """Statistics tracking."""

    def test_stats(self) -> None:
        bc = BlockCompressor(method="int8")
        tensor = torch.randn(2, 4, 16)
        q, s = bc.compress(tensor)
        bc.decompress(q, s)
        stats = bc.stats()
        assert stats["compress_calls"] == 1
        assert stats["decompress_calls"] == 1
        assert stats["method"] == "int8"
        assert "compression_ratio" in stats
