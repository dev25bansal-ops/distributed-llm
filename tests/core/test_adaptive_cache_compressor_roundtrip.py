"""Round-trip regression tests for FP8/INT4 cache compression (audit C11).

Root cause: ``AdaptiveCacheCompressor._compress_fp8`` / ``_compress_int4``
computed per-row amax scales but returned only the quantized tensors,
discarding the scales -- stored blocks were mathematically undecodable and
any consumer got values silently scaled by an unknown factor.

The fix returns a ``CompressedKVBlock`` carrying quantized tensors plus
their scale tensors and original dtype, and adds ``decompress()`` which
multiplies the scales back in.
"""

from __future__ import annotations

import pytest
import torch

from distllm.core.adaptive_cache_compressor import (
    AdaptiveCacheCompressor,
    CompressedKVBlock,
)

HAS_FP8 = hasattr(torch, "float8_e4m3fn")

requires_fp8 = pytest.mark.skipif(
    not HAS_FP8, reason="torch build lacks float8_e4m3fn"
)


def _kv_data(
    layers: int = 2,
    seq: int = 16,
    heads: int = 4,
    dim: int = 32,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> dict:
    gen = torch.Generator().manual_seed(seed)
    return {
        f"layer_{i}": (
            torch.randn(seq, heads, dim, dtype=dtype, generator=gen),
            torch.randn(seq, heads, dim, dtype=dtype, generator=gen),
        )
        for i in range(layers)
    }


def _assert_close(orig: torch.Tensor, rec: torch.Tensor, atol: float, rtol: float):
    assert rec.dtype == orig.dtype
    assert rec.shape == orig.shape
    torch.testing.assert_close(rec, orig, rtol=rtol, atol=atol)


class TestFP8RoundTrip:
    @requires_fp8
    def test_round_trip_within_fp8_tolerance(self):
        compressor = AdaptiveCacheCompressor()
        kv = _kv_data()
        block = compressor.compress_for_tier(kv, "gpu")

        # Scales must now travel with the payload...
        assert isinstance(block, CompressedKVBlock)
        assert block.method == "fp8"
        assert len(block.entries) == len(kv)
        for entry in block.entries.values():
            assert entry["k_scale"] is not None
            assert entry["v_scale"] is not None

        restored = compressor.decompress(block)
        for key, (k, v) in kv.items():
            rk, rv = restored[key]
            amax = max(k.abs().max().item(), v.abs().max().item(), 1e-6)
            # e4m3 relative error <= 2^-4 (~6.25%); small elements may flush
            # to zero, covered by the amplitude-scaled atol.
            _assert_close(k, rk, atol=0.02 * amax, rtol=0.15)
            _assert_close(v, rv, atol=0.02 * amax, rtol=0.15)

    @requires_fp8
    def test_scales_match_compress_time_amax(self):
        """Stored scale equals amax/448 per row -- i.e. real decode data."""
        compressor = AdaptiveCacheCompressor()
        kv = _kv_data(layers=1)
        k, _ = kv["layer_0"]
        block = compressor.compress_for_tier(kv, "gpu")
        expected = k.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 448.0
        torch.testing.assert_close(
            block.entries["layer_0"]["k_scale"], expected
        )

    @requires_fp8
    def test_fp16_originals_restore_to_fp16(self):
        compressor = AdaptiveCacheCompressor()
        kv = _kv_data(dtype=torch.float16)
        restored = compressor.decompress(compressor.compress_for_tier(kv, "gpu"))
        for key, (k, _) in kv.items():
            assert restored[key][0].dtype == torch.float16
            amax = k.abs().max().item()
            _assert_close(
                k, restored[key][0], atol=0.02 * max(amax, 1e-6), rtol=0.15
            )

    @requires_fp8
    def test_zero_tensor_round_trips_exactly(self):
        compressor = AdaptiveCacheCompressor()
        kv = {"z": (torch.zeros(4, 8), torch.zeros(4, 8))}
        restored = compressor.decompress(compressor.compress_for_tier(kv, "gpu"))
        assert restored["z"][0].abs().sum().item() == 0.0
        assert restored["z"][1].abs().sum().item() == 0.0


class TestInt4RoundTrip:
    def test_round_trip_within_int4_tolerance(self):
        compressor = AdaptiveCacheCompressor()
        kv = _kv_data(seed=7)
        block = compressor.compress_for_tier(kv, "disk")

        assert isinstance(block, CompressedKVBlock)
        assert block.method == "int4"
        for entry in block.entries.values():
            assert entry["k"].dtype == torch.int8
            assert entry["k_scale"] is not None

        restored = compressor.decompress(block)
        for key, (k, v) in kv.items():
            rk, rv = restored[key]
            amax = max(k.abs().max().item(), v.abs().max().item(), 1e-6)
            # Pre-existing quantization semantics: (x / scale).to(int8)
            # TRUNCATES toward zero, so |err| < 1 full step = scale =
            # amax/7 (~14.3% of amax); small elements may flush to 0,
            # covered by the amplitude-scaled atol.
            _assert_close(k, rk, atol=0.16 * amax, rtol=0.05)
            _assert_close(v, rv, atol=0.16 * amax, rtol=0.05)

    def test_int4_values_stay_in_4bit_range(self):
        compressor = AdaptiveCacheCompressor()
        block = compressor.compress_for_tier(_kv_data(), "disk")
        for entry in block.entries.values():
            assert entry["k"].abs().max().item() <= 7
            assert entry["v"].abs().max().item() <= 7


class TestContainerAndFallbacks:
    def test_non_quantizable_values_pass_through_and_restore(self):
        compressor = AdaptiveCacheCompressor()
        sentinel_k = object()  # no '.to' attribute -> passthrough
        kv = {
            "meta": ("not-a-tensor-pair",),
            "odd": (sentinel_k, 42),
            "good": (torch.randn(8, 16), torch.randn(8, 16)),
        }
        block = compressor.compress_for_tier(kv, "disk")
        assert isinstance(block, CompressedKVBlock)
        assert "good" in block.entries
        assert set(block.passthrough) == {"meta", "odd"}

        restored = compressor.decompress(block)
        assert restored["meta"] == ("not-a-tensor-pair",)
        assert restored["odd"][0] is sentinel_k
        assert "good" in restored

    def test_unknown_tier_returns_input_uncompressed(self):
        compressor = AdaptiveCacheCompressor()
        kv = {"a": 1}
        assert compressor.compress_for_tier(kv, "tape") is kv

    def test_decompress_of_raw_payload_is_identity(self):
        """Payloads returned uncompressed (no fp8 support / fallback paths)
        must survive decompress unchanged."""
        compressor = AdaptiveCacheCompressor()
        raw = {"k": (torch.randn(4, 4), torch.randn(4, 4))}
        assert compressor.decompress(raw) is raw

    def test_stats_counters_still_incremented(self):
        compressor = AdaptiveCacheCompressor()
        if HAS_FP8:
            compressor.compress_for_tier(_kv_data(layers=2), "gpu")
        compressor.compress_for_tier(_kv_data(layers=3), "disk")
        stats = compressor.get_stats()
        assert stats["disk"]["compressed"] == 3
        if HAS_FP8:
            assert stats["gpu"]["compressed"] == 2

    def test_peer_tier_shape_preserved(self):
        compressor = AdaptiveCacheCompressor()
        t = torch.randn(8, 4, 6)
        out = compressor.compress_for_tier({"h": t}, "peer")
        assert out["h"].shape == t.shape
