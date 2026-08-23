"""Regression tests for audit finding F-021.

``KVCache.get()`` only dequantized the bulk-compress path when
``_quant_fp8`` was True.  ``compress("int8")``, ``compress("int4")`` and
``AdaptiveQuantizer.apply()`` all set ``_quant_fp8=False`` while populating
``_scale_k``/``_scale_v``, so ``get()`` returned the raw int8 quantized
tensors (values in [-127, 127] or [-7, 7]) to attention instead of the
dequantized approximations of the original fp16 values — silently corrupting
generation whenever non-FP8 bulk compression was used.

These tests assert that ``get()`` after every bulk/adaptive compression path
recovers approximately the original values (not merely a cast).
"""

from __future__ import annotations

import pytest
import torch

from distllm.core.kv_cache import KVCache


def _make_cache(num_layers: int = 2) -> tuple[KVCache, list[tuple[torch.Tensor, torch.Tensor]]]:
    torch.manual_seed(0)
    cache = KVCache()
    cache.init_cache(num_layers=num_layers, batch_size=1, num_heads=2, head_dim=8, device="cpu")
    originals = []
    for layer_idx in range(num_layers):
        k = torch.randn(1, 2, 4, 8)
        v = torch.randn(1, 2, 4, 8)
        cache.cache[layer_idx] = (k, v)
        originals.append((k, v))
    return cache, originals


@pytest.mark.parametrize(
    "method,atol",
    [
        ("int8", 0.15),
        # INT4 bulk compress() truncates (no round) before casting, so the
        # reconstruction error approaches one full step (amax / 7).
        ("int4", 0.6),
    ],
)
def test_get_dequantizes_bulk_int_compression(method, atol):
    """get() after compress('int8'/'int4') must return dequantized floats."""
    cache, originals = _make_cache()
    stats = cache.compress(method)
    assert stats["method"] == method

    for layer_idx, (k, v) in enumerate(originals):
        k_out, v_out = cache.get(layer_idx)
        # Must NOT be the raw int8 tensor anymore.
        assert k_out.dtype.is_floating_point, (
            f"{method}: get() returned raw quantized dtype {k_out.dtype}"
        )
        assert torch.allclose(k_out, k, atol=atol), f"{method} key mismatch on layer {layer_idx}"
        assert torch.allclose(v_out, v, atol=atol), f"{method} value mismatch on layer {layer_idx}"


def test_get_dequantizes_fp8_still_works():
    """The previously-working fp8 path must keep working (fp8 dtype preserved)."""
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch.float8_e4m3fn not available")
    cache, originals = _make_cache(num_layers=1)
    stats = cache.compress("fp8")
    assert stats["method"] == "fp8"
    k, v = originals[0]
    k_out, v_out = cache.get(0)
    assert k_out.dtype == torch.float8_e4m3fn  # legacy contract: stays fp8
    # E4M3 has ~6% relative precision; compare after promoting to float.
    assert torch.allclose(k_out.float(), k, atol=0.3)
    assert torch.allclose(v_out.float(), v, atol=0.3)


def test_get_adaptive_mixed_plan():
    """AdaptiveQuantizer.apply() with an int4/int8/fp16 plan must serve
    dequantized values for quantized layers and pass fp16 layers through."""
    from distllm.core.kv_cache import AdaptiveQuantizer

    cache, originals = _make_cache(num_layers=2)
    # Layer 0 -> int4, layer 1 -> int8
    plan = {0: "int4", 1: "int8"}
    stats = AdaptiveQuantizer().apply(cache, plan)
    assert stats["method"] == "adaptive_mixed"

    k0, v0 = originals[0]
    k0_out, v0_out = cache.get(0)
    assert k0_out.dtype.is_floating_point, f"raw dtype served: {k0_out.dtype}"
    assert torch.allclose(k0_out, k0, atol=0.35)
    assert torch.allclose(v0_out, v0, atol=0.35)

    k1, v1 = originals[1]
    k1_out, v1_out = cache.get(1)
    assert k1_out.dtype.is_floating_point, f"raw dtype served: {k1_out.dtype}"
    assert torch.allclose(k1_out, k1, atol=0.15)
    assert torch.allclose(v1_out, v1, atol=0.15)


def test_get_adaptive_fp16_layer_passthrough():
    """fp16 layers inside an adaptive plan have no scale and must be
    returned unchanged (no spurious multiplication)."""
    from distllm.core.kv_cache import AdaptiveQuantizer

    cache, originals = _make_cache(num_layers=1)
    AdaptiveQuantizer().apply(cache, {0: "fp16"})
    k, v = originals[0]
    k_out, v_out = cache.get(0)
    assert k_out is k
    assert v_out is v


def test_adaptive_quantizer_apply_sets_no_scale_for_fp16():
    """Contract check: adaptive apply() stores None scales for fp16 layers,
    which get() must treat as 'no dequantization needed'."""
    from distllm.core.kv_cache import AdaptiveQuantizer

    cache, _ = _make_cache(num_layers=2)
    AdaptiveQuantizer().apply(cache, {0: "int8", 1: "fp16"})
    sk, sv = cache.get_scales()
    assert sk[0] is not None and sv[0] is not None
    assert sk[1] is None and sv[1] is None
