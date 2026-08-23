"""Regression tests for the C2 release-blocker.

``kv_cache.py`` imports ``apply_kv_cache_quantization`` /
``dequantize_kv_cache`` from ``quantization_selector``, but those functions had
been removed in the refactor of ``quantization_selector.py`` into a pure
selector.  Enabling KV quantization therefore crashed with ``ImportError``.
These tests exercise the restored functions and the quantized-KV round trip.
"""

import pytest
import torch

from distllm.core.kv_cache import KVCache
from distllm.core.quantization_selector import (
    apply_kv_cache_quantization,
    dequantize_kv_cache,
)


def test_imports_exist():
    """The two symbols kv_cache.py imports must be importable (C2)."""
    assert callable(apply_kv_cache_quantization)
    assert callable(dequantize_kv_cache)


@pytest.mark.parametrize(
    "bits,atol",
    [
        (8, 0.15),   # INT8: fine-grained
        (4, 0.35),   # INT4: 7 levels -> larger inherent step error
    ],
)
def test_quantize_dequantize_round_trip(bits, atol):
    key = torch.randn(2, 3, 8, 4)
    (qk, sk), (qv, sv) = apply_kv_cache_quantization(key, key, bits=bits)
    assert qk.dtype == torch.int8
    assert qk.shape == key.shape
    assert sk.shape[-1] == 1  # per-row scale

    k_restored = dequantize_kv_cache(qk, sk, bits=bits)
    assert k_restored.shape == key.shape
    # Symmetric quantization round-trips within the step error.
    assert torch.allclose(k_restored, key, atol=atol)


def test_quantized_kv_append_get_round_trip():
    """KVCache with ``_quantized=True`` must append and dequantize (C2)."""
    cache = KVCache(max_seq_len=0)
    cache.init_cache(num_layers=1, batch_size=1, num_heads=2, head_dim=4, device="cpu")

    cache._quantized = True
    cache._quant_bits = 8

    k = torch.randn(1, 2, 4, 4)
    v = torch.randn(1, 2, 4, 4)

    new_k, new_v = cache.update(0, k, v)
    assert new_k.shape == k.shape
    assert new_v.shape == v.shape

    layers = cache.get_all()
    assert len(layers) == 1
    all_k, all_v = layers[0]
    assert all_k.shape == k.shape
    assert torch.allclose(all_k, k, atol=0.15)
    assert torch.allclose(all_v, v, atol=0.15)

    # get(layer_idx) dequantizes on demand too
    gk, gv = cache.get(0)
    assert torch.allclose(gk, k, atol=0.15)
