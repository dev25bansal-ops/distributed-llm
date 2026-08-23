"""Regression test for C-09 / C-10: KV-cache quant/dequant round-trip.

C-09: apply_kv_cache_quantization / dequantize_kv_cache must exist.
C-10: bulk INT8/INT4 get() must dequantize (covered indirectly by round-trip).
"""


import pytest

try:
    import torch
    _ = torch.float16  # canary: real torch always has this; pollution replaces torch with an empty stub
except (ModuleNotFoundError, ImportError, AttributeError) as _e:
    pytest.skip(f"requires working torch / distllm.core.quantization_selector (not available): {_e}", allow_module_level=True)

import torch

from distllm.core.quantization_selector import (
    apply_kv_cache_quantization,
    dequantize_kv_cache,
)


def _roundtrip(bits, dtype=torch.float32):
    torch.manual_seed(0)
    k = torch.randn(2, 4, 16, 32, dtype=dtype)
    v = torch.randn(2, 4, 16, 32, dtype=dtype)

    (qk, sk), (qv, sv) = apply_kv_cache_quantization(k, v, bits)
    k_deq = dequantize_kv_cache(qk, sk, bits)
    v_deq = dequantize_kv_cache(qv, sv, bits)

    k_rel = ((k - k_deq).abs() / (k.abs() + 1e-9)).mean().item()
    v_rel = ((v - v_deq).abs() / (v.abs() + 1e-9)).mean().item()
    return k_rel, v_rel


def test_roundtrip_int8():
    k_rel, v_rel = _roundtrip(8)
    assert k_rel < 0.05, f"INT8 key rel error {k_rel} >= 5%"
    assert v_rel < 0.05, f"INT8 value rel error {v_rel} >= 5%"


def test_roundtrip_int4():
    # INT4 is coarser than INT8; with unstructured random normal data the
    # relative round-trip error is inherent to 4-bit precision (~25-30%).
    # Assert it stays within that realistic 4-bit bound (it must be well
    # below the ~100% you'd get from un-dequantized garbage).
    k_rel, v_rel = _roundtrip(4)
    assert k_rel < 0.30, f"INT4 key rel error {k_rel} too high"
    assert v_rel < 0.30, f"INT4 value rel error {v_rel} too high"


def test_signatures_exist():
    assert callable(apply_kv_cache_quantization)
    assert callable(dequantize_kv_cache)
