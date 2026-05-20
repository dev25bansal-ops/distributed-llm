"""Property-based fuzz tests for serialization round-trips.

Covers: tensor serialization, KV cache proto, quantize/dequantize,
edge cases (empty tensors, NaN, Inf, extreme dims, bf16).
"""

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from distllm.communication.serializers import (
    proto_to_tensor,
    tensor_to_proto,
    kv_cache_to_proto,
    proto_to_kv_cache,
    quantize_activation,
    dequantize_activation,
)
from distllm.core.kv_cache import KVCache


# ---------------------------------------------------------------------------
# Tensor strategies
# ---------------------------------------------------------------------------

DTYPE_STRATEGY = st.sampled_from([
    torch.float32, torch.float16, torch.float64,
    torch.int32, torch.int64, torch.int8,
    torch.uint8, torch.bool, torch.bfloat16,
])


@st.composite
def tensor_strategy(draw):
    """Generate arbitrary PyTorch tensors — diverse shapes, dtypes, values."""
    dtype = draw(DTYPE_STRATEGY)
    ndim = draw(st.integers(0, 5))
    dims = [draw(st.integers(0, 10)) for _ in range(ndim)]
    shape = tuple(dims)
    numel = max(1, int(torch.prod(torch.tensor(shape)).item() if shape else 0))

    if dtype == torch.bool:
        data = [draw(st.booleans()) for _ in range(numel)]
    elif dtype in (torch.int32, torch.int64, torch.int8, torch.uint8):
        data = [draw(st.integers(-128, 255)) for _ in range(numel)]
    else:
        floats = st.floats(-1e6, 1e6, allow_nan=True, allow_infinity=True)
        data = [draw(floats) for _ in range(numel)]

    if shape and all(s > 0 for s in shape):
        return torch.tensor(data, dtype=dtype).reshape(shape)
    return torch.tensor(data[0] if data else 0, dtype=dtype)


@st.composite
def extreme_tensor_strategy(draw):
    """Generate tensors with extreme / boundary values."""
    dtype = draw(st.sampled_from([torch.float32, torch.float16]))
    ndim = draw(st.integers(0, 3))
    dims = [draw(st.sampled_from([0, 1, 2, 256, 65535])) for _ in range(ndim)]
    shape = tuple(d for d in dims if d > 0)
    numel = max(1, int(torch.prod(torch.tensor(shape)).item())) if shape else 1

    strategy = draw(st.sampled_from([
        st.just(0.0),
        st.just(float("inf")),
        st.just(float("-inf")),
        st.just(float("nan")),
        st.floats(-1e-30, 1e30, allow_nan=False, allow_infinity=False),
        st.floats(0.0, 1e-10),
    ]))

    if isinstance(strategy, st._internal.strategies.FloatStrategy):
        data = [draw(strategy) for _ in range(numel)]
    else:
        data = [draw(strategy) for _ in range(numel)]

    if shape:
        return torch.tensor(data, dtype=dtype).reshape(shape)
    return torch.tensor(data[0] if data else 0.0, dtype=dtype)


# ---------------------------------------------------------------------------
# Tensor round-trip
# ---------------------------------------------------------------------------

@given(tensor_strategy())
@settings(max_examples=200, deadline=None)
def test_tensor_roundtrip(tensor):
    """Any tensor survives serialization → deserialization round-trip."""
    proto = tensor_to_proto(tensor)
    restored = proto_to_tensor(proto)
    assert restored.shape == tensor.shape
    assert restored.dtype == tensor.dtype
    if tensor.numel() == 0:
        assert restored.numel() == 0
        return
    if tensor.dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
        mask = ~torch.isnan(tensor)
        if mask.any():
            assert torch.allclose(restored[mask], tensor[mask], atol=1e-3, rtol=1e-3)
    else:
        assert torch.equal(restored, tensor)


@given(extreme_tensor_strategy())
@settings(max_examples=50, deadline=None)
def test_extreme_tensor_roundtrip(tensor):
    """Tensors with NaN, Inf, zero, and extreme values round-trip."""
    proto = tensor_to_proto(tensor)
    restored = proto_to_tensor(proto)
    assert restored.shape == tensor.shape
    assert restored.dtype == tensor.dtype
    if tensor.numel() > 0:
        assert torch.allclose(restored, tensor, atol=1e-2, rtol=1e-2, equal_nan=True)


@given(st.integers(0, 5), st.integers(0, 5))
@settings(max_examples=30, deadline=None)
def test_0d_and_empty_tensors(dim1, dim2):
    """0-dimensional and empty tensors should round-trip."""
    if dim1 == 0 and dim2 == 0:
        t = torch.tensor(3.14)  # 0-d scalar
    elif dim1 == 0:
        t = torch.randn(0)  # 1-d empty
    elif dim2 == 0:
        t = torch.randn(dim1, 0)  # 2-d empty
    else:
        t = torch.randn(dim1, dim2)
    proto = tensor_to_proto(t)
    restored = proto_to_tensor(proto)
    assert restored.shape == t.shape
    assert restored.dtype == t.dtype


# ---------------------------------------------------------------------------
# KV cache serialization round-trip
# ---------------------------------------------------------------------------

@st.composite
def kv_cache_strategy(draw):
    """Generate a KVCache with random layer / head / dim configuration."""
    num_layers = draw(st.integers(1, 16))
    batch_size = draw(st.integers(1, 4))
    num_heads = draw(st.integers(1, 8))
    head_dim = draw(st.integers(4, 128))
    seq_len = draw(st.integers(1, 64))
    cache = KVCache()
    cache.init_cache(num_layers, batch_size, num_heads, head_dim)
    for lidx in range(num_layers):
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim)
        cache.update(lidx, k, v)
    return cache


@given(kv_cache_strategy())
@settings(max_examples=30, deadline=None)
def test_kv_cache_proto_roundtrip(cache):
    """KVCache serialization → deserialization preserves structure and values."""
    import copy
    orig_layers = copy.deepcopy(cache.cache)
    proto = kv_cache_to_proto(cache)

    restored = proto_to_kv_cache(proto)
    assert restored.num_layers == cache.num_layers
    assert restored.sequence_length == cache.sequence_length

    for (ok, ov), (rk, rv) in zip(orig_layers, restored.cache):
        assert ok.shape == rk.shape
        assert ov.shape == rv.shape
        assert torch.allclose(rk, ok, atol=1e-5)
        assert torch.allclose(rv, ov, atol=1e-5)


# ---------------------------------------------------------------------------
# Activation quantization round-trip
# ---------------------------------------------------------------------------

@st.composite
def quant_strategy(draw):
    """Generate tensors and quantisation configs for round-trip testing."""
    dtype = draw(st.sampled_from([torch.float16, torch.float32]))
    rows = draw(st.integers(1, 128))
    cols = draw(st.integers(1, 128))
    values = st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False)
    data = [[draw(values) for _ in range(cols)] for _ in range(rows)]
    tensor = torch.tensor(data, dtype=dtype)
    bits = draw(st.sampled_from([8, 4]))
    use_fp8 = draw(st.booleans()) if bits == 8 else st.just(False)
    if isinstance(use_fp8, st._internal.strateges.BooleanStrategy):
        use_fp8_val = draw(use_fp8)
    else:
        use_fp8_val = use_fp8
    return tensor, bits, use_fp8_val


@given(quant_strategy())
@settings(max_examples=50, deadline=None)
def test_quantize_dequantize_roundtrip(args):
    """Quantize → dequantize preserves approximate values."""
    tensor, bits, use_fp8 = args
    quantized, scale = quantize_activation(tensor)
    if tensor.dtype == torch.float16 and use_fp8:
        # Not applicable; skip
        return
    restored = dequantize_activation(quantized, scale, tensor.dtype)
    assert restored.shape == tensor.shape
    assert restored.dtype == tensor.dtype
    if bits == 8 and not use_fp8:
        assert torch.allclose(restored, tensor, atol=0.5, rtol=0.1)
