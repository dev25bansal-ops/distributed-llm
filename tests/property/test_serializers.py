"""Property-based tests for tensor serialization round-trips."""

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from distllm.communication.serializers import proto_to_tensor, tensor_to_proto


@st.composite
def tensor_strategy(draw):
    """Generate arbitrary PyTorch tensors with diverse shapes and dtypes."""
    dtype = draw(
        st.sampled_from(
            [
                torch.float32,
                torch.float16,
                torch.float64,
                torch.int32,
                torch.int64,
                torch.int8,
                torch.bool,
            ]
        )
    )
    ndim = draw(st.integers(0, 4))
    # Limit sizes to keep tests fast
    dims = [draw(st.integers(1, 8)) for _ in range(ndim)]
    shape = tuple(dims)

    if dtype == torch.bool:
        n = int(torch.prod(torch.tensor(shape)).item() if shape else 1)
        data = [draw(st.booleans()) for _ in range(n)]
        if shape:
            return torch.tensor(data, dtype=dtype).reshape(shape)
        else:
            return torch.tensor(data[0] if data else False, dtype=dtype)
    elif dtype in (torch.int32, torch.int64, torch.int8):
        num_elements = int(torch.prod(torch.tensor(shape)).item() if shape else 1)
        data = [draw(st.integers(-127, 127)) for _ in range(num_elements)]
        if shape:
            return torch.tensor(data, dtype=dtype).reshape(shape)
        else:
            return torch.tensor(data[0] if data else 0, dtype=dtype)
    else:
        num_elements = int(torch.prod(torch.tensor(shape)).item() if shape else 1)
        floats = st.floats(-1e4, 1e4, allow_nan=False, allow_infinity=False)
        data = [draw(floats) for _ in range(num_elements)]
        if shape:
            return torch.tensor(data, dtype=dtype).reshape(shape)
        else:
            return torch.tensor(data[0] if data else 0.0, dtype=dtype)


@given(tensor_strategy())
@settings(max_examples=100, deadline=None)
def test_tensor_roundtrip(tensor):
    """Any tensor should survive serialization → deserialization round-trip."""
    proto = tensor_to_proto(tensor)
    restored = proto_to_tensor(proto)

    assert restored.shape == tensor.shape, f"Shape mismatch: {restored.shape} vs {tensor.shape}"
    assert restored.dtype == tensor.dtype, f"Dtype mismatch: {restored.dtype} vs {tensor.dtype}"

    # For float types, allow small numerical differences
    if tensor.dtype in (torch.float16, torch.float32, torch.float64):
        if tensor.numel() > 0:
            assert torch.allclose(restored, tensor, atol=1e-3, rtol=1e-3)
    else:
        if tensor.numel() > 0:
            assert torch.equal(restored, tensor)


def test_none_tensor():
    """None should serialize to empty proto."""
    proto = tensor_to_proto(None)
    assert proto.shape == []
    assert proto.dtype == "none"


@given(st.integers(1, 32), st.integers(1, 32))
@settings(max_examples=30)
def test_2d_tensor_various_shapes(rows, cols):
    """2D tensors of arbitrary size should round-trip correctly."""
    tensor = torch.randn(rows, cols, dtype=torch.float32)
    proto = tensor_to_proto(tensor)
    restored = proto_to_tensor(proto)
    assert restored.shape == (rows, cols)
    assert torch.allclose(restored, tensor, atol=1e-5)


def test_bfloat16_roundtrip():
    """bfloat16 tensors should round-trip without data loss."""
    tensor = torch.randn(8, 16, dtype=torch.bfloat16)
    proto = tensor_to_proto(tensor)
    assert proto.dtype == "torch.bfloat16"
    restored = proto_to_tensor(proto)
    assert restored.dtype == torch.bfloat16
    assert restored.shape == tensor.shape
    assert torch.equal(restored, tensor)
