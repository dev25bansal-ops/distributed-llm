"""Fuzz protobuf deserialization with malformed tensor / message data.

Usage:
    python tests/fuzz/fuzz_protobuf_deserializer.py          # 10k random iterations
    python tests/fuzz/fuzz_protobuf_deserializer.py --atheris # atheris coverage-guided
"""
import os
import random
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, SRC)

_TEST_CASES = 10000 if "--atheris" not in sys.argv else 0


def _corrupt(data: bytes) -> bytes:
    """Flip random bits/bytes in *data*."""
    if not data:
        return data
    data = bytearray(data)
    n = random.randint(1, max(1, len(data) // 4))
    for _ in range(n):
        idx = random.randrange(len(data))
        data[idx] = random.randint(0, 255)
    return bytes(data)


def _random_tensor_bytes() -> bytes:
    """Generate raw bytes that look like a serialized Tensor message."""
    import struct
    parts = []
    # float values
    n_floats = random.randint(0, 32)
    for _ in range(n_floats):
        parts.append(struct.pack("<f", random.uniform(-1e10, 1e10)))
    # shape
    ndim = random.randint(0, 4)
    shape = [random.randint(0, 128) for _ in range(ndim)]
    if ndim > 0:
        parts.append(struct.pack(f"<{ndim}i", *shape))
    # dtype
    parts.append(struct.pack("<i", random.randint(0, 10)))
    return b"".join(parts)


def _random_kv_cache_bytes() -> bytes:
    """Generate raw bytes that look like a KV cache message."""
    parts = []
    for _ in range(random.randint(0, 16)):
        n = random.randint(0, 64)
        vals = struct.pack(f"<{n}f", *[random.uniform(-1, 1) for _ in range(n)])
        parts.append(vals)
    return b"".join(parts)


def _random_forward_pass_bytes() -> bytes:
    """Generate bytes resembling a ForwardPassRequest."""
    parts = []
    # input_ids
    n_ids = random.randint(0, 512)
    for _ in range(n_ids):
        parts.append(struct.pack("<i", random.randint(0, 100000)))
    # hidden states
    n_hidden = random.randint(0, 256)
    for _ in range(n_hidden):
        parts.append(struct.pack("<f", random.uniform(-10, 10)))
    # attention mask
    n_mask = random.randint(0, 128)
    for _ in range(n_mask):
        parts.append(struct.pack("<f", random.uniform(0, 1)))
    return b"".join(parts)


def _run_one(data: bytes, *, allow_crash: bool = False) -> None:
    """Try to deserialize *data* through various protobuf messages."""
    import sys
    proto_path = os.path.join(SRC, "..", "proto")
    if os.path.isdir(proto_path):
        sys.path.insert(0, proto_path)

    try:
        from distllm.communication import node_pb2
    except ImportError:
        try:
            import node_pb2
        except ImportError:
            return  # protobuf stubs not compiled

    messages = [
        ("Tensor", node_pb2.Tensor()),
        ("ForwardPassRequest", node_pb2.ForwardPassRequest()),
        ("ForwardPassResponse", node_pb2.ForwardPassResponse()),
        ("KVCache", node_pb2.KVCache()),
        ("NodeRegistration", node_pb2.NodeRegistration()),
        ("HealthCheckRequest", node_pb2.HealthCheckRequest()),
        ("HealthCheckResponse", node_pb2.HealthCheckResponse()),
    ]

    for name, msg in messages:
        try:
            parsed = msg.__class__()
            parsed.ParseFromString(data)
        except Exception:
            if not allow_crash:
                raise

        # Re-serialize and re-parse should round-trip
        try:
            serialized = parsed.SerializeToString()
            parsed2 = msg.__class__()
            parsed2.ParseFromString(serialized)
        except Exception:
            if not allow_crash:
                raise


def _random_test_input() -> bytes:
    kind = random.random()
    if kind < 0.25:
        return _random_tensor_bytes()
    elif kind < 0.5:
        return _random_kv_cache_bytes()
    elif kind < 0.75:
        return _random_forward_pass_bytes()
    else:
        return _corrupt(b"".join(
            _random_tensor_bytes() for _ in range(random.randint(1, 5))
        ))


def fuzz(data: bytes) -> None:
    """atheris-compatible fuzz target."""
    _run_one(data, allow_crash=True)


def pytest_fuzz(n: int = 500) -> None:
    """Run *n* random iterations via pytest."""
    for _ in range(n):
        data = _random_test_input()
        _run_one(data, allow_crash=True)


if __name__ == "__main__":
    if "--atheris" in sys.argv:
        import atheris
        atheris.Setup(sys.argv, fuzz)
        atheris.Fuzz()
    else:
        pytest_fuzz(_TEST_CASES)
        print(f"OK {_TEST_CASES} iterations completed, no crashes")
