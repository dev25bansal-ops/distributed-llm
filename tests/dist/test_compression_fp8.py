"""Regression tests for F-031: AdaptiveSerializer ZSTD path corrupting large FP8 tensors.

Two defects were adversarially verified:

1. ``SerializationController.send_tensor`` with the *ZSTD* method took the
   already-quantised, scale-packed, zstd-compressed ``FP8_ZSTD`` payload
   produced by :meth:`AdaptiveSerializer.serialize` for large tensors and
   re-compressed it (double compression) and relabelled it as plain ``ZSTD``,
   so the receiver never unpacks the scale and never dequantises.

2. The ``FP8_ZSTD`` deserialisation path reconstructed the quantised chunk as
   ``uint8`` (from ``_deserialize_raw``), so :func:`tensor_dequantize` could
   not recognise the ``float8_e4m3fn`` dtype and reinterpreted the FP8 bytes as
   uint8->float16 garbage (wrong values, scale effectively lost).

These tests pin FP8 tensors round-tripping correctly (FP8-scale/precision
accurate) through both the direct and the negotiated-ZSTD paths.
"""

from __future__ import annotations

import zstandard

import pytest
import torch

from distllm.dist.pipeline.compression_negotiation import (
    AdaptiveSerializer,
    CompressionMethod,
    SerializationController,
    SerializationFormat,
)

# Skip the whole module if this torch build lacks FP8 (older torch).
pytestmark = pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn"),
    reason="torch.float8_e4m3fn not available",
)


class _LoopbackTransport:
    """Real in-process transport: bytes stored per peer, no mocks."""

    def __init__(self) -> None:
        self._buf: dict[str, bytes] = {}

    def send(self, peer: str, wire: bytes) -> None:
        self._buf[peer] = wire

    def recv(self, peer: str) -> bytes:
        return self._buf.pop(peer)


def _fp8_roundtrip_reference(tensor: torch.Tensor) -> torch.Tensor:
    """The exact values FP8 serialisation is expected to reproduce."""
    return tensor.to(torch.float8_e4m3fn).to(tensor.dtype)


def _make_serializer() -> AdaptiveSerializer:
    # Tiny thresholds so a modest tensor exercises the FP8_ZSTD (large) path,
    # mirroring what the default 100 MB threshold does for real model tensors.
    return AdaptiveSerializer(
        small_threshold=100, large_threshold=1000, fp8_zstd_level=1
    )


# ---------------------------------------------------------------------------
# Direct AdaptiveSerializer FP8_ZSTD round-trip
# ---------------------------------------------------------------------------


def test_direct_fp8_zstd_roundtrip_is_fp8_accurate() -> None:
    ser = _make_serializer()
    tensor = (torch.randn(64, 64, dtype=torch.float16) * 10.0).abs()

    fmt, data = ser.serialize(tensor)
    assert fmt == SerializationFormat.FP8_ZSTD

    out = ser.deserialize(fmt, data, dtype=torch.float16, shape=list(tensor.shape))

    # Right shape/dtype, and values match the FP8 cast exactly (no corruption).
    assert tuple(out.shape) == tuple(tensor.shape)
    assert out.dtype == tensor.dtype
    torch.testing.assert_close(
        out, _fp8_roundtrip_reference(tensor), rtol=0, atol=0
    )


def test_direct_fp8_zstd_roundtrip_without_compression() -> None:
    # _fp8_zstd_level=0 exercises the uncompressed inner payload path.
    ser = AdaptiveSerializer(
        small_threshold=100, large_threshold=1000, fp8_zstd_level=0
    )
    tensor = torch.randn(32, 32, dtype=torch.float16)

    fmt, data = ser.serialize(tensor)
    assert fmt == SerializationFormat.FP8_ZSTD
    out = ser.deserialize(fmt, data, dtype=torch.float16, shape=list(tensor.shape))
    torch.testing.assert_close(
        out, _fp8_roundtrip_reference(tensor), rtol=0, atol=0
    )


# ---------------------------------------------------------------------------
# SerializationController negotiated-ZSTD path (the double-compression bug)
# ---------------------------------------------------------------------------


def test_negotiated_zstd_keeps_fp8_zstd_format_and_roundtrips() -> None:
    transport = _LoopbackTransport()
    ctl = SerializationController()
    ctl.set_transport(transport.send, transport.recv)
    ctl._adaptive = _make_serializer()

    tensor = (torch.randn(64, 64, dtype=torch.float16) * 10.0).abs()
    ctl.send_tensor("p1", tensor, method=CompressionMethod.ZSTD)

    # The wire envelope must preserve the FP8_ZSTD format tag, NOT relabel it
    # as plain ZSTD (which would drop the scale on the receiver).
    wire = transport._buf["p1"]
    fmt_byte = wire[0]
    assert chr(fmt_byte) == SerializationFormat.FP8_ZSTD.value[0]

    out = ctl.recv_tensor(
        "p1", dtype=torch.float16, shape=list(tensor.shape)
    )
    torch.testing.assert_close(
        out, _fp8_roundtrip_reference(tensor), rtol=0, atol=0
    )


def test_negotiated_zstd_no_double_compression() -> None:
    # The FP8_ZSTD payload must be a single valid zstd frame.  Re-compressing
    # it (the F-031 bug) would leave an outer frame wrapping a second frame and
    # the payload would not decompress to the FP8 byte chunk.
    transport = _LoopbackTransport()
    ctl = SerializationController()
    ctl.set_transport(transport.send, transport.recv)
    ctl._adaptive = _make_serializer()

    tensor = torch.randn(64, 64, dtype=torch.float16)
    ctl.send_tensor("p1", tensor, method=CompressionMethod.ZSTD)
    wire = transport._buf["p1"]

    plen = int.from_bytes(wire[2:6], "little")
    payload = wire[6 : 6 + plen]
    dctx = zstandard.ZstdDecompressor()

    # Unpack the scale header, then decompress the exactly-once-compressed
    # payload to the raw FP8 byte chunk (one little-endian byte per element).
    scale_len = int.from_bytes(payload[:4], "little")
    inner = payload[4 : 4 + scale_len]
    rest = payload[4 + scale_len :]
    if inner:
        # int8-with-scale path would have a non-empty scale; FP8 uses none.
        pass
    raw = dctx.decompress(rest)
    assert len(raw) == tensor.numel(), "payload is not a single clean FP8 chunk"
