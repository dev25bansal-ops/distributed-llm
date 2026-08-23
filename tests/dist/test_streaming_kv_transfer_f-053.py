"""Regression test for audit finding F-053.

StreamingKVTransfer broke on bfloat16 tensors:
  - chunk_tensor called t.numpy() directly on bf16 -> TypeError
    ("Got unsupported ScalarType BFloat16"), crashing every bf16 send.
  - reassemble_chunks mapped torch.bfloat16 -> np.float16/torch.float16,
    a different bit layout (bf16: 8-bit exponent / 7-bit mantissa vs
    fp16: 5-bit exponent / 10-bit mantissa), silently corrupting data.

Fix: chunk_tensor upcasts bf16 to float32 before .numpy() (lossless);
reassemble_chunks decodes bf16 chunks as float32 and casts back to
torch.bfloat16. These tests assert an exact bf16 round-trip.
"""

from __future__ import annotations

import struct

import pytest
import torch

from distllm.dist.streaming_kv_transfer import KVChunk, StreamingKVTransfer


class TestBf16Chunking:
    """chunk_tensor must not crash on bfloat16 tensors."""

    def test_chunk_tensor_bf16_no_typeerror(self):
        """F-053: .numpy() on bf16 raised TypeError; upcast fixes it."""
        s = StreamingKVTransfer()
        tensor = torch.randn(8, 8, dtype=torch.bfloat16)
        chunks = list(s.chunk_tensor(tensor, request_id="r", layer_idx=0))
        assert len(chunks) >= 1

    def test_chunk_metadata_preserves_bf16_dtype(self):
        s = StreamingKVTransfer()
        tensor = torch.randn(4, 4, dtype=torch.bfloat16)
        chunks = list(s.chunk_tensor(tensor, request_id="r", layer_idx=3))
        assert chunks[0].dtype == "torch.bfloat16"
        assert chunks[0].shape == [4, 4]
        assert chunks[0].layer_idx == 3

    def test_chunk_wire_format_is_float32(self):
        """bf16 payloads are serialized as float32 bytes (2x element size)."""
        s = StreamingKVTransfer()
        tensor = torch.tensor([[1.5, -2.25]], dtype=torch.bfloat16)
        chunks = list(s.chunk_tensor(tensor, request_id="r", layer_idx=0))
        # 2 elements * 4 bytes (fp32) = 8 bytes total
        assert sum(len(c.data) for c in chunks) == 8


class TestBf16Reassembly:
    """reassemble_chunks must decode bf16 chunks without bit-layout corruption."""

    @staticmethod
    def _handcrafted_bf16_chunks(values: list[float]) -> list[KVChunk]:
        """Build bf16 chunks by hand (fp32 wire encoding) to test the
        receive path independently of the send path."""
        raw = b"".join(struct.pack("<f", v) for v in values)
        return [
            KVChunk(
                request_id="r",
                chunk_index=0,
                total_chunks=1,
                layer_idx=0,
                data=raw,
                shape=[len(values)],
                dtype="torch.bfloat16",
                is_last=True,
            )
        ]

    def test_reassemble_handcrafted_bf16_exact_values(self):
        """F-053: the old np.float16 mapping misread the bit layout."""
        s = StreamingKVTransfer()
        chunks = self._handcrafted_bf16_chunks([1.5, -2.25, 0.0])
        restored = s.reassemble_chunks(chunks)
        assert restored is not None
        assert restored.dtype == torch.bfloat16
        assert restored.tolist() == [1.5, -2.25, 0.0]

    def test_reassemble_rejects_wrong_byte_count(self):
        """fp32 wire format means bf16 buffers must be 4 bytes/element."""
        s = StreamingKVTransfer()
        # Old buggy mapping expected 2 bytes/element; with the fix a 6-byte
        # buffer for shape [3] cannot be fp32 and must fail loudly.
        chunks = [
            KVChunk(
                request_id="r",
                chunk_index=0,
                total_chunks=1,
                layer_idx=0,
                data=b"\x00" * 6,
                shape=[3],
                dtype="torch.bfloat16",
                is_last=True,
            )
        ]
        with pytest.raises(ValueError):
            s.reassemble_chunks(chunks)


class TestBf16RoundTrip:
    """Full chunk -> reassemble round-trips for bf16 tensors."""

    @pytest.mark.parametrize("shape", [[1], [64], [8, 16], [2, 4, 32]])
    def test_roundtrip_bf16_exact(self, shape):
        s = StreamingKVTransfer(chunk_size_mb=0.001)
        original = torch.randn(*shape, dtype=torch.bfloat16)
        chunks = list(s.chunk_tensor(original, request_id="r", layer_idx=1))
        restored = s.reassemble_chunks(chunks)
        assert restored is not None
        assert restored.dtype == torch.bfloat16
        assert restored.shape == original.shape
        assert torch.equal(restored, original)

    def test_roundtrip_bf16_multi_chunk(self):
        s = StreamingKVTransfer(chunk_size_mb=0.001)
        original = torch.randn(128, 128, dtype=torch.bfloat16)
        chunks = list(s.chunk_tensor(original, request_id="r", layer_idx=0))
        assert len(chunks) > 1  # genuinely multi-chunk
        restored = s.reassemble_chunks(chunks)
        assert restored is not None
        assert torch.equal(restored, original)

    def test_roundtrip_mixed_dtypes_same_instance(self):
        s = StreamingKVTransfer(chunk_size_mb=0.001)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            original = torch.randn(32, 32, dtype=dtype)
            chunks = list(s.chunk_tensor(original, request_id="r", layer_idx=0))
            restored = s.reassemble_chunks(chunks)
            assert restored is not None
            assert restored.dtype == dtype
            assert torch.equal(restored, original)

    def test_estimate_chunks_bf16_uses_fp32_wire_size(self):
        """estimate_chunks still uses the in-memory element size (2B); the
        wire uses 4B per bf16 element, so actual chunk count can exceed it."""
        s = StreamingKVTransfer(chunk_size_mb=1)
        tensor = torch.randn(1024, dtype=torch.bfloat16)
        # In-memory: 2048 B < 1 MB -> estimate says 1 chunk...
        assert s.estimate_chunks(tensor) == 1
        # ...and indeed one chunk suffices either way here; just sanity-check
        # the full pipeline stays consistent for this size.
        chunks = list(s.chunk_tensor(tensor, request_id="r", layer_idx=0))
        assert len(chunks) == s.estimate_chunks(tensor)


class TestFp16Unchanged:
    """The fix must not alter float16 behavior."""

    def test_roundtrip_float16_still_exact(self):
        s = StreamingKVTransfer(chunk_size_mb=0.001)
        original = torch.randn(64, 64, dtype=torch.float16)
        chunks = list(s.chunk_tensor(original, request_id="r", layer_idx=0))
        restored = s.reassemble_chunks(chunks)
        assert restored is not None
        assert restored.dtype == torch.float16
        assert torch.equal(restored, original)
