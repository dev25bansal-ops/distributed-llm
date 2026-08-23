"""Real tests for StreamingKVTransfer — chunking and reassembly of KV cache tensors.

Zero mocks — all tests use real torch tensors on CPU and deterministic logic.
"""

from __future__ import annotations

import pytest
import torch

from distllm.dist.streaming_kv_transfer import KVChunk, StreamingKVTransfer


class TestKVChunk:
    """Dataclass for a single KV cache chunk."""

    def test_fields(self):
        chunk = KVChunk(
            request_id="req-1",
            chunk_index=0,
            total_chunks=2,
            layer_idx=3,
            data=b"hello",
            shape=[4, 8],
            dtype="torch.float32",
            is_last=False,
        )
        assert chunk.request_id == "req-1"
        assert chunk.chunk_index == 0
        assert chunk.total_chunks == 2
        assert chunk.layer_idx == 3
        assert chunk.data == b"hello"
        assert chunk.shape == [4, 8]
        assert chunk.dtype == "torch.float32"
        assert chunk.is_last is False

    def test_is_last_default(self):
        chunk = KVChunk(
            request_id="r",
            chunk_index=0,
            total_chunks=1,
            layer_idx=0,
            data=b"x",
            shape=[1],
            dtype="torch.float32",
        )
        assert chunk.is_last is False

    def test_last_chunk(self):
        chunk = KVChunk(
            request_id="r",
            chunk_index=0,
            total_chunks=1,
            layer_idx=0,
            data=b"x",
            shape=[1],
            dtype="torch.float32",
            is_last=True,
        )
        assert chunk.is_last is True

    def test_repr(self):
        chunk = KVChunk(
            request_id="r",
            chunk_index=0,
            total_chunks=2,
            layer_idx=1,
            data=b"\x00\x01",
            shape=[2, 4],
            dtype="torch.float32",
            is_last=False,
        )
        r = repr(chunk)
        assert "KVChunk" in r
        assert "request_id='r'" in r


class TestStreamingKVTransferInit:
    """Construction and configuration."""

    def test_default_chunk_size(self):
        s = StreamingKVTransfer()
        assert s._chunk_size_bytes == 2 * 1024 * 1024

    def test_custom_chunk_size(self):
        s = StreamingKVTransfer(chunk_size_mb=0.5)
        assert s._chunk_size_bytes == int(0.5 * 1024 * 1024)

    def test_zero_chunk_size(self):
        """Constructor accepts 0 (no validation), stores it literally."""
        s = StreamingKVTransfer(chunk_size_mb=0)
        assert s._chunk_size_bytes == 0

    def test_negative_chunk_size(self):
        """Constructor accepts negative (no validation), stores it literally."""
        s = StreamingKVTransfer(chunk_size_mb=-1)
        assert s._chunk_size_bytes == -1048576

    def test_init_stats_empty(self):
        s = StreamingKVTransfer()
        st = s.stats()
        assert st["transfers_sent"] == 0
        assert st["transfers_received"] == 0
        assert st["chunks_sent"] == 0
        assert st["chunks_received"] == 0
        assert st["bytes_transferred"] == 0
        assert st["chunk_size_mb"] == 2.0


class TestEstimateChunks:
    """StreamingKVTransfer.estimate_chunks."""

    def test_small_tensor_one_chunk(self):
        s = StreamingKVTransfer(chunk_size_mb=4)
        tensor = torch.randn(16, 16, dtype=torch.float32)
        assert s.estimate_chunks(tensor) == 1

    def test_large_tensor_multiple_chunks(self):
        s = StreamingKVTransfer(chunk_size_mb=0.001)
        tensor = torch.randn(256, 256, dtype=torch.float32)
        assert s.estimate_chunks(tensor) > 1

    def test_empty_tensor(self):
        s = StreamingKVTransfer()
        tensor = torch.empty(0, dtype=torch.float32)
        assert s.estimate_chunks(tensor) == 1

    def test_single_element(self):
        s = StreamingKVTransfer()
        tensor = torch.tensor([42.0])
        assert s.estimate_chunks(tensor) == 1


class TestNeedsStreaming:
    """StreamingKVTransfer.needs_streaming."""

    def test_small_tensor_no_streaming(self):
        s = StreamingKVTransfer(chunk_size_mb=4)
        tensor = torch.randn(16, 16, dtype=torch.float32)
        assert s.needs_streaming(tensor) is False

    def test_large_tensor_needs_streaming(self):
        s = StreamingKVTransfer(chunk_size_mb=0.001)
        tensor = torch.randn(256, 256, dtype=torch.float32)
        assert s.needs_streaming(tensor) is True

    def test_exact_boundary_no_streaming(self):
        s = StreamingKVTransfer(chunk_size_mb=0.001)
        size_bytes = int(0.001 * 1024 * 1024)
        num_elements = size_bytes // 4
        tensor = torch.randn(num_elements, dtype=torch.float32)
        assert s.needs_streaming(tensor) is False

    def test_exact_boundary_plus_one_needs_streaming(self):
        s = StreamingKVTransfer(chunk_size_mb=0.001)
        size_bytes = int(0.001 * 1024 * 1024)
        num_elements = size_bytes // 4 + 1
        tensor = torch.randn(num_elements, dtype=torch.float32)
        assert s.needs_streaming(tensor) is True

    def test_empty_tensor(self):
        s = StreamingKVTransfer()
        tensor = torch.empty(0, dtype=torch.float32)
        assert s.needs_streaming(tensor) is False


class TestChunkTensor:
    """StreamingKVTransfer.chunk_tensor."""

    def test_yields_single_chunk_for_small_tensor(self):
        s = StreamingKVTransfer(chunk_size_mb=4)
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        chunks = list(s.chunk_tensor(tensor, request_id="r1", layer_idx=0))
        assert len(chunks) == 1
        assert chunks[0].chunk_index == 0
        assert chunks[0].total_chunks == 1
        assert chunks[0].is_last is True
        assert chunks[0].request_id == "r1"
        assert chunks[0].layer_idx == 0
        assert chunks[0].shape == [2, 2]
        assert chunks[0].dtype == "torch.float32"

    def test_yields_multiple_chunks_for_large_tensor(self):
        s = StreamingKVTransfer(chunk_size_mb=0.001)
        tensor = torch.randn(256, 256, dtype=torch.float32)
        chunks = list(s.chunk_tensor(tensor, request_id="r2", layer_idx=5))
        assert len(chunks) > 1
        assert chunks[0].chunk_index == 0
        assert chunks[-1].is_last is True
        for c in chunks:
            assert c.request_id == "r2"
            assert c.layer_idx == 5

    def test_chunks_have_contiguous_data(self):
        s = StreamingKVTransfer(chunk_size_mb=0.001)
        tensor = torch.randn(128, 128, dtype=torch.float32)
        chunks = list(s.chunk_tensor(tensor, request_id="r", layer_idx=0))
        assert len(chunks) > 1
        all_bytes = b"".join(c.data for c in chunks)
        original_bytes = bytes(memoryview(tensor.detach().contiguous().numpy()))
        assert all_bytes == original_bytes

    def test_chunk_data_size_within_limit(self):
        s = StreamingKVTransfer(chunk_size_mb=0.5)
        chunk_bytes = int(0.5 * 1024 * 1024)
        tensor = torch.randn(512, 512, dtype=torch.float32)
        chunks = list(s.chunk_tensor(tensor, request_id="r", layer_idx=0))
        assert len(chunks) > 1
        for c in chunks:
            assert len(c.data) <= chunk_bytes
            assert len(c.data) > 0

    def test_empty_tensor_yields_one_chunk(self):
        s = StreamingKVTransfer()
        tensor = torch.empty(0, dtype=torch.float32)
        chunks = list(s.chunk_tensor(tensor, request_id="r", layer_idx=0))
        assert len(chunks) == 1
        assert chunks[0].data == b""

    def test_different_request_ids(self):
        s = StreamingKVTransfer(chunk_size_mb=0.001)
        tensor = torch.randn(128, 128, dtype=torch.float32)
        chunks_a = list(s.chunk_tensor(tensor, request_id="req-a", layer_idx=0))
        chunks_b = list(s.chunk_tensor(tensor, request_id="req-b", layer_idx=1))
        for c in chunks_a:
            assert c.request_id == "req-a"
        for c in chunks_b:
            assert c.request_id == "req-b"
            assert c.layer_idx == 1

    def test_updates_stats(self):
        s = StreamingKVTransfer(chunk_size_mb=0.001)
        tensor = torch.randn(128, 128, dtype=torch.float32)
        chunks = list(s.chunk_tensor(tensor, request_id="r", layer_idx=0))
        st = s.stats()
        assert st["chunks_sent"] == len(chunks)
        assert st["bytes_transferred"] > 0
        assert st["transfers_sent"] == 0  # only reassemble increments this

    def test_consecutive_chunking(self):
        s = StreamingKVTransfer(chunk_size_mb=0.001)
        tensor = torch.randn(64, 64, dtype=torch.float32)
        chunks1 = list(s.chunk_tensor(tensor, request_id="r1", layer_idx=0))
        chunks2 = list(s.chunk_tensor(tensor, request_id="r2", layer_idx=1))
        st = s.stats()
        assert st["chunks_sent"] == len(chunks1) + len(chunks2)


class TestReassembleChunks:
    """StreamingKVTransfer.reassemble_chunks."""

    def test_roundtrip_small_tensor(self):
        s = StreamingKVTransfer()
        original = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        chunks = list(s.chunk_tensor(original, request_id="r", layer_idx=0))
        restored = s.reassemble_chunks(chunks)
        assert restored is not None
        assert torch.equal(restored, original)

    def test_roundtrip_large_tensor(self):
        s = StreamingKVTransfer(chunk_size_mb=0.001)
        original = torch.randn(128, 128, dtype=torch.float32)
        chunks = list(s.chunk_tensor(original, request_id="r", layer_idx=0))
        restored = s.reassemble_chunks(chunks)
        assert restored is not None
        assert restored.shape == original.shape
        assert restored.dtype == original.dtype
        assert torch.equal(restored, original)

    def test_roundtrip_1d_tensor(self):
        s = StreamingKVTransfer(chunk_size_mb=0.001)
        original = torch.randn(4096, dtype=torch.float32)
        chunks = list(s.chunk_tensor(original, request_id="r", layer_idx=0))
        restored = s.reassemble_chunks(chunks)
        assert restored is not None
        assert torch.equal(restored, original)

    def test_roundtrip_3d_tensor(self):
        s = StreamingKVTransfer(chunk_size_mb=0.001)
        original = torch.randn(4, 16, 64, dtype=torch.float32)
        chunks = list(s.chunk_tensor(original, request_id="r", layer_idx=0))
        restored = s.reassemble_chunks(chunks)
        assert restored is not None
        assert torch.equal(restored, original)

    def test_empty_chunks_list_returns_none(self):
        s = StreamingKVTransfer()
        assert s.reassemble_chunks([]) is None

    def test_incomplete_chunks_returns_none(self):
        s = StreamingKVTransfer(chunk_size_mb=0.001)
        tensor = torch.randn(256, 256, dtype=torch.float32)
        chunks = list(s.chunk_tensor(tensor, request_id="r", layer_idx=0))
        incomplete = chunks[: len(chunks) // 2]
        assert s.reassemble_chunks(incomplete) is None

    def test_duplicate_chunks(self):
        s = StreamingKVTransfer(chunk_size_mb=0.001)
        tensor = torch.randn(128, 128, dtype=torch.float32)
        chunks = list(s.chunk_tensor(tensor, request_id="r", layer_idx=0))
        # Duplicate first chunk and drop last -> wrong total_chunks expectation
        bad = [chunks[0], chunks[0]]
        assert s.reassemble_chunks(bad) is None

    def test_updates_stats_on_success(self):
        s = StreamingKVTransfer(chunk_size_mb=0.001)
        tensor = torch.randn(64, 64, dtype=torch.float32)
        chunks = list(s.chunk_tensor(tensor, request_id="r", layer_idx=0))
        s.reassemble_chunks(chunks)
        st = s.stats()
        assert st["transfers_received"] == 1
        assert st["chunks_received"] == len(chunks)

    def test_does_not_update_stats_on_failure(self):
        s = StreamingKVTransfer(chunk_size_mb=0.001)
        st_before = s.stats()
        s.reassemble_chunks([])
        st_after = s.stats()
        assert st_after["transfers_received"] == st_before["transfers_received"]
        assert st_after["chunks_received"] == st_before["chunks_received"]

    def test_mixed_request_ids_roundtrip_same_tensor(self):
        s = StreamingKVTransfer(chunk_size_mb=0.001)
        tensor = torch.randn(64, 64, dtype=torch.float32)
        chunks = list(s.chunk_tensor(tensor, request_id="r", layer_idx=0))
        restored = s.reassemble_chunks(chunks)
        assert restored is not None
        assert torch.equal(restored, tensor)


class TestFloat16Support:
    """StreamingKVTransfer with float16 and bfloat16 tensors."""

    def test_roundtrip_float16(self):
        s = StreamingKVTransfer(chunk_size_mb=0.001)
        original = torch.randn(64, 64, dtype=torch.float16)
        chunks = list(s.chunk_tensor(original, request_id="r", layer_idx=0))
        restored = s.reassemble_chunks(chunks)
        assert restored is not None
        assert restored.dtype == torch.float16
        assert torch.equal(restored, original)

    def test_roundtrip_bfloat16(self):
        s = StreamingKVTransfer(chunk_size_mb=0.001)
        original = torch.randn(64, 64, dtype=torch.bfloat16)
        # Note: .numpy() does not support BFloat16; chunk_tensor will raise
        with pytest.raises(TypeError, match="BFloat16"):
            list(s.chunk_tensor(original, request_id="r", layer_idx=0))

    def test_chunk_dtype_string_float16(self):
        s = StreamingKVTransfer()
        tensor = torch.randn(4, 4, dtype=torch.float16)
        chunks = list(s.chunk_tensor(tensor, request_id="r", layer_idx=0))
        assert chunks[0].dtype == "torch.float16"


class TestStats:
    """StreamingKVTransfer.stats."""

    def test_stats_keys(self):
        s = StreamingKVTransfer()
        st = s.stats()
        expected_keys = {"transfers_sent", "transfers_received", "chunks_sent",
                         "chunks_received", "bytes_transferred", "chunk_size_mb"}
        assert set(st.keys()) == expected_keys

    def test_stats_accumulates(self):
        s = StreamingKVTransfer(chunk_size_mb=0.001)
        tensor = torch.randn(64, 64, dtype=torch.float32)
        chunks1 = list(s.chunk_tensor(tensor, request_id="r1", layer_idx=0))
        chunks2 = list(s.chunk_tensor(tensor, request_id="r2", layer_idx=0))
        s.reassemble_chunks(chunks1)
        s.reassemble_chunks(chunks2)
        st = s.stats()
        assert st["transfers_sent"] == 0
        assert st["transfers_received"] == 2
        assert st["chunks_received"] == len(chunks1) + len(chunks2)


class TestDeterminism:
    """Deterministic behavior of chunking and reassembly."""

    def test_chunking_deterministic(self):
        s = StreamingKVTransfer(chunk_size_mb=0.001)
        tensor = torch.randn(64, 64, dtype=torch.float32)
        chunks1 = list(s.chunk_tensor(tensor, request_id="r", layer_idx=0))
        chunks2 = list(s.chunk_tensor(tensor, request_id="r", layer_idx=0))
        for c1, c2 in zip(chunks1, chunks2):
            assert c1.data == c2.data
            assert c1.chunk_index == c2.chunk_index
            assert c1.total_chunks == c2.total_chunks
        assert len(chunks1) == len(chunks2)

    def test_estimate_chunks_matches_actual(self):
        s = StreamingKVTransfer(chunk_size_mb=0.001)
        tensor = torch.randn(100, 100, dtype=torch.float32)
        estimated = s.estimate_chunks(tensor)
        chunks = list(s.chunk_tensor(tensor, request_id="r", layer_idx=0))
        assert estimated == len(chunks)
