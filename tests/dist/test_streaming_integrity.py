"""Regression: streaming layer-weight transfer must verify integrity.

F-049: the streaming path (used for large models) concatenated chunks with no
ordering/completeness validation, so a reordered, truncated, or duplicated
stream was accepted and later torch.load'd — silently corrupting/poisoning the
loaded weights.  The client now rejects out-of-order, incomplete (missing/extra),
or non-final-terminated streams.
"""

from __future__ import annotations

from unittest import mock

from distllm.dist.node_client import request_layer_weights_stream


def _chunk(index: int, total: int, data: bytes, final: bool = False):
    rsp = mock.Mock()
    rsp.success = True
    rsp.state_dict_bytes = data
    rsp.chunk_index = index
    rsp.total_chunks = total
    rsp.is_final_chunk = final
    rsp.error_message = ""
    return rsp


def _run(responses):
    client = mock.Mock()
    client.stub.TransferWeightsStream.return_value = iter(responses)
    with mock.patch(
        "distllm.dist.node_client.create_node_client", return_value=client
    ):
        return request_layer_weights_stream("h", 1, "m", 0, 2)


class TestStreamingIntegrity:
    def test_valid_stream_returns_concat(self):
        out = _run([_chunk(0, 2, b"hello"), _chunk(1, 2, b" world", final=True)])
        assert out == b"hello world"

    def test_out_of_order_chunk_rejected(self):
        out = _run([_chunk(1, 2, b" world", final=True), _chunk(0, 2, b"hello")])
        assert out is None

    def test_incomplete_stream_rejected(self):
        # Only chunk 0 of 2, no final chunk.
        out = _run([_chunk(0, 2, b"hello")])
        assert out is None

    def test_changed_total_chunks_rejected(self):
        out = _run([
            _chunk(0, 2, b"hello"),
            _chunk(1, 3, b" world", final=True),  # total changed mid-stream
        ])
        assert out is None

    def test_empty_stream_rejected(self):
        out = _run([])
        assert out is None
