"""Regression: streaming layer-weight transfer must verify integrity.

F-049: the streaming path (used for large models) concatenated chunks with no
ordering/completeness validation, so a reordered, truncated, or duplicated
stream was accepted and later torch.load'd — silently corrupting/poisoning the
loaded weights.  The client now rejects out-of-order, incomplete (missing/extra),
or non-final-terminated streams.

SEC-A5 additionally requires the sender to sign the assembled payload with
HMAC-SHA256(cluster_key, payload) as trailing metadata; a receiver holding the
cluster key fails closed on a missing/invalid tag.  A "valid" stream below is
therefore one that is ordered, complete, final-terminated, AND correctly
signed — modelled on the honest sender in tests/security/
test_weight_transfer_tls_hmac.py.
"""

from __future__ import annotations

from unittest import mock

from distllm.dist.node_client import (
    WEIGHTS_HMAC_METADATA_KEY,
    compute_weights_hmac,
    request_layer_weights_stream,
)

CLUSTER_KEY = "integrity-test-cluster-key"
MODEL = "m"


def _chunk(index: int, total: int, data: bytes, final: bool = False):
    rsp = mock.Mock()
    rsp.success = True
    rsp.state_dict_bytes = data
    rsp.chunk_index = index
    rsp.total_chunks = total
    rsp.is_final_chunk = final
    rsp.error_message = ""
    return rsp


class _FakeStreamCall:
    """Mimics a grpc unary-stream call: iterable + trailing_metadata()."""

    def __init__(self, responses, trailing):
        self._responses = responses
        self._trailing = list(trailing)

    def __iter__(self):
        return iter(self._responses)

    def trailing_metadata(self):
        return list(self._trailing)


def _run(responses, trailing=None):
    client = mock.Mock()
    client.stub.TransferWeightsStream.return_value = (
        _FakeStreamCall(responses, trailing)
        if trailing is not None else iter(responses)
    )
    with mock.patch(
        "distllm.dist.node_client.create_node_client", return_value=client
    ), mock.patch.dict(
        "os.environ", {"DISTLLM_CLUSTER_KEY": CLUSTER_KEY}
    ):
        return request_layer_weights_stream("h", 1, MODEL, 0, 2)


class TestStreamingIntegrity:
    def test_valid_stream_returns_concat(self):
        """Ordered, complete, final-terminated stream with a valid HMAC over
        the assembled payload is accepted and concatenated."""
        full = b"hello world"
        chunks = [_chunk(0, 2, b"hello"), _chunk(1, 2, b" world", final=True)]
        tag = compute_weights_hmac(CLUSTER_KEY, full, MODEL, 0, 2)
        out = _run(chunks, trailing=[(WEIGHTS_HMAC_METADATA_KEY, tag)])
        assert out == full

    def test_unsigned_stream_rejected_when_keyed(self):
        """F-049 + SEC-A5: even an ordered/complete stream is rejected
        (fail-closed) when the receiver holds a cluster key but the sender
        sent no HMAC tag."""
        chunks = [_chunk(0, 2, b"hello"), _chunk(1, 2, b" world", final=True)]
        out = _run(chunks, trailing=[])
        assert out is None

    def test_out_of_order_chunk_rejected(self):
        chunks = [_chunk(1, 2, b" world", final=True), _chunk(0, 2, b"hello")]
        # Signed over what an honest ordered sender would have assembled.
        tag = compute_weights_hmac(CLUSTER_KEY, b"hello world", MODEL, 0, 2)
        out = _run(chunks, trailing=[(WEIGHTS_HMAC_METADATA_KEY, tag)])
        assert out is None

    def test_incomplete_stream_rejected(self):
        # Only chunk 0 of 2, no final chunk.
        out = _run([_chunk(0, 2, b"hello")], trailing=[])
        assert out is None

    def test_changed_total_chunks_rejected(self):
        chunks = [
            _chunk(0, 2, b"hello"),
            _chunk(1, 3, b" world", final=True),  # total changed mid-stream
        ]
        tag = compute_weights_hmac(CLUSTER_KEY, b"hello world", MODEL, 0, 2)
        out = _run(chunks, trailing=[(WEIGHTS_HMAC_METADATA_KEY, tag)])
        assert out is None

    def test_tampered_chunk_content_rejected(self):
        # Sender signed honest bytes; the delivered stream was tampered with.
        chunks = [_chunk(0, 2, b"hello"), _chunk(1, 2, b"EVIL!!", final=True)]
        tag = compute_weights_hmac(CLUSTER_KEY, b"hello world", MODEL, 0, 2)
        out = _run(chunks, trailing=[(WEIGHTS_HMAC_METADATA_KEY, tag)])
        assert out is None

    def test_empty_stream_rejected(self):
        out = _run([], trailing=[])
        assert out is None
