"""SEC-A5 regression tests: weight-transfer TLS + authenticated integrity.

P0 finding: ``request_layer_weights`` / ``request_layer_weights_stream``
constructed their node clients without ever passing ``use_tls``/``ca_cert``,
so model weights (high-value IP) always moved over PLAINTEXT gRPC — even in
fully TLS-enabled deployments.  Integrity was a bare SHA-256 checksum sent on
the same channel, recomputable by any on-path attacker; the streaming path
had no payload integrity check at all.

Fix verified here:
  1. TLS params thread through to ``create_node_client`` and are resolved
     from the shared env contract (``DISTLLM_PIPELINE_TLS`` /
     ``DISTLLM_TLS_CA_CERT_FILE``).
  2. Payload integrity is HMAC-SHA256 keyed by the cluster secret
     (``x-weights-hmac-sha256`` trailing metadata); receivers holding the
     key fail closed on absence/mismatch.
  3. Plaintext fallback still works when TLS is off but warns loudly.
"""

from __future__ import annotations

import hashlib
import hmac as hmac_mod
from unittest import mock

import pytest

from distllm.dist import node_client
from distllm.dist.node_client import (
    LEGACY_CHECKSUM_METADATA_KEY,
    WEIGHTS_HMAC_METADATA_KEY,
    compute_weights_hmac,
    request_layer_weights,
    request_layer_weights_stream,
    resolve_pipeline_tls,
)

CLUSTER_KEY = "unit-test-cluster-key"
MODEL = "test-model"


def _tag(payload: bytes, key: str = CLUSTER_KEY,
         model: str = MODEL, start: int = 0, end: int = 2) -> str:
    return compute_weights_hmac(key, payload, model, start, end)


# ---------------------------------------------------------------------------
# Helpers: resolve_pipeline_tls + compute_weights_hmac
# ---------------------------------------------------------------------------


class TestResolvePipelineTLS:
    def test_default_is_plaintext(self, monkeypatch):
        monkeypatch.delenv("DISTLLM_PIPELINE_TLS", raising=False)
        monkeypatch.delenv("DISTLLM_TLS_CA_CERT_FILE", raising=False)
        monkeypatch.delenv("DISTLLM_TLS_CA_CERT", raising=False)
        use_tls, ca_cert = resolve_pipeline_tls()
        assert use_tls is False
        assert ca_cert is None

    def test_env_toggle_enables_tls(self, monkeypatch):
        monkeypatch.setenv("DISTLLM_PIPELINE_TLS", "1")
        monkeypatch.setenv("DISTLLM_TLS_CA_CERT_FILE", "/ca/cluster.pem")
        use_tls, ca_cert = resolve_pipeline_tls()
        assert use_tls is True
        assert ca_cert == "/ca/cluster.pem"

    def test_legacy_ca_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("DISTLLM_PIPELINE_TLS", "1")
        monkeypatch.delenv("DISTLLM_TLS_CA_CERT_FILE", raising=False)
        monkeypatch.setenv("DISTLLM_TLS_CA_CERT", "/ca/legacy.pem")
        _, ca_cert = resolve_pipeline_tls()
        assert ca_cert == "/ca/legacy.pem"

    def test_non_1_value_disables_tls(self, monkeypatch):
        monkeypatch.setenv("DISTLLM_PIPELINE_TLS", "true")
        use_tls, _ = resolve_pipeline_tls()
        assert use_tls is False


class TestComputeWeightsHmac:
    def test_deterministic_and_hex(self):
        t1 = compute_weights_hmac(CLUSTER_KEY, b"payload", MODEL, 0, 2)
        t2 = compute_weights_hmac(CLUSTER_KEY, b"payload", MODEL, 0, 2)
        assert t1 == t2
        int(t1, 16)  # hex digest
        assert len(t1) == 64

    def test_none_without_key(self):
        assert compute_weights_hmac(None, b"x") is None
        assert compute_weights_hmac("", b"x") is None

    def test_key_binds_tag(self):
        good = compute_weights_hmac(CLUSTER_KEY, b"payload")
        other = compute_weights_hmac("other-key", b"payload")
        assert good != other

    def test_payload_binds_tag(self):
        good = compute_weights_hmac(CLUSTER_KEY, b"payload-a")
        tampered = compute_weights_hmac(CLUSTER_KEY, b"payload-b")
        assert good != tampered

    def test_request_context_binds_tag(self):
        # A captured tag must not be replayable for another model/layers.
        base = compute_weights_hmac(CLUSTER_KEY, b"p", MODEL, 0, 2)
        assert compute_weights_hmac(CLUSTER_KEY, b"p", "other-model", 0, 2) != base
        assert compute_weights_hmac(CLUSTER_KEY, b"p", MODEL, 0, 3) != base
        assert compute_weights_hmac(CLUSTER_KEY, b"p", MODEL, 1, 2) != base


# ---------------------------------------------------------------------------
# Shared mock scaffolding for the two RPC helpers
# ---------------------------------------------------------------------------


def _make_client(payload: bytes = b"weights-bytes",
                 trailing: list[tuple[str, str]] | None = None,
                 success: bool = True):
    """Mock NodeClient whose TransferWeights returns *payload* with metadata."""
    resp = mock.Mock()
    resp.success = success
    resp.state_dict_bytes = payload if success else b""
    resp.error_message = "" if success else "boom"

    call = mock.Mock()
    call.trailing_metadata.return_value = list(trailing or [])

    client = mock.Mock()
    client.stub.TransferWeights.with_call.return_value = (resp, call)
    return client


class _FakeStreamCall:
    """Mimics grpc unary-stream call object: iterable + trailing_metadata()."""

    def __init__(self, responses: list, trailing: list[tuple[str, str]]):
        self._responses = responses
        self._trailing = list(trailing)

    def __iter__(self):
        return iter(self._responses)

    def trailing_metadata(self):
        return list(self._trailing)


def _make_stream_client(chunks: list[bytes],
                        trailing: list[tuple[str, str]] | None = None):
    total = len(chunks)
    responses = [
        mock.Mock(success=True, state_dict_bytes=c, error_message="",
                  chunk_index=i, total_chunks=total,
                  is_final_chunk=(i == total - 1))
        for i, c in enumerate(chunks)
    ]
    client = mock.Mock()
    client.stub.TransferWeightsStream.return_value = _FakeStreamCall(
        responses, list(trailing or []))
    return client


@pytest.fixture(autouse=True)
def _loguru_capture(monkeypatch):
    """Capture loguru records (caplog cannot see loguru)."""
    from loguru import logger as loguru_logger

    records: list = []

    def _sink(message):
        records.append(message.record)

    handler_id = loguru_logger.add(_sink, level="WARNING")
    yield records
    loguru_logger.remove(handler_id)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Keep ambient env from leaking into per-test expectations."""
    monkeypatch.delenv("DISTLLM_PIPELINE_TLS", raising=False)
    monkeypatch.delenv("DISTLLM_TLS_CA_CERT_FILE", raising=False)
    monkeypatch.delenv("DISTLLM_TLS_CA_CERT", raising=False)
    monkeypatch.delenv("DISTLLM_CLUSTER_KEY", raising=False)


# ---------------------------------------------------------------------------
# TLS threading into create_node_client
# ---------------------------------------------------------------------------


class TestWeightTransferTLSThreading:
    @pytest.mark.parametrize("func", [request_layer_weights,
                                      request_layer_weights_stream])
    def test_explicit_tls_params_reach_channel_construction(self, func, monkeypatch):
        captured: dict = {}

        def fake_create(host, port, **kwargs):
            captured.update(kwargs)
            raise RuntimeError("stop-here")

        monkeypatch.setattr(node_client, "create_node_client", fake_create)
        func("h", 1, MODEL, 0, 2, cluster_key=CLUSTER_KEY,
             use_tls=True, ca_cert="/ca/cluster.pem")  # type: ignore[operator]
        assert captured.get("use_tls") is True
        assert captured.get("ca_cert") == "/ca/cluster.pem"
        assert captured.get("cluster_key") == CLUSTER_KEY

    @pytest.mark.parametrize("func", [request_layer_weights,
                                      request_layer_weights_stream])
    def test_env_enables_tls_when_not_explicit(self, func, monkeypatch):
        captured: dict = {}

        def fake_create(host, port, **kwargs):
            captured.update(kwargs)
            raise RuntimeError("stop-here")

        monkeypatch.setenv("DISTLLM_PIPELINE_TLS", "1")
        monkeypatch.setenv("DISTLLM_TLS_CA_CERT_FILE", "/ca/env.pem")
        monkeypatch.setattr(node_client, "create_node_client", fake_create)
        func("h", 1, MODEL, 0, 2, cluster_key=CLUSTER_KEY)  # type: ignore[operator]
        assert captured.get("use_tls") is True
        assert captured.get("ca_cert") == "/ca/env.pem"

    @pytest.mark.parametrize("func", [request_layer_weights,
                                      request_layer_weights_stream])
    def test_no_tls_by_default(self, func, monkeypatch):
        captured: dict = {}

        def fake_create(host, port, **kwargs):
            captured.update(kwargs)
            raise RuntimeError("stop-here")

        monkeypatch.setattr(node_client, "create_node_client", fake_create)
        func("h", 1, MODEL, 0, 2, cluster_key=CLUSTER_KEY)  # type: ignore[operator]
        assert captured.get("use_tls") is False
        assert captured.get("ca_cert") is None

    @pytest.mark.parametrize("func", [request_layer_weights,
                                      request_layer_weights_stream])
    def test_plaintext_fallback_warns_loudly(self, func, monkeypatch, _loguru_capture):
        def fake_create(host, port, **kwargs):
            raise RuntimeError("stop-here")

        monkeypatch.setattr(node_client, "create_node_client", fake_create)
        func("h", 1, MODEL, 0, 2, cluster_key=CLUSTER_KEY)  # type: ignore[operator]
        assert any(
            "PLAINTEXT" in r["message"] and "weights" in r["message"].lower()
            for r in _loguru_capture
        )

    @pytest.mark.parametrize("func", [request_layer_weights,
                                      request_layer_weights_stream])
    def test_tls_on_does_not_warn(self, func, monkeypatch, _loguru_capture):
        def fake_create(host, port, **kwargs):
            raise RuntimeError("stop-here")

        monkeypatch.setattr(node_client, "create_node_client", fake_create)
        func("h", 1, MODEL, 0, 2, cluster_key=CLUSTER_KEY,
             use_tls=True, ca_cert="/ca.pem")  # type: ignore[operator]
        assert not any("PLAINTEXT" in r["message"] for r in _loguru_capture)


# ---------------------------------------------------------------------------
# HMAC verification — unary TransferWeights path
# ---------------------------------------------------------------------------


class TestUnaryHMACVerification:
    def test_valid_hmac_accepted(self, monkeypatch):
        payload = b"state-dict-bytes"
        trailing = [
            (WEIGHTS_HMAC_METADATA_KEY, _tag(payload)),
            (LEGACY_CHECKSUM_METADATA_KEY, hashlib.sha256(payload).hexdigest()),
        ]
        client = _make_client(payload, trailing)
        monkeypatch.setattr(node_client, "create_node_client",
                            lambda *a, **k: client)
        out = request_layer_weights("h", 1, MODEL, 0, 2, cluster_key=CLUSTER_KEY)
        assert out == payload

    def test_missing_hmac_rejected_fail_closed(self, monkeypatch):
        payload = b"state-dict-bytes"
        # Legacy checksum present and correct — still NOT enough.
        trailing = [(LEGACY_CHECKSUM_METADATA_KEY,
                     hashlib.sha256(payload).hexdigest())]
        client = _make_client(payload, trailing)
        monkeypatch.setattr(node_client, "create_node_client",
                            lambda *a, **k: client)
        out = request_layer_weights("h", 1, MODEL, 0, 2, cluster_key=CLUSTER_KEY)
        assert out is None

    def test_tampered_payload_rejected(self, monkeypatch):
        signed = b"honest-state-dict"
        delivered = b"evil-state-dict!"
        client = _make_client(delivered, [
            (WEIGHTS_HMAC_METADATA_KEY, _tag(signed)),
        ])
        monkeypatch.setattr(node_client, "create_node_client",
                            lambda *a, **k: client)
        out = request_layer_weights("h", 1, MODEL, 0, 2, cluster_key=CLUSTER_KEY)
        assert out is None

    def test_checksum_alone_cannot_authenticate_tampered_payload(self, monkeypatch):
        # The SEC-A5 attack: attacker flips bytes AND recomputes SHA-256.
        delivered = b"evil-state-dict!"
        client = _make_client(delivered, [
            (LEGACY_CHECKSUM_METADATA_KEY,
             hashlib.sha256(delivered).hexdigest()),
        ])
        monkeypatch.setattr(node_client, "create_node_client",
                            lambda *a, **k: client)
        out = request_layer_weights("h", 1, MODEL, 0, 2, cluster_key=CLUSTER_KEY)
        assert out is None

    def test_wrong_cluster_key_rejected(self, monkeypatch):
        payload = b"state-dict-bytes"
        client = _make_client(payload, [
            (WEIGHTS_HMAC_METADATA_KEY,
             compute_weights_hmac("different-key", payload, MODEL, 0, 2)),
        ])
        monkeypatch.setattr(node_client, "create_node_client",
                            lambda *a, **k: client)
        out = request_layer_weights("h", 1, MODEL, 0, 2, cluster_key=CLUSTER_KEY)
        assert out is None

    def test_tag_bound_to_request_context(self, monkeypatch):
        # Tag computed for layers 0-2 must not validate layers 5-9.
        payload = b"state-dict-bytes"
        client = _make_client(payload, [
            (WEIGHTS_HMAC_METADATA_KEY, _tag(payload)),  # start=0, end=2
        ])
        monkeypatch.setattr(node_client, "create_node_client",
                            lambda *a, **k: client)
        out = request_layer_weights("h", 1, MODEL, 5, 9, cluster_key=CLUSTER_KEY)
        assert out is None

    def test_trailing_metadata_bytes_decoded(self, monkeypatch):
        # Real gRPC hands metadata values back as bytes for binary-valued
        # keys; tolerate either representation.
        payload = b"state-dict-bytes"
        tag = _tag(payload)
        client = _make_client(payload, [
            (WEIGHTS_HMAC_METADATA_KEY.encode(), tag.encode()),
        ])
        monkeypatch.setattr(node_client, "create_node_client",
                            lambda *a, **k: client)
        out = request_layer_weights("h", 1, MODEL, 0, 2, cluster_key=CLUSTER_KEY)
        assert out == payload

    def test_non_ascii_metadata_does_not_raise(self, monkeypatch):
        # SEC-B2 class bug: compare_digest(str-with-non-ascii) raises.
        payload = b"state-dict-bytes"
        client = _make_client(payload, [
            (WEIGHTS_HMAC_METADATA_KEY, "täg-with-nön-ascii"),
        ])
        monkeypatch.setattr(node_client, "create_node_client",
                            lambda *a, **k: client)
        # Must reject cleanly, not blow up inside the try block... it does
        # catch Exception, but rejection should be deterministic None.
        out = request_layer_weights("h", 1, MODEL, 0, 2, cluster_key=CLUSTER_KEY)
        assert out is None


# ---------------------------------------------------------------------------
# Keyless receiver policy
# ---------------------------------------------------------------------------


class TestKeylessReceiverPolicy:
    def test_keyless_with_signed_payload_refused(self, monkeypatch):
        payload = b"signed-weights"
        client = _make_client(payload, [
            (WEIGHTS_HMAC_METADATA_KEY, _tag(payload)),
        ])
        monkeypatch.setattr(node_client, "create_node_client",
                            lambda *a, **k: client)
        out = request_layer_weights("h", 1, MODEL, 0, 2, cluster_key=None)
        assert out is None

    def test_keyless_unauthenticated_fallback_warns(self, monkeypatch, _loguru_capture):
        payload = b"unsigned-weights"
        client = _make_client(payload, [])
        monkeypatch.setattr(node_client, "create_node_client",
                            lambda *a, **k: client)
        out = request_layer_weights("h", 1, MODEL, 0, 2, cluster_key=None)
        assert out == payload  # legacy behaviour preserved
        assert any(
            "UNAUTHENTICATED" in r["message"] or "no cluster key" in r["message"]
            for r in _loguru_capture
        )

    def test_env_cluster_key_used_without_arg(self, monkeypatch):
        payload = b"state-dict-bytes"
        tag = compute_weights_hmac(CLUSTER_KEY, payload, MODEL, 0, 2)
        client = _make_client(payload, [(WEIGHTS_HMAC_METADATA_KEY, tag)])
        captured: dict = {}
        monkeypatch.setattr(
            node_client, "create_node_client",
            lambda host, port, **kw: captured.update(kw) or client,
        )
        monkeypatch.setenv("DISTLLM_CLUSTER_KEY", CLUSTER_KEY)
        out = request_layer_weights("h", 1, MODEL, 0, 2, cluster_key=None)
        assert out == payload
        assert captured.get("cluster_key") == CLUSTER_KEY


# ---------------------------------------------------------------------------
# HMAC verification — streaming path
# ---------------------------------------------------------------------------


class TestStreamingHMACVerification:
    def test_valid_hmac_over_assembled_buffer_accepted(self, monkeypatch):
        full = b"chunk-one-" + b"chunk-two"
        client = _make_stream_client(
            [b"chunk-one-", b"chunk-two"],
            [(WEIGHTS_HMAC_METADATA_KEY, _tag(full))],
        )
        monkeypatch.setattr(node_client, "create_node_client",
                            lambda *a, **k: client)
        out = request_layer_weights_stream("h", 1, MODEL, 0, 2,
                                           cluster_key=CLUSTER_KEY)
        assert out == full

    def test_missing_hmac_rejected_fail_closed(self, monkeypatch):
        client = _make_stream_client([b"a", b"b"], [])
        monkeypatch.setattr(node_client, "create_node_client",
                            lambda *a, **k: client)
        out = request_layer_weights_stream("h", 1, MODEL, 0, 2,
                                           cluster_key=CLUSTER_KEY)
        assert out is None

    def test_tampered_chunk_content_rejected(self, monkeypatch):
        # Sender signed honest bytes; MITM swapped chunk content mid-flight.
        honest_full = b"aaaabbbb"
        client = _make_stream_client(
            [b"aaaa", b"XXXX"],  # second chunk replaced
            [(WEIGHTS_HMAC_METADATA_KEY, _tag(honest_full))],
        )
        monkeypatch.setattr(node_client, "create_node_client",
                            lambda *a, **k: client)
        out = request_layer_weights_stream("h", 1, MODEL, 0, 2,
                                           cluster_key=CLUSTER_KEY)
        assert out is None

    def test_wrong_context_tag_rejected(self, monkeypatch):
        full = b"chunk-one-chunk-two"
        client = _make_stream_client(
            [b"chunk-one-", b"chunk-two"],
            [_tag(full, model="other-model") and
             (WEIGHTS_HMAC_METADATA_KEY,
              compute_weights_hmac(CLUSTER_KEY, full, "other-model", 0, 2))],
        )
        monkeypatch.setattr(node_client, "create_node_client",
                            lambda *a, **k: client)
        out = request_layer_weights_stream("h", 1, MODEL, 0, 2,
                                           cluster_key=CLUSTER_KEY)
        assert out is None

    def test_stream_without_trailing_metadata_attr_rejects(self, monkeypatch):
        # Plain iterators (old mocks/tests) expose no trailing_metadata();
        # a keyed receiver must still fail closed rather than skip checks.
        responses = [
            mock.Mock(success=True, state_dict_bytes=b"data", error_message="",
                      chunk_index=0, total_chunks=1, is_final_chunk=True),
        ]
        client = mock.Mock()
        client.stub.TransferWeightsStream.return_value = iter(responses)
        monkeypatch.setattr(node_client, "create_node_client",
                            lambda *a, **k: client)
        out = request_layer_weights_stream("h", 1, MODEL, 0, 2,
                                           cluster_key=CLUSTER_KEY)
        assert out is None

    def test_keyless_stream_fallback_still_works(self, monkeypatch):
        client = _make_stream_client([b"a", b"b"], [])
        monkeypatch.setattr(node_client, "create_node_client",
                            lambda *a, **k: client)
        out = request_layer_weights_stream("h", 1, MODEL, 0, 2, cluster_key=None)
        assert out == b"ab"


# ---------------------------------------------------------------------------
# Sender side: NodeServicer signs payloads
# ---------------------------------------------------------------------------


class TestSenderSigning:
    def _servicer_with_model(self):
        """NodeServicer backed by a tiny model so TransferWeights succeeds."""
        import torch
        from distllm.dist.node_service import NodeServicer

        class _MiniModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList(
                    [torch.nn.Linear(4, 4) for _ in range(4)])

        wrapper = mock.Mock()
        wrapper.partitioner = mock.Mock()
        wrapper.partitioner.full_model = _MiniModel()

        servicer = NodeServicer.__new__(NodeServicer)
        servicer._node = wrapper
        servicer._cluster_key = CLUSTER_KEY
        servicer._e2e = None
        servicer._profile_rate_tokens = NodeServicer.PROFILE_RATE_LIMIT
        servicer._profile_rate_last = 0.0
        import threading as _threading
        servicer._profile_rate_lock = _threading.Lock()
        return servicer, NodeServicer

    def _fake_ctx(self):
        ctx = mock.Mock()
        ctx.trailing_metadata = []
        ctx.send_trailing_metadata = lambda md: ctx.trailing_metadata.extend(md)
        return ctx

    def _request(self, start=0, end=2):
        from distllm.dist import node_pb2
        return node_pb2.TransferWeightsRequest(
            model_name=MODEL, start_layer=start, end_layer=end,
            cluster_key=CLUSTER_KEY,
        )

    def test_transfer_weights_signs_payload(self):
        servicer, NS = self._servicer_with_model()
        ctx = self._fake_ctx()
        resp = servicer.TransferWeights(self._request(), ctx)
        assert resp.success is True
        tags = dict(ctx.trailing_metadata)
        expected = compute_weights_hmac(
            CLUSTER_KEY, resp.state_dict_bytes, MODEL, 0, 2)
        assert tags.get(WEIGHTS_HMAC_METADATA_KEY) == expected
        # Legacy checksum still present for older receivers.
        checksums = dict(ctx.trailing_metadata)
        assert checksums.get(LEGACY_CHECKSUM_METADATA_KEY) == (
            hashlib.sha256(resp.state_dict_bytes).hexdigest())
        # Round-trip: the exact verification the client performs accepts it.
        from distllm.dist.node_client import _verify_weight_payload
        md_keys = {k: v for k, v in ctx.trailing_metadata}
        assert _verify_weight_payload(
            resp.state_dict_bytes, CLUSTER_KEY, md_keys,
            "test", MODEL, 0, 2,
        ) == resp.state_dict_bytes

    def test_transfer_weights_stream_signs_assembled_payload(self):
        servicer, NS = self._servicer_with_model()
        ctx = self._fake_ctx()
        responses = list(servicer.TransferWeightsStream(self._request(), ctx))
        assert responses and all(r.success for r in responses)
        tags = dict(ctx.trailing_metadata)
        assembled = b"".join(r.state_dict_bytes for r in responses)
        expected = compute_weights_hmac(
            CLUSTER_KEY, assembled, MODEL, 0, 2)
        assert tags.get(WEIGHTS_HMAC_METADATA_KEY) == expected

    def test_transfer_weights_without_key_fails_closed(self):
        # A servicer without a cluster key cannot sign OR authenticate any
        # request — _check_auth rejects everything before signing runs.
        servicer, NS = self._servicer_with_model()
        servicer._cluster_key = None
        ctx = self._fake_ctx()
        resp = servicer.TransferWeights(self._request(), ctx)
        assert resp.success is False
        assert "authentication failed" in resp.error_message

    def test_hmac_helper_requires_key(self):
        from distllm.dist.node_client import compute_weights_hmac as cwh
        assert cwh(None, b"payload") is None
        assert cwh("", b"payload") is None
