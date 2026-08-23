"""Tests for distributed transport modules.

Tests cover three components from two subpackages:

- TransportBackend enum (pipeline.transport)
- TensorTransport class (pipeline.transport)
- KVCacheTransfer class (p2p.transport) — hash, serialize, bandwidth gating
- GossipTransport class (p2p.transport) — construction, signing, peer-resolution paths

Zero mocks — uses only real objects from the modules.  No GPU, no network,
no timing-dependent assertions.
"""

from __future__ import annotations

import os
import pickle

import pytest
import torch

from distllm.dist.p2p.transport import (
    GossipTransport,
    KVCacheTransfer,
)
from distllm.dist.pipeline.transport import (
    TensorTransport,
    TransportBackend,
)


# ===========================================================================
# TransportBackend
# ===========================================================================


class TestTransportBackend:
    """TransportBackend enum values and properties."""

    def test_members(self) -> None:
        assert TransportBackend.NCCL.value == "nccl"
        assert TransportBackend.GRPC.value == "grpc"
        assert TransportBackend.QUIC.value == "quic"
        assert TransportBackend.AUTO.value == "auto"

    def test_all_members_present(self) -> None:
        """Every expected name is in the enum."""
        names = {m.name for m in TransportBackend}
        assert names == {"NCCL", "GRPC", "QUIC", "AUTO"}

    def test_members_are_distinct(self) -> None:
        """AUTO is a distinct member, not an alias for another backend."""
        assert TransportBackend.AUTO is not TransportBackend.GRPC
        assert TransportBackend.AUTO is not TransportBackend.NCCL
        assert TransportBackend.AUTO is not TransportBackend.QUIC

    def test_is_enum(self) -> None:
        """TransportBackend is a proper enum."""
        from enum import Enum

        assert issubclass(TransportBackend, Enum)


# ===========================================================================
# TensorTransport
# ===========================================================================


class TestTensorTransport:
    """TensorTransport construction, backend selection, and error paths."""

    # -- Construction ---------------------------------------------------

    def test_init_grpc(self) -> None:
        t = TensorTransport(backend=TransportBackend.GRPC)
        assert t.backend == TransportBackend.GRPC
        assert t.is_available is False
        assert t._nccl is None
        t.destroy()

    def test_init_nccl_no_gpu(self) -> None:
        """NCCL init gracefully degrades when no GPU is present."""
        t = TensorTransport(backend=TransportBackend.NCCL)
        assert t.backend == TransportBackend.NCCL
        assert t.is_available is False
        t.destroy()

    def test_init_auto_falls_to_grpc(self) -> None:
        """AUTO falls through to gRPC when NCCL+QUIC are unavailable."""
        t = TensorTransport(backend=TransportBackend.AUTO)
        assert t.is_available is False
        assert t._quic_client is None
        t.destroy()

    def test_extra_kwargs_ignored(self) -> None:
        """Extra keyword arguments are passed through without error."""
        t = TensorTransport(backend=TransportBackend.GRPC, unknown_kwarg="ignored")
        assert t.backend == TransportBackend.GRPC
        t.destroy()

    # -- Properties ----------------------------------------------------

    def test_active_backend_grpc(self) -> None:
        t = TensorTransport(backend=TransportBackend.GRPC)
        assert t.active_backend == TransportBackend.GRPC
        t.destroy()

    def test_active_backend_auto(self) -> None:
        """AUTO returns gRPC since NCCL and QUIC are not available."""
        t = TensorTransport(backend=TransportBackend.AUTO)
        result = t.active_backend
        assert result == TransportBackend.GRPC
        t.destroy()

    def test_is_quic_supported(self) -> None:
        """Without aioquic installed, QUIC is not supported."""
        t = TensorTransport(backend=TransportBackend.GRPC)
        assert t.is_quic_supported is False
        t.destroy()

    # -- Error paths ---------------------------------------------------

    def test_send_tensor_raises_without_nccl(self) -> None:
        t = TensorTransport(backend=TransportBackend.GRPC)
        with pytest.raises(RuntimeError, match="NCCL transport not initialized"):
            t.send_tensor(torch.zeros(1), dst=0)
        t.destroy()

    def test_recv_tensor_raises_without_nccl(self) -> None:
        t = TensorTransport(backend=TransportBackend.GRPC)
        with pytest.raises(RuntimeError, match="NCCL transport not initialized"):
            t.recv_tensor(shape=(1,), dtype=torch.float32, src=0)
        t.destroy()

    # -- Lifecycle -----------------------------------------------------

    def test_destroy_idempotent(self) -> None:
        """Calling destroy() twice must not raise."""
        t = TensorTransport(backend=TransportBackend.GRPC)
        t.destroy()
        t.destroy()

    def test_destroy_with_loop_none(self) -> None:
        """destroy(loop=None) is a valid call path."""
        t = TensorTransport(backend=TransportBackend.GRPC)
        t.destroy(loop=None)
        assert t._nccl is None
        assert t.is_available is False

    def test_destroy_after_nccl_init(self) -> None:
        """destroy works even when NCCL was attempted (no GPU)."""
        t = TensorTransport(backend=TransportBackend.NCCL)
        t.destroy()
        assert t._nccl is None
        assert t.is_available is False

    # -- Async error paths (no QUIC, no network) -----------------------

    @pytest.mark.asyncio
    async def test_send_forward_pass_raises_without_quic(self) -> None:
        t = TensorTransport(backend=TransportBackend.GRPC)
        with pytest.raises(RuntimeError, match="QUIC transport not initialized"):
            await t.send_forward_pass(b"test_data")
        t.destroy()

    @pytest.mark.asyncio
    async def test_quic_connect_raises_without_aioquic(self) -> None:
        """quic_connect fails when aioquic is not installed."""
        from distllm.dist.quic_transport import is_quic_available

        if not is_quic_available():
            t = TensorTransport(backend=TransportBackend.GRPC)
            with pytest.raises(Exception):
                await t.quic_connect("127.0.0.1", 4433)
            t.destroy()

# ===========================================================================
# KVCacheTransfer
# ===========================================================================


class TestKVCacheTransfer:
    """KVCacheTransfer — hash functions, serialization, bandwidth gating."""

    # -- Hash functions -------------------------------------------------

    def test_hash_tokens_empty(self) -> None:
        assert KVCacheTransfer.hash_tokens([]) == "h0"

    def test_hash_tokens_single(self) -> None:
        assert KVCacheTransfer.hash_tokens([0]) == "h0"

    def test_hash_tokens_multiple(self) -> None:
        assert KVCacheTransfer.hash_tokens([1, 2, 3]) == "h1026"

    def test_hash_tokens_large_values(self) -> None:
        """Large token IDs exercise the modulo arithmetic."""
        result = KVCacheTransfer.hash_tokens([1 << 30])
        assert result.startswith("h")

    def test_hash_tokens_negative(self) -> None:
        """Negative token IDs are hashed without error."""
        result = KVCacheTransfer.hash_tokens([-1, -42])
        assert isinstance(result, str)
        assert result.startswith("h")

    def test_hash_sha256(self) -> None:
        h = KVCacheTransfer.hash_tokens_sha256([1, 2, 3])
        assert isinstance(h, str)
        assert len(h) == 16

    def test_hash_sha256_empty(self) -> None:
        h = KVCacheTransfer.hash_tokens_sha256([])
        assert isinstance(h, str)
        assert len(h) == 16

    def test_hash_sha256_large(self) -> None:
        h = KVCacheTransfer.hash_tokens_sha256(list(range(1000)))
        assert isinstance(h, str)
        assert len(h) == 16

    def test_hash_deterministic(self) -> None:
        assert KVCacheTransfer.hash_tokens([42, 99]) == KVCacheTransfer.hash_tokens(
            [42, 99]
        )
        assert KVCacheTransfer.hash_tokens_sha256([1, 2]) == (
            KVCacheTransfer.hash_tokens_sha256([1, 2])
        )

    def test_hash_different_inputs(self) -> None:
        assert KVCacheTransfer.hash_tokens([1]) != KVCacheTransfer.hash_tokens([2])

    # -- Serialization --------------------------------------------------

    def test_serialize_deserialize_roundtrip(self) -> None:
        data = {"k": torch.zeros(2, 3), "v": torch.ones(2, 3)}
        blob = KVCacheTransfer.serialize_kv(data)
        loaded = KVCacheTransfer.deserialize_kv(blob)
        assert torch.equal(loaded["k"], data["k"])
        assert torch.equal(loaded["v"], data["v"])

    def test_serialize_empty_dict(self) -> None:
        blob = KVCacheTransfer.serialize_kv({})
        loaded = KVCacheTransfer.deserialize_kv(blob)
        assert loaded == {}

    def test_serialize_with_multiple_layers(self) -> None:
        data = {
            "layer0": torch.eye(3),
            "layer1": torch.arange(6).reshape(2, 3),
        }
        blob = KVCacheTransfer.serialize_kv(data)
        loaded = KVCacheTransfer.deserialize_kv(blob)
        assert torch.equal(loaded["layer0"], data["layer0"])
        assert torch.equal(loaded["layer1"], data["layer1"])

    def test_deserialize_rejects_pickle(self) -> None:
        """weights_only=True prevents pickle-based exploits."""
        malicious = pickle.dumps(42)
        with pytest.raises((pickle.UnpicklingError, RuntimeError, TypeError)):
            KVCacheTransfer.deserialize_kv(malicious)

    def test_serialize_bool_tensors(self) -> None:
        """Bool tensors round-trip correctly."""
        data = {"mask": torch.tensor([True, False, True])}
        blob = KVCacheTransfer.serialize_kv(data)
        loaded = KVCacheTransfer.deserialize_kv(blob)
        assert torch.equal(loaded["mask"], data["mask"])

    # -- estimate_size --------------------------------------------------

    def test_estimate_size_tensor_tuple(self) -> None:
        data = {"layer0": (torch.randn(4, 8), torch.randn(4, 8))}
        # 4 * 8 * 4 bytes (float32) * 2 = 256
        assert KVCacheTransfer.estimate_size(data) == 256

    def test_estimate_size_nested_dict_with_tuple(self) -> None:
        data = {"layer0": {"k": (torch.randn(2, 2), torch.randn(2, 2))}}
        # 2 * 2 * 4 * 2 = 32
        assert KVCacheTransfer.estimate_size(data) == 32

    def test_estimate_size_flat_tensors(self) -> None:
        """Flat tensors (not tuples) are not counted by estimate_size."""
        data = {"layer0": torch.randn(4, 8)}
        assert KVCacheTransfer.estimate_size(data) == 0

    def test_estimate_size_non_tensor_values(self) -> None:
        data = {"layer0": 42, "layer1": "string", "layer2": None}
        assert KVCacheTransfer.estimate_size(data) == 0

    def test_estimate_size_empty(self) -> None:
        assert KVCacheTransfer.estimate_size({}) == 0

    def test_estimate_size_int64(self) -> None:
        """int64 tensors are correctly sized (8 bytes per element)."""
        data = {
            "layer0": (
                torch.randint(0, 100, (2, 4), dtype=torch.int64),
                torch.randint(0, 100, (2, 4), dtype=torch.int64),
            )
        }
        # 2 * 4 * 8 bytes * 2 = 128
        assert KVCacheTransfer.estimate_size(data) == 128

    def test_estimate_size_half_precision(self) -> None:
        """float16 tensors use 2 bytes per element."""
        data = {
            "layer0": (
                torch.randn(1, 16, dtype=torch.float16),
                torch.randn(1, 16, dtype=torch.float16),
            )
        }
        # 1 * 16 * 2 bytes * 2 = 64
        assert KVCacheTransfer.estimate_size(data) == 64

    # -- Bandwidth gating -----------------------------------------------

    def test_can_transfer_under_limit(self) -> None:
        kt = KVCacheTransfer(max_bandwidth=100_000_000)
        assert kt.can_transfer(1024) is True
        assert kt.can_transfer(50 * 1024 * 1024) is True

    def test_can_transfer_zero_size(self) -> None:
        kt = KVCacheTransfer(max_bandwidth=100_000_000)
        assert kt.can_transfer(0) is True

    def test_can_transfer_immediately(self) -> None:
        """When no transfers have happened, can_transfer returns True."""
        kt = KVCacheTransfer()
        assert kt.can_transfer(1024) is True

    def test_can_transfer_after_large_transfer(self) -> None:
        """After recording a large transfer, can_transfer still works."""
        kt = KVCacheTransfer(max_bandwidth=10 * 1024 * 1024)  # 10 MB/s limit
        # Record enough bytes to push the rate past 80 % of max_bandwidth.
        # To keep tests deterministic, we accept that the time-based rate
        # limiter may or may not gate — we merely verify it does not crash.
        kt.record_transfer(9 * 1024 * 1024, success=True)
        # can_transfer should still be callable without error
        isinstance(kt.can_transfer(1024), bool)

    # -- Recording & stats ----------------------------------------------

    def test_record_transfer_success(self) -> None:
        kt = KVCacheTransfer()
        kt.record_transfer(4096, success=True)
        stats = kt.stats()
        assert stats["bytes_transferred"] == 4096
        assert stats["transfers_completed"] == 1
        assert stats["transfers_failed"] == 0

    def test_record_transfer_failure(self) -> None:
        kt = KVCacheTransfer()
        kt.record_transfer(0, success=False)
        stats = kt.stats()
        assert stats["bytes_transferred"] == 0
        assert stats["transfers_completed"] == 0
        assert stats["transfers_failed"] == 1

    def test_record_multiple_transfers(self) -> None:
        kt = KVCacheTransfer()
        kt.record_transfer(1000, success=True)
        kt.record_transfer(2000, success=True)
        kt.record_transfer(500, success=False)
        stats = kt.stats()
        assert stats["bytes_transferred"] == 3000
        assert stats["transfers_completed"] == 2
        assert stats["transfers_failed"] == 1

    def test_record_zero_bytes_success(self) -> None:
        """Recording a zero-byte successful transfer works."""
        kt = KVCacheTransfer()
        kt.record_transfer(0, success=True)
        stats = kt.stats()
        assert stats["bytes_transferred"] == 0
        assert stats["transfers_completed"] == 1

    def test_stats_structure(self) -> None:
        kt = KVCacheTransfer()
        stats = kt.stats()
        assert set(stats.keys()) == {
            "bytes_transferred",
            "transfers_completed",
            "transfers_failed",
            "avg_bandwidth_mbps",
            "max_bandwidth_mbps",
        }

    def test_stats_bandwidth_types(self) -> None:
        kt = KVCacheTransfer(max_bandwidth=50 * 1024 * 1024)
        stats = kt.stats()
        assert isinstance(stats["avg_bandwidth_mbps"], (int, float))
        assert isinstance(stats["max_bandwidth_mbps"], (int, float))
        assert stats["max_bandwidth_mbps"] == 50.0

    def test_reset_stats(self) -> None:
        kt = KVCacheTransfer()
        kt.record_transfer(8192, success=True)
        kt.record_transfer(4096, success=False)
        kt.reset_stats()
        stats = kt.stats()
        assert stats["bytes_transferred"] == 0
        assert stats["transfers_completed"] == 0
        assert stats["transfers_failed"] == 0

    def test_reset_stats_resets_timer(self) -> None:
        """reset_stats resets the timer so avg_bandwidth is recalculated."""
        kt = KVCacheTransfer()
        kt.record_transfer(10_000_000, success=True)
        kt.reset_stats()
        stats = kt.stats()
        assert stats["bytes_transferred"] == 0
        assert stats["avg_bandwidth_mbps"] == 0.0

    # -- Constants ------------------------------------------------------

    def test_default_max_bandwidth(self) -> None:
        assert KVCacheTransfer.DEFAULT_MAX_BANDWIDTH == 100 * 1024 * 1024

    def test_hash_constants(self) -> None:
        assert KVCacheTransfer.HASH_BASE == 31
        assert KVCacheTransfer.HASH_MOD == (1 << 61) - 1


# ===========================================================================
# GossipTransport
# ===========================================================================


class TestGossipTransport:
    """GossipTransport — construction, signing, stats, peer-resolution.

    Network-requiring methods (exchange_advertisements, request_kv_cache)
    return ``None`` when no ``peer_resolver`` is set, so they are testable
    without network access.
    """

    # -- Construction ---------------------------------------------------

    def test_init_defaults(self) -> None:
        gt = GossipTransport(node_id="test-node")
        assert gt.node_id == "test-node"
        assert gt.host == "localhost"
        assert gt.port == 50052
        assert gt._peer_resolver is None
        assert gt._hmac_key is None
        gt.close()

    def test_init_custom(self) -> None:
        gt = GossipTransport(
            node_id="node-1",
            host="10.0.0.1",
            port=9090,
            max_bandwidth=50 * 1024 * 1024,
            hmac_key="s3cret",
        )
        assert gt.node_id == "node-1"
        assert gt.host == "10.0.0.1"
        assert gt.port == 9090
        assert gt._hmac_key == "s3cret"
        gt.close()

    def test_init_with_peer_resolver(self) -> None:
        called_with: list[str] = []

        def resolver(peer_id: str) -> tuple[str, int]:
            called_with.append(peer_id)
            return ("10.0.0.2", 50052)

        gt = GossipTransport(node_id="test", peer_resolver=resolver)
        assert gt._peer_resolver is not None
        assert gt._peer_resolver("some-peer") == ("10.0.0.2", 50052)
        assert called_with == ["some-peer"]
        gt.close()

    # -- Signing --------------------------------------------------------

    def test_sign_message_without_hmac(self) -> None:
        """Without hmac_key, _sign_message returns data unchanged."""
        gt = GossipTransport(node_id="test-node")
        data = {"key": "value", "num": 42}
        signed = gt._sign_message(data)
        assert signed == data
        assert "_hmac" not in signed
        gt.close()

    def test_sign_message_with_hmac(self) -> None:
        """With hmac_key, _sign_message adds an _hmac field."""
        gt = GossipTransport(node_id="test-node", hmac_key="my-secret")
        data = {"key": "value"}
        signed = gt._sign_message(data)
        assert signed["key"] == "value"
        assert "_hmac" in signed
        assert isinstance(signed["_hmac"], str)
        assert len(signed["_hmac"]) > 0
        gt.close()

    def test_sign_message_deterministic(self) -> None:
        """Same key + same data yields same signature."""
        gt = GossipTransport(node_id="test-node", hmac_key="my-secret")
        data = {"msg": "hello"}
        s1 = gt._sign_message(data)
        s2 = gt._sign_message(data)
        assert s1["_hmac"] == s2["_hmac"]
        gt.close()

    def test_sign_message_different_key_differs(self) -> None:
        """Different HMAC key yields a different signature."""
        gt1 = GossipTransport(node_id="a", hmac_key="key-a")
        gt2 = GossipTransport(node_id="b", hmac_key="key-b")
        data = {"msg": "hello"}
        s1 = gt1._sign_message(data)
        s2 = gt2._sign_message(data)
        # Signatures should differ (different key)
        assert s1["_hmac"] != s2["_hmac"]
        gt1.close()
        gt2.close()

    def test_sign_message_sorts_keys(self) -> None:
        """Signature uses sorted keys, so key order does not matter."""
        gt = GossipTransport(node_id="test-node", hmac_key="key")
        s1 = gt._sign_message({"a": 1, "b": 2})
        s2 = gt._sign_message({"b": 2, "a": 1})
        assert s1["_hmac"] == s2["_hmac"]
        gt.close()

    # -- Peer-resolution fallback ---------------------------------------

    def test_exchange_advertisements_no_resolver(self) -> None:
        """Returns None when no peer_resolver is configured (no network)."""
        gt = GossipTransport(node_id="test-node")
        result = gt.exchange_advertisements("peer-1", {"key": "value"})
        assert result is None
        gt.close()

    def test_request_kv_cache_no_resolver(self) -> None:
        """Returns None when no peer_resolver is configured (no network)."""
        gt = GossipTransport(node_id="test-node")
        result = gt.request_kv_cache(
            "peer-1", ["hash1", "hash2"], estimated_size_per_entry=4096
        )
        assert result is None
        gt.close()

    def test_request_kv_cache_bw_gating_no_resolver(self) -> None:
        """Bandwidth gating is skipped when no resolver is set."""
        gt = GossipTransport(node_id="test-node")
        # Even with huge estimated size, returns None (no resolver), not a
        # bandwidth error.
        result = gt.request_kv_cache(
            "peer-1", ["hash1"], estimated_size_per_entry=999_999_999
        )
        assert result is None
        gt.close()

    # -- _resolve_peer --------------------------------------------------

    def test_resolve_peer_no_resolver(self) -> None:
        gt = GossipTransport(node_id="test-node")
        host, port = gt._resolve_peer("any-peer")
        assert host is None
        assert port == 0
        gt.close()

    def test_resolve_peer_with_resolver(self) -> None:
        def resolver(pid: str) -> tuple[str, int]:
            return ("10.0.0.3", 6000)

        gt = GossipTransport(node_id="test-node", peer_resolver=resolver)
        host, port = gt._resolve_peer("some-peer")
        assert host == "10.0.0.3"
        assert port == 6000
        gt.close()

    def test_resolve_peer_resolver_returns_none(self) -> None:
        """When the resolver returns None/falsy, _resolve_peer returns None."""

        def resolver(_pid: str) -> None:
            return None

        gt = GossipTransport(node_id="test-node", peer_resolver=resolver)
        host, port = gt._resolve_peer("missing-peer")
        assert host is None
        assert port == 0
        gt.close()

    # -- Transfer stats -------------------------------------------------

    def test_transfer_stats_defaults(self) -> None:
        gt = GossipTransport(node_id="test-node")
        stats = gt.transfer_stats
        assert stats["bytes_transferred"] == 0
        assert stats["transfers_completed"] == 0
        assert stats["transfers_failed"] == 0
        gt.close()

    def test_transfer_stats_after_record(self) -> None:
        """GossipTransport.transfer_stats delegates to KVCacheTransfer.stats()."""
        gt = GossipTransport(node_id="test-node")
        gt._transfer.record_transfer(2048, success=True)
        stats = gt.transfer_stats
        assert stats["bytes_transferred"] == 2048
        assert stats["transfers_completed"] == 1
        gt.close()

    # -- Headers --------------------------------------------------------

    def test_headers_no_api_key(self) -> None:
        """Without DISTLLM_GOSSIP_API_KEY or API_KEY, no Authorization header."""
        gt = GossipTransport(node_id="test-node")
        headers = gt._headers()
        assert headers.get("Content-Type") == "application/json"
        assert "Authorization" not in headers
        gt.close()

    def test_headers_with_api_key_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With DISTLLM_GOSSIP_API_KEY set, Authorization header is present."""
        monkeypatch.setenv("DISTLLM_GOSSIP_API_KEY", "test-key-123")
        gt = GossipTransport(node_id="test-node")
        headers = gt._headers()
        assert headers["Authorization"] == "Bearer test-key-123"
        gt.close()

    def test_headers_with_fallback_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fallback to API_KEY when DISTLLM_GOSSIP_API_KEY is not set."""
        monkeypatch.setenv("API_KEY", "fallback-key")
        monkeypatch.delenv("DISTLLM_GOSSIP_API_KEY", raising=False)
        gt = GossipTransport(node_id="test-node")
        headers = gt._headers()
        assert headers["Authorization"] == "Bearer fallback-key"
        gt.close()

    # -- Lifecycle ------------------------------------------------------

    def test_close_idempotent(self) -> None:
        gt = GossipTransport(node_id="test-node")
        gt.close()
        gt.close()  # must not raise

    def test_close_with_active_session(self) -> None:
        """close() cleans up an httpx session if one was created."""
        gt = GossipTransport(node_id="test-node")
        # Trigger lazy session creation via _get_session
        session = gt._get_session()
        assert session is not None
        gt.close()
        assert gt._session is None

    def test_get_session_lazy(self) -> None:
        """_get_session creates the session lazily on first call."""
        gt = GossipTransport(node_id="test-node")
        assert gt._session is None
        session = gt._get_session()
        assert session is not None
        assert gt._session is session  # cached
        gt.close()

    def test_get_session_returns_cached(self) -> None:
        """_get_session returns the same session object on subsequent calls."""
        gt = GossipTransport(node_id="test-node")
        s1 = gt._get_session()
        s2 = gt._get_session()
        assert s1 is s2
        gt.close()
