"""Tests for CrossClusterForwarder — KV cache replication and cross-cluster
request forwarding for federated inference.

Tests use real objects from the module. No mocks, no GPU, no active network
target required (unreachable hosts are expected and tested).
"""

from __future__ import annotations

import asyncio
import json
from urllib.error import URLError

import pytest

from distllm.dist.cross_cluster import CrossClusterForwarder


# ── Constructor & basic properties ──────────────────────────────────────


class TestConstructon:
    def test_defaults(self) -> None:
        fwd = CrossClusterForwarder()
        assert fwd.timeout_s == 120.0
        assert fwd.max_retries == 2
        assert fwd.retry_delay_s == 1.0
        assert fwd._ray_workers == {}

    def test_custom_values(self) -> None:
        fwd = CrossClusterForwarder(timeout_s=5.0, max_retries=1, retry_delay_s=0.1)
        assert fwd.timeout_s == 5.0
        assert fwd.max_retries == 1
        assert fwd.retry_delay_s == 0.1

    def test_zero_retries_allowed(self) -> None:
        fwd = CrossClusterForwarder(max_retries=0)
        assert fwd.max_retries == 0


# ── Ray workers management ──────────────────────────────────────────────


class TestSetRayWorkers:
    def test_set_workers_updates_dict(self) -> None:
        fwd = CrossClusterForwarder()
        fwd.set_ray_workers({"cluster-a": []})
        assert "cluster-a" in fwd._ray_workers
        assert fwd._ray_workers["cluster-a"] == []

    def test_set_workers_merges(self) -> None:
        fwd = CrossClusterForwarder()
        fwd.set_ray_workers({"a": []})
        fwd.set_ray_workers({"b": []})
        assert "a" in fwd._ray_workers
        assert "b" in fwd._ray_workers

    def test_set_workers_overwrites(self) -> None:
        fwd = CrossClusterForwarder()
        fwd.set_ray_workers({"x": ["old"]})
        fwd.set_ray_workers({"x": ["new"]})
        assert fwd._ray_workers["x"] == ["new"]

    def test_set_workers_empty(self) -> None:
        fwd = CrossClusterForwarder()
        fwd.set_ray_workers({})
        assert fwd._ray_workers == {}


# ── _call_ray_worker (internal) ────────────────────────────────────────


class TestCallRayWorker:
    def test_no_workers_for_cluster_returns_none(self) -> None:
        fwd = CrossClusterForwarder()
        result = fwd._call_ray_worker("nonexistent", {"prompt": "hello"})
        assert result is None

    def test_empty_workers_list_returns_none(self) -> None:
        fwd = CrossClusterForwarder()
        fwd.set_ray_workers({"cluster-a": []})
        result = fwd._call_ray_worker("cluster-a", {"prompt": "hello"})
        assert result is None


# ── forward_request (HTTP fallback; no active server) ───────────────────


class TestForwardRequest:
    def test_raises_on_unreachable_no_cluster(self) -> None:
        """No cluster_id set, HTTP fallback: expect URLError."""
        fwd = CrossClusterForwarder(max_retries=0, timeout_s=0.1)
        with pytest.raises(URLError):
            fwd.forward_request(
                "http://127.0.0.1:1",
                {"model": "test", "prompt": "hello"},
            )

    def test_raises_on_unreachable_with_cluster_id(self) -> None:
        """cluster_id given but no ray workers => HTTP fallback => error."""
        fwd = CrossClusterForwarder(max_retries=0, timeout_s=0.1)
        with pytest.raises(URLError):
            fwd.forward_request(
                "http://127.0.0.1:1",
                {"model": "test", "prompt": "hello"},
                cluster_id="nonexistent",
            )

    def test_raises_on_unreachable_with_timeout_override(self) -> None:
        fwd = CrossClusterForwarder(max_retries=0, timeout_s=120.0)
        with pytest.raises(URLError):
            fwd.forward_request(
                "http://127.0.0.1:1",
                {"model": "test"},
                timeout_s=0.1,
            )

    def test_rejects_bad_url_scheme(self) -> None:
        fwd = CrossClusterForwarder(max_retries=0)
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            fwd.forward_request(
                "ftp://127.0.0.1:1",
                {"model": "test"},
            )


# ── forward_streaming (async; no active server) ────────────────────────


class TestForwardStreaming:
    """Uses real httpx (no mocking) — targets an unreachable port."""

    @pytest.mark.asyncio
    async def test_yields_error_on_connection_failure(self) -> None:
        fwd = CrossClusterForwarder(timeout_s=0.2)
        results: list[str] = []
        async for chunk in fwd.forward_streaming(
            "http://127.0.0.1:1",
            {"model": "test", "prompt": "hello"},
        ):
            results.append(chunk)

        assert len(results) == 1
        payload = json.loads(results[0])
        assert "error" in payload

    @pytest.mark.asyncio
    async def test_long_timeout_still_errors_on_unreachable(self) -> None:
        """Even with generous timeout, unreachable port fails fast."""
        fwd = CrossClusterForwarder(timeout_s=10.0)
        results: list[str] = []
        # Use a port that is very unlikely to be listening
        async for chunk in fwd.forward_streaming(
            "http://127.0.0.1:1",
            {"model": "test"},
            timeout_s=0.1,
        ):
            results.append(chunk)

        assert len(results) == 1
        payload = json.loads(results[0])
        assert "error" in payload


# ── _kv_to_protobuf (internal serialization) ────────────────────────────


class TestKvToProtobuf:
    def test_empty_dict(self) -> None:
        fwd = CrossClusterForwarder()
        result = fwd._kv_to_protobuf({})
        assert isinstance(result, str)
        assert len(result) == 0  # no layers -> empty protobuf -> empty b64

    def test_empty_list(self) -> None:
        fwd = CrossClusterForwarder()
        result = fwd._kv_to_protobuf([])
        assert isinstance(result, str)

    def test_layers_with_no_tensors(self) -> None:
        """Layers with string keys (not tensors) — no shape attr, so skipped."""
        fwd = CrossClusterForwarder()
        result = fwd._kv_to_protobuf({"layers": [{"key": "abc", "value": "def"}]})
        assert isinstance(result, str)
        # Decode and check layer exists but has no key_states
        import base64

        from distllm.dist import node_pb2

        pb = node_pb2.KVCacheProto()
        pb.ParseFromString(base64.b64decode(result))
        assert len(pb.layers) == 1
        assert not pb.layers[0].HasField("key_states")

    def test_layers_with_tensors(self) -> None:
        import torch

        fwd = CrossClusterForwarder()
        kv_data = {
            "layers": [
                {"key": torch.randn(2, 4, 8), "value": torch.randn(2, 4, 8)},
            ]
        }
        result = fwd._kv_to_protobuf(kv_data)
        import base64

        from distllm.dist import node_pb2

        pb = node_pb2.KVCacheProto.FromString(base64.b64decode(result))
        assert len(pb.layers) == 1
        assert pb.layers[0].HasField("key_states")
        assert pb.layers[0].HasField("value_states")
        assert list(pb.layers[0].key_states.shape) == [2, 4, 8]

    def test_layers_with_tensors_and_non_tensor_mixed(self) -> None:
        import torch

        fwd = CrossClusterForwarder()
        kv_data = {
            "layers": [
                {"key": torch.randn(1, 2), "value": torch.randn(1, 2)},
                {"key": "not-a-tensor", "value": "also-not"},
            ]
        }
        result = fwd._kv_to_protobuf(kv_data)
        import base64

        from distllm.dist import node_pb2

        pb = node_pb2.KVCacheProto.FromString(base64.b64decode(result))
        assert len(pb.layers) == 2
        assert pb.layers[0].HasField("key_states")
        assert not pb.layers[1].HasField("key_states")

    def test_layers_with_missing_key_or_value(self) -> None:
        import torch

        fwd = CrossClusterForwarder()
        kv_data = {
            "layers": [
                {"key": torch.randn(1, 2)},  # no "value"
                {"value": torch.randn(1, 2)},  # no "key"
            ]
        }
        result = fwd._kv_to_protobuf(kv_data)
        import base64

        from distllm.dist import node_pb2

        pb = node_pb2.KVCacheProto.FromString(base64.b64decode(result))
        assert len(pb.layers) == 2
        assert not pb.layers[0].HasField("key_states")
        assert not pb.layers[1].HasField("key_states")


# ── forward_kv_cache (no active server) ─────────────────────────────────


class TestForwardKvCache:
    def test_returns_false_on_unreachable(self) -> None:
        fwd = CrossClusterForwarder()
        result = fwd.forward_kv_cache(
            "http://127.0.0.1:1",
            "abc123",
            {"layers": []},
        )
        assert result is False

    def test_returns_false_with_tensor_data(self) -> None:
        import torch

        fwd = CrossClusterForwarder()
        result = fwd.forward_kv_cache(
            "http://127.0.0.1:1",
            "prefix-xyz",
            {
                "layers": [
                    {"key": torch.randn(1, 2), "value": torch.randn(1, 2)},
                ]
            },
        )
        assert result is False

    def test_returns_false_empty_prefix(self) -> None:
        fwd = CrossClusterForwarder()
        result = fwd.forward_kv_cache(
            "http://127.0.0.1:1",
            "",
            {"layers": []},
        )
        assert result is False


# ── replicate_kv_batch (no active server) ───────────────────────────────


class TestReplicateKvBatch:
    def test_returns_zero_on_unreachable_single(self) -> None:
        fwd = CrossClusterForwarder()
        count = fwd.replicate_kv_batch(
            [{"prefix_hash": "h1", "kv_data": {"layers": []}}],
            ["http://127.0.0.1:1"],
        )
        assert count == 0

    def test_returns_zero_on_multiple_unreachable(self) -> None:
        fwd = CrossClusterForwarder()
        count = fwd.replicate_kv_batch(
            [{"prefix_hash": "h1", "kv_data": {"layers": []}}],
            ["http://127.0.0.1:1", "http://127.0.0.1:2"],
        )
        assert count == 0

    def test_returns_zero_with_empty_entries(self) -> None:
        fwd = CrossClusterForwarder()
        count = fwd.replicate_kv_batch(
            [],
            ["http://127.0.0.1:1"],
        )
        assert count == 0

    def test_returns_zero_with_tensor_data(self) -> None:
        import torch

        fwd = CrossClusterForwarder()
        count = fwd.replicate_kv_batch(
            [
                {
                    "prefix_hash": "abc",
                    "kv_data": {
                        "layers": [
                            {
                                "key": torch.randn(1, 4),
                                "value": torch.randn(1, 4),
                            }
                        ]
                    },
                }
            ],
            ["http://127.0.0.1:1"],
        )
        assert count == 0

    def test_entries_with_missing_fields(self) -> None:
        fwd = CrossClusterForwarder()
        count = fwd.replicate_kv_batch(
            [{"prefix_hash": "h1"}, {}],
            ["http://127.0.0.1:1"],
        )
        assert count == 0


# ── Integration: protobuf round-trip with torch tensors ─────────────────


class TestProtobufRoundTrip:
    def test_kv_to_protobuf_round_trip(self) -> None:
        """Write KV data to protobuf, read it back, verify shape/dtype."""
        import torch

        from distllm.dist.pipeline.serialization import from_proto_tensor

        fwd = CrossClusterForwarder()
        k = torch.arange(16, dtype=torch.float32).reshape(2, 8)
        v = torch.arange(16, 32, dtype=torch.float32).reshape(2, 8)
        kv_data = {"layers": [{"key": k, "value": v}]}

        serialized = fwd._kv_to_protobuf(kv_data)
        import base64

        from distllm.dist import node_pb2

        pb = node_pb2.KVCacheProto.FromString(base64.b64decode(serialized))

        assert len(pb.layers) == 1
        k_restored = from_proto_tensor(pb.layers[0].key_states, device="cpu")
        assert k_restored.shape == (2, 8)
        assert k_restored.dtype == torch.float32
