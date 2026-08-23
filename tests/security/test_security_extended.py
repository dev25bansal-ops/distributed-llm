"""Security tests: proto fuzzing, cluster_key forgery, input validation.

Tests:
1. Malformed protobuf handling (giant tensors, invalid shapes)
2. Cluster key forgery (wrong key, empty key, timing attack)
3. Input size validation (DoS via oversized inputs)
4. Rate limiting on Profile() RPC
"""

from __future__ import annotations

import sys
import time
import types
from unittest.mock import MagicMock, patch

import pytest
import torch

sys.path.insert(0, "src")

from distllm.dist import node_pb2
from distllm.dist import node_service as node_service_module
from distllm.dist.node_service import NodeServicer, tensor_from_proto


# ── Proto Fuzzing ──────────────────────────────────────────────────


class TestProtoFuzzing:
    """Test handling of malformed protobuf messages."""

    def test_empty_tensor_proto(self):
        """Empty tensor proto should return empty tensor."""
        proto = node_pb2.TensorProto()
        result = tensor_from_proto(proto)
        assert result.numel() == 0

    def test_giant_shape_proto(self):
        """Giant shape should be rejected by validation."""
        proto = node_pb2.TensorProto(
            shape=[1, 1000000],
            dtype="torch.float32",
            raw_data=b"\x00" * 100,
        )
        # A payload far smaller than the declared shape is rejected with a
        # clear error (never a cryptic torch reshape failure or a huge
        # zero-allocation from attacker-declared shapes).
        with pytest.raises(ValueError, match="payload mismatch"):
            tensor_from_proto(proto)

    def test_mismatched_shape_and_data(self):
        """Mismatched shape/data should not crash."""
        proto = node_pb2.TensorProto(
            shape=[10, 10, 10],
            dtype="torch.float32",
            raw_data=b"\x00" * 4,  # Too small for 1000 elements
        )
        try:
            result = tensor_from_proto(proto)
        except Exception:
            pass  # Expected

    def test_invalid_dtype_proto(self):
        """Invalid dtype should fall back to float32."""
        proto = node_pb2.TensorProto(
            shape=[2, 2],
            dtype="invalid_dtype",
            raw_data=b"\x00" * 16,
        )
        result = tensor_from_proto(proto)
        assert result.dtype == torch.float32

    def test_zero_dimension_proto(self):
        """Zero dimension should be handled."""
        proto = node_pb2.TensorProto(
            shape=[0, 10],
            dtype="torch.float32",
            raw_data=b"",
        )
        result = tensor_from_proto(proto)
        assert result.numel() == 0


# ── Cluster Key Forgery ────────────────────────────────────────────


class TestClusterKeyForgery:
    """Test authentication with forged or missing cluster keys."""

    def _make_servicer(self, cluster_key="valid-key-123"):
        worker = MagicMock()
        worker.node_id = "test-node"
        worker._get_device.return_value = "cpu"
        return NodeServicer(worker, cluster_key=cluster_key)

    def test_valid_key_accepted(self):
        """Valid cluster key should be accepted."""
        servicer = self._make_servicer()
        req = MagicMock()
        req.cluster_key = "valid-key-123"
        assert servicer._check_auth(req) is True

    def test_wrong_key_rejected(self):
        """Wrong cluster key should be rejected."""
        servicer = self._make_servicer()
        req = MagicMock()
        req.cluster_key = "wrong-key"
        assert servicer._check_auth(req) is False

    def test_empty_key_rejected(self):
        """Empty cluster key should be rejected when key is required."""
        servicer = self._make_servicer()
        req = MagicMock()
        req.cluster_key = ""
        assert servicer._check_auth(req) is False

    def test_none_key_rejected(self):
        """None cluster key should be rejected when key is required."""
        servicer = self._make_servicer()
        req = MagicMock()
        req.cluster_key = None
        assert servicer._check_auth(req) is False

    def test_no_key_fails_closed(self):
        """No cluster key configured -> every RPC is rejected (fail closed).

        Previously a keyless servicer accepted everything, which let any LAN
        attacker call TransferWeights and exfiltrate model weights.
        """
        servicer = self._make_servicer(cluster_key=None)
        req = MagicMock()
        req.cluster_key = "anything"
        assert servicer._check_auth(req) is False

    def test_timing_attack_resistance(self):
        """Auth check should use constant-time comparison."""
        servicer = self._make_servicer()

        # Both should take similar time (constant-time comparison)
        req_correct = MagicMock()
        req_correct.cluster_key = "valid-key-123"

        req_wrong = MagicMock()
        req_wrong.cluster_key = "wrong-key-xxx"

        times_correct = []
        times_wrong = []

        for _ in range(100):
            t0 = time.perf_counter_ns()
            servicer._check_auth(req_correct)
            times_correct.append(time.perf_counter_ns() - t0)

            t0 = time.perf_counter_ns()
            servicer._check_auth(req_wrong)
            times_wrong.append(time.perf_counter_ns() - t0)

        avg_correct = sum(times_correct) / len(times_correct)
        avg_wrong = sum(times_wrong) / len(times_wrong)

        # Timing difference should be < 50% (constant-time)
        ratio = max(avg_correct, avg_wrong) / max(min(avg_correct, avg_wrong), 1)
        assert ratio < 2.0, f"Timing side channel detected: ratio={ratio:.2f}"


# ── Input Size Validation ──────────────────────────────────────────


class TestInputValidation:
    """Test input size limits to prevent DoS."""

    def _make_servicer(self):
        worker = MagicMock()
        worker.node_id = "test-node"
        worker._get_device.return_value = "cpu"
        worker.forward_fn.return_value = (
            torch.randn(1, 1, 32000),
            [],
        )
        # Keyed servicer so auth passes and the input-validation logic runs.
        return NodeServicer(worker, cluster_key="valid-key-123")

    def test_batch_size_limit(self):
        """Batch size exceeding MAX_BATCH_SIZE should be rejected."""
        servicer = self._make_servicer()

        # Create request with giant batch
        req = MagicMock()
        req.cluster_key = "valid-key-123"
        req.input_ids = list(range(1024 * 131072 + 1))  # Over limit
        req.hidden_states = None
        req.attention_mask = None
        req.position_ids = None
        req.kv_cache = None
        req.request_id = "test"

        context = MagicMock()
        resp = servicer.ForwardPass(req, context)
        assert resp.success is False
        assert "too large" in resp.error_message.lower()

    def test_kv_cache_layer_limit(self):
        """KV cache with too many layers should be rejected."""
        servicer = self._make_servicer()

        req = MagicMock()
        req.cluster_key = "valid-key-123"
        req.input_ids = []
        req.hidden_states = None
        req.request_id = "test"

        # Create KV cache with too many layers
        kv_cache = MagicMock()
        kv_cache.layers = [MagicMock() for _ in range(NodeServicer.MAX_KV_LAYERS + 1)]
        for layer in kv_cache.layers:
            layer.key_states = MagicMock()
            layer.key_states.shape = [1, 32, 100, 128]
            layer.key_states.raw_data = b"\x00"
            layer.value_states = MagicMock()
            layer.value_states.shape = [1, 32, 100, 128]
            layer.value_states.raw_data = b"\x00"

        req.kv_cache = kv_cache
        req.attention_mask = None
        req.position_ids = None

        context = MagicMock()
        resp = servicer.ForwardPass(req, context)
        assert resp.success is False
        assert "too many layers" in resp.error_message.lower()


# ── Rate Limiting ──────────────────────────────────────────────────


class TestRateLimiting:
    """Test rate limiting on Profile() RPC."""

    def _make_servicer(self):
        worker = MagicMock()
        worker.node_id = "test-node"
        return NodeServicer(worker, cluster_key="valid-key-123")

    def test_profile_rate_limit(self):
        """Profile() should rate limit after burst."""
        servicer = self._make_servicer()
        context = MagicMock()
        req = MagicMock()
        req.cluster_key = "valid-key-123"
        req.node_id = "test"

        # Freeze the limiter's clock: real Profile calls take tens of ms on
        # GPU hosts, and the token bucket refills at PROFILE_RATE_LIMIT/s,
        # so a wall-clock burst can outrun the limit.  A frozen clock makes
        # the burst deterministic (no refill during the loop).
        clock = {"t": time.monotonic()}
        fake_time = types.SimpleNamespace(monotonic=lambda: clock["t"])
        with patch.object(node_service_module, "time", fake_time):
            # Burst: should succeed up to limit
            for _ in range(NodeServicer.PROFILE_RATE_LIMIT):
                resp = servicer.Profile(req, context)
                assert resp.gpu_name != "rate_limited"

            # Next request should be rate limited
            resp = servicer.Profile(req, context)
            assert resp.gpu_name == "rate_limited"

    def test_rate_limit_recovers(self):
        """Rate limit should recover after cooldown."""
        servicer = self._make_servicer()
        context = MagicMock()
        req = MagicMock()
        req.cluster_key = None
        req.node_id = "test"

        # Exhaust rate limit
        for _ in range(NodeServicer.PROFILE_RATE_LIMIT + 5):
            servicer.Profile(req, context)

        # Wait for cooldown
        time.sleep(1.1)

        # Should work again
        resp = servicer.Profile(req, context)
        assert resp.gpu_name != "rate_limited"
