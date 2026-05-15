"""Benchmark tests for Feature 20: Benchmark Regression Gate.

Uses pytest-benchmark to measure core operations and detect regressions.
"""

import struct

import pytest
import torch

from distllm.communication.serializers import tensor_to_proto, proto_to_tensor
from distllm.core.batch_scheduler import BatchScheduler, Sequence


class TestTokenGenerationSpeed:
    """Benchmark token generation throughput."""

    def test_batch_scheduling_throughput(self, benchmark):
        """Measure how many sequences can be scheduled per second."""

        def add_and_schedule():
            scheduler = BatchScheduler(max_batch_size=32, max_tokens_per_batch=4096)
            for i in range(100):
                seq = Sequence(request_id=f"req-{i}", prompt_tokens=[1] * 10)
                scheduler.add(seq)
            return scheduler.schedule()

        result = benchmark(add_and_schedule)
        assert result is not None


class TestSerializationSpeed:
    """Benchmark tensor serialization/deserialization speed."""

    @pytest.fixture
    def sample_tensor(self):
        return torch.randn(1, 128, 768)  # Typical hidden state shape

    def test_tensor_to_proto_speed(self, benchmark, sample_tensor):
        """Measure serialization speed (ops/sec)."""
        result = benchmark(tensor_to_proto, sample_tensor)
        assert result is not None

    def test_proto_to_tensor_speed(self, benchmark, sample_tensor):
        """Measure deserialization speed (ops/sec)."""
        proto = tensor_to_proto(sample_tensor)

        def deserialize():
            return proto_to_tensor(proto)

        result = benchmark(deserialize)
        assert result.shape == sample_tensor.shape

    def test_roundtrip_speed(self, benchmark, sample_tensor):
        """Measure full round-trip speed."""

        def roundtrip():
            proto = tensor_to_proto(sample_tensor)
            return proto_to_tensor(proto)

        result = benchmark(roundtrip)
        assert result.shape == sample_tensor.shape


class TestBatchOperations:
    """Benchmark batch-level operations."""

    def test_sequence_creation(self, benchmark):
        """Measure sequence creation speed."""
        result = benchmark(
            lambda: Sequence(
                request_id="bench-req",
                prompt_tokens=list(range(100)),
                max_new_tokens=256,
            )
        )
        assert result.request_id == "bench-req"

    def test_batch_merge(self, benchmark):
        """Measure batch merge/schedule speed."""

        def add_and_schedule():
            scheduler = BatchScheduler(max_batch_size=16, max_tokens_per_batch=2048)
            for i in range(8):
                seq = Sequence(request_id=f"req-{i}", prompt_tokens=[1] * 20)
                scheduler.add(seq)
            return scheduler.schedule()

        result = benchmark(add_and_schedule)
        assert result is not None
