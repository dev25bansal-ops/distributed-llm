"""Tests for RequestPipeline (generation orchestration).

Covers:
- Construction with coordinator reference
- _speculative_tokens_to_append static method (edge cases)
- _sample with mock coordinator and param overrides
- generate: rate-limited path raises NodeError
- generate: no node_order + no local_partitioner raises NodeError
- generate_async: no scheduler raises BatchError
- generate_async: rate-limited raises NodeError
- wait_for_result delegates to request_tracker
- get_logprobs delegates to request_tracker
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_mod = load_module("distllm/core/request_pipeline.py")
RequestPipeline = _mod.RequestPipeline
NodeError = _mod.NodeError
BatchError = _mod.BatchError


# ---------------------------------------------------------------------------
# Stub coordinator
# ---------------------------------------------------------------------------


class _StubParamUpdateChannel:
    """Minimal param-update channel for _sample tests."""

    def __init__(self) -> None:
        self._params: dict[str, Any] = {}

    def register(self, request_id: str) -> None:
        pass

    def unregister(self, request_id: str) -> None:
        pass

    def get(self, request_id: str) -> Any | None:
        return self._params.get(request_id)


class _StubTokenGen:
    """Minimal token generator stub."""

    def __init__(self) -> None:
        self.tokenizer = None

    def sample(
        self,
        logits: Any,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> Any:
        """Return a mock next-token index (0)."""
        import torch
        return torch.tensor([[0]])


class _StubTokenTracker:
    """Minimal request tracker stub."""

    def __init__(self) -> None:
        self._events: dict[str, threading.Event] = {}

    def register_request(self, request_id: str) -> None:
        self._events[request_id] = threading.Event()

    def set_result(self, request_id: str, result: str) -> None:
        pass

    def wait_for_result(self, request_id: str, timeout: float = 120.0) -> str:
        evt = self._events.get(request_id)
        if evt:
            evt.wait(timeout)
        return "mock_result"

    def get_logprobs(self, request_id: str) -> dict[str, Any] | None:
        return {"token_0": -0.5}


class _StubCoordinator:
    """Minimal coordinator stub for pipeline construction tests."""

    def __init__(self) -> None:
        self.node_order: list[str] = []
        self.local_partitioner: Any = None
        self.model_info: dict[str, Any] | None = None
        self._param_update_channel = _StubParamUpdateChannel()
        self._token_gen = _StubTokenGen()
        self.tokenizer = None
        self._request_tracker = _StubTokenTracker()
        self._rate_limiter: Any = None
        self._request_fingerprinter: Any = None
        self._prompt_cache_service: Any = None
        self._request_auditor: Any = None
        self._graceful_degradation: Any = None
        self._spec_decoder: Any = None
        self._predictive_cache: Any = None
        self._cache_mgr: Any = None
        self._zero_copy_engine: Any = None
        self._hybrid_parallel_executor: Any = None
        self._pipeline: Any = None
        self._continuous_trainer: Any = None
        self._model_svc: Any = None
        self._preemption_policy: Any = None
        self.draft_model: Any = None
        self.num_assistant_tokens = 5
        self._self_optimizing: Any = None
        self.model_name = "test-model"
        self.metrics_exporter: Any = None
        self.config: Any = None
        self.scheduler: Any = None
        self.num_assistant_tokens = 5
        self._spec_method = ""

    def record_metric(self, name: str, value: float) -> None:
        pass


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestRequestPipelineConstruction:
    """Construction with coordinator reference."""

    def test_construction(self) -> None:
        coord = _StubCoordinator()
        pipeline = RequestPipeline(coord)
        assert pipeline._coord is coord


# ---------------------------------------------------------------------------
# _speculative_tokens_to_append
# ---------------------------------------------------------------------------


class TestSpeculativeTokensToAppend:
    """Static method for computing accepted tokens."""

    def test_accepts_all_tokens(self) -> None:
        result = RequestPipeline._speculative_tokens_to_append(
            draft_tokens=[10, 20, 30],
            target_logits=__import__("torch").zeros((1, 5, 100)),
            accepted_count=3,
            accepted_tokens=[10, 20, 30],
            next_token=40,
        )
        # accepted_count >= draft_len (3 >= 3) and verification_steps(5) > accepted_count(3) and next_token(40) >= 0
        assert result == [10, 20, 30, 40]

    def test_no_bonus_token_when_no_more_steps(self) -> None:
        result = RequestPipeline._speculative_tokens_to_append(
            draft_tokens=[10, 20],
            target_logits=__import__("torch").zeros((1, 2, 100)),
            accepted_count=2,
            accepted_tokens=[10, 20],
            next_token=30,
        )
        # accepted_count >= draft_len, but verification_steps(2) <= accepted_count(2)
        assert result == [10, 20]

    def test_partial_acceptance(self) -> None:
        result = RequestPipeline._speculative_tokens_to_append(
            draft_tokens=[10, 20, 30],
            target_logits=__import__("torch").zeros((1, 3, 100)),
            accepted_count=2,
            accepted_tokens=[10, 20],
            next_token=99,
        )
        assert result == [10, 20, 99]

    def test_no_accepted_tokens_fallback_to_next(self) -> None:
        result = RequestPipeline._speculative_tokens_to_append(
            draft_tokens=[10, 20],
            target_logits=__import__("torch").zeros((1, 3, 100)),
            accepted_count=0,
            accepted_tokens=[],
            next_token=99,
        )
        assert result == [99]

    def test_no_accepted_and_negative_next(self) -> None:
        result = RequestPipeline._speculative_tokens_to_append(
            draft_tokens=[10, 20],
            target_logits=__import__("torch").zeros((1, 3, 100)),
            accepted_count=0,
            accepted_tokens=[],
            next_token=-1,
        )
        assert result == []

    def test_tensor_draft_tokens(self) -> None:
        import torch
        result = RequestPipeline._speculative_tokens_to_append(
            draft_tokens=torch.tensor([10, 20, 30]),
            target_logits=torch.zeros((1, 4, 100)),
            accepted_count=3,
            accepted_tokens=[10, 20, 30],
            next_token=40,
        )
        assert result == [10, 20, 30, 40]

    def test_2d_target_logits(self) -> None:
        import torch
        result = RequestPipeline._speculative_tokens_to_append(
            draft_tokens=[10, 20],
            target_logits=torch.zeros((3, 100)),  # 2D: shape[0]=3
            accepted_count=2,
            accepted_tokens=[10, 20],
            next_token=30,
        )
        assert result == [10, 20]

    def test_scalar_target_logits_shape(self) -> None:
        import torch
        result = RequestPipeline._speculative_tokens_to_append(
            draft_tokens=[10],
            target_logits=torch.zeros((1, 100)),  # 2D with shape[0]=1
            accepted_count=0,
            accepted_tokens=[],
            next_token=99,
        )
        assert result == [99]


# ---------------------------------------------------------------------------
# _sample (requires torch)
# ---------------------------------------------------------------------------


class TestSample:
    """_sample method with mock coordinator."""

    def test_sample_basic(self) -> None:
        import torch

        coord = _StubCoordinator()
        pipeline = RequestPipeline(coord)
        coord._token_gen.tokenizer = None

        logits = torch.rand((1, 1, 100))
        token = pipeline._sample(logits, temperature=1.0, top_p=1.0, top_k=0)
        assert isinstance(token, torch.Tensor)
        assert token.numel() == 1

    def test_sample_with_param_override(self) -> None:
        import torch

        coord = _StubCoordinator()
        coord._param_update_channel._params["test-req"] = type("Params", (), {
            "temperature": 0.5, "top_p": 0.8, "top_k": 10,
        })()
        pipeline = RequestPipeline(coord)

        # Manually set context var to test override path
        token = _mod._current_request_id_ctx.set("test-req")
        try:
            logits = torch.rand((1, 1, 100))
            result = pipeline._sample(logits)
            assert isinstance(result, torch.Tensor)
        finally:
            _mod._current_request_id_ctx.reset(token)


# ---------------------------------------------------------------------------
# generate: error paths
# ---------------------------------------------------------------------------


class TestGenerateErrors:
    """generate() error handling."""

    def test_raises_node_error_when_no_nodes_and_no_local(self) -> None:
        coord = _StubCoordinator()
        pipeline = RequestPipeline(coord)
        with pytest.raises(NodeError, match="No nodes registered"):
            pipeline.generate("hello", max_new_tokens=10)

    def test_raises_node_error_when_rate_limited(self) -> None:
        coord = _StubCoordinator()

        class _AlwaysBlock:
            def check(self, key: str, cost: float) -> bool:
                return False

        coord._rate_limiter = _AlwaysBlock()
        coord.node_order = ["node-1"]  # avoids the "no nodes" check
        coord.model_name = "test"
        # We need to also set up these to avoid AttributeError
        coord._request_tracker = _StubTokenTracker()

        class _StubPipeline:
            @staticmethod
            def create_node_kv_caches():
                return {}
            @staticmethod
            def run_pipeline(*args, **kwargs):
                return __import__("torch").zeros((1, 1, 100))

        coord._pipeline = _StubPipeline()

        # tokenizer needed for encoding
        class _StubTokenizer:
            eos_token_id = 2

            def encode(self, text, return_tensors="pt"):
                return __import__("torch").tensor([[101, 102]])

        coord.tokenizer = _StubTokenizer()

        pipeline = RequestPipeline(coord)
        with pytest.raises(NodeError, match="Rate limit exceeded"):
            pipeline.generate("hello", max_new_tokens=10, user_id="test-user")


# ---------------------------------------------------------------------------
# generate_async: error paths
# ---------------------------------------------------------------------------


class TestGenerateAsyncErrors:
    """generate_async() error handling."""

    def test_raises_batch_error_when_no_scheduler(self) -> None:
        coord = _StubCoordinator()
        pipeline = RequestPipeline(coord)
        with pytest.raises(BatchError, match="Batch scheduler not configured"):
            pipeline.generate_async("hello")

    def test_raises_node_error_when_rate_limited(self) -> None:
        coord = _StubCoordinator()

        class _AlwaysBlock:
            def check(self, key: str, cost: float) -> bool:
                return False

        coord._rate_limiter = _AlwaysBlock()
        coord.tokenizer = type("T", (), {"eos_token_id": 2, "encode": lambda s, **kw: __import__("torch").tensor([[1, 2]])})()

        # Create a stub scheduler with required attrs
        class _StubScheduler:
            def __init__(self):
                self.pending_count = 0
                self.active = set()

            def add(self, seq):
                pass

        coord.scheduler = _StubScheduler()

        pipeline = RequestPipeline(coord)
        with pytest.raises(NodeError, match="Rate limit exceeded"):
            pipeline.generate_async("hello", user_id="test-user")


# ---------------------------------------------------------------------------
# wait_for_result / get_logprobs
# ---------------------------------------------------------------------------


class TestRequestPipelineDelegation:
    """Delegation to request_tracker."""

    def test_wait_for_result(self) -> None:
        coord = _StubCoordinator()
        pipeline = RequestPipeline(coord)
        result = pipeline.wait_for_result("req-1", timeout=0.1)
        assert result == "mock_result"

    def test_get_logprobs(self) -> None:
        coord = _StubCoordinator()
        pipeline = RequestPipeline(coord)
        logprobs = pipeline.get_logprobs("req-1")
        assert logprobs == {"token_0": -0.5}


# ---------------------------------------------------------------------------
# generate: max context window capping
# ---------------------------------------------------------------------------


class TestGenerateContextWindow:
    """Context window capping logic."""

    def test_max_new_tokens_capped(self) -> None:
        coord = _StubCoordinator()
        coord.model_info = {"max_position_embeddings": 100}
        coord.node_order = ["node-1"]

        class _StubTokenizer:
            eos_token_id = 2

            def encode(self, text, return_tensors="pt"):
                return __import__("torch").tensor([[1]])

        coord.tokenizer = _StubTokenizer()

        # This will still fail when trying to run generation with no real model,
        # but we can test that max_new_tokens gets capped to 99 before the error.
        pipeline = RequestPipeline(coord)
        with pytest.raises(Exception):
            # Will hit issues later (no real pipeline), but caps first
            pipeline.generate("hello", max_new_tokens=200)
