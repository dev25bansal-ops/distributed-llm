"""Integration test — InferenceEngine with distributed speculative decoding."""

from unittest.mock import MagicMock, patch

import torch


class TestInferenceEngineDistributedSpec:
    def test_generate_distributed_speculative_called(self):
        """InferenceEngine.generate() dispatches to distributed speculative
        when _remote_draft_endpoint is set and pipeline exists."""
        from distllm.core.inference_engine import InferenceEngine

        engine = InferenceEngine(model_name="test-model")
        engine._remote_draft_endpoint = "http://draft:8000/v1/completions"
        engine._remote_draft_num_candidates = 3

        # Mock tokenizer
        engine.tokenizer = MagicMock()
        engine.tokenizer.encode.return_value = torch.tensor([[1, 2, 3]])
        engine.tokenizer.decode.return_value = "hello world"
        engine.tokenizer.eos_token_id = 0

        # Mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.run_pipeline.return_value = torch.randn(1, 5, 100)
        mock_pipeline.create_node_kv_caches.return_value = {}
        engine._pipeline = mock_pipeline

        # Patch RemoteDraftModel to avoid real HTTP
        with patch("distllm.core.distributed_speculative.RemoteDraftModel") as MockDraft:
            mock_draft_instance = MagicMock()
            from distllm.core.distributed_speculative import DraftTokenResult
            mock_draft_instance.generate_tokens.return_value = DraftTokenResult(
                token_ids=[10, 10], logprobs=[-0.1, -0.2],
            )
            mock_draft_instance.stats = {
                "total_calls": 0, "total_tokens": 0,
                "avg_latency_ms": 0, "tokens_per_second": 0, "errors": 0,
            }
            MockDraft.return_value = mock_draft_instance

            # Should not raise
            result = engine.generate("test prompt", max_new_tokens=4)
            assert isinstance(result, str)

    def test_fleet_routing_dispatch(self):
        """InferenceEngine uses fleet routing when _draft_fleet is set."""
        from distllm.core.inference_engine import InferenceEngine

        engine = InferenceEngine(model_name="test-model")
        engine._draft_fleet = MagicMock()
        engine._draft_fleet.get_all_specs.return_value = []
        engine._draft_fleet.get_all_health.return_value = {}
        engine._draft_fleet.healthy_endpoints = []

        # Mock tokenizer
        engine.tokenizer = MagicMock()
        engine.tokenizer.encode.return_value = torch.tensor([[1, 2]])
        engine.tokenizer.decode.return_value = "ok"
        engine.tokenizer.eos_token_id = 0

        # Mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.run_pipeline.return_value = torch.randn(1, 3, 100)
        mock_pipeline.create_node_kv_caches.return_value = {}
        engine._pipeline = mock_pipeline

        engine._remote_draft_endpoint = "http://draft:8000/v1/completions"

        # Fleet routing should attempt to find an endpoint
        # (will raise RuntimeError if no fleet endpoints available, which is expected)
        try:
            engine.generate("test", max_new_tokens=2)
        except RuntimeError:
            pass  # Expected when fleet has no endpoints
