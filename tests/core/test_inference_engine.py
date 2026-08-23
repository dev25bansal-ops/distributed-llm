"""Tests for InferenceEngine using real objects via load_module pattern."""

from __future__ import annotations

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_inf_mod = load_module("distllm/core/inference_engine.py")
InferenceEngine = _inf_mod.InferenceEngine
GenerationStrategy = _inf_mod.GenerationStrategy


class TestInferenceEngineConstruction:
    """InferenceEngine can be constructed with minimal parameters."""

    def test_minimal_construction(self) -> None:
        engine = InferenceEngine(model_name="test-model")
        assert engine.model_name == "test-model"
        assert engine.dtype == "float16"
        assert engine.tokenizer is None
        assert engine._pipeline is None
        assert engine.local_partitioner is None

    def test_custom_dtype(self) -> None:
        engine = InferenceEngine(model_name="test-model", dtype="bfloat16")
        assert engine.dtype == "bfloat16"

    def test_with_tokenizer_and_pipeline(self) -> None:
        engine = InferenceEngine(
            model_name="test-model",
            pipeline=object(),
            batch_scheduler=object(),
        )
        assert engine._pipeline is not None
        assert engine._batch_scheduler is not None

    def test_revision_default(self) -> None:
        engine = InferenceEngine(model_name="test-model")
        assert engine.model_revision == "main"

    def test_trust_remote_code_stored(self) -> None:
        engine = InferenceEngine(model_name="test-model", trust_remote_code=True)
        assert engine.trust_remote_code is True

    def test_default_strategies_initialized(self) -> None:
        engine = InferenceEngine(model_name="test-model")
        assert hasattr(engine, "_local_strategy")
        assert hasattr(engine, "_speculative_strategy")
        assert hasattr(engine, "_distributed_spec_strategy")
        assert hasattr(engine, "_distributed_strategy")
        assert hasattr(engine, "_prompt_lookup_strategy")


class TestInferenceEngineSelectStrategy:
    """_select_strategy returns correct strategy based on config."""

    def test_distributed_when_no_local_or_draft(self) -> None:
        engine = InferenceEngine(model_name="test-model")
        strategy = engine._select_strategy()
        # Without local_partitioner, draft_model_fn, or remote endpoint
        # and without pipeline, the distributed strategy needs node_order.
        # But _select_strategy checks local_partitioner first.
        assert isinstance(strategy, _inf_mod._DistributedStrategy)

    def test_prompt_lookup_when_local(self) -> None:
        engine = InferenceEngine(model_name="test-model")
        engine.local_partitioner = object()  # type: ignore[assignment]
        strategy = engine._select_strategy()
        assert isinstance(strategy, _inf_mod._PromptLookupStrategy)

    def test_speculative_when_draft_and_pipeline(self) -> None:
        engine = InferenceEngine(
            model_name="test-model",
            pipeline=object(),
            draft_model_fn=lambda x: x,
        )
        strategy = engine._select_strategy()
        assert isinstance(strategy, _inf_mod._SpeculativeStrategy)

    def test_dist_spec_when_remote_endpoint_and_pipeline(self) -> None:
        engine = InferenceEngine(
            model_name="test-model",
            pipeline=object(),
        )
        engine._remote_draft_endpoint = "http://draft:8000"
        strategy = engine._select_strategy()
        assert isinstance(strategy, _inf_mod._DistributedSpeculativeStrategy)


class TestInferenceEngineDeterministicMode:
    """Deterministic mode toggling."""

    def test_deterministic_default_disabled(self) -> None:
        engine = InferenceEngine(model_name="test-model")
        assert engine._deterministic_mode.is_enabled is False

    def test_enable_deterministic(self) -> None:
        engine = InferenceEngine(model_name="test-model")
        engine.set_deterministic_mode(enabled=True)
        assert engine._deterministic_mode.is_enabled is True

    def test_disable_deterministic(self) -> None:
        engine = InferenceEngine(model_name="test-model")
        engine.set_deterministic_mode(enabled=True)
        engine.set_deterministic_mode(enabled=False)
        assert engine._deterministic_mode.is_enabled is False

    def test_deterministic_custom_seed(self) -> None:
        engine = InferenceEngine(model_name="test-model")
        engine.set_deterministic_mode(enabled=True, seed=99)
        assert engine._deterministic_mode._seed == 99


class TestInferenceEngineReplayBuffer:
    """Request replay buffer integration."""

    def test_recent_requests_empty_initially(self) -> None:
        engine = InferenceEngine(model_name="test-model")
        recent = engine.get_recent_requests(n=10)
        assert isinstance(recent, list)
        assert len(recent) == 0

    def test_replay_buffer_exists(self) -> None:
        engine = InferenceEngine(model_name="test-model")
        assert engine._replay_buffer is not None

    def test_warmup_returns_zero_when_no_tokenizer(self) -> None:
        engine = InferenceEngine(model_name="test-model")
        result = engine.warmup(num_tokens=8)
        assert result == 0.0
