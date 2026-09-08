"""Contract tests for InferenceEngine — documented invariants.

Contract tests pin the *public* contract of the engine: constructor
parameters, attribute wiring, default strategy selection, deterministic
mode, and resource lifecycle. They use the same ``load_module`` pattern as
``tests/core/test_inference_engine.py`` and never touch a real model.
"""

from __future__ import annotations

import concurrent.futures

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_inf_mod = load_module("distllm/core/inference_engine.py")
InferenceEngine = _inf_mod.InferenceEngine


# ---------------------------------------------------------------------------
# Constructor contract
# ---------------------------------------------------------------------------

class TestConstructorContract:
    def test_documented_params_accepted(self) -> None:
        engine = InferenceEngine(
            model_name="contract-model",
            dtype="float16",
            trust_remote_code=False,
            model_revision="main",
            pipeline=None,
            batch_scheduler=None,
        )
        assert engine.model_name == "contract-model"

    def test_dtype_stored_verbatim(self) -> None:
        for dtype in ("float16", "bfloat16", "float32"):
            engine = InferenceEngine(model_name="m", dtype=dtype)
            assert engine.dtype == dtype

    def test_model_revision_stored(self) -> None:
        engine = InferenceEngine(model_name="m", model_revision="v2")
        assert engine.model_revision == "v2"

    def test_trust_remote_code_stored(self) -> None:
        engine = InferenceEngine(model_name="m", trust_remote_code=True)
        assert engine.trust_remote_code is True

    def test_optional_deps_default_none(self) -> None:
        engine = InferenceEngine(model_name="m")
        assert engine.tokenizer is None
        assert engine._pipeline is None
        assert engine._batch_scheduler is None
        assert engine._recovery_manager is None
        assert engine._draft_model_fn is None

    def test_injected_pipeline_and_scheduler_wired(self) -> None:
        pipeline, scheduler = object(), object()
        engine = InferenceEngine(
            model_name="m", pipeline=pipeline, batch_scheduler=scheduler
        )
        assert engine._pipeline is pipeline
        assert engine._batch_scheduler is scheduler

    def test_draft_model_fn_stored(self) -> None:
        fn = lambda tokens: tokens  # noqa: E731
        engine = InferenceEngine(model_name="m", draft_model_fn=fn)
        assert engine._draft_model_fn is fn

    def test_draft_model_fns_stored(self) -> None:
        fns = [lambda t: t, lambda t: t]
        engine = InferenceEngine(model_name="m", draft_model_fns=fns)
        assert engine._draft_model_fns is fns


# ---------------------------------------------------------------------------
# Resource lifecycle contract
# ---------------------------------------------------------------------------

class TestResourceContract:
    def test_stream_pool_exists_and_bounded(self) -> None:
        engine = InferenceEngine(model_name="m")
        assert isinstance(engine._stream_pool, concurrent.futures.ThreadPoolExecutor)
        expected = min(32, (__import__("os").cpu_count() or 1) * 2)
        assert engine._stream_pool._max_workers == expected

    def test_draft_fleet_defaults(self) -> None:
        engine = InferenceEngine(model_name="m")
        assert engine._draft_fleet is None
        assert engine._draft_adaptive is False
        assert engine._draft_min_candidates == 2


# ---------------------------------------------------------------------------
# Strategy selection contract
# ---------------------------------------------------------------------------

class TestStrategyContract:
    def test_prompt_lookup_precedence_over_speculative(self) -> None:
        """Prompt-lookup wins when local partitioner is present."""
        engine = InferenceEngine(model_name="m", draft_model_fn=lambda t: t)
        engine.local_partitioner = object()
        strategy = engine._select_strategy()
        assert isinstance(strategy, _inf_mod._PromptLookupStrategy)

    def test_speculative_with_draft_and_pipeline(self) -> None:
        engine = InferenceEngine(
            model_name="m", pipeline=object(), draft_model_fn=lambda t: t
        )
        assert isinstance(engine._select_strategy(), _inf_mod._SpeculativeStrategy)

    def test_distributed_spec_with_remote_endpoint(self) -> None:
        engine = InferenceEngine(model_name="m", pipeline=object())
        engine._remote_draft_endpoint = "http://draft:8000"
        assert isinstance(
            engine._select_strategy(), _inf_mod._DistributedSpeculativeStrategy
        )


# ---------------------------------------------------------------------------
# Deterministic mode contract
# ---------------------------------------------------------------------------

class TestDeterministicContract:
    def test_disabled_by_default(self) -> None:
        engine = InferenceEngine(model_name="m")
        assert engine._deterministic_mode.is_enabled is False

    def test_round_trip(self) -> None:
        engine = InferenceEngine(model_name="m")
        engine.set_deterministic_mode(enabled=True, seed=7)
        assert engine._deterministic_mode.is_enabled is True
        assert engine._deterministic_mode._seed == 7
        engine.set_deterministic_mode(enabled=False)
        assert engine._deterministic_mode.is_enabled is False


# ---------------------------------------------------------------------------
# Replay buffer contract
# ---------------------------------------------------------------------------

class TestReplayContract:
    def test_recent_requests_starts_empty(self) -> None:
        engine = InferenceEngine(model_name="m")
        assert engine.get_recent_requests(n=5) == []

    def test_warmup_no_tokenizer_returns_zero(self) -> None:
        engine = InferenceEngine(model_name="m")
        assert engine.warmup(num_tokens=8) == 0.0
