"""Unit tests for PipelineExecutor (pipeline_composer.py)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from distllm.core.pipeline_composer import (
    PipelineExecutor,
    PipelineSpec,
    PipelineStep,
    StepType,
)


class TestPipelineSpec:
    def test_minimal_spec(self):
        spec = PipelineSpec(pipeline_id="pipe-1", steps=[])
        assert spec.pipeline_id == "pipe-1"
        assert spec.steps == []
        assert spec.fallback_pipeline_id is None

    def test_spec_with_fallback(self):
        spec = PipelineSpec(
            pipeline_id="pipe-1",
            steps=[PipelineStep(model="m1", step_type=StepType.generate)],
            fallback_pipeline_id="pipe-0",
        )
        assert spec.fallback_pipeline_id == "pipe-0"


class TestPipelineStep:
    def test_minimal_step(self):
        step = PipelineStep(model="m1", step_type=StepType.generate)
        assert step.model == "m1"
        assert step.step_type == StepType.generate
        assert step.params == {}
        assert step.timeout_ms is None

    def test_step_with_params(self):
        step = PipelineStep(
            model="m1",
            step_type=StepType.embedding,
            params={"batch_size": 32},
            timeout_ms=1000.0,
        )
        assert step.params == {"batch_size": 32}
        assert step.timeout_ms == 1000.0


class TestPipelineExecutorRegister:
    def test_register_and_get(self):
        executor = PipelineExecutor()
        spec = PipelineSpec(pipeline_id="pipe-1", steps=[])
        executor.register(spec)
        assert executor.get("pipe-1") is spec

    def test_get_unknown_pipeline(self):
        executor = PipelineExecutor()
        assert executor.get("pipe-missing") is None

    def test_fallback_chain(self):
        executor = PipelineExecutor()
        fallback = PipelineSpec(pipeline_id="pipe-fallback", steps=[])
        primary = PipelineSpec(
            pipeline_id="pipe-primary",
            steps=[],
            fallback_pipeline_id="pipe-fallback",
        )
        executor.register(fallback)
        executor.register(primary)

        # Getting the fallback by primary ID should work
        assert executor.get("pipe-primary") is primary
        # Getting the fallback by its own ID should work
        assert executor.get("pipe-fallback") is fallback

    def test_fallback_chain_missing_fallback(self):
        executor = PipelineExecutor()
        primary = PipelineSpec(
            pipeline_id="pipe-primary",
            steps=[],
            fallback_pipeline_id="pipe-missing",
        )
        executor.register(primary)
        # fallback doesn't exist, so get returns None for that ID
        assert executor.get("pipe-missing") is None


class TestPipelineExecutorExecute:
    @pytest.mark.asyncio
    async def test_unknown_pipeline(self):
        executor = PipelineExecutor()
        results = [r async for r in executor.execute("pipe-unknown", "hello")]
        assert len(results) == 1
        assert results[0]["step_type"] == "error"
        assert "Unknown pipeline" in results[0]["error"]

    @pytest.mark.asyncio
    async def test_single_transform_step(self):
        executor = PipelineExecutor()
        spec = PipelineSpec(
            pipeline_id="pipe-1",
            steps=[PipelineStep(
                model="identity",
                step_type=StepType.transform,
                params={"type": "identity"},
            )],
        )
        executor.register(spec)

        results = [r async for r in executor.execute("pipe-1", "hello")]
        assert len(results) == 2  # step + complete
        assert results[0]["step_index"] == 0
        assert results[0]["step_type"] == "transform"
        assert results[0]["output"] == "hello"
        assert results[0]["error"] is None
        assert results[1]["step_type"] == "complete"

    @pytest.mark.asyncio
    async def test_transform_truncate(self):
        executor = PipelineExecutor()
        spec = PipelineSpec(
            pipeline_id="pipe-1",
            steps=[PipelineStep(
                model="identity",
                step_type=StepType.transform,
                params={"type": "truncate", "max_chars": 5},
            )],
        )
        executor.register(spec)

        results = [r async for r in executor.execute("pipe-1", "hello world")]
        assert results[0]["output"] == "hello"

    @pytest.mark.asyncio
    async def test_transform_prepend(self):
        executor = PipelineExecutor()
        spec = PipelineSpec(
            pipeline_id="pipe-1",
            steps=[PipelineStep(
                model="identity",
                step_type=StepType.transform,
                params={"type": "prepend", "text": "PREFIX: "},
            )],
        )
        executor.register(spec)

        results = [r async for r in executor.execute("pipe-1", "world")]
        assert results[0]["output"] == "PREFIX: world"

    @pytest.mark.asyncio
    async def test_transform_append(self):
        executor = PipelineExecutor()
        spec = PipelineSpec(
            pipeline_id="pipe-1",
            steps=[PipelineStep(
                model="identity",
                step_type=StepType.transform,
                params={"type": "append", "text": " SUFFIX"},
            )],
        )
        executor.register(spec)

        results = [r async for r in executor.execute("pipe-1", "hello")]
        assert results[0]["output"] == "hello SUFFIX"

    @pytest.mark.asyncio
    async def test_embedding_step(self):
        coord = MagicMock()
        coord.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
        executor = PipelineExecutor(coordinator=coord)
        spec = PipelineSpec(
            pipeline_id="pipe-1",
            steps=[PipelineStep(model="emb1", step_type=StepType.embedding)],
        )
        executor.register(spec)

        results = [r async for r in executor.execute("pipe-1", "text")]
        assert results[0]["output"] == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_reranker_step(self):
        coord = MagicMock()
        coord.rerank = AsyncMock(return_value=[{"text": "result1", "score": 0.9}])
        executor = PipelineExecutor(coordinator=coord)
        spec = PipelineSpec(
            pipeline_id="pipe-1",
            steps=[PipelineStep(model="rank1", step_type=StepType.reranker)],
        )
        executor.register(spec)

        results = [r async for r in executor.execute("pipe-1", "query")]
        assert results[0]["output"] == [{"text": "result1", "score": 0.9}]

    @pytest.mark.asyncio
    async def test_generate_step(self):
        coord = MagicMock()
        coord.generate_async = AsyncMock(return_value="generated text")
        executor = PipelineExecutor(coordinator=coord)
        spec = PipelineSpec(
            pipeline_id="pipe-1",
            steps=[PipelineStep(
                model="gen1",
                step_type=StepType.generate,
                params={"temperature": 0.7},
            )],
        )
        executor.register(spec)

        results = [r async for r in executor.execute("pipe-1", "prompt")]
        assert results[0]["output"] == "generated text"
        coord.generate_async.assert_called_once_with(prompt="prompt", temperature=0.7)

    @pytest.mark.asyncio
    async def test_step_failure(self):
        coord = MagicMock()
        coord.embed = AsyncMock(side_effect=RuntimeError("embed failed"))
        executor = PipelineExecutor(coordinator=coord)
        spec = PipelineSpec(
            pipeline_id="pipe-1",
            steps=[PipelineStep(model="emb1", step_type=StepType.embedding)],
        )
        executor.register(spec)

        results = [r async for r in executor.execute("pipe-1", "text")]
        assert len(results) == 1  # step fails, no complete yield
        assert results[0]["error"] == "embed failed"
        assert results[0]["output"] is None

    @pytest.mark.asyncio
    async def test_multi_step_pipeline(self):
        coord = MagicMock()
        coord.embed = AsyncMock(return_value=[0.1, 0.2])
        coord.generate_async = AsyncMock(return_value="final output")
        executor = PipelineExecutor(coordinator=coord)
        spec = PipelineSpec(
            pipeline_id="pipe-1",
            steps=[
                PipelineStep(model="emb1", step_type=StepType.embedding),
                PipelineStep(model="gen1", step_type=StepType.generate),
            ],
        )
        executor.register(spec)

        results = [r async for r in executor.execute("pipe-1", "input")]
        assert len(results) == 3  # embed + generate + complete
        assert results[0]["step_index"] == 0
        assert results[0]["step_type"] == "embedding"
        assert results[1]["step_index"] == 1
        assert results[1]["step_type"] == "generate"
        assert results[2]["step_type"] == "complete"

    @pytest.mark.asyncio
    async def test_embedding_step_no_coordinator(self):
        executor = PipelineExecutor(coordinator=None)
        spec = PipelineSpec(
            pipeline_id="pipe-1",
            steps=[PipelineStep(model="emb1", step_type=StepType.embedding)],
        )
        executor.register(spec)

        results = [r async for r in executor.execute("pipe-1", "text")]
        assert "Coordinator required" in results[0]["error"]

    @pytest.mark.asyncio
    async def test_generate_step_no_coordinator(self):
        executor = PipelineExecutor(coordinator=None)
        spec = PipelineSpec(
            pipeline_id="pipe-1",
            steps=[PipelineStep(model="gen1", step_type=StepType.generate)],
        )
        executor.register(spec)

        results = [r async for r in executor.execute("pipe-1", "prompt")]
        assert "Coordinator required" in results[0]["error"]
