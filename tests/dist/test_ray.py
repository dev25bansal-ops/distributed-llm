"""Tests for the Ray-based pipeline execution module.

Tests the ``RayPipeline`` public API without requiring a running Ray
cluster.  All tests use real objects from the module -- no mocks.
"""

from __future__ import annotations

import pytest
import torch

from distllm.dist.backends.ray import RayPipeline


# ── RayPipeline ──


class TestRayPipelineInit:
    """Constructor and initial state."""

    def test_initial_workers_empty(self) -> None:
        pipeline = RayPipeline()
        assert pipeline.workers == []

    def test_initial_worker_ids_empty(self) -> None:
        pipeline = RayPipeline()
        assert pipeline.worker_ids == []

    def test_initial_num_gpus_zero(self) -> None:
        pipeline = RayPipeline()
        assert pipeline._num_gpus == 0

    def test_initial_num_stages_zero(self) -> None:
        pipeline = RayPipeline()
        assert pipeline.num_stages == 0
        assert pipeline.get_num_stages() == 0


class TestRayPipelineAddWorker:
    """``add_worker`` -- appends workers and node ids."""

    def test_add_single_worker(self) -> None:
        pipeline = RayPipeline()
        worker = object()
        pipeline.add_worker(worker, "node-0")
        assert len(pipeline.workers) == 1
        assert pipeline.workers[0] is worker
        assert pipeline.worker_ids == ["node-0"]

    def test_add_multiple_workers(self) -> None:
        pipeline = RayPipeline()
        w0 = object()
        w1 = object()
        pipeline.add_worker(w0, "node-0")
        pipeline.add_worker(w1, "node-1")
        assert len(pipeline.workers) == 2
        assert pipeline.workers == [w0, w1]
        assert pipeline.worker_ids == ["node-0", "node-1"]

    def test_add_worker_increments_num_stages(self) -> None:
        pipeline = RayPipeline()
        assert pipeline.num_stages == 0
        pipeline.add_worker(object(), "n0")
        assert pipeline.num_stages == 1
        pipeline.add_worker(object(), "n1")
        assert pipeline.num_stages == 2

    def test_add_worker_accepts_any_object_type(self) -> None:
        """``add_worker`` does not enforce the ``ActorHandle`` type at
        runtime because of ``from __future__ import annotations``."""
        pipeline = RayPipeline()
        pipeline.add_worker("string-as-worker", "string-node")
        pipeline.add_worker(42, "int-node")
        pipeline.add_worker(None, "none-node")
        assert pipeline.num_stages == 3


class TestRayPipelineNumStages:
    """``num_stages`` property and ``get_num_stages`` method."""

    def test_empty(self) -> None:
        pipeline = RayPipeline()
        assert pipeline.num_stages == 0
        assert pipeline.get_num_stages() == 0

    def test_after_add(self) -> None:
        pipeline = RayPipeline()
        pipeline.add_worker(object(), "n0")
        assert pipeline.num_stages == 1
        assert pipeline.get_num_stages() == 1

    def test_property_and_method_agree(self) -> None:
        pipeline = RayPipeline()
        for i in range(5):
            pipeline.add_worker(object(), f"n{i}")
            assert pipeline.num_stages == pipeline.get_num_stages()


class TestRayPipelineRunPipeline:
    """``run_pipeline`` -- raises on empty pipeline.

    Note: tests with registered workers require a live Ray cluster
    and are therefore excluded from this deterministic unit suite.
    """

    def test_empty_pipeline_raises(self) -> None:
        pipeline = RayPipeline()
        input_ids = torch.zeros(1, 4)
        with pytest.raises(RuntimeError, match="No workers registered"):
            pipeline.run_pipeline(input_ids, "req-1")


class TestRayPipelineRunPipelineAsync:
    """``run_pipeline_async`` -- raises on empty pipeline."""

    def test_empty_pipeline_raises(self) -> None:
        pipeline = RayPipeline()
        input_ids = torch.zeros(1, 4)
        with pytest.raises(RuntimeError, match="No workers registered"):
            pipeline.run_pipeline_async(input_ids, "req-1")


class TestRayPipelineClearCache:
    """KV cache clearing methods -- safe no-ops on empty pipeline."""

    def test_clear_kv_cache_empty_pipeline(self) -> None:
        pipeline = RayPipeline()
        pipeline.clear_kv_cache("req-1")  # should not raise

    def test_clear_all_kv_caches_empty_pipeline(self) -> None:
        pipeline = RayPipeline()
        pipeline.clear_all_kv_caches()  # should not raise
