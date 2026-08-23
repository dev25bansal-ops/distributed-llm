"""Tests for distllm.dist.wandb_integration.

These tests use only real objects from the module — no mocking.
Since wandb is not installed in the test environment, _WANDB_AVAILABLE
is False and all logging methods are graceful no-ops. This makes the
tests fully deterministic with no network or GPU dependencies.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from distllm.dist.wandb_integration import WandbExperiment, _WANDB_AVAILABLE


# ── Module-level constant ──────────────────────────────────────────────


class TestWANDB_AVAILABLE:
    """The _WANDB_AVAILABLE flag should be False when wandb is absent."""

    def test_not_available(self) -> None:
        assert _WANDB_AVAILABLE is False


# ── WandbExperiment construction ──────────────────────────────────────


class TestWandbExperimentConstruction:
    """Test the constructor with various argument combinations."""

    def test_defaults(self) -> None:
        exp = WandbExperiment()
        assert exp.project == "distllm"
        assert exp.config == {}
        assert exp.run_name is None
        assert exp.tags is None
        assert exp.group is None
        assert exp._external_run is False
        assert exp._run is None
        assert exp._step == 0

    def test_all_params_explicit(self) -> None:
        exp = WandbExperiment(
            project="my-project",
            config={"lr": 0.01, "epochs": 10},
            run_name="run-001",
            tags=["test", "quant"],
            group="experiment-group",
        )
        assert exp.project == "my-project"
        assert exp.config == {"lr": 0.01, "epochs": 10}
        assert exp.run_name == "run-001"
        assert exp.tags == ["test", "quant"]
        assert exp.group == "experiment-group"
        assert exp._external_run is False

    def test_config_none_becomes_empty_dict(self) -> None:
        exp = WandbExperiment(config=None)
        assert exp.config == {}

    def test_empty_config(self) -> None:
        exp = WandbExperiment(config={})
        assert exp.config == {}

    def test_empty_tags_list(self) -> None:
        exp = WandbExperiment(tags=[])
        assert exp.tags == []

    def test_project_empty_string(self) -> None:
        exp = WandbExperiment(project="")
        assert exp.project == ""

    def test_external_run_sets_flag(self) -> None:
        external = SimpleNamespace()  # any object
        exp = WandbExperiment(run=external)
        assert exp._external_run is True
        assert exp._run is external

    def test_external_run_with_other_params(self) -> None:
        external = SimpleNamespace()
        exp = WandbExperiment(
            project="ext-project",
            config={"key": "val"},
            run_name="ext-run",
            tags=["ext"],
            group="ext-group",
            run=external,
        )
        assert exp.project == "ext-project"
        assert exp.config == {"key": "val"}
        assert exp.run_name == "ext-run"
        assert exp.tags == ["ext"]
        assert exp.group == "ext-group"
        assert exp._external_run is True
        assert exp._run is external

    def test_run_url_when_run_is_none(self) -> None:
        exp = WandbExperiment()
        assert exp.run_url is None


# ── Context manager protocol ──────────────────────────────────────────


class TestWandbExperimentContextManager:
    """Test __enter__ / __exit__ behavior (no-op path)."""

    def test_enter_returns_self(self) -> None:
        exp = WandbExperiment()
        result = exp.__enter__()
        assert result is exp
        # _run remains None because wandb is unavailable
        assert exp._run is None

    def test_enter_exit_no_exception(self) -> None:
        exp = WandbExperiment()
        exp.__enter__()
        # Should not raise regardless of _run state
        exp.__exit__(None, None, None)

    def test_enter_exit_with_external_run(self) -> None:
        external = SimpleNamespace()
        exp = WandbExperiment(run=external)
        exp.__enter__()
        # __exit__ should be a no-op for external runs
        exp.__exit__(None, None, None)
        assert exp._external_run is True
        assert exp._run is external

    def test_enter_exit_chained_no_crash(self) -> None:
        """Multiple enter/exit pairs should not crash."""
        for _ in range(10):
            with WandbExperiment() as exp:
                assert isinstance(exp, WandbExperiment)


# ── log_metrics ────────────────────────────────────────────────────────


class TestLogMetrics:
    """Test log_metrics in the no-op path."""

    def test_empty_dict(self) -> None:
        exp = WandbExperiment()
        exp.__enter__()
        exp.log_metrics({})  # no-op, must not raise

    def test_single_metric(self) -> None:
        exp = WandbExperiment()
        exp.__enter__()
        exp.log_metrics({"loss": 0.5})

    def test_multiple_metrics(self) -> None:
        exp = WandbExperiment()
        exp.__enter__()
        exp.log_metrics({"loss": 0.5, "accuracy": 0.92, "f1": 0.88})

    def test_with_explicit_step(self) -> None:
        exp = WandbExperiment()
        exp.__enter__()
        exp.log_metrics({"loss": 0.3}, step=5)

    def test_negative_step(self) -> None:
        exp = WandbExperiment()
        exp.__enter__()
        exp.log_metrics({"loss": 0.4}, step=-1)

    def test_zero_step(self) -> None:
        exp = WandbExperiment()
        exp.__enter__()
        exp.log_metrics({"loss": 0.5}, step=0)

    def test_large_step(self) -> None:
        exp = WandbExperiment()
        exp.__enter__()
        exp.log_metrics({"loss": 0.1}, step=999999)

    def test_step_type_as_string(self) -> None:
        """Type errors are not raised in no-op path."""
        exp = WandbExperiment()
        exp.__enter__()
        exp.log_metrics({"loss": 0.5}, step="invalid")  # no-op

    def test_metrics_is_none(self) -> None:
        """None metrics — no-op path ignores it."""
        exp = WandbExperiment()
        exp.__enter__()
        exp.log_metrics({})  # not None, but empty

    def test_float_values(self) -> None:
        exp = WandbExperiment()
        exp.__enter__()
        exp.log_metrics({"val": 3.14159})

    def test_integer_values(self) -> None:
        exp = WandbExperiment()
        exp.__enter__()
        exp.log_metrics({"val": 42})

    def test_negative_metric_value(self) -> None:
        exp = WandbExperiment()
        exp.__enter__()
        exp.log_metrics({"val": -1.0})

    def test_without_enter(self) -> None:
        """Calling log_metrics without __enter__ should be a no-op."""
        exp = WandbExperiment()
        exp.log_metrics({"loss": 0.5})  # _run is None -> no-op


# ── log_partition_plan ────────────────────────────────────────────────


class TestLogPartitionPlan:
    """Test log_partition_plan in the no-op path."""

    def test_none_plan(self) -> None:
        exp = WandbExperiment()
        exp.__enter__()
        exp.log_partition_plan(None)

    def test_empty_assignments_via_node_assignments(self) -> None:
        plan = SimpleNamespace(node_assignments=[], throughput_estimate=100.0)
        exp = WandbExperiment()
        exp.__enter__()
        exp.log_partition_plan(plan)

    def test_empty_assignments_via_assignments_fallback(self) -> None:
        plan = SimpleNamespace(assignments=[], throughput=50.0)
        exp = WandbExperiment()
        exp.__enter__()
        exp.log_partition_plan(plan)

    def test_dict_assignments(self) -> None:
        plan = SimpleNamespace(
            node_assignments=[
                {"node_id": "node-0", "start_layer": 0, "end_layer": 9,
                 "compute_ms": 12.3, "memory_gb": 4.5},
                {"node_id": "node-1", "start_layer": 10, "end_layer": 19,
                 "compute_ms": 15.7, "memory_gb": 6.2},
            ],
            throughput_estimate=142.5,
        )
        exp = WandbExperiment()
        exp.__enter__()
        exp.log_partition_plan(plan)

    def test_object_assignments(self) -> None:
        assignment_0 = SimpleNamespace(
            node_id="gpu-0", start_layer=0, end_layer=7,
            compute_ms=10.0, memory_gb=3.0,
        )
        assignment_1 = SimpleNamespace(
            node_id="gpu-1", start_layer=8, end_layer=15,
            compute_ms=20.0, memory_gb=5.0,
        )
        plan = SimpleNamespace(node_assignments=[assignment_0, assignment_1])
        exp = WandbExperiment()
        exp.__enter__()
        exp.log_partition_plan(plan)

    def test_mixed_dict_and_object_raises_in_real_path_but_noop_here(self) -> None:
        """The no-op path ignores type mismatches."""
        plan = SimpleNamespace(
            assignments=[
                {"node_id": "n0", "start_layer": 0},
                SimpleNamespace(node_id="n1", start_layer=10),
            ]
        )
        exp = WandbExperiment()
        exp.__enter__()
        exp.log_partition_plan(plan)

    def test_neither_assignments_nor_node_assignments(self) -> None:
        plan = SimpleNamespace(throughput_estimate=200.0)
        exp = WandbExperiment()
        exp.__enter__()
        exp.log_partition_plan(plan)

    def test_plan_with_partial_dict_keys(self) -> None:
        plan = SimpleNamespace(
            node_assignments=[
                {"node_id": "n0"},  # missing other keys
            ]
        )
        exp = WandbExperiment()
        exp.__enter__()
        exp.log_partition_plan(plan)

    def test_plan_with_throughput_fallback(self) -> None:
        plan = SimpleNamespace(
            node_assignments=[],
            throughput=300.0,  # fallback attr
        )
        exp = WandbExperiment()
        exp.__enter__()
        exp.log_partition_plan(plan)

    def test_without_enter(self) -> None:
        exp = WandbExperiment()
        exp.log_partition_plan(SimpleNamespace(node_assignments=[]))


# ── log_quant_results ─────────────────────────────────────────────────


class TestLogQuantResults:
    """Test log_quant_results in the no-op path."""

    def test_none_results(self) -> None:
        exp = WandbExperiment()
        exp.__enter__()
        exp.log_quant_results(None)

    def test_empty_list(self) -> None:
        exp = WandbExperiment()
        exp.__enter__()
        exp.log_quant_results([])

    def test_single_dict(self) -> None:
        exp = WandbExperiment()
        exp.__enter__()
        exp.log_quant_results({
            "layer_name": "mlp.0",
            "method": "gptq",
            "accuracy": 0.98,
            "speedup": 1.5,
        })

    def test_list_of_dicts(self) -> None:
        exp = WandbExperiment()
        exp.__enter__()
        exp.log_quant_results([
            {"layer_name": "attn.0", "method": "awq", "accuracy": 0.97, "speedup": 1.4},
            {"layer_name": "mlp.1", "method": "gptq", "accuracy": 0.96, "speedup": 1.6},
        ])

    def test_dict_with_layer_key(self) -> None:
        """The key 'layer' should also be accepted (fallback)."""
        exp = WandbExperiment()
        exp.__enter__()
        exp.log_quant_results({
            "layer": "embedding",
            "method": "fp8",
            "accuracy": 0.99,
            "speedup": 2.0,
        })

    def test_dict_with_perplexity_as_accuracy(self) -> None:
        """If 'accuracy' is missing, it falls back to 'perplexity'."""
        exp = WandbExperiment()
        exp.__enter__()
        exp.log_quant_results({
            "layer_name": "lm_head",
            "method": "int8",
            "perplexity": 5.2,
            "speedup": 1.2,
        })

    def test_dict_with_throughput_gain_as_speedup(self) -> None:
        exp = WandbExperiment()
        exp.__enter__()
        exp.log_quant_results({
            "layer_name": "norm",
            "method": "int4",
            "accuracy": 0.95,
            "throughput_gain": 1.8,
        })

    def test_partial_dicts(self) -> None:
        exp = WandbExperiment()
        exp.__enter__()
        exp.log_quant_results([
            {"layer_name": "a", "method": "fp16"},
            {"layer_name": "b"},  # no method
            {},  # completely empty
        ])

    def test_flat_dict_list_unpacking(self) -> None:
        """When a single dict is passed, it gets wrapped in a list."""
        exp = WandbExperiment()
        exp.__enter__()
        exp.log_quant_results({"layer_name": "x", "method": "y", "accuracy": 1.0, "speedup": 1.0})

    def test_no_accuracy_no_perplexity(self) -> None:
        """avg_accuracy should be skipped when no accuracy data."""
        exp = WandbExperiment()
        exp.__enter__()
        exp.log_quant_results([
            {"layer_name": "a", "method": "fp8"},
            {"layer_name": "b", "method": "int8"},
        ])

    def test_without_enter(self) -> None:
        exp = WandbExperiment()
        exp.log_quant_results({"layer_name": "x", "method": "y"})


# ── log_artifacts ─────────────────────────────────────────────────────


class TestLogArtifacts:
    """Test log_artifacts in the no-op path."""

    def test_non_existent_path(self) -> None:
        exp = WandbExperiment()
        exp.__enter__()
        exp.log_artifacts("/nonexistent/path/model.bin")

    def test_empty_string_path(self) -> None:
        exp = WandbExperiment()
        exp.__enter__()
        exp.log_artifacts("")

    def test_directory_path_string(self) -> None:
        exp = WandbExperiment()
        exp.__enter__()
        exp.log_artifacts("/some/directory/")

    def test_custom_artifact_type(self) -> None:
        exp = WandbExperiment()
        exp.__enter__()
        exp.log_artifacts("/tmp/weights.pt", artifact_type="weights")

    def test_without_enter(self) -> None:
        exp = WandbExperiment()
        exp.log_artifacts("path/to/model.bin")


# ── watch_model ────────────────────────────────────────────────────────


class TestWatchModel:
    """Test watch_model in the no-op path."""

    def test_none_model(self) -> None:
        exp = WandbExperiment()
        exp.__enter__()
        exp.watch_model(None)

    def test_simple_object(self) -> None:
        exp = WandbExperiment()
        exp.__enter__()
        exp.watch_model(SimpleNamespace())

    def test_custom_log_freq(self) -> None:
        exp = WandbExperiment()
        exp.__enter__()
        exp.watch_model(SimpleNamespace(), log_freq=50)

    def test_default_log_freq(self) -> None:
        exp = WandbExperiment()
        exp.__enter__()
        exp.watch_model(SimpleNamespace())  # default log_freq=100

    def test_zero_log_freq(self) -> None:
        exp = WandbExperiment()
        exp.__enter__()
        exp.watch_model(SimpleNamespace(), log_freq=0)

    def test_without_enter(self) -> None:
        exp = WandbExperiment()
        exp.watch_model(SimpleNamespace())


# ── run_url property ────────────────────────────────────────────────────


class TestRunUrl:
    """Test the run_url property in no-op / edge cases."""

    def test_returns_none_when_no_run(self) -> None:
        exp = WandbExperiment()
        assert exp.run_url is None

    def test_returns_none_when_run_has_no_get_url(self) -> None:
        exp = WandbExperiment(run=SimpleNamespace())
        assert exp.run_url is None

    def test_returns_none_when_get_url_raises(self) -> None:
        class BrokenRun:
            def get_url(self) -> str:
                raise RuntimeError("broken")
        exp = WandbExperiment(run=BrokenRun())
        assert exp.run_url is None

    def test_returns_url_from_external_run(self) -> None:
        class RunWithUrl:
            def get_url(self) -> str:
                return "https://wandb.ai/test/run/abc123"
        exp = WandbExperiment(run=RunWithUrl())
        assert exp.run_url == "https://wandb.ai/test/run/abc123"


# ── Integration / full flow ────────────────────────────────────────────


class TestFullWorkflow:
    """Simulate real usage patterns with the no-op path."""

    def test_quantization_tuning_workflow(self) -> None:
        with WandbExperiment(
            project="distllm-quant",
            config={"model": "llama-70b", "method": "gptq"},
            tags=["test"],
        ) as exp:
            exp.log_metrics({"total_layers": 80})
            exp.log_quant_results([
                {"layer_name": f"layer.{i}", "method": "gptq",
                 "accuracy": 0.95 + i * 0.001, "speedup": 1.5}
                for i in range(5)
            ])
            exp.log_metrics({"avg_accuracy": 0.96, "avg_speedup": 1.5})

    def test_partition_workflow(self) -> None:
        assignments = [
            {"node_id": "a", "start_layer": 0, "end_layer": 3,
             "compute_ms": 10.0, "memory_gb": 2.0},
            {"node_id": "b", "start_layer": 4, "end_layer": 7,
             "compute_ms": 12.0, "memory_gb": 2.5},
        ]
        plan = SimpleNamespace(
            node_assignments=assignments,
            throughput_estimate=95.3,
        )
        with WandbExperiment(
            project="distllm-partition",
            config={"model": "llama-70b", "num_layers": 8},
            run_name="partition-run",
            group="test-group",
        ) as exp:
            exp.log_partition_plan(plan)
            exp.log_metrics({"total_tflops": 95.3})

    def test_external_run_no_op_finish(self) -> None:
        """An external run passed to the constructor should not be finished
        on __exit__."""
        with WandbExperiment(
            run=SimpleNamespace(),
        ) as exp:
            assert exp._external_run is True
            exp.log_metrics({"test": 1.0})
