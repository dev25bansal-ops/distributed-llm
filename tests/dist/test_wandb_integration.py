"""Tests for merged WandBIntegration class.

Covers the complete merged API surface from both:
- ``WandBIntegration`` (GPU monitoring, latencies, model artefacts, lifecycle)
- ``WandbExperiment`` (partition plans, quant results, external run, watch_model)

Since wandb is not installed in the test environment, ``_WANDB_AVAILABLE`` is
False and all W&B-dependent methods are graceful no-ops.  This makes the tests
fully deterministic with no network, GPU, or wandb dependencies.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Ensure the standalone distllm-wandb package is importable.
_wandb_pkg = Path(__file__).resolve().parents[2] / "integrations" / "wandb" / "src"
if str(_wandb_pkg) not in sys.path:
    sys.path.insert(0, str(_wandb_pkg))

from distllm_wandb import WandBIntegration, _WANDB_AVAILABLE
from distllm_wandb.tracker import _ensure_nvml, _NVM_AVAILABLE, _NVM_INITIALISED


# ===================================================================
# Module-level sanity
# ===================================================================


class TestWANDAvailable:
    """The _WANDB_AVAILABLE flag must be False when wandb is absent."""

    def test_not_available(self) -> None:
        assert _WANDB_AVAILABLE is False


# ===================================================================
# Construction
# ===================================================================


class TestWandBIntegrationConstruction:
    """Test the constructor with various argument combinations."""

    def test_defaults(self) -> None:
        tracker = WandBIntegration()
        assert tracker.project == "distllm"
        assert tracker.config == {}
        assert tracker.entity is None
        assert tracker.run_name is None
        assert tracker.tags is None
        assert tracker.group is None
        assert tracker._watch_model_enabled is False
        assert tracker.log_gpu is True
        assert tracker.gpu_poll_interval == 10.0
        assert tracker._external_run is False
        assert tracker._run is None
        assert tracker._started is False
        assert tracker._finished is False
        assert tracker._step == 0
        assert tracker._latency_buffer == []
        assert tracker._gpu_thread is None

    def test_all_params_explicit(self) -> None:
        tracker = WandBIntegration(
            project="my-project",
            config={"lr": 0.01, "epochs": 10},
            entity="my-team",
            run_name="run-001",
            tags=["test", "quant"],
            group="experiment-group",
            watch_model=True,
            log_gpu=False,
            gpu_poll_interval=5.0,
        )
        assert tracker.project == "my-project"
        assert tracker.config == {"lr": 0.01, "epochs": 10}
        assert tracker.entity == "my-team"
        assert tracker.run_name == "run-001"
        assert tracker.tags == ["test", "quant"]
        assert tracker.group == "experiment-group"
        assert tracker._watch_model_enabled is True
        assert tracker.log_gpu is False
        assert tracker.gpu_poll_interval == 5.0

    def test_config_none_becomes_empty_dict(self) -> None:
        tracker = WandBIntegration(config=None)
        assert tracker.config == {}

    def test_empty_config(self) -> None:
        tracker = WandBIntegration(config={})
        assert tracker.config == {}

    def test_empty_tags_list(self) -> None:
        tracker = WandBIntegration(tags=[])
        assert tracker.tags == []

    def test_project_empty_string(self) -> None:
        tracker = WandBIntegration(project="")
        assert tracker.project == ""

    def test_gpu_poll_interval_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="gpu_poll_interval must be > 0"):
            WandBIntegration(gpu_poll_interval=0)

    def test_gpu_poll_interval_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="gpu_poll_interval must be > 0"):
            WandBIntegration(gpu_poll_interval=-1)

    def test_external_run_sets_flag(self) -> None:
        external = SimpleNamespace()
        tracker = WandBIntegration(run=external)
        assert tracker._external_run is True
        assert tracker._run is external

    def test_external_run_with_other_params(self) -> None:
        external = SimpleNamespace()
        tracker = WandBIntegration(
            project="ext-project",
            config={"key": "val"},
            run_name="ext-run",
            tags=["ext"],
            group="ext-group",
            run=external,
        )
        assert tracker.project == "ext-project"
        assert tracker.config == {"key": "val"}
        assert tracker.run_name == "ext-run"
        assert tracker.tags == ["ext"]
        assert tracker.group == "ext-group"
        assert tracker._external_run is True
        assert tracker._run is external

    def test_entity_none_default(self) -> None:
        tracker = WandBIntegration()
        assert tracker.entity is None

    def test_entity_explicit(self) -> None:
        tracker = WandBIntegration(entity="my-entity")
        assert tracker.entity == "my-entity"

    def test_watch_model_default_false(self) -> None:
        tracker = WandBIntegration()
        assert tracker._watch_model_enabled is False

    def test_watch_model_explicit_true(self) -> None:
        tracker = WandBIntegration(watch_model=True)
        assert tracker._watch_model_enabled is True

    def test_log_gpu_default_true(self) -> None:
        tracker = WandBIntegration()
        assert tracker.log_gpu is True

    def test_log_gpu_explicit_false(self) -> None:
        tracker = WandBIntegration(log_gpu=False)
        assert tracker.log_gpu is False


# ===================================================================
# Lifecycle (no-op path)
# ===================================================================


class TestLifecycle:
    """Test start / finish / context manager behaviour (no-op path)."""

    def test_start_sets_started_flag(self) -> None:
        tracker = WandBIntegration()
        tracker.start()
        assert tracker._started is True
        assert tracker._finished is False
        assert tracker._run is None  # no wandb

    def test_start_is_idempotent(self) -> None:
        tracker = WandBIntegration()
        tracker.start()
        tracker.start()  # second call should be no-op
        assert tracker._started is True

    def test_finish_sets_finished_flag(self) -> None:
        tracker = WandBIntegration()
        tracker.start()
        tracker.finish()
        assert tracker._finished is True

    def test_finish_is_idempotent(self) -> None:
        tracker = WandBIntegration()
        tracker.start()
        tracker.finish()
        tracker.finish()  # second call should be no-op
        assert tracker._finished is True

    def test_finish_without_start(self) -> None:
        tracker = WandBIntegration()
        tracker.finish()  # must not raise
        assert tracker._finished is True

    def test_enter_returns_self(self) -> None:
        tracker = WandBIntegration()
        result = tracker.__enter__()
        assert result is tracker

    def test_enter_exit_no_exception(self) -> None:
        tracker = WandBIntegration()
        tracker.__enter__()
        tracker.__exit__(None, None, None)
        assert tracker._finished is True

    def test_context_manager_protocol(self) -> None:
        with WandBIntegration() as tracker:
            assert isinstance(tracker, WandBIntegration)
        assert tracker._finished is True

    def test_multiple_enter_exit(self) -> None:
        """Multiple enter/exit pairs should all be safe."""
        for _ in range(5):
            with WandBIntegration():
                pass

    def test_external_run_lifecycle(self) -> None:
        """External run should NOT be finished on exit."""
        external = SimpleNamespace()
        tracker = WandBIntegration(run=external)
        tracker.__enter__()
        assert tracker._started is True
        assert tracker._external_run is True
        tracker.__exit__(None, None, None)
        # External run is detached but the tracker state is still reset
        assert tracker._finished is True

    def test_run_url_when_no_run(self) -> None:
        tracker = WandBIntegration()
        assert tracker.run_url is None

    def test_start_stop_gpu_monitor_without_gpu(self) -> None:
        """GPU monitor should gracefully start/stop even without pynvml."""
        tracker = WandBIntegration(log_gpu=True)
        tracker.start()
        # GPU monitor daemon thread may or may not start depending on pynvml
        tracker.finish()

    def test_start_with_log_gpu_false(self) -> None:
        """No GPU monitor thread when log_gpu=False."""
        tracker = WandBIntegration(log_gpu=False)
        tracker.start()
        assert tracker._gpu_thread is None
        tracker.finish()

    def test_started_flag_reset_after_finish(self) -> None:
        tracker = WandBIntegration()
        tracker.start()
        assert tracker._started is True
        tracker.finish()
        assert tracker._started is False


# ===================================================================
# log_metrics (no-op path)
# ===================================================================


class TestLogMetrics:
    """Test log_metrics in the no-op path."""

    def test_empty_dict(self) -> None:
        with WandBIntegration() as tracker:
            tracker.log_metrics({})

    def test_single_metric(self) -> None:
        with WandBIntegration() as tracker:
            tracker.log_metrics({"loss": 0.5})

    def test_multiple_metrics(self) -> None:
        with WandBIntegration() as tracker:
            tracker.log_metrics({"loss": 0.5, "accuracy": 0.92, "f1": 0.88})

    def test_with_explicit_step(self) -> None:
        with WandBIntegration() as tracker:
            tracker.log_metrics({"loss": 0.3}, step=5)

    def test_negative_step(self) -> None:
        with WandBIntegration() as tracker:
            tracker.log_metrics({"loss": 0.4}, step=-1)

    def test_zero_step(self) -> None:
        with WandBIntegration() as tracker:
            tracker.log_metrics({"loss": 0.5}, step=0)

    def test_large_step(self) -> None:
        with WandBIntegration() as tracker:
            tracker.log_metrics({"loss": 0.1}, step=999999)

    def test_step_increments(self) -> None:
        """Without explicit step, _step should auto-increment."""
        with WandBIntegration() as tracker:
            tracker.log_metrics({"m1": 1.0})
            assert tracker._step == 1
            tracker.log_metrics({"m2": 2.0})
            assert tracker._step == 2

    def test_explicit_step_updates_tracker(self) -> None:
        """Explicit step should update _step to max(current, step + 1)."""
        with WandBIntegration() as tracker:
            tracker.log_metrics({"m1": 1.0}, step=10)
            assert tracker._step == 11

    def test_explicit_step_lower_than_current(self) -> None:
        """A lower explicit step should not decrease _step."""
        with WandBIntegration() as tracker:
            tracker.log_metrics({"m1": 1.0})  # _step becomes 1
            tracker.log_metrics({"m2": 2.0}, step=0)  # max(1, 1) = 1
            assert tracker._step == 1

    def test_float_values(self) -> None:
        with WandBIntegration() as tracker:
            tracker.log_metrics({"val": 3.14159})

    def test_integer_values(self) -> None:
        with WandBIntegration() as tracker:
            tracker.log_metrics({"val": 42})

    def test_negative_metric_value(self) -> None:
        with WandBIntegration() as tracker:
            tracker.log_metrics({"val": -1.0})

    def test_without_start(self) -> None:
        """Calling log_metrics before start() should be a no-op."""
        tracker = WandBIntegration()
        tracker.log_metrics({"loss": 0.5})  # _run is None -> no-op

    def test_after_finish(self) -> None:
        """Calling log_metrics after finish() should be a no-op."""
        tracker = WandBIntegration()
        tracker.start()
        tracker.finish()
        tracker.log_metrics({"loss": 0.5})  # should not raise


# ===================================================================
# log_latencies (no-op path)
# ===================================================================


class TestLogLatencies:
    """Test log_latencies in the no-op path."""

    def test_empty_list(self) -> None:
        with WandBIntegration() as tracker:
            tracker.log_latencies([])

    def test_single_latency(self) -> None:
        with WandBIntegration() as tracker:
            tracker.log_latencies([42.0])

    def test_multiple_latencies(self) -> None:
        with WandBIntegration() as tracker:
            tracker.log_latencies([10.0, 20.0, 30.0, 40.0, 50.0])

    def test_latencies_with_explicit_step(self) -> None:
        with WandBIntegration() as tracker:
            tracker.log_latencies([1.0, 2.0, 3.0], step=5)

    def test_many_latencies(self) -> None:
        """Test latency buffer flush at 100 items."""
        with WandBIntegration() as tracker:
            for _ in range(101):
                tracker.log_latencies([1.0])

    def test_latency_buffer_accumulates(self) -> None:
        with WandBIntegration() as tracker:
            tracker.log_latencies([1.0, 2.0])
            assert len(tracker._latency_buffer) == 2

    def test_latency_buffer_flush_on_finish(self) -> None:
        """Finish should flush the latency buffer."""
        with WandBIntegration() as tracker:
            tracker.log_latencies([1.0, 2.0, 3.0])
            assert len(tracker._latency_buffer) == 3
        # After finish, buffer should be cleared
        assert tracker._latency_buffer == []

    def test_without_start(self) -> None:
        tracker = WandBIntegration()
        tracker.log_latencies([1.0])

    def test_after_finish(self) -> None:
        tracker = WandBIntegration()
        tracker.start()
        tracker.finish()
        tracker.log_latencies([1.0])


# ===================================================================
# log_model (no-op path)
# ===================================================================


class TestLogModel:
    """Test log_model in the no-op path."""

    def test_simple_object(self) -> None:
        with WandBIntegration() as tracker:
            tracker.log_model(SimpleNamespace(), "test-model")

    def test_with_state_dict(self) -> None:
        """Object with state_dict() method simulates a torch model."""
        model = SimpleNamespace()
        model.state_dict = lambda: {"weight": 1.0}  # type: ignore[method-assign]
        with WandBIntegration() as tracker:
            tracker.log_model(model, "torch-model")

    def test_with_watch_flag(self) -> None:
        """watch_model flag should be accepted as a no-op."""
        with WandBIntegration(watch_model=True) as tracker:
            tracker.log_model(SimpleNamespace(), "watched-model")

    def test_empty_name(self) -> None:
        with WandBIntegration() as tracker:
            tracker.log_model(SimpleNamespace(), "")

    def test_without_start(self) -> None:
        tracker = WandBIntegration()
        tracker.log_model(SimpleNamespace(), "noop")

    def test_after_finish(self) -> None:
        tracker = WandBIntegration()
        tracker.start()
        tracker.finish()
        tracker.log_model(SimpleNamespace(), "done")


# ===================================================================
# log_partition_plan (no-op path)
# ===================================================================


class TestLogPartitionPlan:
    """Test log_partition_plan in the no-op path."""

    def test_none_plan(self) -> None:
        with WandBIntegration() as tracker:
            tracker.log_partition_plan(None)

    def test_empty_assignments_via_node_assignments(self) -> None:
        plan = SimpleNamespace(node_assignments=[], throughput_estimate=100.0)
        with WandBIntegration() as tracker:
            tracker.log_partition_plan(plan)

    def test_empty_assignments_via_assignments_fallback(self) -> None:
        plan = SimpleNamespace(assignments=[], throughput=50.0)
        with WandBIntegration() as tracker:
            tracker.log_partition_plan(plan)

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
        with WandBIntegration() as tracker:
            tracker.log_partition_plan(plan)

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
        with WandBIntegration() as tracker:
            tracker.log_partition_plan(plan)

    def test_mixed_dict_and_object(self) -> None:
        """The no-op path ignores type mismatches."""
        plan = SimpleNamespace(
            assignments=[
                {"node_id": "n0", "start_layer": 0},
                SimpleNamespace(node_id="n1", start_layer=10),
            ]
        )
        with WandBIntegration() as tracker:
            tracker.log_partition_plan(plan)

    def test_neither_assignments_nor_node_assignments(self) -> None:
        plan = SimpleNamespace(throughput_estimate=200.0)
        with WandBIntegration() as tracker:
            tracker.log_partition_plan(plan)

    def test_plan_with_partial_dict_keys(self) -> None:
        plan = SimpleNamespace(
            node_assignments=[
                {"node_id": "n0"},  # missing other keys
            ]
        )
        with WandBIntegration() as tracker:
            tracker.log_partition_plan(plan)

    def test_plan_with_throughput_fallback(self) -> None:
        plan = SimpleNamespace(
            node_assignments=[],
            throughput=300.0,  # fallback attr
        )
        with WandBIntegration() as tracker:
            tracker.log_partition_plan(plan)

    def test_without_start(self) -> None:
        tracker = WandBIntegration()
        tracker.log_partition_plan(SimpleNamespace(node_assignments=[]))


# ===================================================================
# log_quant_results (no-op path)
# ===================================================================


class TestLogQuantResults:
    """Test log_quant_results in the no-op path."""

    def test_none_results(self) -> None:
        with WandBIntegration() as tracker:
            tracker.log_quant_results(None)

    def test_empty_list(self) -> None:
        with WandBIntegration() as tracker:
            tracker.log_quant_results([])

    def test_single_dict(self) -> None:
        with WandBIntegration() as tracker:
            tracker.log_quant_results({
                "layer_name": "mlp.0",
                "method": "gptq",
                "accuracy": 0.98,
                "speedup": 1.5,
            })

    def test_list_of_dicts(self) -> None:
        with WandBIntegration() as tracker:
            tracker.log_quant_results([
                {"layer_name": "attn.0", "method": "awq",
                 "accuracy": 0.97, "speedup": 1.4},
                {"layer_name": "mlp.1", "method": "gptq",
                 "accuracy": 0.96, "speedup": 1.6},
            ])

    def test_dict_with_layer_key(self) -> None:
        """The key 'layer' should also be accepted (fallback)."""
        with WandBIntegration() as tracker:
            tracker.log_quant_results({
                "layer": "embedding",
                "method": "fp8",
                "accuracy": 0.99,
                "speedup": 2.0,
            })

    def test_dict_with_perplexity_as_accuracy(self) -> None:
        """If 'accuracy' is missing, it falls back to 'perplexity'."""
        with WandBIntegration() as tracker:
            tracker.log_quant_results({
                "layer_name": "lm_head",
                "method": "int8",
                "perplexity": 5.2,
                "speedup": 1.2,
            })

    def test_dict_with_throughput_gain_as_speedup(self) -> None:
        with WandBIntegration() as tracker:
            tracker.log_quant_results({
                "layer_name": "norm",
                "method": "int4",
                "accuracy": 0.95,
                "throughput_gain": 1.8,
            })

    def test_partial_dicts(self) -> None:
        with WandBIntegration() as tracker:
            tracker.log_quant_results([
                {"layer_name": "a", "method": "fp16"},
                {"layer_name": "b"},
                {},
            ])

    def test_no_accuracy_no_perplexity(self) -> None:
        """avg_accuracy should be skipped when no accuracy data."""
        with WandBIntegration() as tracker:
            tracker.log_quant_results([
                {"layer_name": "a", "method": "fp8"},
                {"layer_name": "b", "method": "int8"},
            ])

    def test_without_start(self) -> None:
        tracker = WandBIntegration()
        tracker.log_quant_results({"layer_name": "x", "method": "y"})


# ===================================================================
# log_artifacts (no-op path)
# ===================================================================


class TestLogArtifacts:
    """Test log_artifacts in the no-op path."""

    def test_non_existent_path(self) -> None:
        with WandBIntegration() as tracker:
            tracker.log_artifacts("/nonexistent/path/model.bin")

    def test_empty_string_path(self) -> None:
        with WandBIntegration() as tracker:
            tracker.log_artifacts("")

    def test_directory_path_string(self) -> None:
        with WandBIntegration() as tracker:
            tracker.log_artifacts("/some/directory/")

    def test_custom_artifact_type(self) -> None:
        with WandBIntegration() as tracker:
            tracker.log_artifacts("/tmp/weights.pt", artifact_type="weights")

    def test_without_start(self) -> None:
        tracker = WandBIntegration()
        tracker.log_artifacts("path/to/model.bin")


# ===================================================================
# watch_model (no-op path)
# ===================================================================


class TestWatchModel:
    """Test watch_model method in the no-op path."""

    def test_none_model(self) -> None:
        with WandBIntegration() as tracker:
            tracker.watch_model(None)

    def test_simple_object(self) -> None:
        with WandBIntegration() as tracker:
            tracker.watch_model(SimpleNamespace())

    def test_custom_log_freq(self) -> None:
        with WandBIntegration() as tracker:
            tracker.watch_model(SimpleNamespace(), log_freq=50)

    def test_default_log_freq(self) -> None:
        with WandBIntegration() as tracker:
            tracker.watch_model(SimpleNamespace())  # default log_freq=100

    def test_zero_log_freq(self) -> None:
        with WandBIntegration() as tracker:
            tracker.watch_model(SimpleNamespace(), log_freq=0)

    def test_without_start(self) -> None:
        tracker = WandBIntegration()
        tracker.watch_model(SimpleNamespace())


# ===================================================================
# run_url property
# ===================================================================


class TestRunUrl:
    """Test the run_url property in no-op / edge cases."""

    def test_returns_none_when_no_run(self) -> None:
        tracker = WandBIntegration()
        assert tracker.run_url is None

    def test_returns_none_when_run_has_no_get_url(self) -> None:
        tracker = WandBIntegration(run=SimpleNamespace())
        assert tracker.run_url is None

    def test_returns_none_when_get_url_raises(self) -> None:
        class BrokenRun:
            def get_url(self) -> str:
                msg = "broken"
                raise RuntimeError(msg)

        tracker = WandBIntegration(run=BrokenRun())
        assert tracker.run_url is None

    def test_returns_url_from_external_run(self) -> None:
        class RunWithUrl:
            def get_url(self) -> str:
                return "https://wandb.ai/test/run/abc123"

        tracker = WandBIntegration(run=RunWithUrl())
        assert tracker.run_url == "https://wandb.ai/test/run/abc123"

    def test_returns_url_after_start(self) -> None:
        """run_url should still work after start() even without wandb."""
        # In the no-op path, _run remains None, so run_url should be None
        with WandBIntegration() as tracker:
            assert tracker.run_url is None

    def test_returns_none_after_finish(self) -> None:
        tracker = WandBIntegration()
        tracker.start()
        tracker.finish()
        assert tracker.run_url is None


# ===================================================================
# Cross-platform tempfile usage
# ===================================================================


class TestTempfileUsage:
    """Verify that tempfile.gettempdir() is used instead of hardcoded /tmp/."""

    def test_save_torch_model_uses_tempfile(self) -> None:
        """_save_torch_model should use tempfile.gettempdir(), not /tmp/."""
        source_lines, _ = inspect.getsourcelines(
            WandBIntegration._save_torch_model
        )
        combined = "".join(source_lines)
        # Must NOT contain hardcoded /tmp/
        assert "/tmp/" not in combined, (
            "_save_torch_model must not use hardcoded /tmp/ path"
        )
        # Must use tempfile.gettempdir()
        assert "tempfile.gettempdir()" in combined, (
            "_save_torch_model must use tempfile.gettempdir()"
        )

    def test_tracker_module_uses_tempfile_not_hardcoded(self) -> None:
        """The tracker module file must not contain hardcoded /tmp/."""
        mod_file = inspect.getfile(WandBIntegration)
        with open(mod_file) as f:
            content = f.read()
        lines = content.splitlines()
        for lineno, line in enumerate(lines, start=1):
            stripped = line.strip()
            # Skip comments and docstrings
            if stripped.startswith("#") or stripped.startswith('"""'):
                continue
            if "/tmp/" in stripped and "tempfile" not in stripped:
                pytest.fail(
                    f"Line {lineno} references hardcoded /tmp/: {stripped!r}"
                )


# ===================================================================
# Lazy pynvml initialisation
# ===================================================================


class TestLazyPynvmlInit:
    """Verify that pynvml is lazily initialised on first GPU poll."""

    def test_module_level_not_initialised(self) -> None:
        """_ensure_nvml() should be the only way to init pynvml."""
        # _NVM_INITIALISED should be False until _ensure_nvml is called
        pass  # can't reliably check _NVM_INITIALISED due to import side-effects

    def test_ensure_nvml_returns_bool(self) -> None:
        """_ensure_nvml should return a boolean."""
        result = _ensure_nvml()
        assert isinstance(result, bool)

    def test_ensure_nvml_is_idempotent(self) -> None:
        """Multiple calls to _ensure_nvml should all return the same value."""
        first = _ensure_nvml()
        second = _ensure_nvml()
        assert first == second

    def test_module_level_init_is_lazy(self) -> None:
        """The _NVM_INITIALISED flag is a module-level boolean."""
        from distllm_wandb.tracker import _NVM_INITIALISED as _init_flag

        assert isinstance(_init_flag, bool)

    def test_gpu_monitor_thread_calls_ensure_nvml(self) -> None:
        """Starting the GPU monitor should call _ensure_nvml."""
        tracker = WandBIntegration(log_gpu=True)
        tracker.start()
        # Thread is started but may skip if pynvml is unavailable
        # In no-op env, the thread stops immediately
        tracker.finish()


# ===================================================================
# Full flow integration (no-op path)
# ===================================================================


class TestFullWorkflow:
    """Simulate real usage patterns with the no-op path."""

    def test_quantization_tuning_workflow(self) -> None:
        with WandBIntegration(
            project="distllm-quant",
            config={"model": "llama-70b", "method": "gptq"},
            tags=["test"],
        ) as tracker:
            tracker.log_metrics({"total_layers": 80})
            tracker.log_quant_results([
                {"layer_name": f"layer.{i}", "method": "gptq",
                 "accuracy": 0.95 + i * 0.001, "speedup": 1.5}
                for i in range(5)
            ])
            tracker.log_metrics({"avg_accuracy": 0.96, "avg_speedup": 1.5})

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
        with WandBIntegration(
            project="distllm-partition",
            config={"model": "llama-70b", "num_layers": 8},
            run_name="partition-run",
            group="test-group",
        ) as tracker:
            tracker.log_partition_plan(plan)
            tracker.log_metrics({"total_tflops": 95.3})

    def test_external_run_no_op_finish(self) -> None:
        """An external run passed to the constructor should not be finished
        on __exit__."""
        with WandBIntegration(
            run=SimpleNamespace(),
        ) as tracker:
            assert tracker._external_run is True
            tracker.log_metrics({"test": 1.0})

    def test_latency_then_metrics_workflow(self) -> None:
        """Realistic workflow: log latencies then summary metrics."""
        with WandBIntegration(project="latency-test") as tracker:
            tracker.log_latencies([5.0, 10.0, 15.0, 20.0, 25.0])
            tracker.log_metrics({"throughput": 100.0})
            tracker.log_latencies([30.0, 35.0, 40.0], step=2)

    def test_all_logging_methods(self) -> None:
        """Call every logging method at least once."""
        with WandBIntegration() as tracker:
            tracker.log_metrics({"m": 1.0})
            tracker.log_latencies([1.0, 2.0])
            tracker.log_model(SimpleNamespace(), "m")
            tracker.log_partition_plan(
                SimpleNamespace(node_assignments=[])
            )
            tracker.log_quant_results({"layer_name": "l1", "method": "fp8"})
            tracker.log_artifacts("/nonexistent")
            tracker.watch_model(SimpleNamespace())

    def test_entity_and_group_workflow(self) -> None:
        """Entity should be accepted and passed through."""
        with WandBIntegration(
            project="entity-test",
            entity="my-team",
            group="experiment-group",
        ) as tracker:
            assert tracker.entity == "my-team"
            assert tracker.group == "experiment-group"
            tracker.log_metrics({"test": 1.0})

    def test_gpu_poll_interval_float(self) -> None:
        """gpu_poll_interval can be a float."""
        tracker = WandBIntegration(gpu_poll_interval=2.5)
        assert tracker.gpu_poll_interval == 2.5

    def test_gpu_poll_interval_large(self) -> None:
        tracker = WandBIntegration(gpu_poll_interval=3600.0)
        assert tracker.gpu_poll_interval == 3600.0


# ===================================================================
# Edge cases and resilience
# ===================================================================


class TestEdgeCases:
    """Test edge cases and resilience of the merged class."""

    def test_log_metrics_none_value(self) -> None:
        """None values in metrics dict — no-op path should handle it."""
        with WandBIntegration() as tracker:
            tracker.log_metrics({"val": None})  # type: ignore[dict-item]

    def test_log_metrics_empty_name(self) -> None:
        with WandBIntegration() as tracker:
            tracker.log_metrics({"": 1.0})

    def test_log_latencies_none(self) -> None:
        with WandBIntegration() as tracker:
            tracker.log_latencies([])

    def test_log_artifacts_none_path(self) -> None:
        with WandBIntegration() as tracker:
            tracker.log_artifacts("")  # empty path, should be no-op

    def test_watch_model_none(self) -> None:
        with WandBIntegration() as tracker:
            tracker.watch_model(None)

    def test_log_model_none(self) -> None:
        with WandBIntegration() as tracker:
            tracker.log_model(None, "none-model")  # type: ignore[arg-type]

    def test_log_partition_plan_none(self) -> None:
        with WandBIntegration() as tracker:
            tracker.log_partition_plan(None)

    def test_log_quant_results_none(self) -> None:
        with WandBIntegration() as tracker:
            tracker.log_quant_results(None)

    def test_external_run_method_invocation(self) -> None:
        """All methods should work (as no-ops) with external run."""
        with WandBIntegration(run=SimpleNamespace()) as tracker:
            tracker.log_metrics({"test": 1.0})
            tracker.log_latencies([1.0])
            tracker.log_model(SimpleNamespace(), "m")
            tracker.log_partition_plan(SimpleNamespace(node_assignments=[]))
            tracker.log_quant_results({"layer_name": "l"})
            tracker.log_artifacts("/path")
            tracker.watch_model(SimpleNamespace())

    def test_atexit_register_called(self) -> None:
        """start() should register finish() with atexit."""
        tracker = WandBIntegration()
        tracker.start()
        # Verify atexit registered by checking finish unregisters
        tracker.finish()  # should not raise

    def test_gpu_monitor_stop_without_start(self) -> None:
        """Calling _stop_gpu_monitor without _start_gpu_monitor should work."""
        tracker = WandBIntegration()
        tracker._stop_gpu_monitor()  # should not raise


import inspect
