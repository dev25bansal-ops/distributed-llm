"""Tests for the optional MLflow integration.

Covers graceful no-op behaviour when mlflow is not installed, context
manager lifecycle, failure handling in _RunContext, model dispatch
logic, and get_runs output.
"""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest

from distllm.integrations.mlflow_tracking import MLflowIntegration


# ---------------------------------------------------------------------------
# Graceful noop when mlflow is not available
# ---------------------------------------------------------------------------


@contextmanager
def _force_noop() -> Generator[None, None, None]:
    """Temporarily force _MLFLOW_AVAILABLE to False so we can test noop paths
    even when mlflow is installed on the host system."""
    import distllm.integrations.mlflow_tracking as mt

    orig = mt._MLFLOW_AVAILABLE
    mt._MLFLOW_AVAILABLE = False
    try:
        yield
    finally:
        mt._MLFLOW_AVAILABLE = orig


def test_noop_constructor_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    """MLflowIntegration logs a warning on construction when mlflow missing."""
    with _force_noop():
        with caplog.at_level(logging.INFO, logger="distllm.mlflow"):
            tracker = MLflowIntegration()
            assert "mlflow is not installed" in caplog.text
        # All public methods are no-ops.
        with tracker.log_run("test") as run:
            assert run.info is None
            tracker.log_model(object(), "model")
            tracker.log_artifact("/fake/path")
            tracker.log_artifacts_from_dict({"a": 1})
            tracker.set_tag("key", "val")
        assert tracker.get_runs() == []


def test_noop_returns_empty_list() -> None:
    """get_runs returns empty list when mlflow unavailable."""
    with _force_noop():
        tracker = MLflowIntegration()
        assert tracker.get_runs("anything") == []
        assert tracker.get_runs() == []


# ---------------------------------------------------------------------------
# Mock-based tests for real-path behaviour
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_mlflow() -> Generator[dict[str, MagicMock], None, None]:
    """Replace the mlflow module with mocks so we can test real-path logic.

    Also stubs *torch* and *transformers* in ``sys.modules`` so that
    ``log_model`` dispatch checks do not trigger real (heavy) imports.
    """
    import distllm.integrations.mlflow_tracking as mt

    orig_flag = mt._MLFLOW_AVAILABLE

    mock_module = MagicMock()
    mock_module.exceptions.MlflowException = Exception  # re-use built-in
    mock_module.pyfunc = MagicMock()
    mock_module.pytorch = MagicMock()
    mock_module.transformers = MagicMock()
    mock_module.sklearn = MagicMock()

    mock_experiment = MagicMock()
    mock_experiment.experiment_id = "exp-123"

    mock_run = MagicMock()
    mock_run.info.run_id = "run-abc"
    mock_run.info.experiment_id = "exp-123"
    mock_run.info.status = "FINISHED"
    mock_run.info.start_time = 1000
    mock_run.data.params = {"batch_size": "32"}
    mock_run.data.metrics = {"throughput": "123.4"}
    mock_run.data.tags = {"env": "test"}

    def set_experiment_side_effect(name: str) -> MagicMock:
        return mock_experiment

    def search_runs_side_effect(**kwargs: Any) -> list[MagicMock]:
        return [mock_run]

    mock_module.set_experiment.side_effect = set_experiment_side_effect
    mock_module.start_run.return_value = mock_run
    mock_module.search_runs.side_effect = search_runs_side_effect

    # Stub mlflow.exceptions so that ``from mlflow.exceptions import
    # MlflowException`` inside the reloaded module uses our mock Exception
    # instead of loading the real mlflow.exceptions module from disk.
    mock_exceptions = MagicMock()
    mock_exceptions.MlflowException = Exception
    mock_module.exceptions = mock_exceptions

    # Stub torch and transformers so log_model dispatch does not trigger
    # real (heavy) imports.  Ensure torch.nn.Module is a real type so that
    # isinstance() checks work correctly.
    mock_torch = MagicMock()
    mock_torch_nn = MagicMock()
    mock_torch_nn.Module = type("Module", (), {})
    mock_torch.nn = mock_torch_nn  # wire .nn attribute to the stubbed sub-module
    mock_transformers = MagicMock()
    mock_transformers.PreTrainedModel = type("PreTrainedModel", (), {})

    stubs: dict[str, MagicMock] = {
        "mlflow": mock_module,
        "mlflow.exceptions": mock_exceptions,
        "torch": mock_torch,
        "torch.nn": mock_torch_nn,
        "transformers": mock_transformers,
    }

    patcher = patch.dict(sys.modules, stubs, clear=False)
    patcher.start()

    # Reload the module so the mock is picked up by the _MLFLOW_AVAILABLE check
    import importlib

    importlib.reload(mt)

    yield {
        "mlflow": mock_module,
        "experiment": mock_experiment,
        "run": mock_run,
    }

    patcher.stop()
    mt._MLFLOW_AVAILABLE = orig_flag


# ---------------------------------------------------------------------------
# log_run context manager lifecycle
# ---------------------------------------------------------------------------


def test_log_run_creates_experiment_and_run(
    mock_mlflow: dict[str, MagicMock],
) -> None:
    """log_run calls set_experiment, start_run, and end_run."""
    tracker = MLflowIntegration()
    mock = mock_mlflow["mlflow"]

    with tracker.log_run("my-exp", params={"lr": 0.01}, metrics={"acc": 0.95}) as run:
        assert run is not None
        assert run.info.run_id == "run-abc"

    mock.set_experiment.assert_called_once_with("my-exp")
    mock.start_run.assert_called_once()
    mock.log_params.assert_called_once_with({"lr": 0.01})
    mock.log_metrics.assert_called_once_with({"acc": 0.95})
    mock.end_run.assert_called_once()


def test_log_run_creates_experiment_on_missing(
    mock_mlflow: dict[str, MagicMock],
) -> None:
    """log_run creates the experiment when set_experiment raises MlflowException."""
    import distllm.integrations.mlflow_tracking as mt

    tracker = MLflowIntegration()
    mock = mock_mlflow["mlflow"]

    # Use a callable side_effect that raises MlflowException on the first
    # call, then returns the expected experiment on subsequent calls.
    # Access MlflowException from the reloaded module so it matches what
    # the except clause is checking against.
    MlflowException = mt.MlflowException
    call_count: list[int] = [0]

    def set_experiment_fn(name: str) -> MagicMock:
        call_count[0] += 1
        if call_count[0] == 1:
            raise MlflowException("not found")
        return mock_mlflow["experiment"]

    mock.set_experiment.side_effect = set_experiment_fn

    with tracker.log_run("new-exp"):
        pass

    mock.create_experiment.assert_called_once_with("new-exp")
    # set_experiment called twice: first raises, second after create
    assert mock.set_experiment.call_count == 2


# ---------------------------------------------------------------------------
# _RunContext exit handling
# ---------------------------------------------------------------------------


def test_run_context_failure_status(
    mock_mlflow: dict[str, MagicMock],
) -> None:
    """__exit__ passes status='FAILED' when an exception propagates."""
    tracker = MLflowIntegration()
    mock = mock_mlflow["mlflow"]

    try:
        with tracker.log_run("failing-exp"):
            msg = "something went wrong"
            raise RuntimeError(msg)
    except RuntimeError:
        pass

    mock.end_run.assert_called_once_with(status="FAILED")


def test_run_context_success_no_failed_status(
    mock_mlflow: dict[str, MagicMock],
) -> None:
    """__exit__ calls end_run() without arguments on success."""
    tracker = MLflowIntegration()
    mock = mock_mlflow["mlflow"]

    with tracker.log_run("ok-exp"):
        pass

    mock.end_run.assert_called_once_with()


# ---------------------------------------------------------------------------
# log_model dispatch
# ---------------------------------------------------------------------------


def test_log_model_pytorch_module(
    mock_mlflow: dict[str, MagicMock],
) -> None:
    """log_model dispatches to mlflow.pytorch.log_model for torch.nn.Module."""
    tracker = MLflowIntegration()
    mock = mock_mlflow["mlflow"]

    # The fixture stubs torch with torch.nn.Module as a real type.
    import torch

    model = torch.nn.Module()
    tracker.log_model(model, "my-model")

    mock.pytorch.log_model.assert_called_once()


def test_log_model_huggingface_transformers(
    mock_mlflow: dict[str, MagicMock],
) -> None:
    """log_model dispatches to mlflow.transformers.log_model for HF models."""
    tracker = MLflowIntegration()
    mock = mock_mlflow["mlflow"]

    # The fixture stubs transformers with PreTrainedModel as a real type.
    import transformers

    model = transformers.PreTrainedModel()
    tracker.log_model(model, "hf-model")

    mock.transformers.log_model.assert_called_once()


def test_log_model_fallback_pyfunc(
    mock_mlflow: dict[str, MagicMock],
) -> None:
    """log_model falls back to mlflow.pyfunc.log_model for generic objects."""
    tracker = MLflowIntegration()
    mock = mock_mlflow["mlflow"]

    # A plain dict is neither a torch.nn.Module nor a PreTrainedModel.
    generic_obj = {"a": 1}
    tracker.log_model(generic_obj, "generic")

    mock.pyfunc.log_model.assert_called_once_with(generic_obj, "generic")


# ---------------------------------------------------------------------------
# get_runs
# ---------------------------------------------------------------------------


def test_get_runs_returns_list_of_dicts(
    mock_mlflow: dict[str, MagicMock],
) -> None:
    """get_runs returns runs in the expected dict format."""
    tracker = MLflowIntegration()
    mock = mock_mlflow["mlflow"]

    # Manually set active run so experiment_ids fallback works
    active_run = MagicMock()
    active_run.info.experiment_id = "exp-123"
    mock.active_run.return_value = active_run

    runs = tracker.get_runs()
    assert len(runs) == 1
    entry = runs[0]
    assert entry["run_id"] == "run-abc"
    assert entry["experiment_id"] == "exp-123"
    assert entry["status"] == "FINISHED"
    assert entry["start_time"] == 1000
    assert entry["params"] == {"batch_size": "32"}
    assert entry["metrics"] == {"throughput": "123.4"}
    assert entry["tags"] == {"env": "test"}

    mock.search_runs.assert_called_once()
    _, kwargs = mock.search_runs.call_args
    assert kwargs["output_format"] == "list"


def test_get_runs_with_experiment_name(
    mock_mlflow: dict[str, MagicMock],
) -> None:
    """get_runs filters by experiment name when provided."""
    tracker = MLflowIntegration()
    mock = mock_mlflow["mlflow"]

    mock.get_experiment_by_name.return_value = mock_mlflow["experiment"]

    runs = tracker.get_runs(experiment="my-exp")
    assert len(runs) == 1

    mock.get_experiment_by_name.assert_called_once_with("my-exp")


def test_get_runs_unknown_experiment(
    mock_mlflow: dict[str, MagicMock],
) -> None:
    """get_runs returns empty list and logs warning for unknown experiment."""
    tracker = MLflowIntegration()
    mock = mock_mlflow["mlflow"]

    mock.get_experiment_by_name.return_value = None

    with patch.object(tracker, "_autologged", True):
        runs = tracker.get_runs(experiment="missing-exp")

    assert runs == []
    mock.get_experiment_by_name.assert_called_once_with("missing-exp")


# ---------------------------------------------------------------------------
# Thread safety of _maybe_autolog
# ---------------------------------------------------------------------------


def test_maybe_autolog_thread_safe(
    mock_mlflow: dict[str, MagicMock],
) -> None:
    """_maybe_autolog only calls mlflow.pytorch.autolog once, even under
    concurrent access."""
    tracker = MLflowIntegration()
    mock = mock_mlflow["mlflow"]

    # Ensure autolog is flagged as requested
    tracker._autolog = True
    tracker._autologged = False

    # torch is already stubbed by the fixture; ensure the mock is accessible.
    with patch.dict(sys.modules, {"torch": MagicMock()}):
        import torch  # noqa: F401

        # Simulate concurrent calls
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(tracker._maybe_autolog) for _ in range(8)]
            concurrent.futures.wait(futures)

    # autolog should only be called once
    assert mock.pytorch.autolog.call_count == 1
    assert tracker._autologged is True
