"""Optional MLflow integration for DistLLM experiment tracking.

Auto-logs DistLLM runs as MLflow experiments, tracking model
performance metrics (latency, throughput, GPU utilization) and
persisting artifacts (config, metrics, model binaries).

Usage:
    from distllm.integrations.mlflow_tracking import MLflowIntegration

    tracker = MLflowIntegration()
    with tracker.log_run("my-experiment", params={...}, metrics={...}) as run:
        tracker.log_model(my_model, "artifact_model")
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("distllm.mlflow")

try:
    import mlflow
    from mlflow.exceptions import MlflowException

    _MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MLFLOW_AVAILABLE = False

    class MlflowException(Exception):  # type: ignore[no-redef]
        """Stub replacement when mlflow is not installed."""


class MLflowIntegration:
    """Track DistLLM experiments with MLflow.

    All public methods are no-ops when mlflow is not installed, making
    the integration safe to use in environments where MLflow may or may
    not be available.

    Typical workflow::

        tracker = MLflowIntegration(tracking_uri="http://localhost:5000")
        with tracker.log_run(
            experiment_name="distllm-bench",
            params={"model": "llama-7b", "backend": "onnx"},
            metrics={"throughput": 123.4, "latency_p50_ms": 42.0},
        ) as run:
            tracker.log_model(model, "model")
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        registry_uri: Optional[str] = None,
        autolog: bool = True,
    ) -> None:
        """Initialise the MLflow integration.

        Parameters
        ----------
        tracking_uri:
            MLflow tracking server URI.  Falls back to the
            ``MLFLOW_TRACKING_URI`` environment variable, then to the
            local ``mlruns/`` directory.
        registry_uri:
            MLflow model registry URI.
        autolog:
            If True and PyTorch is available, call
            ``mlflow.pytorch.autolog()`` (once) on first use.
        """
        self._autolog = autolog
        self._autologged = False

        if not _MLFLOW_AVAILABLE:
            logger.info("mlflow is not installed — all MLflowIntegration calls are no-ops")
            return

        uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
        if uri:
            mlflow.set_tracking_uri(uri)

        if registry_uri:
            mlflow.set_registry_uri(registry_uri)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_run(
        self,
        experiment_name: str,
        params: Optional[dict[str, Any]] = None,
        metrics: Optional[dict[str, float]] = None,
        description: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
    ) -> "_RunContext":
        """Start a new MLflow run under *experiment_name*.

        Returns a context manager that automatically ends the run on
        exit.  Inside the ``with`` block you may call ``log_model``,
        ``log_artifact``, ``log_metric``, etc.

        Parameters
        ----------
        experiment_name:
            MLflow experiment name.  Created automatically if it does
            not exist.
        params:
            Key-value pairs logged as MLflow parameters (e.g. model
            name, batch size, precision).
        metrics:
            Key-value pairs logged as MLflow metrics (e.g. throughput,
            latency).
        description:
            Optional run description.
        tags:
            Optional tags to attach to the run.

        Returns
        -------
        _RunContext
            A context manager that wraps the active MLflow run.
        """
        if not _MLFLOW_AVAILABLE:
            return _RunContext(None)

        self._maybe_autolog()

        try:
            experiment = mlflow.set_experiment(experiment_name)
        except MlflowException:
            experiment_id = mlflow.create_experiment(experiment_name)
            experiment = mlflow.set_experiment(experiment_name)

        run = mlflow.start_run(
            experiment_id=experiment.experiment_id,
            description=description,
            tags=tags,
        )

        if params:
            mlflow.log_params(params)

        if metrics:
            mlflow.log_metrics(metrics)

        return _RunContext(run)

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        **kwargs: Any,
    ) -> None:
        """Log a model as an MLflow artifact.

        Delegates to ``mlflow.pytorch.log_model`` when ``model`` is a
        ``torch.nn.Module``, otherwise falls back to a generic pickle.

        Parameters
        ----------
        model:
            The model object to persist.
        artifact_path:
            Relative path inside the run's artifact directory.
        **kwargs:
            Forwarded to the underlying ``mlflow.*.log_model`` call.
        """
        if not _MLFLOW_AVAILABLE:
            return

        try:
            import torch

            if isinstance(model, torch.nn.Module):
                mlflow.pytorch.log_model(model, artifact_path, **kwargs)
                return
        except ImportError:
            pass

        # Generic fallback via pickle.
        mlflow.sklearn.log_model(model, artifact_path, **kwargs)

    def get_runs(
        self,
        experiment: Optional[str] = None,
        order_by: Optional[list[str]] = None,
        max_results: int = 100,
    ) -> list[dict[str, Any]]:
        """Retrieve runs for *experiment* as plain dictionaries.

        Parameters
        ----------
        experiment:
            Experiment name.  If None, the currently set experiment is
            used.
        order_by:
            MLflow order clauses, e.g. ``["metrics.throughput DESC"]``.
        max_results:
            Maximum number of runs to return.

        Returns
        -------
        list[dict]
            Each dict contains ``run_id``, ``experiment_id``,
            ``status``, ``start_time``, ``params``, ``metrics``, and
            ``tags``.
        """
        if not _MLFLOW_AVAILABLE:
            return []

        experiment_obj = mlflow.get_experiment_by_name(experiment) if experiment else None
        if experiment_obj is None and experiment is not None:
            logger.warning("Experiment %r not found", experiment)
            return []

        experiment_ids = (
            [experiment_obj.experiment_id]
            if experiment_obj
            else [mlflow.active_run().info.experiment_id]
        )

        runs = mlflow.search_runs(
            experiment_ids=experiment_ids,
            order_by=order_by or [],
            max_results=max_results,
        )

        results: list[dict[str, Any]] = []
        for _, row in runs.iterrows():
            results.append(
                {
                    "run_id": row.get("run_id", ""),
                    "experiment_id": row.get("experiment_id", ""),
                    "status": row.get("status", ""),
                    "start_time": row.get("start_time", ""),
                    "params": {
                        k: row[k]
                        for k in row.index
                        if k.startswith("params.")
                    },
                    "metrics": {
                        k: row[k]
                        for k in row.index
                        if k.startswith("metrics.") and not isinstance(row[k], str)
                    },
                    "tags": {
                        k: row[k]
                        for k in row.index
                        if k.startswith("tags.")
                    },
                }
            )

        return results

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def log_artifact(self, local_path: str) -> None:
        """Upload a local file or directory as a run artifact."""
        if not _MLFLOW_AVAILABLE:
            return
        mlflow.log_artifact(local_path)

    def log_artifacts_from_dict(
        self,
        data: dict[str, Any],
        filename: str = "config.json",
    ) -> None:
        """Write *data* to a temporary JSON file and log as artifact.

        This is useful for persisting runtime configuration alongside a
        run.
        """
        if not _MLFLOW_AVAILABLE:
            return
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
            encoding="utf-8",
        ) as f:
            json.dump(data, f, indent=2, default=str)
            tmp_path = f.name
        try:
            mlflow.log_artifact(tmp_path, artifact_path=Path(filename).parent.as_posix() or ".")
        finally:
            os.unlink(tmp_path)

    def set_tag(self, key: str, value: str) -> None:
        """Set a single tag on the currently active run."""
        if not _MLFLOW_AVAILABLE:
            return
        mlflow.set_tag(key, value)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_autolog(self) -> None:
        """Call ``mlflow.pytorch.autolog()`` once if PyTorch is available."""
        if not _MLFLOW_AVAILABLE or self._autologged or not self._autolog:
            return
        try:
            import torch  # noqa: F401

            mlflow.pytorch.autolog()
            self._autologged = True
            logger.info("mlflow.pytorch.autolog() enabled")
        except ImportError:
            pass


class _RunContext:
    """Context manager wrapping an MLflow ``ActiveRun``.

    Ends the run on ``__exit__`` so callers do not have to remember.
    """

    def __init__(self, run: Any) -> None:
        self._run = run

    @property
    def info(self) -> Any:
        """Access the underlying ``mlflow.ActiveRun.info`` or None."""
        if self._run is None or not _MLFLOW_AVAILABLE:
            return None
        return self._run.info

    def __enter__(self) -> "_RunContext":
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
    ) -> None:
        if self._run is not None and _MLFLOW_AVAILABLE:
            mlflow.end_run()
