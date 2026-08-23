"""MLflow experiment tracking plugin for DistLLM runs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False


@dataclass
class MLflowConfig:
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "distllm"
    run_name: str | None = None
    tags: dict[str, str] = field(default_factory=lambda: {"framework": "distllm"})


class MLflowPlugin:
    def __init__(self, config: MLflowConfig | None = None):
        self.config = config or MLflowConfig()
        self._run_id: str | None = None

    def log_run_start(self, params: dict[str, Any]) -> bool:
        if not _MLFLOW_AVAILABLE:
            return False
        try:
            mlflow.set_tracking_uri(self.config.tracking_uri)
            mlflow.set_experiment(self.config.experiment_name)
            run = mlflow.start_run(run_name=self.config.run_name)
            self._run_id = run.info.run_id
            mlflow.set_tags(self.config.tags)
            for k, v in params.items():
                mlflow.log_param(k, v)
            return True
        except Exception:
            return False

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> bool:
        if not _MLFLOW_AVAILABLE or not self._run_id:
            return False
        try:
            mlflow.log_metrics(metrics, step=step)
            return True
        except Exception:
            return False

    def log_run_end(self, status: str = "FINISHED") -> bool:
        if not _MLFLOW_AVAILABLE or not self._run_id:
            return False
        try:
            mlflow.end_run(status=status)
            self._run_id = None
            return True
        except Exception:
            return False

    def log_artifacts(self, local_dir: str) -> bool:
        if not _MLFLOW_AVAILABLE or not self._run_id:
            return False
        try:
            mlflow.log_artifacts(local_dir)
            return True
        except Exception:
            return False

    def get_run_id(self) -> str | None:
        return self._run_id
