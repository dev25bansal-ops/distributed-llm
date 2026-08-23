"""Weights & Biases real-time inference monitor."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


@dataclass
class WandBConfig:
    project: str = "distllm"
    entity: str | None = None
    api_key: str | None = None
    tags: list[str] = field(default_factory=lambda: ["distllm"])
    log_interval_s: float = 5.0


class WandBMonitor:
    def __init__(self, config: WandBConfig | None = None):
        self.config = config or WandBConfig()
        self._run = None

    def start(self, metrics_provider: Callable | None = None) -> bool:
        if not _WANDB_AVAILABLE:
            return False
        try:
            if self.config.api_key:
                wandb.login(key=self.config.api_key)
            self._run = wandb.init(project=self.config.project, entity=self.config.entity, tags=self.config.tags)
            return True
        except Exception:
            return False

    def log_metrics(self, metrics: dict[str, float]) -> bool:
        if not _WANDB_AVAILABLE or not self._run:
            return False
        try:
            wandb.log(metrics)
            return True
        except Exception:
            return False

    def log_eval_results(self, results: dict[str, Any]) -> bool:
        if not _WANDB_AVAILABLE or not self._run:
            return False
        try:
            import pandas as pd
            df = pd.DataFrame([results]) if isinstance(results, dict) else pd.DataFrame(results)
            wandb.log({"evaluation": wandb.Table(dataframe=df)})
            return True
        except Exception:
            return False

    def stop(self) -> None:
        if _WANDB_AVAILABLE and self._run:
            try:
                wandb.finish()
            except Exception:
                pass
            self._run = None
