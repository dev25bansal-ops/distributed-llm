"""Re-export shim for the merged WandBIntegration.

This module provides a thin re-export of :class:`WandBIntegration` from the
standalone ``distllm-wandb`` package.  If the package is not installed, it
gracefully falls back to a no-op stub that satisfies import-time references
while logging a warning at first use.

Usage::

    from distllm.integrations.wandb import WandBIntegration

    tracker = WandBIntegration(project="distllm-experiment")
"""

from __future__ import annotations

from typing import Any

import logging

logger = logging.getLogger("distllm")

try:
    from distllm_wandb import WandBIntegration
except ImportError:
    logger.info(
        "distllm-wandb package is not installed. "
        "Install it with: pip install distllm-wandb"
    )

    class WandBIntegration:  # type: ignore[no-redef]
        """No-op fallback when distllm-wandb is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
            pass

        def __enter__(self) -> WandBIntegration:
            return self

        def __exit__(self, *args: Any) -> None:  # noqa: ANN401
            pass

        def start(self) -> None:
            pass

        def finish(self) -> None:
            pass

        def log_metrics(
            self, metrics: dict[str, float], step: int | None = None
        ) -> None:
            pass

        def log_latencies(
            self, latencies: list[float], step: int | None = None
        ) -> None:
            pass

        def log_model(self, model: Any, name: str) -> None:  # noqa: ANN401
            pass

        def log_partition_plan(self, plan: Any) -> None:  # noqa: ANN401
            pass

        def log_quant_results(self, results: Any) -> None:  # noqa: ANN401
            pass

        def log_artifacts(
            self, local_path: str, artifact_type: str = "model"
        ) -> None:
            pass

        def watch_model(self, model: Any, log_freq: int = 100) -> None:  # noqa: ANN401
            pass

        @property
        def run_url(self) -> str | None:
            return None


__all__ = ["WandBIntegration"]
