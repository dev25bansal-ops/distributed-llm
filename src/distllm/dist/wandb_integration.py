"""Weights & Biases integration for experiment tracking.

Hooks into :class:`QuantizationAutoTuner` and :class:`PartitionOptimizer`
to log experiments — model architectures, quantization plans, partition
solutions, and performance metrics — to W&B runs for comparison and
analysis.

Usage::

    from distllm.dist.wandb_integration import WandbExperiment

    # Log a partition experiment:
    with WandbExperiment(
        project="distllm-partition",
        config={"model": "llama-70b", "num_layers": 80},
    ) as exp:
        plan = optimizer.solve()
        exp.log_partition_plan(plan)
        exp.log_metrics({"total_tflops": plan.throughput_estimate})

    # Log a quantization experiment:
    with WandbExperiment(
        project="distllm-quant",
        config={"model": "llama-70b", "method": "gptq"},
    ) as exp:
        results = tuner.tune()
        exp.log_quant_results(results)

Gracefully degrades when ``wandb`` is not installed — all methods
become no-ops.
"""

from __future__ import annotations

import os
import time
from typing import Any

from loguru import logger

try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


class WandbExperiment:
    """Context manager for a single W&B experiment run.

    Accepts an optional *run* for continuation; otherwise creates one.

    Usage::

        with WandbExperiment(project="distllm-partition") as exp:
            exp.log_metrics({"throughput": 142.3})
    """

    def __init__(
        self,
        project: str = "distllm",
        config: dict[str, Any] | None = None,
        run_name: str | None = None,
        tags: list[str] | None = None,
        group: str | None = None,
        run: Any = None,
    ):
        self.project = project
        self.config = config or {}
        self.run_name = run_name
        self.tags = tags
        self.group = group
        self._external_run = run is not None
        self._run = run
        self._step = 0

    def __enter__(self) -> WandbExperiment:
        if not _WANDB_AVAILABLE:
            logger.info("wandb not installed — experiment logging is a no-op")
            return self

        if self._run is None:
            try:
                self._run = wandb.init(
                    project=self.project,
                    config=self.config,
                    name=self.run_name,
                    tags=self.tags,
                    group=self.group,
                    reinit=True,
                )
                logger.info(
                    "W&B run started: {}/{}",
                    self.project, self._run.name or self.run_name,
                )
            except Exception as e:
                logger.warning("Failed to init W&B run: {}", e)
                self._run = None
        return self

    def __exit__(self, *args: Any) -> None:
        if not _WANDB_AVAILABLE or self._external_run:
            return
        if self._run is not None:
            try:
                self._run.finish()
                logger.debug("W&B run finished")
            except Exception as e:
                logger.warning("Failed to finish W&B run: {}", e)

    # ── Logging helpers ───────────────────────────────────────────────

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log scalar metrics to the current run."""
        if not _WANDB_AVAILABLE or self._run is None:
            return
        try:
            self._run.log(metrics, step=step or self._step)
            self._step += 1
        except Exception as e:
            logger.warning("Failed to log metrics to W&B: {}", e)

    def log_partition_plan(self, plan: Any) -> None:
        """Log a partition plan as a W&B Table.

        Args:
            plan: A ``PartitionSolution`` or similar object with
                ``node_assignments`` and ``throughput_estimate``.
        """
        if not _WANDB_AVAILABLE or self._run is None:
            return
        try:
            columns = ["node_id", "start_layer", "end_layer", "compute_ms", "memory_gb"]
            data: list[list[Any]] = []

            assignments = getattr(plan, "node_assignments", None) or getattr(plan, "assignments", [])
            for assignment in assignments:
                if isinstance(assignment, dict):
                    data.append([
                        assignment.get("node_id", ""),
                        assignment.get("start_layer", 0),
                        assignment.get("end_layer", 0),
                        round(assignment.get("compute_ms", 0), 1),
                        round(assignment.get("memory_gb", 0), 1),
                    ])
                elif hasattr(assignment, "node_id"):
                    data.append([
                        assignment.node_id,
                        getattr(assignment, "start_layer", 0),
                        getattr(assignment, "end_layer", 0),
                        round(getattr(assignment, "compute_ms", 0), 1),
                        round(getattr(assignment, "memory_gb", 0), 1),
                    ])

            if data:
                table = wandb.Table(columns=columns, data=data)
                self._run.log({"partition_plan": table})

            throughput = getattr(plan, "throughput_estimate", None) or getattr(plan, "throughput", None)
            if throughput is not None:
                self._run.log({"throughput_tokens_per_sec": throughput})
        except Exception as e:
            logger.warning("Failed to log partition plan to W&B: {}", e)

    def log_quant_results(self, results: Any) -> None:
        """Log quantization tuning results.

        Args:
            results: Dict or list of per-layer quantization results
                with ``layer_name``, ``method``, ``accuracy``, ``speedup``.
        """
        if not _WANDB_AVAILABLE or self._run is None:
            return
        try:
            if isinstance(results, dict):
                results = [results]
            if isinstance(results, list):
                columns = ["layer", "method", "accuracy", "speedup"]
                data = []
                for r in results:
                    if isinstance(r, dict):
                        data.append([
                            r.get("layer", r.get("layer_name", "")),
                            r.get("method", ""),
                            r.get("accuracy", r.get("perplexity", 0)),
                            r.get("speedup", r.get("throughput_gain", 0)),
                        ])
                if data:
                    table = wandb.Table(columns=columns, data=data)
                    self._run.log({"quantization_results": table})

            total_accuracy = 0
            count = 0
            for r in (results if isinstance(results, list) else [results]):
                acc = r.get("accuracy") if isinstance(r, dict) else getattr(r, "accuracy", None)
                if acc is not None:
                    total_accuracy += acc
                    count += 1
            if count > 0:
                self._run.log({"avg_accuracy": total_accuracy / count})
        except Exception as e:
            logger.warning("Failed to log quant results to W&B: {}", e)

    def log_artifacts(self, local_path: str, artifact_type: str = "model") -> None:
        """Log a local file/dir as a W&B artifact."""
        if not _WANDB_AVAILABLE or self._run is None:
            return
        try:
            artifact = wandb.Artifact(
                name=f"{self.project}-{int(time.time())}",
                type=artifact_type,
            )
            if os.path.isfile(local_path):
                artifact.add_file(local_path)
            elif os.path.isdir(local_path):
                artifact.add_dir(local_path)
            self._run.log_artifact(artifact)
        except Exception as e:
            logger.warning("Failed to log artifact to W&B: {}", e)

    def watch_model(self, model: Any, log_freq: int = 100) -> None:
        """Watch a PyTorch model's gradients and topology."""
        if not _WANDB_AVAILABLE or self._run is None:
            return
        try:
            wandb.watch(model, log_freq=log_freq)
        except Exception as e:
            logger.warning("Failed to watch model: {}", e)

    @property
    def run_url(self) -> str | None:
        if self._run is not None and hasattr(self._run, "get_url"):
            try:
                return self._run.get_url()
            except Exception:
                pass
        return None
