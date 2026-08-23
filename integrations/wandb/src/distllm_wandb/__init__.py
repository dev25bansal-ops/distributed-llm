"""distllm-wandb — merged W&B integration for DistLLM.

Provides :class:`WandBIntegration` for experiment tracking, GPU monitoring,
latency histograms, partition plans, quantization reporting, and model
artefact logging.

Usage::

    from distllm_wandb import WandBIntegration

    tracker = WandBIntegration(
        project="distllm-experiment",
        config={"learning_rate": 1e-4, "batch_size": 32},
    )
    with tracker:
        tracker.log_metrics({"loss": 0.05, "accuracy": 0.97})

Gracefully degrades when ``wandb`` is not installed — every public method
becomes a no-op with a logged warning at first use.
"""

from distllm_wandb.tracker import WandBIntegration, _WANDB_AVAILABLE

__all__ = ["WandBIntegration", "_WANDB_AVAILABLE"]
