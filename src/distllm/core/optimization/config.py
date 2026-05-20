from __future__ import annotations

from pydantic import BaseModel, Field


class TrialRunnerConfig(BaseModel):
    warmup_seconds: float = Field(default=5.0, ge=0.0, description="Warmup duration before each trial")
    cooldown_seconds: float = Field(default=2.0, ge=0.0, description="Cooldown between trials")
    benchmark_requests: int = Field(default=50, ge=1, description="Requests per benchmark run")


class BayesianOptimizerConfig(BaseModel):
    n_startup_trials: int = Field(default=10, ge=1, description="Random trials before Bayesian updates")
    n_ei_candidates: int = Field(default=24, ge=1, description="EI candidates for TPE sampler")
    seed: int = Field(default=42, description="RNG seed for reproducibility")
    n_trials: int = Field(default=50, ge=1, description="Total trials per optimization run")
    storage: str | None = Field(default=None, description="optuna storage URL for study persistence")


class OptimizationConfig(BaseModel):
    """Top-level configuration for the Bayesian optimization engine."""
    enabled: bool = Field(default=False, description="Enable self-optimizing configuration")
    study_name: str = Field(default="distllm_opt", description="Name for this optimization study")
    output_dir: str = Field(default="~/.distllm/optimization", description="Directory for trial results")
    maximize_throughput: bool = Field(default=True, description="Maximize throughput (vs minimize latency)")
    bayesian: BayesianOptimizerConfig = Field(default_factory=BayesianOptimizerConfig)
    runner: TrialRunnerConfig = Field(default_factory=TrialRunnerConfig)
