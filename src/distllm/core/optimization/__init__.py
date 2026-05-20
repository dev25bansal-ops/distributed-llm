"""Self-optimizing configuration engine using Bayesian optimization.

Uses optuna (TPE sampler) to search over the configuration space of:
    - Batch size
    - Tensor parallelism degree
    - Number of pipeline stages
    - Quantization level (FP16 / INT8 / FP8)
    - Speculation length
    - Chunk size for chunked prefill

Package structure:
    space.py    — Parameter search space definitions
    bayesian.py — Bayesian optimization engine (optuna wrapper)
    runner.py   — Trial runner: applies configs and collects metrics
    tracker.py  — Results tracking, persistence, best-config selection
    config.py   — Configuration models for optimization settings
"""

from distllm.core.optimization.space import (
    ParamDomain,
    IntDomain,
    CategoricalDomain,
    SearchSpace,
    default_search_space,
)
from distllm.core.optimization.bayesian import BayesianOptimizer, ObjectiveDirection
from distllm.core.optimization.runner import TrialRunner, TrialResult
from distllm.core.optimization.tracker import (
    TrialRecord,
    OptimizationTracker,
    BestConfig,
)
from distllm.core.optimization.config import (
    OptimizationConfig,
    BayesianOptimizerConfig,
    TrialRunnerConfig,
)

__all__ = [
    "ParamDomain",
    "IntDomain",
    "CategoricalDomain",
    "SearchSpace",
    "default_search_space",
    "BayesianOptimizer",
    "ObjectiveDirection",
    "TrialRunner",
    "TrialResult",
    "TrialRecord",
    "OptimizationTracker",
    "BestConfig",
    "OptimizationConfig",
    "BayesianOptimizerConfig",
    "TrialRunnerConfig",
]
