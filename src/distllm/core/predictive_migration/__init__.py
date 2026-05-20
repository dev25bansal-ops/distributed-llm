"""Predictive KV Cache Migration.

Pre-warms KV cache on target nodes before the request arrives by:
1. Tracking prompt prefix frequency across the cluster
2. Using a Markov chain to predict the next likely prefix
3. Pre-migrating KV cache to optimal nodes
4. Deduplicating cache via content-addressable storage

Package structure:
    tracker.py   — PrefixFrequencyTracker: observes and scores prefixes
    predictor.py — MarkovChainPredictor: predicts next likely prefixes
    store.py     — ContentAddressableStore: hash-deduplicated KV cache
    migration.py — PreMigrationScheduler: schedules cache transfers
    engine.py    — PredictiveMigrationEngine: orchestrates the full loop
    config.py    — Configuration models
"""

from distllm.core.predictive_migration.tracker import (
    PrefixRecord,
    PrefixFrequencyTracker,
)
from distllm.core.predictive_migration.predictor import (
    Transition,
    MarkovChainPredictor,
    Prediction,
)
from distllm.core.predictive_migration.store import (
    ContentEntry,
    ContentAddressableStore,
)
from distllm.core.predictive_migration.migration import (
    MigrationTask,
    PreMigrationScheduler,
)
from distllm.core.predictive_migration.engine import PredictiveMigrationEngine
from distllm.core.predictive_migration.config import (
    PredictiveMigrationConfig,
    TrackerConfig,
    PredictorConfig,
    StoreConfig,
    MigrationConfig,
)

__all__ = [
    "PrefixRecord",
    "PrefixFrequencyTracker",
    "Transition",
    "MarkovChainPredictor",
    "Prediction",
    "ContentEntry",
    "ContentAddressableStore",
    "MigrationTask",
    "PreMigrationScheduler",
    "PredictiveMigrationEngine",
    "PredictiveMigrationConfig",
    "TrackerConfig",
    "PredictorConfig",
    "StoreConfig",
    "MigrationConfig",
]
