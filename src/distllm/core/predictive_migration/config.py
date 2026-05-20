from __future__ import annotations

from pydantic import BaseModel, Field


class TrackerConfig(BaseModel):
    enabled: bool = Field(default=True, description="Enable prefix frequency tracking")
    min_prefix_len: int = Field(default=8, ge=1, description="Minimum prefix length to track")
    max_prefixes: int = Field(default=10000, ge=100, description="Maximum number of tracked prefixes")
    decay_hours: float = Field(default=24.0, ge=0.1, description="Frequency decay half-life in hours")


class PredictorConfig(BaseModel):
    enabled: bool = Field(default=True, description="Enable Markov chain prediction")
    order: int = Field(default=1, ge=1, le=2, description="Markov chain order (1 or 2)")
    window_size: int = Field(default=10000, ge=100, description="Max transitions before reset")
    decay_hours: float = Field(default=24.0, ge=0.1, description="Transition decay half-life in hours")
    min_observations: int = Field(default=2, ge=1, description="Minimum observations before prediction")


class StoreConfig(BaseModel):
    enabled: bool = Field(default=True, description="Enable content-addressable KV cache store")
    max_entries: int = Field(default=10000, ge=100, description="Maximum cache entries")
    default_ttl_secs: float = Field(default=3600.0, ge=60.0, description="Default entry TTL in seconds")


class MigrationConfig(BaseModel):
    enabled: bool = Field(default=True, description="Enable pre-migration scheduling")
    max_concurrent: int = Field(default=4, ge=1, le=64, description="Max concurrent migrations")
    max_bandwidth_mbps: float = Field(default=1000.0, ge=1.0, description="Network bandwidth limit in Mbps")
    migrated_ttl_secs: float = Field(default=600.0, ge=60.0, description="TTL for migrated entries")
    confidence_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum confidence to trigger migration")
    top_k_predictions: int = Field(default=10, ge=1, le=100, description="Number of top predictions to evaluate")


class PredictiveMigrationConfig(BaseModel):
    """Top-level configuration for predictive KV cache migration."""
    enabled: bool = Field(default=False, description="Enable predictive KV cache migration")
    source_node: str = Field(default="local", description="This node's identifier")
    target_nodes: list[str] = Field(default_factory=lambda: ["node-a", "node-b"], description="Candidate target nodes for migration")
    observe_interval: float = Field(default=10.0, ge=1.0, description="Observation collection interval in seconds")
    predict_interval: float = Field(default=30.0, ge=5.0, description="Prediction interval in seconds")
    migrate_interval: float = Field(default=15.0, ge=5.0, description="Migration execution interval in seconds")
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    predictor: PredictorConfig = Field(default_factory=PredictorConfig)
    store: StoreConfig = Field(default_factory=StoreConfig)
    migration: MigrationConfig = Field(default_factory=MigrationConfig)
