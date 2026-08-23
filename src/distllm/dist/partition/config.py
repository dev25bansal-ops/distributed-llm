from __future__ import annotations

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    hidden_size: int = Field(default=4096, ge=64, description="Model hidden dimension")
    intermediate_size: int = Field(default=11008, ge=1, description="MLP intermediate dimension")
    num_layers: int = Field(default=32, ge=1, description="Number of transformer layers")
    num_heads: int = Field(default=32, ge=1, description="Number of attention heads")
    head_dim: int = Field(default=128, ge=1, description="Attention head dimension")
    vocab_size: int = Field(default=32000, ge=1, description="Vocabulary size")
    max_seq_len: int = Field(default=4096, ge=1, description="Maximum sequence length")


class ProfilerConfig(BaseModel):
    enabled: bool = Field(default=True, description="Enable hardware profiling")
    benchmark_matmul: bool = Field(default=True, description="Benchmark matmul TFLOPS at runtime")
    benchmark_bandwidth: bool = Field(default=True, description="Benchmark memory bandwidth at runtime")
    ping_timeout_seconds: float = Field(default=2.0, ge=0.5, description="Timeout for latency probes")
    bandwidth_test_bytes: int = Field(default=8388608, ge=1024, description="Bytes for bandwidth test")


class OptimizerConfig(BaseModel):
    batch_size: int = Field(default=1, ge=1, description="Target batch size for cost model")
    seq_len: int = Field(default=4096, ge=1, description="Target sequence length for cost model")
    allow_oom: bool = Field(default=False, description="Allow partitions that exceed GPU memory")
    compare_to_baselines: bool = Field(default=True, description="Compare DP vs equal/proportional")


class ParetoConfig(BaseModel):
    """Configuration for multi-objective Pareto optimization."""
    enabled: bool = Field(default=False, description="Enable Pareto multi-objective optimization")
    objectives: list[str] = Field(
        default=["latency", "memory", "cost"],
        description="Objectives to optimize (latency, throughput, memory, quality, cost)",
    )
    weights: dict[str, float] = Field(
        default={"latency": 0.6, "memory": 0.2, "cost": 0.2},
        description="Weights for solution selection from Pareto frontier",
    )
    frontier_limit: int = Field(default=32, ge=4, le=256, description="Max Pareto points per DP cell")
    node_costs_per_hour: dict[str, float] = Field(
        default_factory=dict,
        description="Per-node hourly cost ($/hr) for cost objective",
    )


class LearnedCostConfig(BaseModel):
    """Configuration for ML-based learned cost model."""
    enabled: bool = Field(default=False, description="Enable learned cost model")
    min_samples_to_train: int = Field(default=50, ge=10, description="Minimum observations before training")
    retrain_interval_s: float = Field(default=3600.0, ge=60.0, description="Seconds between automatic retraining")
    save_path: str = Field(default="~/.distllm/learned_cost.json", description="Path to persist the learned model")


class AdaptiveConfig(BaseModel):
    """Configuration for online adaptive re-partitioning."""
    enabled: bool = Field(default=False, description="Enable adaptive re-partitioning")
    straggler_threshold: float = Field(default=1.5, ge=1.1, description="Latency multiplier to trigger re-partition")
    min_repartition_interval_s: float = Field(default=30.0, ge=5.0, description="Min seconds between re-partitions")
    cooldown_after_repartition_s: float = Field(default=60.0, ge=10.0, description="Cooldown after re-partition")
    max_repartitions_per_hour: int = Field(default=10, ge=1, le=100, description="Max re-partitions per hour")
    require_quorum: bool = Field(default=True, description="Require multiple stragglers before re-partitioning")
    quorum_fraction: float = Field(default=0.5, ge=0.1, le=1.0, description="Fraction of straggler nodes needed")


class CloudArbitrageConfig(BaseModel):
    """Configuration for cloud cost optimization."""
    enabled: bool = Field(default=False, description="Enable cloud arbitrage optimization")
    throughput_target_tok_s: float = Field(default=0.0, ge=0.0, description="Minimum throughput target (0 = no limit)")
    latency_target_ms: float = Field(default=0.0, ge=0.0, description="Maximum latency target (0 = no limit)")
    max_budget_per_hour: float = Field(default=1000.0, ge=0.0, description="Maximum $/hr budget")
    max_preemption_risk: float = Field(default=0.15, ge=0.0, le=1.0, description="Max spot preemption probability")
    prefer_spot: bool = Field(default=True, description="Prefer spot/preemptible pricing")
    allowed_providers: list[str] = Field(
        default=["aws", "gcp", "azure"],
        description="Allowed cloud providers",
    )


class AutoPartitionConfig(BaseModel):
    enabled: bool = Field(default=False, description="Enable hardware-aware auto-partitioning")
    strategy: str = Field(
        default="auto",
        pattern=r"^(auto|equal|gpu_aware|dp_minimax|pareto|quant_aware)$",
        description="Partition strategy",
    )
    safety_margin: float = Field(default=0.1, ge=0.0, le=0.5, description="GPU memory safety margin")
    profile_dir: str = Field(default="~/.distllm/partitions", description="Directory for partition plans")
    model: ModelConfig = Field(default_factory=ModelConfig)
    profiler: ProfilerConfig = Field(default_factory=ProfilerConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    pareto: ParetoConfig = Field(default_factory=ParetoConfig)
    learned_cost: LearnedCostConfig = Field(default_factory=LearnedCostConfig)
    adaptive: AdaptiveConfig = Field(default_factory=AdaptiveConfig)
    cloud_arbitrage: CloudArbitrageConfig = Field(default_factory=CloudArbitrageConfig)
