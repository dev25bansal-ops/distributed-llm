from __future__ import annotations

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Model architecture configuration for partition estimation."""
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


class AutoPartitionConfig(BaseModel):
    """Top-level configuration for hardware-aware auto-partitioning."""
    enabled: bool = Field(default=False, description="Enable hardware-aware auto-partitioning")
    strategy: str = Field(default="auto", pattern=r"^(auto|equal|gpu_aware|dp_minimax)$", description="Partition strategy")
    safety_margin: float = Field(default=0.1, ge=0.0, le=0.5, description="GPU memory safety margin")
    profile_dir: str = Field(default="~/.distllm/partitions", description="Directory for partition plans")
    model: ModelConfig = Field(default_factory=ModelConfig)
    profiler: ProfilerConfig = Field(default_factory=ProfilerConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
