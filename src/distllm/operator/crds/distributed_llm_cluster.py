"""Kubernetes Operator CRDs for distributed-llm."""

from dataclasses import dataclass, field


@dataclass
class ModelSpec:
    name: str
    layers: int
    tensor_parallel_size: int = 1
    dtype: str = "float16"


@dataclass
class ResourceSpec:
    gpu: str = "1"
    memory: str = "32Gi"
    cpu: str = "4"


@dataclass
class CoordinatorSpec:
    replicas: int = 1
    resources: ResourceSpec = field(default_factory=ResourceSpec)
    image: str = "distllm/coordinator:latest"
    port: int = 8000
    grpc_port: int = 50050


@dataclass
class NodePoolSpec:
    """A pool of worker nodes handling a specific layer range."""
    start_layer: int = 0
    end_layer: int = 0
    replicas: int = 1
    resources: ResourceSpec = field(default_factory=ResourceSpec)
    image: str = "distllm/worker:latest"
    grpc_port: int = 50051


@dataclass
class HPASpec:
    enabled: bool = False
    target_tps: int | None = None
    min_replicas: int = 1
    max_replicas: int = 10
    metric: str = "tokens_per_second"  # or "queue_depth"


@dataclass
class DistributedLLMClusterSpec:
    """Top-level CRD for a distributed LLM cluster deployment."""
    model: ModelSpec
    coordinator: CoordinatorSpec = field(default_factory=CoordinatorSpec)
    node_pools: list[NodePoolSpec] = field(default_factory=list)
    tls_enabled: bool = False
    api_key_secret: str | None = None
    hpa: HPASpec = field(default_factory=HPASpec)
    labels: dict[str, str] = field(default_factory=dict)
    namespace: str = "default"


@dataclass
class NodePoolCRD:
    """Standalone CRD for managing a node pool independently."""
    cluster_name: str
    start_layer: int
    end_layer: int
    replicas: int = 1
    resources: ResourceSpec = field(default_factory=ResourceSpec)
    image: str = "distllm/worker:latest"
    grpc_port: int = 50051
