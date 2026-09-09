"""CRD spec dataclasses for the DistLLM cluster operator."""

from dataclasses import dataclass, field


@dataclass
class ModelSpec:
    name: str
    layers: int = 32
    revision: str = "main"


@dataclass
class ResourceSpec:
    gpu: str = "1"
    memory: str = "32Gi"
    cpu: str = "4"


@dataclass
class CoordinatorSpec:
    replicas: int = 1
    port: int = 8000
    grpc_port: int = 50050
    resources: ResourceSpec = field(default_factory=ResourceSpec)


@dataclass
class NodePoolSpec:
    name: str = "workers"
    replicas: int = 1
    grpc_port: int = 50051
    node_selector: dict = field(default_factory=dict)
    resources: ResourceSpec = field(default_factory=ResourceSpec)


@dataclass
class DistributedLLMClusterSpec:
    model: ModelSpec
    coordinator: CoordinatorSpec = field(default_factory=CoordinatorSpec)
    node_pools: list[NodePoolSpec] = field(default_factory=list)
    tls_enabled: bool = False
    namespace: str = "default"
