from typing import Optional
from pydantic import BaseModel


class OperatorConfig(BaseModel):
    """Configuration for the DistLLM Kubernetes operator."""

    namespace: str = "default"
    image: str = "distributed-llm:latest"
    image_pull_policy: str = "IfNotPresent"
    coordinator_port: int = 8000
    worker_port: int = 50051
    default_replicas: int = 1
    default_model: str = "distributed-llm"
    default_cpu: str = "4"
    default_memory: str = "16Gi"
    default_gpu: Optional[str] = "1"
