"""Terraform provider resources for DistLLM cluster provisioning.

This module defines the resource schema and CRUD operations for
managing DistLLM resources as Infrastructure as Code via Terraform.

Resources:
- ``distllm_cluster`` — A DistLLM cluster node group
- ``distllm_federation_link`` — A federation link between clusters
- ``distllm_tenant`` — A tenant with SLO configuration
- ``distllm_model_deployment`` — A model deployment on the cluster

Usage (Terraform HCL)::

    provider "distllm" {
        endpoint = "http://coordinator:8000"
    }

    resource "distllm_cluster" "production" {
        name           = "prod-gpu-cluster"
        node_count     = 8
        gpu_type       = "A100-80GB"
        min_bandwidth  = "200Gbps"
    }

    resource "distllm_tenant" "research" {
        tenant_id      = "research-team"
        max_rpm        = 1000
        latency_slo_ms = 500
    }
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from loguru import logger


class GPUType(str, Enum):
    A100_40GB = "A100-40GB"
    A100_80GB = "A100-80GB"
    H100 = "H100"
    H200 = "H200"
    MI300X = "MI300X"
    B200 = "B200"
    L40S = "L40S"
    L4 = "L4"
    V100 = "V100"
    T4 = "T4"


class ClusterStatus(str, Enum):
    CREATING = "creating"
    RUNNING = "running"
    UPDATING = "updating"
    DEGRADED = "degraded"
    DELETED = "deleted"


@dataclass
class ClusterResource:
    """A DistLLM cluster node group managed as IaC.

    Maps to ``distllm_cluster`` in Terraform HCL.
    """
    name: str
    node_count: int = 1
    gpu_type: GPUType = GPUType.A100_80GB
    min_bandwidth_gbps: str = "200"
    status: ClusterStatus = ClusterStatus.CREATING
    coordinator_endpoint: str = "http://localhost:8000"
    labels: dict[str, str] = field(default_factory=dict)

    def create(self) -> dict[str, Any]:
        """Create the cluster (calls DistLLM coordinator API)."""
        import requests
        payload = {
            "name": self.name,
            "node_count": self.node_count,
            "gpu_type": self.gpu_type.value,
            "min_bandwidth_gbps": self.min_bandwidth_gbps,
            "labels": self.labels,
        }
        resp = requests.post(
            f"{self.coordinator_endpoint}/api/v1/clusters",
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        self.status = ClusterStatus.RUNNING
        return {"id": resp.json().get("cluster_id", self.name)}

    def read(self) -> dict[str, Any]:
        """Read the cluster state."""
        import requests
        resp = requests.get(
            f"{self.coordinator_endpoint}/api/v1/clusters/{self.name}",
            timeout=30,
        )
        if resp.status_code == 404:
            return {}
        resp.raise_for_status()
        return resp.json()

    def update(self, changes: dict[str, Any]) -> dict[str, Any]:
        """Update cluster configuration."""
        import requests
        resp = requests.patch(
            f"{self.coordinator_endpoint}/api/v1/clusters/{self.name}",
            json=changes,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()

    def delete(self) -> None:
        """Delete the cluster."""
        import requests
        try:
            requests.delete(
                f"{self.coordinator_endpoint}/api/v1/clusters/{self.name}",
                timeout=60,
            )
        except Exception:
            logger.debug("Failed to delete cluster: {}", self.name)
            pass


@dataclass
class FederationLinkResource:
    """A federation link between two DistLLM clusters.

    Maps to ``distllm_federation_link`` in Terraform HCL.
    """
    name: str
    source_cluster: str = ""
    target_cluster: str = ""
    target_endpoint: str = ""
    bandwidth_gbps: str = "100"
    encrypted: bool = True
    coordinator_endpoint: str = "http://localhost:8000"
    status: str = "pending"

    def create(self) -> dict[str, Any]:
        import requests
        payload = {
            "name": self.name,
            "source_cluster": self.source_cluster,
            "target_cluster": self.target_cluster,
            "target_endpoint": self.target_endpoint,
            "bandwidth_gbps": self.bandwidth_gbps,
            "encrypted": self.encrypted,
        }
        resp = requests.post(
            f"{self.coordinator_endpoint}/api/v1/federation",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def delete(self) -> None:
        import requests
        try:
            requests.delete(
                f"{self.coordinator_endpoint}/api/v1/federation/{self.name}",
                timeout=30,
            )
        except Exception:
            logger.debug("Failed to delete federation link: {}", self.name)
            pass


@dataclass
class TenantResource:
    """A tenant with SLO configuration.

    Maps to ``distllm_tenant`` in Terraform HCL.
    """
    tenant_id: str
    max_rpm: float = 60.0
    latency_slo_ms: float = 1000.0
    max_concurrent: int = 10
    priority_base: float = 1.0
    coordinator_endpoint: str = "http://localhost:8000"

    def create(self) -> dict[str, Any]:
        import requests
        payload = {
            "tenant_id": self.tenant_id,
            "max_rpm": self.max_rpm,
            "latency_slo_ms": self.latency_slo_ms,
            "max_concurrent": self.max_concurrent,
            "priority_base": self.priority_base,
        }
        resp = requests.post(
            f"{self.coordinator_endpoint}/api/v1/tenants",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def delete(self) -> None:
        import requests
        try:
            requests.delete(
                f"{self.coordinator_endpoint}/api/v1/tenants/{self.tenant_id}",
                timeout=30,
            )
        except Exception:
            logger.debug("Failed to delete tenant: {}", self.tenant_id)
            pass


@dataclass
class ModelDeploymentResource:
    """A model deployment on the DistLLM cluster.

    Maps to ``distllm_model_deployment`` in Terraform HCL.
    """
    model_name: str
    model_path: str = ""
    num_replicas: int = 1
    gpu_type: GPUType = GPUType.A100_80GB
    max_batch_size: int = 32
    quantization: str = "fp16"
    coordinator_endpoint: str = "http://localhost:8000"
    status: str = "pending"

    def create(self) -> dict[str, Any]:
        import requests
        payload = {
            "model_name": self.model_name,
            "model_path": self.model_path or self.model_name,
            "num_replicas": self.num_replicas,
            "gpu_type": self.gpu_type.value,
            "max_batch_size": self.max_batch_size,
            "quantization": self.quantization,
        }
        resp = requests.post(
            f"{self.coordinator_endpoint}/api/v1/models",
            json=payload,
            timeout=300,  # model loading can take minutes
        )
        resp.raise_for_status()
        return resp.json()

    def delete(self) -> None:
        import requests
        try:
            requests.delete(
                f"{self.coordinator_endpoint}/api/v1/models/{self.model_name}",
                timeout=300,
            )
        except Exception:
            logger.debug("Failed to delete model deployment: {}", self.model_name)
            pass
