"""Tests: Operator — CRD reconciliation, cluster status, NodePool reconciliation."""

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("distllm.operator", reason="distllm.operator package not installed")

from distllm.operator.crds.distributed_llm_cluster import (
    DistributedLLMClusterSpec, ModelSpec, CoordinatorSpec, ResourceSpec, NodePoolSpec,
)
from distllm.operator.gpu_scheduler import (
    build_gpu_node_selector, build_gpu_affinity, build_pod_anti_affinity,
    build_tensor_parallel_affinity, build_gpu_tolerations,
    select_optimal_node_vram, estimate_gpu_requirements,
)


# ===========================================================================
# CRD Dataclasses
# ===========================================================================


class TestCRDDataclasses:
    def test_model_spec_required(self):
        s = ModelSpec(name="test", layers=32)
        assert s.name == "test"
        assert s.layers == 32

    def test_coordinator_spec_defaults(self):
        s = CoordinatorSpec()
        assert s.replicas == 1
        assert s.port == 8000
        assert s.grpc_port == 50050

    def test_node_pool_spec_defaults(self):
        s = NodePoolSpec()
        assert s.replicas == 1
        assert s.grpc_port == 50051

    def test_resource_spec_defaults(self):
        s = ResourceSpec()
        assert s.gpu == "1"
        assert s.memory == "32Gi"

    def test_cluster_spec_defaults(self):
        model = ModelSpec(name="m", layers=32)
        s = DistributedLLMClusterSpec(model=model)
        assert s.tls_enabled is False
        assert s.namespace == "default"


# ===========================================================================
# GPU Scheduler (no kopf dependency)
# ===========================================================================


class TestGPUScheduler:
    def test_build_node_selector_defaults(self):
        sel = build_gpu_node_selector()
        assert "nvidia.com/gpu.count" in sel

    def test_build_node_selector_with_product(self):
        sel = build_gpu_node_selector(gpu_product="A100")
        assert "nvidia.com/gpu.product" in sel

    def test_build_node_selector_count_default(self):
        sel = build_gpu_node_selector()
        assert sel.get("nvidia.com/gpu.count", "0") >= "1"

    def test_build_gpu_affinity(self):
        aff = build_gpu_affinity()
        assert "nodeAffinity" in aff

    def test_build_pod_anti_affinity(self):
        aff = build_pod_anti_affinity()
        assert "podAntiAffinity" in aff

    def test_tensor_parallel_affinity(self):
        aff = build_tensor_parallel_affinity(tp_size=2)
        assert "podAffinity" in aff

    def test_build_gpu_tolerations(self):
        tols = build_gpu_tolerations()
        assert len(tols) >= 1
        assert tols[0]["key"] == "nvidia.com/gpu"

    def test_select_optimal_node_no_match(self):
        nodes = [
            {"name": "small", "allocatable": {"nvidia.com/gpu.memory": "10000"}},
        ]
        chosen = select_optimal_node_vram(required_vram_gb=40, nodes=nodes)
        assert chosen is None

    def test_estimate_gpu_requirements(self):
        reqs = estimate_gpu_requirements(model_size_params_b=7, quantization_bits=16)
        assert reqs["vram_gb"] > 0
        assert reqs["gpu_count"] >= 1
