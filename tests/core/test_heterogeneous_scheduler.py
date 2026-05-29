"""Tests for heterogeneous scheduling and disaggregated prefill/decode routing."""

import importlib.util
import os
from dataclasses import dataclass
from typing import Any

import pytest


@dataclass
class StubDeviceInfo:
    device_type: str = "cuda"
    device_family: Any = None
    device_id: int = 0
    name: str = "stub"
    total_memory_bytes: int = 8 * 1024**3
    tflops_fp16: float = 100.0
    memory_bandwidth_gbps: float = 500.0


def _get_module():
    import sys
    import types
    import enum

    reg_mod = types.ModuleType("distllm.core.device_registry")
    reg_mod.DeviceInfo = StubDeviceInfo
    reg_mod.detect_all_devices = lambda: []
    sys.modules["distllm.core.device_registry"] = reg_mod

    const_mod = types.ModuleType("distllm.constants")
    DeviceFamily = enum.Enum("DeviceFamily", {"UNKNOWN": "unknown", "NVIDIA": "nvidia", "AMD": "amd"})
    const_mod.DeviceFamily = DeviceFamily
    const_mod.DEVICE_TO_FAMILY = {}
    const_mod.Device = enum.Enum("Device", {"UNKNOWN": "unknown"})
    sys.modules["distllm.constants"] = const_mod

    path = os.path.join("src", "distllm", "core", "heterogeneous_scheduler.py")
    spec = importlib.util.spec_from_file_location("heterogeneous_scheduler", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["heterogeneous_scheduler"] = mod

    mod.logger = types.ModuleType("logger")
    mod.logger.info = lambda *a, **kw: None
    mod.logger.warning = lambda *a, **kw: None

    spec.loader.exec_module(mod)
    return mod


class TestPrefillDecodeRouter:
    @classmethod
    def setup_class(cls):
        cls.mod = _get_module()
        cls.PrefillDecodeRouter = cls.mod.PrefillDecodeRouter
        cls.NodeRole = cls.mod.NodeRole

    def test_not_disaggregated_when_missing_pool(self):
        r = self.PrefillDecodeRouter(prefill_node_ids=["n1"], decode_node_ids=[])
        assert r.is_disaggregated is False

        r2 = self.PrefillDecodeRouter(prefill_node_ids=[], decode_node_ids=["n1"])
        assert r2.is_disaggregated is False

    def test_disaggregated_when_both_pools_exist(self):
        r = self.PrefillDecodeRouter(prefill_node_ids=["n1"], decode_node_ids=["n2"])
        assert r.is_disaggregated is True

    def test_route_prefill_to_prefill_node(self):
        r = self.PrefillDecodeRouter(prefill_node_ids=["prefill-1"], decode_node_ids=["decode-1"])
        route = r.route(is_prefill_step=True)
        assert route.node_id == "prefill-1"
        assert route.is_prefill is True
        assert route.kv_transfer_required is False

    def test_route_decode_to_decode_node(self):
        r = self.PrefillDecodeRouter(prefill_node_ids=["prefill-1"], decode_node_ids=["decode-1"])
        r.record_prefill_node("req-1", "prefill-1")
        route = r.route(is_prefill_step=False, request_id="req-1")
        assert route.node_id == "decode-1"
        assert route.is_prefill is False
        assert route.kv_transfer_required is True
        assert route.source_node_id == "prefill-1"

    def test_route_decode_no_transfer_without_record(self):
        r = self.PrefillDecodeRouter(prefill_node_ids=["prefill-1"], decode_node_ids=["decode-1"])
        route = r.route(is_prefill_step=False)
        assert route.kv_transfer_required is False

    def test_round_robin_prefill(self):
        r = self.PrefillDecodeRouter(prefill_node_ids=["p1", "p2"], decode_node_ids=["d1"])
        assert r.route(is_prefill_step=True).node_id == "p1"
        assert r.route(is_prefill_step=True).node_id == "p2"
        assert r.route(is_prefill_step=True).node_id == "p1"

    def test_round_robin_decode(self):
        r = self.PrefillDecodeRouter(prefill_node_ids=["p1"], decode_node_ids=["d1", "d2"])
        r.record_prefill_node("r1", "p1")
        r.record_prefill_node("r2", "p1")
        assert r.route(is_prefill_step=False, request_id="r1").node_id == "d1"
        assert r.route(is_prefill_step=False, request_id="r2").node_id == "d2"

    def test_auto_nodes_as_fallback(self):
        r = self.PrefillDecodeRouter(prefill_node_ids=[], decode_node_ids=[], auto_node_ids=["a1"])
        route = r.route(is_prefill_step=True)
        assert route.node_id == "a1"

    def test_cleanup_removes_kv_source(self):
        r = self.PrefillDecodeRouter(prefill_node_ids=["p1"], decode_node_ids=["d1"])
        r.record_prefill_node("req-1", "p1")
        assert "req-1" in r._kv_sources
        r.cleanup_request("req-1")
        assert "req-1" not in r._kv_sources

    def test_from_node_roles(self):
        roles = {"n1": self.NodeRole.PREFILL, "n2": self.NodeRole.DECODE, "n3": self.NodeRole.AUTO}
        r = self.PrefillDecodeRouter.from_node_roles(roles)
        assert r._prefill_nodes == ["n1"]
        assert r._decode_nodes == ["n2"]
        assert r._auto_nodes == ["n3"]

    def test_route_non_disaggregated_returns_first_node(self):
        r = self.PrefillDecodeRouter(prefill_node_ids=["only-node"], decode_node_ids=[])
        route = r.route(is_prefill_step=True)
        assert route.node_id == "only-node"
        route2 = r.route(is_prefill_step=False)
        assert route2.node_id == "only-node"


class TestAssignPrefillDecodeRoles:
    @classmethod
    def setup_class(cls):
        cls.mod = _get_module()
        cls.NodeRole = cls.mod.NodeRole
        cls.HeterogeneousCluster = cls.mod.HeterogeneousCluster
        cls.HeterogeneousNode = cls.mod.HeterogeneousNode
        cls.StubDeviceInfo = cls.mod.DeviceInfo

    def test_single_node_returns_auto(self):
        cluster = self.HeterogeneousCluster(nodes=[
            self.HeterogeneousNode(node_id="n1", host="h1", port=1, device_info=self.StubDeviceInfo()),
        ])
        roles = self.mod.assign_prefill_decode_roles(cluster)
        assert len(roles) == 1
        assert roles["n1"] in (self.NodeRole.PREFILL, self.NodeRole.DECODE, self.NodeRole.AUTO)

    def test_two_equal_nodes_get_split(self):
        cluster = self.HeterogeneousCluster(nodes=[
            self.HeterogeneousNode(node_id="n1", host="h1", port=1, device_info=self.StubDeviceInfo(tflops_fp16=100, memory_bandwidth_gbps=500)),
            self.HeterogeneousNode(node_id="n2", host="h2", port=2, device_info=self.StubDeviceInfo(tflops_fp16=100, memory_bandwidth_gbps=500)),
        ])
        roles = self.mod.assign_prefill_decode_roles(cluster)
        assert len(roles) == 2
        assert roles["n1"] == self.NodeRole.PREFILL
        assert roles["n2"] == self.NodeRole.DECODE

    def test_compute_heavy_gets_prefill(self):
        cluster = self.HeterogeneousCluster(nodes=[
            self.HeterogeneousNode(node_id="compute", host="h1", port=1, device_info=self.StubDeviceInfo(tflops_fp16=300, memory_bandwidth_gbps=200)),
            self.HeterogeneousNode(node_id="bandwidth", host="h2", port=2, device_info=self.StubDeviceInfo(tflops_fp16=50, memory_bandwidth_gbps=900)),
        ])
        roles = self.mod.assign_prefill_decode_roles(cluster)
        assert roles["compute"] == self.NodeRole.PREFILL
        assert roles["bandwidth"] == self.NodeRole.DECODE

    def test_empty_cluster(self):
        cluster = self.HeterogeneousCluster(nodes=[])
        roles = self.mod.assign_prefill_decode_roles(cluster)
        assert roles == {}

    def test_all_auto_when_ambiguous(self):
        cluster = self.HeterogeneousCluster(nodes=[
            self.HeterogeneousNode(node_id="n1", host="h1", port=1, device_info=self.StubDeviceInfo(tflops_fp16=100, memory_bandwidth_gbps=500)),
            self.HeterogeneousNode(node_id="n2", host="h2", port=2, device_info=self.StubDeviceInfo(tflops_fp16=100, memory_bandwidth_gbps=500)),
            self.HeterogeneousNode(node_id="n3", host="h3", port=3, device_info=self.StubDeviceInfo(tflops_fp16=100, memory_bandwidth_gbps=500)),
        ])
        roles = self.mod.assign_prefill_decode_roles(cluster)
        assert len(roles) == 3
        assert roles["n1"] == self.NodeRole.PREFILL
        assert roles["n2"] == self.NodeRole.DECODE
        assert roles["n3"] == self.NodeRole.DECODE


class TestBuildDisaggregatedPlan:
    @classmethod
    def setup_class(cls):
        cls.mod = _get_module()

    def test_returns_plan_dict(self):
        plan = self.mod.build_disaggregated_pipeline_plan(
            node_configs=[{"node_id": "n1", "host": "h1", "port": 1, "device_type": "cuda"}],
            total_layers=32,
        )
        assert "roles" in plan
        assert "layer_assignments" in plan
        assert "router" in plan
        assert "is_disaggregated" in plan

    def test_layer_assignments_have_required_keys(self):
        plan = self.mod.build_disaggregated_pipeline_plan(
            node_configs=[{"node_id": "n1", "host": "h1", "port": 1, "device_type": "cuda"}],
            total_layers=20,
        )
        for a in plan["layer_assignments"]:
            assert "node_id" in a
            assert "start_layer" in a
            assert "end_layer" in a
            assert "role" in a
            assert "host" in a
            assert "port" in a

    def test_multinode_plan(self):
        plan = self.mod.build_disaggregated_pipeline_plan(
            node_configs=[
                {"node_id": "n1", "host": "h1", "port": 1, "device_type": "cuda"},
                {"node_id": "n2", "host": "h2", "port": 2, "device_type": "cuda"},
                {"node_id": "n3", "host": "h3", "port": 3, "device_type": "cuda"},
            ],
            total_layers=40,
        )
        assert len(plan["layer_assignments"]) == 3
        total_assigned = sum(
            a["end_layer"] - a["start_layer"] + 1 for a in plan["layer_assignments"]
        )
        assert total_assigned <= 40


def test_module_exports():
    mod = _get_module()
    assert hasattr(mod, "PrefillDecodeRouter")
    assert hasattr(mod, "NodeRole")
    assert hasattr(mod, "assign_prefill_decode_roles")
    assert hasattr(mod, "build_disaggregated_pipeline_plan")
