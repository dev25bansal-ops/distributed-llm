"""Tests for distllm.dist.partition.quant_coordinator — zero mocks, no GPU required.

Uses only real objects from the module and its dependencies
(QuantizationAutoTuner, ReportGenerator, etc.).
"""

from __future__ import annotations

from dataclasses import asdict

import pytest

from distllm.dist.partition.quant_coordinator import (
    CoordinatorState,
    NodeProfile,
    NodeQuantAssignment,
    QuantizationCoordinator,
)


# ---------------------------------------------------------------------------
# NodeProfile
# ---------------------------------------------------------------------------


class TestNodeProfile:
    """Dataclass for GPU profile collected from a remote node."""

    def test_defaults(self) -> None:
        p = NodeProfile(node_id="node-1")
        assert p.node_id == "node-1"
        assert p.gpu_name == ""
        assert p.total_memory_bytes == 0
        assert p.compute_tflops == 0.0
        assert p.bandwidth_gbps == 0.0
        assert p.compute_capability == 0.0
        assert p.is_hopper_or_newer is False
        assert p.status == "online"
        assert p.last_heartbeat == 0.0
        assert p.error == ""

    def test_all_fields_explicit(self) -> None:
        p = NodeProfile(
            node_id="node-x",
            gpu_name="H100",
            total_memory_bytes=80 * 1024**3,
            compute_tflops=989.0,
            bandwidth_gbps=3350.0,
            compute_capability=9.0,
            is_hopper_or_newer=True,
            status="online",
            last_heartbeat=1000.0,
            error="",
        )
        assert p.node_id == "node-x"
        assert p.gpu_name == "H100"
        assert p.total_memory_bytes == 80 * 1024**3
        assert p.compute_tflops == 989.0
        assert p.bandwidth_gbps == 3350.0
        assert p.compute_capability == 9.0
        assert p.is_hopper_or_newer is True
        assert p.status == "online"
        assert p.last_heartbeat == 1000.0

    def test_edge_empty_node_id(self) -> None:
        p = NodeProfile(node_id="")
        assert p.node_id == ""

    def test_edge_negative_memory(self) -> None:
        p = NodeProfile(node_id="n1", total_memory_bytes=-1)
        assert p.total_memory_bytes == -1

    def test_edge_offline_status(self) -> None:
        p = NodeProfile(node_id="n1", status="offline")
        assert p.status == "offline"

    def test_is_dataclass(self) -> None:
        p = NodeProfile(node_id="n1", gpu_name="A100")
        d = asdict(p)
        assert isinstance(d, dict)
        assert d["node_id"] == "n1"
        assert d["gpu_name"] == "A100"

    def test_error_string(self) -> None:
        p = NodeProfile(node_id="n1", error="CUDA OOM")
        assert p.error == "CUDA OOM"


# ---------------------------------------------------------------------------
# NodeQuantAssignment
# ---------------------------------------------------------------------------


class TestNodeQuantAssignment:
    """Dataclass for quantization assignment sent to a node."""

    def test_defaults(self) -> None:
        a = NodeQuantAssignment(node_id="node-1", quant_method="bnb_8bit")
        assert a.node_id == "node-1"
        assert a.quant_method == "bnb_8bit"
        assert a.activation_quant == "none"
        assert a.kv_cache_bits == "none"
        assert a.max_quality_loss == 0.05
        assert a.mixed_precision_plan == {}

    def test_all_fields_explicit(self) -> None:
        a = NodeQuantAssignment(
            node_id="n1",
            quant_method="fp8_e4m3",
            activation_quant="fp8_e4m3",
            kv_cache_bits="fp8",
            max_quality_loss=0.01,
            mixed_precision_plan={"num_layers": 16},
        )
        assert a.quant_method == "fp8_e4m3"
        assert a.activation_quant == "fp8_e4m3"
        assert a.kv_cache_bits == "fp8"
        assert a.max_quality_loss == 0.01
        assert a.mixed_precision_plan == {"num_layers": 16}

    def test_edge_none_method(self) -> None:
        a = NodeQuantAssignment(node_id="n1", quant_method="none")
        assert a.quant_method == "none"

    def test_edge_zero_quality_loss(self) -> None:
        a = NodeQuantAssignment(node_id="n1", quant_method="none", max_quality_loss=0.0)
        assert a.max_quality_loss == 0.0

    def test_is_dataclass(self) -> None:
        a = NodeQuantAssignment(node_id="n1", quant_method="int8")
        d = asdict(a)
        assert isinstance(d, dict)
        assert d["quant_method"] == "int8"


# ---------------------------------------------------------------------------
# CoordinatorState
# ---------------------------------------------------------------------------


class TestCoordinatorState:
    """Full coordinator state dataclass."""

    def test_defaults(self) -> None:
        s = CoordinatorState()
        assert s.nodes == {}
        assert s.assignments == {}
        assert s.plan_json == ""
        assert s.last_update == 0.0
        assert s.model_name == ""
        assert s.model_size_bytes == 0
        assert s.num_layers == 0

    def test_with_initial_values(self) -> None:
        s = CoordinatorState(
            model_name="llama-7b",
            model_size_bytes=14 * 1024**3,
            num_layers=32,
        )
        assert s.model_name == "llama-7b"
        assert s.model_size_bytes == 14 * 1024**3
        assert s.num_layers == 32

    def test_to_dict_empty(self) -> None:
        s = CoordinatorState()
        d = s.to_dict()
        assert d["nodes"] == {}
        assert d["assignments"] == {}
        assert d["model_name"] == ""
        assert d["model_size_bytes"] == 0
        assert d["num_layers"] == 0
        assert d["last_update"] == 0.0

    def test_to_dict_with_nodes(self) -> None:
        s = CoordinatorState(
            model_name="test-model",
            model_size_bytes=1000,
            num_layers=4,
        )
        s.nodes["n1"] = NodeProfile(
            node_id="n1",
            gpu_name="A100",
            total_memory_bytes=80 * 1024**3,
            compute_tflops=312.0,
            status="online",
        )
        s.nodes["n2"] = NodeProfile(
            node_id="n2",
            gpu_name="H100",
            total_memory_bytes=80 * 1024**3,
            compute_tflops=989.0,
            status="online",
        )
        d = s.to_dict()
        assert "n1" in d["nodes"]
        assert "n2" in d["nodes"]
        assert d["nodes"]["n1"]["gpu_name"] == "A100"
        assert d["nodes"]["n2"]["compute_tflops"] == 989.0
        assert d["model_name"] == "test-model"

    def test_to_dict_with_assignments(self) -> None:
        s = CoordinatorState()
        s.assignments["n1"] = NodeQuantAssignment(
            node_id="n1", quant_method="int8"
        )
        d = s.to_dict()
        assert "n1" in d["assignments"]
        assert d["assignments"]["n1"]["quant_method"] == "int8"

    def test_to_dict_omits_optional_profile_fields(self) -> None:
        """to_dict only includes selected fields from NodeProfile."""
        s = CoordinatorState()
        s.nodes["n1"] = NodeProfile(
            node_id="n1",
            gpu_name="A100",
            total_memory_bytes=80 * 1024**3,
            compute_tflops=312.0,
            bandwidth_gbps=2000.0,
            compute_capability=8.0,
            is_hopper_or_newer=False,
            status="online",
            last_heartbeat=42.0,
            error="",
        )
        d = s.to_dict()
        node_dict = d["nodes"]["n1"]
        # Included fields
        assert node_dict["gpu_name"] == "A100"
        assert node_dict["total_memory_bytes"] == 80 * 1024**3
        assert node_dict["compute_tflops"] == 312.0
        assert node_dict["status"] == "online"
        # Excluded fields
        assert "bandwidth_gbps" not in node_dict
        assert "compute_capability" not in node_dict
        assert "is_hopper_or_newer" not in node_dict
        assert "last_heartbeat" not in node_dict
        assert "error" not in node_dict

    def test_is_dataclass(self) -> None:
        s = CoordinatorState()
        d = asdict(s)
        assert isinstance(d, dict)
        assert d["plan_json"] == ""


# ---------------------------------------------------------------------------
# QuantizationCoordinator
# ---------------------------------------------------------------------------


class TestQuantizationCoordinator:
    """Distributed quantization coordinator — full lifecycle tests."""

    # -- Construction -------------------------------------------------------

    def test_default_construction(self) -> None:
        c = QuantizationCoordinator()
        assert c._model_name == ""  # noqa: SLF001
        assert c._model_size_bytes == 0  # noqa: SLF001
        assert c._num_layers == 32  # noqa: SLF001
        assert c._max_quality_loss == 0.05  # noqa: SLF001
        assert c._prefer_speed is False  # noqa: SLF001
        assert c._require_calibration is False  # noqa: SLF001
        assert c._fallback_count == {}  # noqa: SLF001

    def test_construction_with_params(self) -> None:
        c = QuantizationCoordinator(
            model_name="llama2-7b",
            model_size_bytes=14 * 1024**3,
            num_layers=32,
            max_quality_loss=0.02,
            prefer_speed=True,
            require_calibration=True,
        )
        assert c._model_name == "llama2-7b"  # noqa: SLF001
        assert c._model_size_bytes == 14 * 1024**3  # noqa: SLF001
        assert c._num_layers == 32  # noqa: SLF001
        assert c._max_quality_loss == 0.02  # noqa: SLF001
        assert c._prefer_speed is True  # noqa: SLF001
        assert c._require_calibration is True  # noqa: SLF001

    def test_construction_state_reflects_params(self) -> None:
        c = QuantizationCoordinator(
            model_name="test", model_size_bytes=5000, num_layers=8,
        )
        state = c.get_state()
        assert state.model_name == "test"
        assert state.model_size_bytes == 5000
        assert state.num_layers == 8

    # -- register_node ------------------------------------------------------

    def test_register_node(self) -> None:
        c = QuantizationCoordinator()
        p = NodeProfile(
            node_id="node-1",
            gpu_name="A100",
            total_memory_bytes=80 * 1024**3,
            compute_tflops=312.0,
            bandwidth_gbps=2000.0,
            compute_capability=8.0,
        )
        c.register_node(p)

        state = c.get_state()
        assert "node-1" in state.nodes
        registered = state.nodes["node-1"]
        assert registered.gpu_name == "A100"
        assert registered.total_memory_bytes == 80 * 1024**3
        assert registered.status == "online"
        assert registered.last_heartbeat > 0
        assert state.last_update > 0

    def test_register_multiple_nodes(self) -> None:
        c = QuantizationCoordinator()
        c.register_node(NodeProfile(node_id="n1"))
        c.register_node(NodeProfile(node_id="n2"))
        state = c.get_state()
        assert len(state.nodes) == 2
        assert "n1" in state.nodes
        assert "n2" in state.nodes

    def test_register_duplicate_node_updates_profile(self) -> None:
        c = QuantizationCoordinator()
        c.register_node(NodeProfile(
            node_id="n1", gpu_name="A100", total_memory_bytes=80 * 1024**3,
        ))
        c.register_node(NodeProfile(
            node_id="n1", gpu_name="H100", total_memory_bytes=80 * 1024**3,
        ))
        state = c.get_state()
        assert len(state.nodes) == 1
        assert state.nodes["n1"].gpu_name == "H100"

    def test_register_updates_heartbeat_and_timestamp(self) -> None:
        c = QuantizationCoordinator()
        early = c.get_state().last_update
        c.register_node(NodeProfile(node_id="n1"))
        mid = c.get_state().last_update
        c.register_node(NodeProfile(node_id="n1"))
        late = c.get_state().last_update

        assert early <= mid <= late
        assert c.get_state().nodes["n1"].last_heartbeat > 0

    # -- unregister_node ----------------------------------------------------

    def test_unregister_existing_node(self) -> None:
        c = QuantizationCoordinator()
        c.register_node(NodeProfile(node_id="n1"))
        c.unregister_node("n1")
        assert c.get_state().nodes["n1"].status == "offline"

    def test_unregister_nonexistent_node_does_not_raise(self) -> None:
        c = QuantizationCoordinator()
        c.unregister_node("ghost")  # should not raise

    def test_unregister_empty_string_does_not_raise(self) -> None:
        c = QuantizationCoordinator()
        c.unregister_node("")  # should not raise

    def test_unregister_then_register_brings_back_online(self) -> None:
        c = QuantizationCoordinator()
        c.register_node(NodeProfile(node_id="n1"))
        c.unregister_node("n1")
        assert c.get_state().nodes["n1"].status == "offline"
        c.register_node(NodeProfile(node_id="n1"))
        assert c.get_state().nodes["n1"].status == "online"

    # -- status -------------------------------------------------------------

    def test_status_initial(self) -> None:
        c = QuantizationCoordinator(model_name="m", model_size_bytes=2 * 1024**3)
        s = c.status()
        assert s["model"] == "m"
        assert s["model_size_gb"] == 2.0
        assert s["num_layers"] == 32
        assert s["nodes_online"] == 0
        assert s["nodes_total"] == 0
        assert s["nodes_assigned"] == 0
        assert s["max_quality_loss"] == 0.05
        assert s["prefer_speed"] is False
        assert s["last_update"] == 0.0

    def test_status_after_registration(self) -> None:
        c = QuantizationCoordinator()
        c.register_node(NodeProfile(node_id="n1"))
        s = c.status()
        assert s["nodes_online"] == 1
        assert s["nodes_total"] == 1

    def test_status_after_unregister(self) -> None:
        c = QuantizationCoordinator()
        c.register_node(NodeProfile(node_id="n1"))
        c.register_node(NodeProfile(node_id="n2"))
        c.unregister_node("n1")
        s = c.status()
        assert s["nodes_online"] == 1
        assert s["nodes_total"] == 2

    def test_status_prefer_speed_flag(self) -> None:
        c = QuantizationCoordinator(prefer_speed=True)
        assert c.status()["prefer_speed"] is True

    # -- get_assignment -----------------------------------------------------

    def test_get_assignment_none_when_no_plan(self) -> None:
        c = QuantizationCoordinator()
        assert c.get_assignment("any") is None

    def test_get_assignment_after_generate_plan(self) -> None:
        c = QuantizationCoordinator(
            model_size_bytes=1 * 1024**3,  # 1 GB
            num_layers=8,
        )
        c.register_node(NodeProfile(
            node_id="n1",
            gpu_name="A100",
            total_memory_bytes=80 * 1024**3,
            compute_capability=8.0,
            compute_tflops=312.0,
            bandwidth_gbps=2000.0,
        ))
        c.generate_plan()
        a = c.get_assignment("n1")
        assert a is not None
        assert isinstance(a, NodeQuantAssignment)
        assert a.node_id == "n1"

    def test_get_assignment_nonexistent_after_plan(self) -> None:
        c = QuantizationCoordinator(model_size_bytes=1 * 1024**3)
        c.register_node(NodeProfile(
            node_id="n1", gpu_name="A100", total_memory_bytes=80 * 1024**3,
        ))
        c.generate_plan()
        assert c.get_assignment("ghost") is None

    # -- get_state ----------------------------------------------------------

    def test_get_state_returns_coordinator_state(self) -> None:
        c = QuantizationCoordinator()
        state = c.get_state()
        assert isinstance(state, CoordinatorState)

    def test_get_state_is_reflective(self) -> None:
        c = QuantizationCoordinator(model_name="test", model_size_bytes=999)
        c.register_node(NodeProfile(node_id="n1"))
        state = c.get_state()
        assert state.model_name == "test"
        assert state.model_size_bytes == 999
        assert "n1" in state.nodes

    # -- generate_plan ------------------------------------------------------

    def test_generate_plan_no_nodes_returns_error(self) -> None:
        c = QuantizationCoordinator(model_size_bytes=1 * 1024**3)
        result = c.generate_plan()
        assert result == {"error": "No online nodes", "assignments": {}}

    def test_generate_plan_all_offline_returns_error(self) -> None:
        c = QuantizationCoordinator(model_size_bytes=1 * 1024**3)
        c.register_node(NodeProfile(node_id="n1"))
        c.unregister_node("n1")
        result = c.generate_plan()
        assert result["error"] == "No online nodes"

    def test_generate_plan_single_node(self) -> None:
        c = QuantizationCoordinator(model_size_bytes=1 * 1024**3, num_layers=8)
        c.register_node(NodeProfile(
            node_id="n1",
            gpu_name="A100",
            total_memory_bytes=80 * 1024**3,
            compute_capability=8.0,
            compute_tflops=312.0,
            bandwidth_gbps=2000.0,
        ))
        result = c.generate_plan()

        assert "plan" in result
        assert "assignments" in result
        assert "report" in result

        assignments = result["assignments"]
        assert "n1" in assignments
        assert assignments["n1"]["quant_method"] == "none"

        plan = result["plan"]
        assert isinstance(plan, dict)
        assert plan["total_memory_saved_bytes"] == 0

        report = result["report"]
        assert isinstance(report, dict)
        assert "No quantization needed" in report["strategy"]

    def test_generate_plan_updates_state(self) -> None:
        c = QuantizationCoordinator(model_size_bytes=2 * 1024**3)
        c.register_node(NodeProfile(
            node_id="n1",
            gpu_name="A100",
            total_memory_bytes=80 * 1024**3,
            compute_capability=8.0,
        ))
        c.generate_plan()
        state = c.get_state()
        assert "n1" in state.assignments
        assert state.plan_json != ""

    def test_generate_plan_multiple_nodes(self) -> None:
        c = QuantizationCoordinator(model_size_bytes=2 * 1024**3, num_layers=8)
        c.register_node(NodeProfile(
            node_id="n1",
            gpu_name="A100",
            total_memory_bytes=80 * 1024**3,
            compute_capability=8.0,
        ))
        c.register_node(NodeProfile(
            node_id="n2",
            gpu_name="H100",
            total_memory_bytes=80 * 1024**3,
            compute_capability=9.0,
            is_hopper_or_newer=True,
        ))
        result = c.generate_plan()
        assert "n1" in result["assignments"]
        assert "n2" in result["assignments"]

    def test_generate_plan_idempotent(self) -> None:
        c = QuantizationCoordinator(model_size_bytes=1 * 1024**3, num_layers=4)
        c.register_node(NodeProfile(
            node_id="n1",
            gpu_name="A100",
            total_memory_bytes=80 * 1024**3,
            compute_capability=8.0,
        ))
        r1 = c.generate_plan()
        r2 = c.generate_plan()
        # Both calls produce identical assignments
        assert r1["assignments"] == r2["assignments"]

    def test_generate_plan_with_tight_vram_selects_quantization(self) -> None:
        """When model barely fits, a quant method should be selected."""
        # Use an extremely small VRAM so quantization is forced
        c = QuantizationCoordinator(
            model_size_bytes=4 * 1024**3,  # 4 GB model
            num_layers=16,
        )
        c.register_node(NodeProfile(
            node_id="n1",
            gpu_name="A100",
            total_memory_bytes=2 * 1024**3,  # only 2 GB VRAM
            compute_capability=8.0,
        ))
        result = c.generate_plan()
        assignments = result["assignments"]
        assert "n1" in assignments
        # n1 should get a quantized method (not "none") since model > VRAM
        assert assignments["n1"]["quant_method"] != "none"

    # -- report_failure -----------------------------------------------------

    def test_report_failure_increments_counter(self) -> None:
        c = QuantizationCoordinator(model_size_bytes=2 * 1024**3)
        c.register_node(NodeProfile(
            node_id="n1",
            gpu_name="A100",
            total_memory_bytes=80 * 1024**3,
            compute_capability=8.0,
        ))
        c.generate_plan()
        c.report_failure("n1", "int8", "OOM")
        assert c._fallback_count["n1"] == 1  # noqa: SLF001

    def test_report_failure_reduces_quality_loss(self) -> None:
        c = QuantizationCoordinator(model_size_bytes=2 * 1024**3)
        c.register_node(NodeProfile(
            node_id="n1",
            gpu_name="A100",
            total_memory_bytes=80 * 1024**3,
            compute_capability=8.0,
        ))
        c.generate_plan()
        before = c._max_quality_loss  # noqa: SLF001
        c.report_failure("n1", "int8", "error")
        after = c._max_quality_loss  # noqa: SLF001
        # quality_loss is restored after report_failure
        assert after == before

    def test_report_failure_returns_generated_plan(self) -> None:
        c = QuantizationCoordinator(model_size_bytes=2 * 1024**3)
        c.register_node(NodeProfile(
            node_id="n1",
            gpu_name="A100",
            total_memory_bytes=80 * 1024**3,
            compute_capability=8.0,
        ))
        c.generate_plan()
        result = c.report_failure("n1", "int8", "error")
        assert "plan" in result
        assert "assignments" in result

    def test_report_failure_thrice_marks_node_offline(self) -> None:
        c = QuantizationCoordinator(model_size_bytes=2 * 1024**3)
        c.register_node(NodeProfile(
            node_id="n1",
            gpu_name="A100",
            total_memory_bytes=80 * 1024**3,
            compute_capability=8.0,
        ))
        c.generate_plan()
        # Fallback #1
        c.report_failure("n1", "method", "err")
        # Fallback #2
        c.report_failure("n1", "method", "err")
        # Fallback #3: marks offline, returns error-plan
        result = c.report_failure("n1", "method", "err")
        assert c._fallback_count["n1"] == 3  # noqa: SLF001
        assert c.get_state().nodes["n1"].status == "offline"
        # After the node is marked offline, generate_plan returns error
        assert "error" in result

    def test_report_failure_missing_node_still_increments(self) -> None:
        """report_failure works even if the node was never registered."""
        c = QuantizationCoordinator(model_size_bytes=2 * 1024**3)
        result = c.report_failure("ghost", "method", "err")
        assert c._fallback_count["ghost"] == 1  # noqa: SLF001
        # Without online nodes, returns error
        assert "error" in result
