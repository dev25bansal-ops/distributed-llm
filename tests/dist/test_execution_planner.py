"""Tests for execution_planner module.

Tests the public API surface of ExecutionPlan, ExecPlannerProtocol,
SimpleExecutionPlanner, and EdgeAwareExecutionPlanner using only real
objects from the module (zero mocks).
"""

import pytest
from distllm.dist.execution_planner import (
    ExecutionPlan,
    ExecPlannerProtocol,
    SimpleExecutionPlanner,
    EdgeAwareExecutionPlanner,
)

# ---------------------------------------------------------------------------
# Test fakes — minimal collaborators that match the required protocol
# ---------------------------------------------------------------------------


class FakePipeline:
    """Duck-typed pipeline for SimpleExecutionPlanner."""

    def run_pipeline(self, plan):
        return {"result": "ok", "plan_id": plan.request_id}


class FakeEdgeNode:
    def __init__(self, node_id: str):
        self.node_id = node_id


class FakeEdgeManager:
    """Duck-typed edge manager with get_online_nodes and route_inference."""

    def __init__(self, nodes=None):
        self._nodes = nodes or []

    def get_online_nodes(self):
        return self._nodes

    def route_inference(self, prompt, model_name):
        return {"result": "edge_result", "prompt": prompt, "model": model_name}


class FakeClusterPlanner(ExecPlannerProtocol):
    """Concrete cluster planner that complies with ExecPlannerProtocol."""

    def plan_execution(self, request, available_nodes):
        node_ids = [n.node_id for n in available_nodes]
        return ExecutionPlan(
            request_id=getattr(request, "request_id", ""),
            route=node_ids,
            transport="grpc",
        )

    def execute(self, plan):
        return {"result": "cluster_result", "plan_id": plan.request_id}


# ===================================================================
# ExecutionPlan
# ===================================================================


class TestExecutionPlan:
    """Dataclass that models an execution plan."""

    def test_default_values(self):
        plan = ExecutionPlan()
        assert plan.request_id == ""
        assert plan.node_assignments == []
        assert plan.route == []
        assert plan.fallback_strategy == "retry"
        assert plan.estimated_latency_ms == 0.0
        assert plan.transport == "grpc"
        assert plan.metadata == {}

    def test_custom_values(self):
        plan = ExecutionPlan(
            request_id="req-1",
            node_assignments=[("node1", "0-5", "grpc")],
            route=["node1", "node2"],
            fallback_strategy="abort",
            estimated_latency_ms=150.5,
            transport="webrtc",
            metadata={"model": "llama-7b"},
        )
        assert plan.request_id == "req-1"
        assert plan.node_assignments == [("node1", "0-5", "grpc")]
        assert plan.route == ["node1", "node2"]
        assert plan.fallback_strategy == "abort"
        assert plan.estimated_latency_ms == 150.5
        assert plan.transport == "webrtc"
        assert plan.metadata == {"model": "llama-7b"}

    def test_mutable_fields(self):
        """ExecutionPlan is not frozen — fields may be mutated in place."""
        plan = ExecutionPlan()
        plan.request_id = "req-2"
        plan.route.append("node1")
        plan.node_assignments.append(("n1", "0-3", "http"))
        plan.metadata["key"] = "value"
        assert plan.request_id == "req-2"
        assert plan.route == ["node1"]
        assert plan.node_assignments == [("n1", "0-3", "http")]
        assert plan.metadata == {"key": "value"}

    def test_empty_route(self):
        plan = ExecutionPlan(route=[])
        assert plan.route == []
        assert len(plan.route) == 0

    def test_empty_node_assignments(self):
        plan = ExecutionPlan(node_assignments=[])
        assert plan.node_assignments == []

    def test_negative_latency(self):
        """Latency may be negative (realistically invalid but unenforced)."""
        plan = ExecutionPlan(estimated_latency_ms=-1.0)
        assert plan.estimated_latency_ms == -1.0

    def test_strategy_not_retry(self):
        assert ExecutionPlan(fallback_strategy="skip").fallback_strategy == "skip"
        assert ExecutionPlan(fallback_strategy="abort").fallback_strategy == "abort"
        assert ExecutionPlan(fallback_strategy="unknown").fallback_strategy == "unknown"

    def test_with_all_empty_lists(self):
        plan = ExecutionPlan(node_assignments=[], route=[])
        assert plan.node_assignments == []
        assert plan.route == []


# ===================================================================
# ExecPlannerProtocol
# ===================================================================


class TestExecPlannerProtocol:
    """Abstract base protocol for execution planners."""

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            ExecPlannerProtocol()

    def test_concrete_subclass_is_instance(self):
        class MinimalPlanner(ExecPlannerProtocol):
            def plan_execution(self, request, available_nodes):
                return ExecutionPlan()

            def execute(self, plan):
                return None

        planner = MinimalPlanner()
        assert isinstance(planner, ExecPlannerProtocol)

    def test_missing_plan_execution_raises(self):
        with pytest.raises(TypeError):

            class BadPlanner(ExecPlannerProtocol):
                def execute(self, plan):
                    return None

            BadPlanner()  # type: ignore[abstract]

    def test_missing_execute_raises(self):
        with pytest.raises(TypeError):

            class BadPlanner(ExecPlannerProtocol):
                def plan_execution(self, request, available_nodes):
                    return ExecutionPlan()

            BadPlanner()  # type: ignore[abstract]


# ===================================================================
# SimpleExecutionPlanner
# ===================================================================


class TestSimpleExecutionPlanner:
    """Wraps a PipelineOrchestrator for standard pipeline execution."""

    def test_default_construction(self):
        planner = SimpleExecutionPlanner()
        assert planner._pipeline is None
        assert planner._transport == "grpc"

    def test_construction_with_pipeline(self):
        pipeline = FakePipeline()
        planner = SimpleExecutionPlanner(pipeline=pipeline)
        assert planner._pipeline is pipeline

    def test_plan_execution_with_no_nodes(self):
        planner = SimpleExecutionPlanner()

        class Req:
            request_id = "req-1"

        plan = planner.plan_execution(Req(), [])
        assert plan.request_id == "req-1"
        assert plan.route == []
        assert plan.fallback_strategy == "retry"
        assert plan.transport == "grpc"

    def test_plan_execution_with_string_nodes(self):
        planner = SimpleExecutionPlanner()
        plan = planner.plan_execution({}, ["node1", "node2"])
        assert plan.route == ["node1", "node2"]

    def test_plan_execution_request_without_id(self):
        planner = SimpleExecutionPlanner()
        plan = planner.plan_execution("raw_string", ["node1"])
        assert plan.request_id == ""
        assert plan.route == ["node1"]

    def test_plan_execution_with_node_objects(self):
        planner = SimpleExecutionPlanner()
        plan = planner.plan_execution({}, [FakeEdgeNode("n1"), FakeEdgeNode("n2")])
        assert plan.route == ["n1", "n2"]

    def test_execute_no_pipeline_raises(self):
        planner = SimpleExecutionPlanner()
        with pytest.raises(RuntimeError, match="No pipeline configured"):
            planner.execute(ExecutionPlan(request_id="r1"))

    def test_execute_with_pipeline(self):
        planner = SimpleExecutionPlanner(pipeline=FakePipeline())
        result = planner.execute(ExecutionPlan(request_id="r1"))
        assert result == {"result": "ok", "plan_id": "r1"}

    def test_execute_passthrough_plan(self):
        """Ensure the plan object is forwarded to pipeline.run_pipeline."""
        pipeline = FakePipeline()
        planner = SimpleExecutionPlanner(pipeline=pipeline)
        plan = ExecutionPlan(request_id="custom", metadata={"key": "val"})
        result = planner.execute(plan)
        assert result["plan_id"] == "custom"


# ===================================================================
# EdgeAwareExecutionPlanner
# ===================================================================


class TestEdgeAwareExecutionPlanner:
    """Routes small models to edge nodes, falls back to cluster."""

    def test_default_construction(self):
        planner = EdgeAwareExecutionPlanner()
        assert planner._edge is None
        assert planner._cluster is None

    def test_plan_small_model_to_edge(self):
        """model_params_b <= 3 should route to an edge node."""
        edge = FakeEdgeManager([FakeEdgeNode("edge1")])
        planner = EdgeAwareExecutionPlanner(edge_manager=edge)

        class Req:
            request_id = "req-1"
            model_params_b = 1

        plan = planner.plan_execution(Req(), [FakeEdgeNode("cluster1")])
        assert plan.route == ["edge1"]
        assert plan.transport == "webrtc"
        assert plan.fallback_strategy == "abort"

    def test_plan_small_model_model_params_boundary(self):
        """model_params_b == 3 is still small enough for edge."""
        edge = FakeEdgeManager([FakeEdgeNode("edge1")])
        planner = EdgeAwareExecutionPlanner(edge_manager=edge)

        class Req:
            request_id = "req-1"
            model_params_b = 3

        plan = planner.plan_execution(Req(), [])
        assert plan.route == ["edge1"]

    def test_plan_small_model_no_edge_nodes_falls_back(self):
        """Empty edge node list should trigger cluster fallback."""
        edge = FakeEdgeManager([])
        cluster = FakeClusterPlanner()
        planner = EdgeAwareExecutionPlanner(edge_manager=edge, cluster_planner=cluster)

        class Req:
            request_id = "req-1"
            model_params_b = 1

        available = [FakeEdgeNode("cluster1")]
        plan = planner.plan_execution(Req(), available)
        assert plan.transport == "grpc"
        assert plan.route == ["cluster1"]

    def test_plan_small_model_no_edge_manager_attribute(self):
        """Edge manager without get_online_nodes falls back."""
        class BareEdgeManager:
            pass

        cluster = FakeClusterPlanner()
        planner = EdgeAwareExecutionPlanner(
            edge_manager=BareEdgeManager(), cluster_planner=cluster
        )

        class Req:
            request_id = "req-1"
            model_params_b = 1

        available = [FakeEdgeNode("cluster1")]
        plan = planner.plan_execution(Req(), available)
        assert plan.transport == "grpc"
        assert plan.route == ["cluster1"]

    def test_plan_large_model_falls_back_to_cluster(self):
        """model_params_b > 3 should go to cluster."""
        cluster = FakeClusterPlanner()
        planner = EdgeAwareExecutionPlanner(
            edge_manager=FakeEdgeManager([FakeEdgeNode("edge1")]),
            cluster_planner=cluster,
        )

        class Req:
            request_id = "req-1"
            model_params_b = 70

        plan = planner.plan_execution(
            Req(), [FakeEdgeNode("c1"), FakeEdgeNode("c2")]
        )
        assert plan.transport == "grpc"
        assert plan.route == ["c1", "c2"]

    def test_plan_no_backend_returns_empty_route(self):
        """No edge manager and no cluster planner returns fallback plan."""
        planner = EdgeAwareExecutionPlanner()

        class Req:
            request_id = "req-1"
            model_params_b = 1

        plan = planner.plan_execution(Req(), [])
        assert plan.route == []
        assert plan.fallback_strategy == "abort"

    def test_plan_request_without_model_params_b(self):
        """Request without model_params_b defaults to 70 (cluster path)."""
        planner = EdgeAwareExecutionPlanner()

        class Req:
            request_id = "req-1"

        plan = planner.plan_execution(Req(), [])
        assert plan.route == []
        assert plan.fallback_strategy == "abort"

    def test_execute_webrtc_calls_edge_route_inference(self):
        edge = FakeEdgeManager([FakeEdgeNode("edge1")])
        planner = EdgeAwareExecutionPlanner(edge_manager=edge)

        plan = ExecutionPlan(transport="webrtc")
        result = planner.execute(plan)
        assert result == {"result": "edge_result", "prompt": "", "model": ""}

    def test_execute_webrtc_passes_prompt_and_model(self):
        edge = FakeEdgeManager([FakeEdgeNode("edge1")])
        planner = EdgeAwareExecutionPlanner(edge_manager=edge)

        plan = ExecutionPlan(transport="webrtc")
        # The execute method reads plan.prompt and plan.model_name via getattr
        plan.prompt = "hello world"  # type: ignore[attr-defined]
        plan.model_name = "tiny-llama"  # type: ignore[attr-defined]
        result = planner.execute(plan)
        assert result["prompt"] == "hello world"
        assert result["model"] == "tiny-llama"

    def test_execute_falls_back_to_cluster(self):
        cluster = FakeClusterPlanner()
        planner = EdgeAwareExecutionPlanner(
            edge_manager=FakeEdgeManager(), cluster_planner=cluster
        )

        plan = ExecutionPlan(transport="grpc", request_id="req-1")
        result = planner.execute(plan)
        assert result == {"result": "cluster_result", "plan_id": "req-1"}

    def test_execute_no_backend_raises(self):
        planner = EdgeAwareExecutionPlanner()
        with pytest.raises(RuntimeError, match="No execution backend available"):
            planner.execute(ExecutionPlan(transport="grpc"))

    def test_execute_webrtc_no_edge_route_inference_raises(self):
        """Edge manager without route_inference falls through to cluster (or raises)."""

        class BareEdgeManager:
            pass

        planner = EdgeAwareExecutionPlanner(edge_manager=BareEdgeManager())
        with pytest.raises(RuntimeError, match="No execution backend available"):
            planner.execute(ExecutionPlan(transport="webrtc"))

    def test_execute_webrtc_no_edge_attribute(self):
        """When self._edge is None, webrtc falls through to cluster or raises."""
        planner = EdgeAwareExecutionPlanner()
        with pytest.raises(RuntimeError, match="No execution backend available"):
            planner.execute(ExecutionPlan(transport="webrtc"))
