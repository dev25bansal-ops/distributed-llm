"""Execution Planner — unified abstraction for distributed execution paths.

Provides a common protocol (ExecPlannerProtocol) that all distributed
execution engines implement. This replaces the fragmented interfaces of:

- PipelineOrchestrator
- WideAreaPipeline
- DisaggregatedBatchScheduler
- EdgeFederationManager

Usage:
    planner = SimpleExecutionPlanner(pipeline=orchestrator)
    plan = planner.plan_execution(request, available_nodes)
    result = planner.execute(plan)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExecutionPlan:
    """A plan for executing an inference request across available nodes.

    Attributes:
        node_assignments: List of (node_id, layer_range, transport) tuples.
        route: Ordered list of node IDs for the execution path.
        fallback_strategy: How to handle node failures ('retry', 'skip', 'abort').
        estimated_latency_ms: Predicted end-to-end latency.
        transport: Transport protocol ('grpc', 'webrtc', 'http').
    """
    request_id: str = ""
    node_assignments: list[tuple[str, str, str]] = field(default_factory=list)
    route: list[str] = field(default_factory=list)
    fallback_strategy: str = "retry"
    estimated_latency_ms: float = 0.0
    transport: str = "grpc"
    metadata: dict[str, Any] = field(default_factory=dict)


class ExecPlannerProtocol(ABC):
    """Abstract protocol for execution planners.

    Implementors must provide plan_execution() and execute().
    Callers depend on this interface, not on concrete implementations.
    """

    @abstractmethod
    def plan_execution(
        self,
        request: Any,
        available_nodes: list[Any],
    ) -> ExecutionPlan:
        """Create an execution plan for a request.

        Args:
            request: The inference request (prompt, model, parameters).
            available_nodes: Nodes available for execution.

        Returns:
            ExecutionPlan describing how to route and execute.
        """
        ...

    @abstractmethod
    def execute(self, plan: ExecutionPlan) -> Any:
        """Execute a plan and return the result.

        Args:
            plan: The ExecutionPlan to execute.

        Returns:
            Inference result (generated text, embeddings, etc.).
        """
        ...


class SimpleExecutionPlanner(ExecPlannerProtocol):
    """Wraps PipelineOrchestrator for standard pipeline execution."""

    def __init__(self, pipeline: Any = None):
        self._pipeline = pipeline
        self._transport = "grpc"

    def plan_execution(
        self,
        request: Any,
        available_nodes: list[Any],
    ) -> ExecutionPlan:
        node_ids = [getattr(n, "node_id", str(n)) for n in available_nodes]
        return ExecutionPlan(
            request_id=getattr(request, "request_id", ""),
            route=node_ids,
            fallback_strategy="retry",
            transport=self._transport,
        )

    def execute(self, plan: ExecutionPlan) -> Any:
        if self._pipeline is None:
            raise RuntimeError("No pipeline configured")
        return self._pipeline.run_pipeline(plan)


class EdgeAwareExecutionPlanner(ExecPlannerProtocol):
    """Routes to edge nodes first, falls back to cluster."""

    def __init__(self, edge_manager: Any = None, cluster_planner: ExecPlannerProtocol | None = None):
        self._edge = edge_manager
        self._cluster = cluster_planner

    def plan_execution(
        self,
        request: Any,
        available_nodes: list[Any],
    ) -> ExecutionPlan:
        model_params = getattr(request, "model_params_b", 70)
        # Small models can go to edge
        if model_params <= 3 and self._edge is not None:
            edge_nodes = self._edge.get_online_nodes() if hasattr(self._edge, "get_online_nodes") else []
            if edge_nodes:
                return ExecutionPlan(
                    request_id=getattr(request, "request_id", ""),
                    route=[n.node_id for n in edge_nodes[:1]],
                    fallback_strategy="abort",
                    transport="webrtc",
                )
        # Fall back to cluster
        if self._cluster is not None:
            return self._cluster.plan_execution(request, available_nodes)
        return ExecutionPlan(
            request_id=getattr(request, "request_id", ""),
            route=[],
            fallback_strategy="abort",
        )

    def execute(self, plan: ExecutionPlan) -> Any:
        if plan.transport == "webrtc" and self._edge is not None:
            if hasattr(self._edge, "route_inference"):
                return self._edge.route_inference(
                    getattr(plan, "prompt", ""),
                    getattr(plan, "model_name", ""),
                )
        if self._cluster is not None:
            return self._cluster.execute(plan)
        raise RuntimeError("No execution backend available")
