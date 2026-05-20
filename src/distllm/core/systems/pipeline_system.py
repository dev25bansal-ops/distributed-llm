"""PipelineSystem: distributed pipeline execution, transport.

Groups: PipelineOrchestrator, ZeroCopyTransferEngine, LatencyTracker, Rebalancer
"""

from typing import Any

import torch
from loguru import logger


class PipelineSystem:
    """Manages distributed pipeline execution.

    Composes PipelineOrchestrator, ZeroCopyTransferEngine, LatencyTracker,
    and Rebalancer into a single interface.
    """

    def __init__(
        self,
        resource_mgr: Any = None,
        total_layers: int = 0,
        max_workers: int = 4,
        enable_rebalancing: bool = False,
    ):
        from distllm.core.pipeline_orchestrator import PipelineOrchestrator
        from distllm.core.latency_tracker import LatencyTracker
        from distllm.core.rebalancer import Rebalancer
        from distllm.core.zero_copy_transfer import ZeroCopyTransferEngine

        self.orchestrator = PipelineOrchestrator(
            resource_mgr=resource_mgr,
            total_layers=total_layers,
            max_workers=max_workers,
        )
        self.latency_tracker = LatencyTracker()
        self.orchestrator.set_latency_tracker(self.latency_tracker)

        self.rebalancer = Rebalancer(
            pipeline=self.orchestrator,
            latency_tracker=self.latency_tracker,
        ) if enable_rebalancing else None

        self.zero_copy = ZeroCopyTransferEngine()

    @property
    def nodes(self) -> dict:
        return self.orchestrator.nodes

    @property
    def node_order(self) -> list[str]:
        return self.orchestrator.node_order

    @property
    def total_layers(self) -> int:
        return self.orchestrator.total_layers

    def run_pipeline(
        self,
        input_ids: torch.Tensor,
        node_kv_caches: dict[str, list | None],
        request_id: str,
        draft_tokens: list[int] | None = None,
    ) -> torch.Tensor:
        return self.orchestrator.run_pipeline(
            input_ids, node_kv_caches, request_id, draft_tokens,
        )

    async def run_pipeline_async(
        self,
        input_ids: torch.Tensor,
        node_kv_caches: dict[str, list | None],
        request_id: str,
        draft_tokens: list[int] | None = None,
    ) -> torch.Tensor:
        return await self.orchestrator.run_pipeline_async(
            input_ids, node_kv_caches, request_id, draft_tokens,
        )

    def set_tensor_transport(self, transport: Any, node_rank_map: dict[str, int] | None = None) -> None:
        self.orchestrator.set_tensor_transport(transport, node_rank_map)

    def record_latency(self, node_id: str, latency_ms: float) -> None:
        self.latency_tracker.record(node_id, latency_ms)

    def stats(self) -> dict:
        return {
            "nodes": len(self.nodes),
            "total_layers": self.total_layers,
            "zero_copy": self.zero_copy.stats(),
        }
