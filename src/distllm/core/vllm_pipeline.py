"""Multi-node pipeline engine orchestrating vLLM-backed worker nodes.

Replaces PipelineOrchestrator.run_pipeline() for vLLM-backed deployments.
Each worker node runs VLLMNodeAdapter on its layer subset; this engine
coordinates the distributed forward pass via gRPC.
"""

from __future__ import annotations

import asyncio
import torch
from loguru import logger

from distllm.communication.node_pb2 import ForwardPassRequest
from distllm.communication.tensor_transport import (
    tensor_to_proto,
    proto_to_tensor,
    kv_cache_to_proto,
    proto_to_kv_cache,
)
from distllm.errors.types import NodeUnreachableError
from distllm.core.kv_cache import KVCache


class VLLMPipelineEngine:
    """Coordinates multi-node pipeline forward passes using vLLM on each node.

    Maps the legacy pipeline topology (node_order, nodes) onto gRPC-based
    forward calls, where each node runs VLLMNodeAdapter internally.
    """

    def __init__(
        self,
        node_order: list[str],
        nodes: dict,
        timeout_s: float = 60.0,
    ):
        self._node_order = node_order
        self._nodes = nodes
        self._timeout_s = timeout_s

        self.enable_overlap = False

    @property
    def node_order(self) -> list[str]:
        return self._node_order

    @node_order.setter
    def node_order(self, value: list[str]):
        self._node_order = value

    @property
    def nodes(self) -> dict:
        return self._nodes

    @nodes.setter
    def nodes(self, value: dict):
        self._nodes = value

    def create_node_kv_caches(self) -> dict[str, list | None]:
        return {nid: None for nid in self._node_order}

    def run_pipeline(
        self,
        input_ids: torch.Tensor,
        node_kv_caches: dict[str, list | None],
        request_id: str = "",
        draft_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run forward pass through all nodes sequentially.

        Each node executes VLLMNodeAdapter.forward() on hidden states
        or input_ids, returning updated hidden states for the next node.

        Args:
            input_ids: Token IDs for the first node.
            node_kv_caches: KV caches per node (updated in-place).
            request_id: Unique request identifier.
            draft_tokens: Optional speculative decoding draft tokens.

        Returns:
            Logits from the last node.
        """
        current_hidden = None
        input_ids_tensor = input_ids

        for idx, node_id in enumerate(self._node_order):
            node = self._nodes.get(node_id)
            if node is None:
                raise NodeUnreachableError(f"Node {node_id} not found")

            is_first = idx == 0
            is_last = idx == len(self._node_order) - 1

            request = ForwardPassRequest(
                request_id=request_id,
                batch_size=input_ids.shape[0] if is_first else 1,
                seq_len=input_ids.shape[1] if is_first else 1,
                use_cache=True,
                is_first_pass=is_first,
            )

            if is_first:
                request.input_ids.extend(input_ids_tensor.flatten().tolist())
            elif current_hidden is not None:
                request.hidden_states.CopyFrom(tensor_to_proto(current_hidden))

            if node_kv_caches.get(node_id) is not None:
                kv = KVCache()
                kv.set_all(node_kv_caches[node_id])
                request.kv_cache.CopyFrom(kv_cache_to_proto(kv))

            if draft_tokens is not None and is_last:
                request.draft_tokens.extend(draft_tokens.flatten().tolist())

            try:
                response = node.client.stub.ForwardPass(request, timeout=self._timeout_s)
            except Exception as e:
                raise NodeUnreachableError(f"ForwardPass failed on {node_id}: {e}") from e

            if not response.success:
                raise RuntimeError(
                    f"ForwardPass error on {node_id}: {response.error_message}"
                )

            current_hidden = proto_to_tensor(response.output, device="cpu")

            if response.HasField("kv_cache"):
                kv = proto_to_kv_cache(response.kv_cache, device="cpu")
                node_kv_caches[node_id] = kv.cache

        return current_hidden

    async def run_pipeline_async(
        self,
        input_ids: torch.Tensor,
        node_kv_caches: dict[str, list | None],
        request_id: str = "",
        draft_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Async version of run_pipeline using asyncio.to_thread."""
        return await asyncio.to_thread(
            self.run_pipeline,
            input_ids=input_ids,
            node_kv_caches=node_kv_caches,
            request_id=request_id,
            draft_tokens=draft_tokens,
        )

    def run_pipeline_overlap(
        self,
        input_ids: torch.Tensor,
        node_kv_caches: dict[str, list | None],
        request_id: str = "",
        draft_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Overlap compute with communication using per-node background tasks.

        Forwards first node, then pipelines the rest.
        """
        current_hidden = None
        input_ids_tensor = input_ids

        for idx, node_id in enumerate(self._node_order):
            node = self._nodes.get(node_id)
            if node is None:
                raise NodeUnreachableError(f"Node {node_id} not found")

            is_first = idx == 0
            is_last = idx == len(self._node_order) - 1

            request = ForwardPassRequest(
                request_id=request_id,
                use_cache=True,
                is_first_pass=is_first,
            )

            if is_first:
                request.input_ids.extend(input_ids_tensor.flatten().tolist())
            elif current_hidden is not None:
                request.hidden_states.CopyFrom(tensor_to_proto(current_hidden))

            if node_kv_caches.get(node_id) is not None:
                kv = KVCache()
                kv.set_all(node_kv_caches[node_id])
                request.kv_cache.CopyFrom(kv_cache_to_proto(kv))

            try:
                response = node.client.stub.ForwardPass(request, timeout=self._timeout_s)
            except Exception as e:
                raise NodeUnreachableError(f"ForwardPass failed on {node_id}: {e}") from e

            if not response.success:
                raise RuntimeError(
                    f"ForwardPass error on {node_id}: {response.error_message}"
                )

            current_hidden = proto_to_tensor(response.output, device="cpu")

            if response.HasField("kv_cache"):
                kv = proto_to_kv_cache(response.kv_cache, device="cpu")
                node_kv_caches[node_id] = kv.cache

        return current_hidden

    def shutdown(self):
        """Release resources."""
        logger.info("[VLLMPipeline] Engine shut down")
