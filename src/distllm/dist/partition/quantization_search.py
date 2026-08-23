"""Search strategies for quantization: beam/DP solver wrappers and the
AutoMixedPrecisionPipeline runtime.

This module provides the runtime side of quantization planning:

- :class:`AutoMixedPrecisionPipeline` wraps a distributed orchestrator and
  transparently casts hidden states to the per-layer precision assigned by
  a :class:`~distllm.dist.partition.quantization_tuner.MixedPrecisionPlan`.

- Future additions may include beam-search or DP-based quant-method search
  over the full device-cluster configuration space.
"""

from __future__ import annotations

from typing import Any

import torch
from loguru import logger

from distllm.dist.partition.quantization_tuner import MixedPrecisionPlan


# ---------------------------------------------------------------------------
# AutoMixedPrecisionPipeline -- wraps orchestrator with per-layer dtype casting
# ---------------------------------------------------------------------------


class AutoMixedPrecisionPipeline:
    """Pipeline wrapper that applies per-layer mixed precision during
    distributed inference.

    Uses a :class:`MixedPrecisionPlan` to determine the target dtype
    for each layer and inserts dtype casts at layer boundaries so that
    each layer executes in its assigned precision without modifying the
    orchestrator's internal routing logic.

    Typical usage::

        amp_pipeline = AutoMixedPrecisionPipeline(
            orchestrator=orchestrator,
            precision_plan=plan,
        )
        output = amp_pipeline.run(input_ids, kv_caches, "req-1")

    The wrapper transparently casts hidden states between layers using
    :meth:`cast_to_layer_precision` at each node boundary.
    """

    def __init__(
        self,
        orchestrator: Any,
        precision_plan: MixedPrecisionPlan,
        model: torch.nn.Module | None = None,
        device: str = "cuda",
    ):
        self._orchestrator = orchestrator
        self._plan = precision_plan
        self._model = model
        self._device = device

        # Build per-layer dtype map: layer_idx -> torch.dtype
        self._layer_dtype: dict[int, torch.dtype] = {}
        for p in precision_plan.plans:
            self._layer_dtype[p.layer_idx] = self._parse_dtype(p.weight_dtype)

        logger.info(
            f"AutoMixedPrecisionPipeline initialized with "
            f"{len(precision_plan.plans)} layers, "
            f"avg compression {precision_plan.overall_compression_ratio:.1f}x"
        )

    @staticmethod
    def _parse_dtype(dtype_str: str) -> torch.dtype:
        """Convert a precision string to a torch dtype."""
        mapping: dict[str, torch.dtype] = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "fp8_e4m3": torch.float8_e4m3fn,
            "fp8": torch.float8_e4m3fn,
            "int8": torch.int8,
            "nf4": torch.float16,  # NF4 weights stored as fp16 with scale factors
        }
        if dtype_str not in mapping:
            logger.warning(f"Unknown dtype '{dtype_str}', falling back to float16")
            return torch.float16
        return mapping[dtype_str]

    def get_dtype_for_layer(self, layer_idx: int) -> torch.dtype:
        """Return the target dtype for a given layer index."""
        return self._layer_dtype.get(layer_idx, torch.float16)

    def cast_to_layer_precision(
        self,
        tensor: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Cast a tensor to the target dtype for the specified layer.

        Args:
            tensor: Input hidden states.
            layer_idx: Target layer index.

        Returns:
            Tensor cast to the appropriate dtype, on the same device.
        """
        target_dtype = self.get_dtype_for_layer(layer_idx)
        if tensor.dtype != target_dtype:
            return tensor.to(target_dtype)
        return tensor

    def run(
        self,
        input_ids: torch.Tensor,
        kv_caches: dict[str, list | None],
        request_id: str,
        *,
        micro_batched: bool = False,
        micro_batch_size: int | None = None,
    ) -> torch.Tensor:
        """Run the pipeline with per-layer mixed-precision casting.

        The wrapper intercepts the hidden states before each layer and
        casts them to the layer's assigned dtype.  The orchestrator's
        ``run_pipeline`` or ``run_pipeline_microbatched`` handles the
        actual distributed forwarding.

        Note:
            This implementation casts on the coordinator side -- the
            worker nodes receive tensors already in the target dtype.
            For true per-worker casting, see
            :meth:`apply_to_model_weights` which modifies the model
            weights in-place.

        Args:
            input_ids: Input token IDs.
            kv_caches: Per-node KV cache dictionary.
            request_id: Unique request identifier.
            micro_batched: If True, use the micro-batched pipeline.
            micro_batch_size: Optional micro-batch size override.

        Returns:
            Output logits from the last node.
        """
        if micro_batched:
            return self._run_microbatched(
                input_ids, kv_caches, request_id, micro_batch_size,
            )
        return self._run_sequential(input_ids, kv_caches, request_id)

    def _run_sequential(
        self,
        input_ids: torch.Tensor,
        kv_caches: dict[str, list | None],
        request_id: str,
    ) -> torch.Tensor:
        """Sequential pipeline with per-layer precision casts."""
        current = input_ids

        # Gather nodes in layer order
        node_order = self._orchestrator.node_order
        if not node_order:
            raise RuntimeError("No nodes registered in pipeline")

        for node_id in node_order:
            node = self._orchestrator.get_node(node_id)
            if node is None or not node.is_healthy:
                continue

            # Cast input to each layer's precision as we traverse
            # layers assigned to this node
            for layer_idx in range(node.start_layer, node.end_layer + 1):
                current = self.cast_to_layer_precision(current, layer_idx)

            # Forward through the node
            from distllm.dist.node_client import forward_request

            kv_cache = kv_caches.get(node_id)
            current = forward_request(
                host=node.host,
                port=node.port,
                hidden_states=current,
                kv_cache=kv_cache,
                request_id=request_id,
            )
            if current is None:
                raise RuntimeError(f"Node {node_id} returned None")

        return current

    def _run_microbatched(
        self,
        input_ids: torch.Tensor,
        kv_caches: dict[str, list | None],
        request_id: str,
        micro_batch_size: int | None = None,
    ) -> torch.Tensor:
        """Micro-batched pipeline with per-layer precision casts.

        Uses ``run_pipeline_microbatched`` on the underlying orchestrator
        but intercepts the hidden states at each stage boundary.  Since
        the orchestrator handles gRPC routing, we rely on the worker
        nodes to apply per-layer casting when the model is loaded with
        :meth:`apply_to_model_weights`.

        For the current implementation, we cast the entire input before
        sending and rely on the orchestrator's micro-batch split.
        """
        # Cast full input to match the first layer's precision
        if self._plan.plans:
            first = self._plan.plans[0]
            cast_input = input_ids.to(self._parse_dtype(first.weight_dtype))
        else:
            cast_input = input_ids

        # The micro-batched pipeline does not expose per-micro-batch
        # casting hooks in this initial version.  We pass the cast input
        # and rely on the model weights having been set via
        # apply_to_model_weights for per-layer correctness.
        import asyncio

        coro = self._orchestrator.run_pipeline_microbatched(
            cast_input, kv_caches, request_id, micro_batch_size,
        )
        return asyncio.run(coro)

    def apply_to_model_weights(self, model: torch.nn.Module) -> torch.nn.Module:
        """Modify model weights in-place to match the precision plan.

        Casts each layer's parameters to the assigned dtype.  This is
        a one-time operation performed when loading the model onto the
        worker node.  After this call, the model's forward pass runs
        natively in the assigned per-layer precision without runtime
        casting overhead.

        Args:
            model: The transformer model whose weights should be cast.

        Returns:
            The same model with updated weight dtypes (in-place).
        """
        for plan in self._plan.plans:
            layer_name = f"model.layers.{plan.layer_idx}"
            target_dtype = self._parse_dtype(plan.weight_dtype)

            # Resolve submodule
            module: torch.nn.Module = model
            for part in layer_name.split("."):
                module = getattr(module, part, None)
                if module is None:
                    break

            if module is None:
                logger.warning(
                    f"Layer {layer_name} not found in model, skipping"
                )
                continue

            # Cast all parameters in this submodule
            for param in module.parameters(recurse=True):
                if param.dtype != target_dtype:
                    param.data = param.data.to(target_dtype)

            logger.debug(
                f"Casted {layer_name} to {plan.weight_dtype} "
                f"(target dtype: {target_dtype})"
            )

        return model

    @property
    def orchestrator(self) -> Any:
        """The underlying pipeline orchestrator."""
        return self._orchestrator

    @property
    def precision_plan(self) -> MixedPrecisionPlan:
        """The mixed-precision plan."""
        return self._plan

    def summary(self) -> str:
        """Return a human-readable summary of the precision plan."""
        parts = [
            f"AutoMixedPrecisionPipeline: {len(self._plan.plans)} layers",
        ]
        for p in self._plan.plans:
            parts.append(f"  L{p.layer_idx:>3} ({p.layer_type:<10}) -> {p.weight_dtype}")
        parts.append(f"  Avg compression: {self._plan.overall_compression_ratio:.1f}x")
        return "\n".join(parts)
