"""Adapter-aware request routing for multi-LoRA serving.

Extends the ``AdapterManager`` (S-LoRA) with:
- Request-level adapter routing via adapter_id header/metadata
- Integration with ``ContinuousBatchingEngine`` for batched multi-adapter decode
- Adapter-aware batch construction (group requests by adapter for efficiency)
- Dynamic adapter loading/unloading via API

Usage::

    router = AdapterRouter(base_model, tokenizer)
    router.load_adapter("code-lora", "/path/to/code-lora")
    router.load_adapter("chat-lora", "/path/to/chat-lora")

    # During inference:
    router.activate_for_request("req-1", "code-lora")
    router.activate_for_request("req-2", "chat-lora")
    router.activate_for_request("req-3", None)  # base model

    batch = router.build_adapter_batch(requests)
    # batch.groups = [("code-lora", [req1]), ("chat-lora", [req2]), (None, [req3])]
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import torch

from distllm.models.adapter import AdapterManager


@dataclass
class AdapterBatchGroup:
    """A group of requests sharing the same adapter."""
    adapter_id: str | None
    request_ids: list[str] = field(default_factory=list)
    input_ids: list[torch.Tensor] = field(default_factory=list)


@dataclass
class AdapterBatch:
    """A batch partitioned by adapter."""
    groups: list[AdapterBatchGroup] = field(default_factory=list)
    num_adapters: int = 0
    num_requests: int = 0


class AdapterRouter:
    """Routes requests to the correct LoRA adapter.

    Wraps ``AdapterManager`` and adds batch-level adapter grouping
    for efficient multi-adapter inference.
    """

    def __init__(
        self,
        base_model: object | None = None,
        tokenizer: object | None = None,
        max_vram_bytes: int = 0,
        device: str | None = None,
    ):
        self._mgr = AdapterManager(
            base_model=base_model,
            tokenizer=tokenizer,
            max_vram_bytes=max_vram_bytes,
            device=device,
        )
        # Per-request adapter mapping: request_id -> adapter_id | None
        self._request_adapters: dict[str, str | None] = {}
        self._lock = torch.lock if hasattr(torch, "lock") else None  # noqa

    # ── Adapter lifecycle ──────────────────────────────────────────────

    def load_adapter(
        self,
        adapter_id: str,
        adapter_path: str,
        rank: int = 0,
        tenant_id: str = "",
    ) -> None:
        """Load a LoRA adapter into the VRAM pool."""
        self._mgr.load_adapter(adapter_id, adapter_path, rank, tenant_id)

    def unload_adapter(self, adapter_id: str) -> bool:
        """Unload an adapter from the VRAM pool."""
        return self._mgr.unload_adapter(adapter_id)

    def warmup(self, adapters: dict[str, str]) -> list[str]:
        """Pre-load adapters before traffic arrives."""
        return self._mgr.warmup_adapters(adapters)

    def list_adapters(self) -> list[str]:
        return self._mgr.list_adapters()

    # ── Per-request adapter routing ────────────────────────────────────

    def activate_for_request(self, request_id: str, adapter_id: str | None) -> None:
        """Associate a request with an adapter (or None for base model)."""
        self._request_adapters[request_id] = adapter_id

    def get_adapter_for_request(self, request_id: str) -> str | None:
        """Return the adapter ID for a request, or None for base model."""
        return self._request_adapters.get(request_id)

    def clear_request(self, request_id: str) -> None:
        """Remove a request's adapter mapping when it completes."""
        self._request_adapters.pop(request_id, None)

    # ── Batch construction ─────────────────────────────────────────────

    def build_adapter_batch(
        self,
        requests: list[tuple[str, torch.Tensor]],
    ) -> AdapterBatch:
        """Group requests by adapter for efficient batched inference.

        Args:
            requests: List of (request_id, input_ids) tuples.

        Returns:
            An ``AdapterBatch`` partitioned by adapter.
        """
        groups_map: OrderedDict[str | None, AdapterBatchGroup] = OrderedDict()

        for req_id, input_ids in requests:
            adapter_id = self._request_adapters.get(req_id)
            if adapter_id not in groups_map:
                groups_map[adapter_id] = AdapterBatchGroup(adapter_id=adapter_id)
            groups_map[adapter_id].request_ids.append(req_id)
            groups_map[adapter_id].input_ids.append(input_ids)

        groups = list(groups_map.values())
        return AdapterBatch(
            groups=groups,
            num_adapters=len(groups),
            num_requests=len(requests),
        )

    def prepare_for_batch(self, batch: AdapterBatch) -> float:
        """Ensure all adapters needed for a batch are on GPU.

        Returns:
            Time spent swapping (seconds).
        """
        adapter_ids = [
            g.adapter_id for g in batch.groups if g.adapter_id is not None
        ]
        return self._mgr.prepare_batch_adapters(adapter_ids)

    def get_model_for_request(self, request_id: str) -> object | None:
        """Get the PEFT model for a specific request.

        Returns the loaded PeftModel, or None for base model.
        """
        adapter_id = self._request_adapters.get(request_id)
        return self._mgr.get_adapter_for_request(adapter_id)

    # ── Federated training integration ─────────────────────────────────

    def start_federated_training(
        self,
        adapter_id: str,
        local_data_path: str,
        epochs: int = 3,
        learning_rate: float = 2e-4,
    ) -> dict[str, Any]:
        return self._mgr.start_federated_training(
            adapter_id, local_data_path, epochs, learning_rate,
        )

    def export_weights(self, adapter_id: str) -> dict[str, torch.Tensor] | None:
        return self._mgr.export_adapter_weights(adapter_id)

    def import_weights(
        self,
        adapter_id: str,
        weights: dict[str, torch.Tensor],
    ) -> bool:
        return self._mgr.import_adapter_weights(adapter_id, weights)

    # ── Stats ──────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        return {
            **self._mgr.get_stats(),
            "active_routes": len(self._request_adapters),
        }

    def shutdown(self) -> None:
        self._mgr.stop_background_prefetch()
        self._request_adapters.clear()

    @property
    def manager(self) -> AdapterManager:
        return self._mgr
