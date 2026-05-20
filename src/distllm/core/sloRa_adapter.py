"""S-LoRA style multi-adapter serving.

Instead of sequential merge->forward->unmerge per adapter, this module packs
all adapter deltas into a single tensor and applies per-segment during a
single forward pass.

Reference: S-LoRA — Serving Thousands of Concurrent LoRA Adapters
(Sheng et al., 2023)
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SLoRAManager:
    """Manages multiple LoRA adapters using the S-LoRA packing strategy.

    All adapter weight deltas are packed into contiguous buffers so that a
    single forward pass can apply the correct delta to each sequence segment
    without any unmerge / re-merge overhead.

    Parameters
    ----------
    base_model : nn.Module
        The base (foundation) model that adapters are applied on top of.
    max_adapters : int
        Maximum number of adapters that can be registered simultaneously.
    device : str
        Device on which delta buffers reside (e.g. ``"cuda"``).
    """

    def __init__(
        self,
        base_model: nn.Module,
        max_adapters: int = 64,
        device: str = "cuda",
    ) -> None:
        if max_adapters < 1:
            raise ValueError("max_adapters must be >= 1")

        self.base_model = base_model
        self.max_adapters = max_adapters
        self.device = torch.device(device)

        # Delta buffers — dimensions are filled per-slot at registration time.
        # We allocate with zeros so that unregistered slots are no-ops.
        # Actual shapes are [slot, rank, in_dim] / [slot, out_dim, rank].
        # Because rank / in_dim / out_dim can vary across layers, we store
        # a dict keyed by a unique layer identifier.
        self._delta_A: dict[str, torch.Tensor] = {}
        self._delta_B: dict[str, torch.Tensor] = {}

        # adapter_id -> slot index  (0-based)
        self.adapter_map: dict[str, int] = {}

        # slot index -> adapter_id  (reverse mapping)
        self._slot_to_id: dict[int, str] = {}

        # Per-layer bookkeeping: layer_key -> (rank, in_dim, out_dim)
        self._layer_shapes: dict[str, tuple[int, int, int]] = {}

        # Scaling factors per adapter: adapter_id -> float
        self._scalings: dict[str, float] = {}

        # Thread-safety lock for registration / unregistration
        self._lock = threading.Lock()

        logger.info(
            "SLoRAManager initialised: max_adapters=%d, device=%s",
            max_adapters,
            device,
        )

    # ------------------------------------------------------------------
    # Adapter lifecycle
    # ------------------------------------------------------------------

    def register_adapter(
        self,
        adapter_id: str,
        A: torch.Tensor,
        B: torch.Tensor,
        scaling: float = 1.0,
        layer_key: Optional[str] = None,
    ) -> int:
        """Register a LoRA adapter delta into a free slot.

        Parameters
        ----------
        adapter_id : str
            Unique identifier for this adapter.
        A : torch.Tensor
            Low-rank matrix A of shape ``[rank, in_dim]``.
        B : torch.Tensor
            Low-rank matrix B of shape ``[out_dim, rank]``.
        scaling : float
            Adapter scaling factor (often ``alpha / rank``).
        layer_key : str, optional
            Identifier for the layer this delta belongs to.
            If ``None``, a default key ``"default"`` is used, which works
            when all adapters share the same shape.

        Returns
        -------
        int
            The slot index assigned to this adapter.

        Raises
        ------
        ValueError
            If the adapter is already registered or no free slot is available.
        """
        with self._lock:
            if adapter_id in self.adapter_map:
                raise ValueError(
                    f"Adapter '{adapter_id}' is already registered at "
                    f"slot {self.adapter_map[adapter_id]}"
                )

            # Find the first free slot
            free_slot: Optional[int] = None
            for s in range(self.max_adapters):
                if s not in self._slot_to_id:
                    free_slot = s
                    break

            if free_slot is None:
                raise RuntimeError(
                    f"All {self.max_adapters} adapter slots are occupied. "
                    f"Unregister an adapter first."
                )

            if A.ndim != 2 or B.ndim != 2:
                raise ValueError(
                    f"A and B must be 2-D tensors, got A.ndim={A.ndim}, "
                    f"B.ndim={B.ndim}"
                )

            rank_a, in_dim = A.shape
            out_dim, rank_b = B.shape

            if rank_a != rank_b:
                raise ValueError(
                    f"Rank mismatch: A has rank {rank_a}, B has rank {rank_b}"
                )

            rank = rank_a
            key = layer_key if layer_key is not None else "default"

            # Validate / initialise delta buffers for this layer key
            if key not in self._layer_shapes:
                self._layer_shapes[key] = (rank, in_dim, out_dim)
                self._delta_A[key] = torch.zeros(
                    (self.max_adapters, rank, in_dim),
                    device=self.device,
                    dtype=A.dtype,
                )
                self._delta_B[key] = torch.zeros(
                    (self.max_adapters, out_dim, rank),
                    device=self.device,
                    dtype=B.dtype,
                )
                logger.debug(
                    "Allocated delta buffers for layer '%s': "
                    "rank=%d, in_dim=%d, out_dim=%d",
                    key,
                    rank,
                    in_dim,
                    out_dim,
                )
            else:
                expected = self._layer_shapes[key]
                if (rank, in_dim, out_dim) != expected:
                    raise ValueError(
                        f"Shape mismatch for layer '{key}': got "
                        f"(rank={rank}, in={in_dim}, out={out_dim}), "
                        f"expected {expected}"
                    )

            # Write delta into the slot
            self._delta_A[key][free_slot].copy_(A.to(self.device))
            self._delta_B[key][free_slot].copy_(B.to(self.device))

            self.adapter_map[adapter_id] = free_slot
            self._slot_to_id[free_slot] = adapter_id
            self._scalings[adapter_id] = scaling

            logger.info(
                "Registered adapter '%s' at slot %d (scaling=%.4f)",
                adapter_id,
                free_slot,
                scaling,
            )
            return free_slot

    def unregister_adapter(self, adapter_id: str) -> None:
        """Remove an adapter and free its slot.

        Parameters
        ----------
        adapter_id : str
            The adapter to remove.

        Raises
        ------
        KeyError
            If the adapter is not registered.
        """
        with self._lock:
            if adapter_id not in self.adapter_map:
                raise KeyError(
                    f"Adapter '{adapter_id}' is not registered"
                )

            slot = self.adapter_map.pop(adapter_id)
            del self._slot_to_id[slot]
            self._scalings.pop(adapter_id, None)

            # Zero out the delta buffers for every layer key
            for key in self._delta_A:
                self._delta_A[key][slot].zero_()
                self._delta_B[key][slot].zero_()

            logger.info("Unregistered adapter '%s' (slot %d)", adapter_id, slot)

    def list_adapters(self) -> list[str]:
        """Return a list of currently registered adapter IDs."""
        with self._lock:
            return list(self.adapter_map.keys())

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def batch_forward(
        self,
        base_output: torch.Tensor,
        adapter_ids: list[str],
        segment_ranges: list[tuple[int, int]],
        layer_key: Optional[str] = None,
    ) -> torch.Tensor:
        """Apply per-segment LoRA deltas to the base model output.

        Parameters
        ----------
        base_output : torch.Tensor
            Tensor of shape ``[batch, seq, hidden]`` from the base model.
        adapter_ids : list[str]
            One adapter ID per sequence in the batch.  ``None`` or ``""``
            means "use the base model only" for that sequence.
        segment_ranges : list[tuple[int, int]]
            One ``(start, end)`` token-range per sequence.
        layer_key : str, optional
            Which layer's delta buffers to use.  Defaults to ``"default"``.

        Returns
        -------
        torch.Tensor
            Adjusted output of the same shape as ``base_output``.
        """
        key = layer_key if layer_key is not None else "default"

        if base_output.ndim != 3:
            raise ValueError(
                f"base_output must be 3-D [batch, seq, hidden], "
                f"got {base_output.ndim}-D"
            )

        batch, seq_len, hidden = base_output.shape

        if len(adapter_ids) != batch:
            raise ValueError(
                f"Expected {batch} adapter_ids (one per sequence), "
                f"got {len(adapter_ids)}"
            )

        if len(segment_ranges) != batch:
            raise ValueError(
                f"Expected {batch} segment_ranges (one per sequence), "
                f"got {len(segment_ranges)}"
            )

        result = base_output.clone()

        # Check if we have deltas for this layer key
        if key not in self._delta_A:
            logger.debug(
                "No delta buffers for layer '%s'; returning base output",
                key,
            )
            return result

        delta_A = self._delta_A[key]  # [max_adapters, rank, in_dim]
        delta_B = self._delta_B[key]  # [max_adapters, out_dim, rank]

        for idx in range(batch):
            aid = adapter_ids[idx]
            start, end = segment_ranges[idx]

            # Skip if no adapter specified or adapter not registered
            if not aid or aid not in self.adapter_map:
                continue

            slot = self.adapter_map[aid]
            scaling = self._scalings.get(aid, 1.0)

            segment = result[:, start:end, :]  # [batch, seg_len, hidden]
            # segment shape for einsum: [batch, seg_len, in_dim]

            # delta_A[slot]: [rank, in_dim]
            # delta_B[slot]: [out_dim, rank]
            # BA = delta_B[slot] @ delta_A[slot]  => [out_dim, in_dim]
            # delta = (BA * scaling) @ x^T
            A_s = delta_A[slot]  # [rank, in_dim]
            B_s = delta_B[slot]  # [out_dim, rank]

            # Compute: segment @ A_s^T => [batch, seg_len, rank]
            # Then:    (that) @ B_s^T => [batch, seg_len, out_dim]
            # Note: in_dim == hidden == out_dim for a single linear layer
            low_rank = torch.einsum("bsh,ri->bsr", segment, A_s)
            delta_out = torch.einsum("bsr,or->bso", low_rank, B_s)

            result[:, start:end, :] = result[:, start:end, :] + delta_out * scaling

        return result

    # ------------------------------------------------------------------
    # Linear-layer integration
    # ------------------------------------------------------------------

    def apply_to_linear_layer(
        self,
        layer: nn.Linear,
        adapter_ids: list[str],
        layer_input: torch.Tensor,
        segment_ranges: list[tuple[int, int]],
    ) -> torch.Tensor:
        """Forward through a linear layer with per-segment S-LoRA adjustment.

        Parameters
        ----------
        layer : nn.Linear
            The base linear layer.
        adapter_ids : list[str]
            Adapter ID for each sequence in the batch.
        layer_input : torch.Tensor
            Input tensor of shape ``[batch, seq, in_features]``.
        segment_ranges : list[tuple[int, int]]
            Token ranges per sequence.

        Returns
        -------
        torch.Tensor
            Output of shape ``[batch, seq, out_features]`` with adapter
            deltas applied.
        """
        if not isinstance(layer, nn.Linear):
            raise TypeError(
                f"Expected nn.Linear layer, got {type(layer).__name__}"
            )

        base_output = layer(layer_input)
        return self.batch_forward(
            base_output=base_output,
            adapter_ids=adapter_ids,
            segment_ranges=segment_ranges,
        )

    # ------------------------------------------------------------------
    # Model setup helper
    # ------------------------------------------------------------------

    def setup_sloRa_for_model(
        self,
        model: nn.Module,
        adapter_configs: dict[str, dict],
    ) -> None:
        """Register adapters and patch the model's linear layers for S-LoRA.

        Parameters
        ----------
        model : nn.Module
            The model to patch in-place.
        adapter_configs : dict[str, dict]
            Mapping of adapter name to configuration dict.  Each config must
            contain:

            * ``"A"``: ``torch.Tensor`` of shape ``[rank, in_dim]``
            * ``"B"``: ``torch.Tensor`` of shape ``[out_dim, rank]``
            * ``"path"``: dotted attribute path to the ``nn.Linear`` layer
              (e.g. ``"layers.0.attn.q_proj"``)

            Optional keys:

            * ``"scaling"``: ``float`` (default ``1.0``)

        Raises
        ------
        KeyError
            If a required config key is missing.
        AttributeError
            If the dotted path does not resolve to a valid layer.
        """
        for adapter_name, cfg in adapter_configs.items():
            if not isinstance(cfg, dict):
                raise TypeError(
                    f"Config for adapter '{adapter_name}' must be a dict, "
                    f"got {type(cfg).__name__}"
                )

            for required_key in ("A", "B", "path"):
                if required_key not in cfg:
                    raise KeyError(
                        f"Adapter '{adapter_name}' config is missing "
                        f"required key '{required_key}'"
                    )

            A: torch.Tensor = cfg["A"]
            B: torch.Tensor = cfg["B"]
            path: str = cfg["path"]
            scaling: float = cfg.get("scaling", 1.0)

            # Resolve the dotted path to the actual nn.Linear module
            layer = _resolve_dotted_path(model, path)
            if not isinstance(layer, nn.Linear):
                raise AttributeError(
                    f"Path '{path}' resolved to {type(layer).__name__}, "
                    f"expected nn.Linear"
                )

            # Validate shapes against the layer
            rank_a, in_dim = A.shape
            out_dim, rank_b = B.shape
            if in_dim != layer.in_features:
                raise ValueError(
                    f"Adapter '{adapter_name}': A in_dim={in_dim} does not "
                    f"match layer in_features={layer.in_features}"
                )
            if out_dim != layer.out_features:
                raise ValueError(
                    f"Adapter '{adapter_name}': B out_dim={out_dim} does not "
                    f"match layer out_features={layer.out_features}"
                )

            # Register the adapter using the layer path as the layer_key
            self.register_adapter(
                adapter_id=adapter_name,
                A=A,
                B=B,
                scaling=scaling,
                layer_key=path,
            )

            # Wrap the layer's forward method
            original_forward = layer.forward

            def _patched_forward(
                x: torch.Tensor,
                _orig=original_forward,
                _layer=layer,
                _path=path,
            ) -> torch.Tensor:
                base_out = _orig(x)

                # Determine adapter_ids and segment_ranges from the batch.
                # By default we apply the registered adapter to the full
                # sequence range.  Users who need per-segment control should
                # call batch_forward directly.
                batch = x.shape[0]
                seq_len = x.shape[1] if x.ndim >= 2 else 1

                adapter_ids_for_layer = [adapter_name] * batch
                segment_ranges_for_layer = [(0, seq_len)] * batch

                return self.batch_forward(
                    base_output=base_out,
                    adapter_ids=adapter_ids_for_layer,
                    segment_ranges=segment_ranges_for_layer,
                    layer_key=_path,
                )

            layer.forward = _patched_forward
            logger.info(
                "Patched layer at '%s' for adapter '%s'", path, adapter_name
            )

        logger.info(
            "S-LoRA setup complete: %d adapter(s) registered",
            len(self.adapter_map),
        )


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _resolve_dotted_path(module: nn.Module, path: str) -> nn.Module:
    """Resolve a dotted attribute path on an nn.Module.

    Example: ``"layers.0.attn.q_proj"`` walks ``module.layers[0].attn.q_proj``.

    Parameters
    ----------
    module : nn.Module
        The root module to start resolving from.
    path : str
        Dotted path string.

    Returns
    -------
    nn.Module
        The resolved sub-module.
    """
    parts = path.split(".")
    current: nn.Module = module

    for part in parts:
        # Try integer index first (for nn.ModuleList / Sequential)
        try:
            idx = int(part)
            current = current[idx]  # type: ignore[index]
            continue
        except (ValueError, IndexError, TypeError):
            pass

        # Fall back to attribute access
        if not hasattr(current, part):
            raise AttributeError(
                f"Cannot resolve path '{path}': "
                f"'{type(current).__name__}' has no attribute '{part}'"
            )
        current = getattr(current, part)

    return current
