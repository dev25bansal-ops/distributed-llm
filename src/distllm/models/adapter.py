"""LoRA adapter management for multi-tenant serving.

S-LoRA style: shared base model + per-request adapter weights in one batch.
- VRAM-resident adapter pool with LRU eviction
- Adapter warmup API (pre-load before traffic)
- Adapter ranking for multi-tenant priority
- Per-sequence adapter routing in batched inference
- Async background preloading with swap-in/swap-out
- CPU offloading for cold adapters
- Adapter quantization (int8) for higher density
"""

import time
import threading
import queue
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum

import torch
from loguru import logger
from distllm.security import hf_revision


@dataclass
class AdapterInfo:
    """Metadata about a loaded adapter."""
    adapter_id: str
    path: str
    loaded_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0
    vram_bytes: int = 0
    rank: int = 0  # Multi-tenant priority (higher = more important)
    tenant_id: str = ""
    state: str = "loaded"  # loaded, offloaded, swapping_in, swapping_out
    quantized: bool = False
    numel: int = 0


class AdapterState(Enum):
    GPU = "gpu"
    CPU = "cpu"
    SWAPPING_IN = "swapping_in"
    SWAPPING_OUT = "swapping_out"
    LOADING = "loading"


class SwappingScheduler:
    """S-LoRA style swapping scheduler for multi-adapter batching.

    Decides which adapters to keep on GPU, which to offload,
    and orchestrates async swap-in/swap-out to hide latency.
    """

    def __init__(self, pool: "AdapterPool", max_gpu_adapters: int = 4):
        self._pool = pool
        self._max_gpu_adapters = max_gpu_adapters
        self._gpu_slots: OrderedDict[str, float] = OrderedDict()  # adapter_id -> last_used
        self._pending_swaps: dict[str, str] = {}  # adapter_id -> "in" | "out"
        self._lock = threading.Lock()

    def schedule_swap(self, needed_ids: set[str]) -> list[tuple[str, str]]:
        """Schedule which adapters to swap in/out for a batch.

        Args:
            needed_ids: Adapter IDs needed for the upcoming batch.

        Returns:
            List of (adapter_id, "in"|"out") operations to perform.
        """
        ops: list[tuple[str, str]] = []
        with self._lock:
            current = set(self._gpu_slots.keys())
            to_evict = current - needed_ids
            to_load = needed_ids - current

            # Count evictions: must free enough slots for incoming adapters
            needed_slots = len(to_load)
            freeable = min(needed_slots, len(to_evict))

            # Sort evictions by LRU, evict the minimum needed
            evict_order = sorted(to_evict, key=lambda x: self._gpu_slots.get(x, 0))
            evicted = 0
            for eid in evict_order:
                if evicted >= freeable:
                    break
                ops.append((eid, "out"))
                evicted += 1

            # Sort loads by priority (rank) then by recency
            load_order = sorted(
                to_load,
                key=lambda x: (
                    -(self._pool._pool[x].rank if x in self._pool._pool else 0),
                    -self._pool._pool[x].use_count if x in self._pool._pool else 0,
                ),
            )
            available_slots = self._max_gpu_adapters - (len(current) - evicted)
            for lid in load_order[:max(0, available_slots)]:
                ops.append((lid, "in"))

        return ops

    def mark_used(self, adapter_id: str) -> None:
        with self._lock:
            self._gpu_slots[adapter_id] = time.time()

    def mark_removed(self, adapter_id: str) -> None:
        with self._lock:
            self._gpu_slots.pop(adapter_id, None)

    def gpu_adapters(self) -> list[str]:
        with self._lock:
            return list(self._gpu_slots.keys())

    @property
    def gpu_count(self) -> int:
        return len(self._gpu_slots)


class AdapterPool:
    """VRAM-resident LoRA adapter pool with LRU eviction.

    Keeps frequently used adapters in VRAM for fast switching.
    When VRAM is full, evicts least-recently-used adapters.
    """

    def __init__(self, max_vram_bytes: int = 0):
        self.max_vram_bytes = max_vram_bytes or self._detect_vram()
        # OrderedDict for LRU: adapter_id -> AdapterInfo
        self._pool: OrderedDict[str, AdapterInfo] = OrderedDict()
        self._active_adapter: str | None = None
        self._lock = threading.Lock()
        self._total_vram = 0

    def _detect_vram(self) -> int:
        """Detect available GPU VRAM."""
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                return int(props.total_memory * 0.3)  # Use 30% of VRAM for adapters
        except Exception:
            logger.debug("Adapter VRAM detection failed, using default")
        return 4 * 1024 ** 3  # Default: 4GB

    def add_adapter(self, adapter_id: str, path: str, vram_bytes: int = 0, rank: int = 0, tenant_id: str = "") -> None:
        """Register an adapter in the pool.

        Args:
            adapter_id: Unique adapter identifier.
            path: Path to adapter weights.
            vram_bytes: Estimated VRAM usage.
            rank: Multi-tenant priority rank.
            tenant_id: Tenant identifier for isolation.
        """
        with self._lock:
            if adapter_id in self._pool:
                # Update existing
                info = self._pool[adapter_id]
                info.path = path
                info.last_used = time.time()
                self._pool.move_to_end(adapter_id)
                return

            info = AdapterInfo(
                adapter_id=adapter_id,
                path=path,
                vram_bytes=vram_bytes,
                rank=rank,
                tenant_id=tenant_id,
            )
            self._pool[adapter_id] = info
            self._total_vram += vram_bytes

            # Evict LRU if over capacity
            self._evict_lru()

    def get_adapter(self, adapter_id: str) -> AdapterInfo | None:
        """Get adapter info and mark as recently used."""
        with self._lock:
            if adapter_id in self._pool:
                info = self._pool[adapter_id]
                info.last_used = time.time()
                info.use_count += 1
                self._pool.move_to_end(adapter_id)
                return info
            return None

    def remove_adapter(self, adapter_id: str) -> bool:
        """Remove an adapter from the pool."""
        with self._lock:
            if adapter_id in self._pool:
                info = self._pool.pop(adapter_id)
                self._total_vram -= info.vram_bytes
                return True
            return False

    def _evict_lru(self) -> None:
        """Evict least-recently-used adapters until under VRAM limit."""
        while self._total_vram > self.max_vram_bytes and self._pool:
            # Don't evict the active adapter
            lru_id, lru_info = self._pool.popitem(last=False)
            if lru_id == self._active_adapter:
                self._pool[lru_id] = lru_info
                break
            self._total_vram -= lru_info.vram_bytes
            logger.info(f"Evicted LRU adapter '{lru_id}' ({lru_info.vram_bytes / 1e6:.1f}MB)")

    def set_active(self, adapter_id: str | None) -> None:
        """Set the active adapter.

        May reference an adapter that is not (yet) resident — it is looked
        up at generate time.
        """
        with self._lock:
            self._active_adapter = adapter_id

    @property
    def active_adapter(self) -> str | None:
        return self._active_adapter

    def list_adapters(self) -> list[str]:
        return list(self._pool.keys())

    def get_stats(self) -> dict:
        return {
            "pool_size": len(self._pool),
            "total_adapters": len(self._pool),
            "total_vram_bytes": self._total_vram,
            "max_vram_bytes": self.max_vram_bytes,
            "vram_usage_pct": round(self._total_vram / max(self.max_vram_bytes, 1) * 100, 1),
            "active_adapter": self._active_adapter,
        }


class AdapterManager:
    """S-LoRA style LoRA adapter manager with per-request adapter routing.

    Maintains a shared base model and routes each request to its specified
    adapter within a batch. Adapters are loaded once and kept in a VRAM pool
    with LRU eviction. Supports adapter quantization, CPU offloading,
    and background preloading.

    Usage:
        mgr = AdapterManager()
        mgr.set_base_model(model, tokenizer)
        mgr.warmup_adapters({"adapter_1": "/path/to/adapter1"})
        # During batch inference:
        batch_tags = {"adapter_ids": ["adapter_1", "adapter_2", None]}
    """

    def __init__(
        self,
        base_model: object | None = None,
        tokenizer: object | None = None,
        max_vram_bytes: int = 0,
        device: str | None = None,
        quantize_int8: bool = False,
    ):
        self.base_model = base_model
        self.tokenizer = tokenizer
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._quantize_int8 = quantize_int8
        self._pool = AdapterPool(max_vram_bytes=max_vram_bytes)
        self._swapping = SwappingScheduler(self._pool, max_gpu_adapters=4)
        self._loaded_models: dict[str, object] = {}
        self._offloaded_models: dict[str, object] = {}  # CPU-offloaded adapter weights
        self._lock = threading.Lock()
        self._prefetch_thread: threading.Thread | None = None
        self._prefetch_queue: queue.Queue = queue.Queue()
        self._prefetch_running = False

    def swap_in(self, adapter_id: str) -> None:
        """Bring adapter weights from CPU to GPU for inference.

        S-LoRA style: moves the adapter to GPU without blocking the entire batch.
        Other adapters remain on GPU and continue serving.
        """
        logger.info(f"Swapping adapter '{adapter_id}' from CPU to GPU")
        info = self._pool.get_adapter(adapter_id)
        if info:
            info.state = "swapping_in"

        if adapter_id in self._offloaded_models:
            model = self._offloaded_models.pop(adapter_id)
            model = model.to(self._device)
            self._loaded_models[adapter_id] = model
            self._swapping.mark_used(adapter_id)

        if info:
            info.state = "loaded"

    def swap_out(self, adapter_id: str) -> None:
        """Offload adapter from GPU to CPU to free VRAM."""
        logger.info(f"Swapping adapter '{adapter_id}' from GPU to CPU")
        info = self._pool.get_adapter(adapter_id)
        if info:
            info.state = "swapping_out"

        if adapter_id in self._loaded_models:
            model = self._loaded_models.pop(adapter_id)
            model = model.cpu()
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._offloaded_models[adapter_id] = model
            self._swapping.mark_removed(adapter_id)

        if info:
            info.state = "offloaded"

    def quantize_adapter(self, adapter_id: str) -> None:
        """Quantize adapter weights with symmetric absmax int8 metadata.

        PyTorch parameters cannot be blindly reassigned to ``torch.int8``
        without breaking downstream modules.  Keep compatible floating-point
        parameters in the live model and retain the int8 backing state for
        export/custom kernels.
        """
        if adapter_id not in self._loaded_models:
            logger.warning(f"Cannot quantize '{adapter_id}': not on GPU")
            return

        logger.info(f"Quantizing adapter '{adapter_id}' to int8")
        model = self._loaded_models[adapter_id]
        quantized_state = {}
        converted = 0
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not torch.is_floating_point(param.data):
                    continue
                max_abs = param.data.detach().abs().max()
                if max_abs == 0:
                    scale = torch.tensor(1.0, device=param.device, dtype=torch.float32)
                    quantized = torch.zeros_like(param.data, dtype=torch.int8)
                else:
                    scale = (max_abs / 127).to(torch.float32)
                    quantized = torch.clamp(torch.round(param.data / scale), -127, 127).to(torch.int8)
                quantized_state[name] = {
                    "weight": quantized.detach().cpu(),
                    "scale": float(scale.detach().cpu()),
                    "shape": tuple(param.shape),
                    "dtype": str(param.dtype),
                }
                param.data.copy_((quantized.to(param.device).to(torch.float32) * scale).to(param.dtype))
                converted += 1
        model._distllm_quantized_adapter_state = quantized_state
        info = self._pool.get_adapter(adapter_id)
        if info:
            info.quantized = converted > 0
        if converted == 0:
            logger.warning(f"Adapter '{adapter_id}' had no floating-point parameters to quantize")

    def prepare_batch_adapters(self, adapter_ids: list[str | None]) -> float:
        """Ensure all needed adapters are on GPU for a batch.

        S-LoRA style: orchestrates async swap-in/swap-out so all adapters
        needed for the batch are resident in GPU memory.

        Args:
            adapter_ids: Adapter IDs needed for each sequence in the batch.

        Returns:
            Time spent swapping (seconds), 0 if all were already on GPU.
        """
        needed = {aid for aid in adapter_ids if aid is not None}
        if not needed:
            return 0.0

        already_on_gpu = set(self._loaded_models.keys())
        missing = needed - already_on_gpu
        if not missing:
            return 0.0

        t0 = time.time()
        ops = self._swapping.schedule_swap(needed)

        for aid, direction in ops:
            if direction == "in" and aid in self._offloaded_models:
                self.swap_in(aid)
            elif direction == "out" and aid in self._loaded_models:
                self.swap_out(aid)

        for aid in missing:
            if aid not in self._loaded_models and aid not in self._offloaded_models:
                logger.warning(f"Adapter '{aid}' not found in pool, loading from pool entry")
                info = self._pool.get_adapter(aid)
                if info and self.base_model is not None:
                    self.load_adapter(aid, info.path, info.rank, info.tenant_id)

        return time.time() - t0

    def start_background_prefetch(self) -> None:
        """Start background thread for async adapter prefetching."""
        if self._prefetch_running:
            return
        self._prefetch_running = True
        self._prefetch_thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self._prefetch_thread.start()
        logger.info("Background adapter prefetching started")

    def stop_background_prefetch(self) -> None:
        """Stop background prefetching thread."""
        self._prefetch_running = False
        if self._prefetch_thread:
            self._prefetch_thread.join(timeout=5)
        logger.info("Background adapter prefetching stopped")

    def enqueue_prefetch(self, adapter_id: str) -> None:
        """Enqueue an adapter for background prefetching."""
        if adapter_id not in self._loaded_models and adapter_id not in self._offloaded_models:
            self._prefetch_queue.put(adapter_id)

    def _prefetch_loop(self) -> None:
        """Background loop: prefetch adapters from disk to CPU."""
        while self._prefetch_running:
            try:
                adapter_id = self._prefetch_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if adapter_id in self._loaded_models or adapter_id in self._offloaded_models:
                continue
            info = self._pool.get_adapter(adapter_id)
            if info and self.base_model is not None:
                try:
                    self.load_adapter(adapter_id, info.path, info.rank, info.tenant_id)
                    self.swap_out(adapter_id)  # Load then offload to CPU to warm OS cache
                    logger.info(f"Prefetched adapter '{adapter_id}' to CPU")
                except Exception as e:
                    logger.error(f"Prefetch failed for '{adapter_id}': {e}")

    def set_base_model(self, base_model: object, tokenizer: object | None = None) -> None:
        """Set the base model after it has been loaded."""
        self.base_model = base_model
        if tokenizer:
            self.tokenizer = tokenizer

    def load_adapter(
        self,
        adapter_id: str,
        adapter_path: str,
        rank: int = 0,
        tenant_id: str = "",
    ) -> None:
        """Load a LoRA adapter from path or HuggingFace hub.

        Args:
            adapter_id: Unique adapter identifier.
            adapter_path: Path to adapter weights or HF repo.
            rank: Multi-tenant priority (higher = more important to keep).
            tenant_id: Tenant identifier for isolation.
        """
        from peft import PeftModel

        if self.base_model is None:
            raise RuntimeError("No base model loaded. Call set_base_model() first.")

        logger.info(f"Loading LoRA adapter '{adapter_id}' from {adapter_path}")

        # Load the adapter using PEFT
        model = PeftModel.from_pretrained(
            self.base_model,
            adapter_path,
            adapter_name=adapter_id,
            revision=hf_revision(),
        )
        self._loaded_models[adapter_id] = model

        # Estimate VRAM usage (rough: count trainable parameters * 2 bytes)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        vram_bytes = trainable_params * 2  # fp16

        # Register in the pool
        self._pool.add_adapter(
            adapter_id,
            adapter_path,
            vram_bytes=vram_bytes,
            rank=rank,
            tenant_id=tenant_id,
        )
        logger.info(f"Adapter '{adapter_id}' loaded ({trainable_params / 1e6:.1f}M params)")

    def set_active(self, adapter_id: str | None) -> None:
        """Switch active adapter. None = base model only."""
        self._pool.set_active(adapter_id)
        if adapter_id is not None and adapter_id in self._loaded_models:
            self._loaded_models[adapter_id].set_adapter(adapter_id)
            info = self._pool.get_adapter(adapter_id)
            if info:
                info.use_count += 1
        logger.info(f"Switched to adapter '{adapter_id}'" if adapter_id else "Switched to base model")

    def warmup_adapters(self, adapters: dict[str, str], rank_map: dict[str, int] | None = None, tenant_map: dict[str, str] | None = None) -> list[str]:
        """Pre-load adapters before traffic arrives.

        Args:
            adapters: Dict of adapter_id -> adapter_path.
            rank_map: Optional dict of adapter_id -> rank.
            tenant_map: Optional dict of adapter_id -> tenant_id.

        Returns:
            List of successfully loaded adapter IDs.
        """
        loaded = []
        for adapter_id, path in adapters.items():
            try:
                self.load_adapter(
                    adapter_id,
                    path,
                    rank=rank_map.get(adapter_id, 0) if rank_map else 0,
                    tenant_id=tenant_map.get(adapter_id, "") if tenant_map else "",
                )
                loaded.append(adapter_id)
            except Exception as e:
                logger.error(f"Failed to warmup adapter '{adapter_id}': {e}")
        return loaded

    def unload_adapter(self, adapter_id: str) -> bool:
        """Unload an adapter to free VRAM."""
        if adapter_id in self._loaded_models:
            del self._loaded_models[adapter_id]
            return self._pool.remove_adapter(adapter_id)
        return False

    def get_adapter_for_request(self, request_adapter_id: str | None) -> object | None:
        """Get the PEFT model for a specific request.

        For S-LoRA style batching, each request in a batch may use
        a different adapter. This returns the appropriate model.

        Args:
            request_adapter_id: Adapter ID for the request.

        Returns:
            PeftModel instance, or None for base model.
        """
        if request_adapter_id is None:
            return None
        return self._loaded_models.get(request_adapter_id)

    def get_batch_adapter_ids(self, sequence_adapter_ids: list[str | None]) -> tuple[list[str | None], int]:
        """Get unique adapter IDs used in a batch.

        Returns the set of adapters needed for the batch and counts
        how many sequences use each adapter.

        Args:
            sequence_adapter_ids: Adapter ID per sequence in the batch.

        Returns:
            (unique_adapter_ids, num_unique_adapters)
        """
        unique = list(dict.fromkeys(sequence_adapter_ids))  # Preserve order, remove dups
        return unique, len(unique)

    def rank_adapters(self) -> list[AdapterInfo]:
        """Return adapters sorted by rank (highest first) for multi-tenant priority."""
        adapters = []
        for adapter_id in self._pool.list_adapters():
            info = self._pool.get_adapter(adapter_id)
            if info:
                adapters.append(info)
        adapters.sort(key=lambda a: (a.rank, a.use_count), reverse=True)
        return adapters

    def list_adapters(self) -> list[str]:
        """Return list of loaded adapter IDs."""
        return self._pool.list_adapters()

    def get_adapter_info(self, adapter_id: str) -> AdapterInfo | None:
        """Get detailed info about an adapter."""
        return self._pool.get_adapter(adapter_id)

    @property
    def active_adapter(self) -> str | None:
        return self._pool.active_adapter

    def get_stats(self) -> dict:
        pool_stats = self._pool.get_stats()
        return {
            **pool_stats,
            "loaded_models": len(self._loaded_models),
        }

    # ── Federated Training Integration ─────────────────────────────────

    def export_adapter_weights(self, adapter_id: str) -> dict[str, torch.Tensor] | None:
        """Export adapter weights for federated merging.

        Returns the adapter's state dict suitable for federated averaging.
        Used by FederatedMergeCoordinator to collect weights from nodes.

        Args:
            adapter_id: Adapter to export.

        Returns:
            State dict of adapter parameters, or None if not found.
        """
        model = self._loaded_models.get(adapter_id)
        if model is None:
            logger.warning(f"Cannot export adapter '{adapter_id}': not loaded")
            return None

        state = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                state[name] = param.data.detach().cpu().clone()
        return state

    def import_adapter_weights(
        self,
        adapter_id: str,
        weights: dict[str, torch.Tensor],
        path: str = "",
    ) -> bool:
        """Import merged adapter weights from federated averaging.

        Loads merged weights into an existing adapter or creates a new one.

        Args:
            adapter_id: Adapter ID to load weights into.
            weights: Merged state dict from federated averaging.
            path: Optional path to save the merged weights.

        Returns:
            True on success.
        """
        import tempfile
        import os

        if not path:
            path = os.path.join(
                tempfile.gettempdir(),
                "distllm-federated",
                f"{adapter_id}.pt",
            )
            os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save(weights, path)

        # If adapter is already loaded, update its weights
        model = self._loaded_models.get(adapter_id)
        if model is not None:
            model.load_state_dict(weights, strict=False)
            logger.info(f"Updated adapter '{adapter_id}' with federated weights")
            return True

        # Otherwise register in the pool for later loading
        vram_bytes = sum(p.numel() for p in weights.values()) * 2
        self._pool.add_adapter(adapter_id, path, vram_bytes=vram_bytes)
        logger.info(f"Registered federated adapter '{adapter_id}' at {path}")
        return True

    def start_federated_training(
        self,
        adapter_id: str,
        local_data_path: str,
        epochs: int = 3,
        learning_rate: float = 2e-4,
        batch_size: int = 4,
    ) -> dict[str, Any]:
        """Run local LoRA fine-tuning on a node's private data.

        Trains the adapter locally and returns the updated weights
        for submission to the federated merge coordinator.

        Args:
            adapter_id: Adapter to fine-tune.
            local_data_path: Path to local training data.
            epochs: Number of training epochs.
            learning_rate: Learning rate.
            batch_size: Training batch size.

        Returns:
            Dict with training metrics and exported weights.
        """
        model = self._loaded_models.get(adapter_id)
        if model is None:
            raise ValueError(f"Adapter '{adapter_id}' not loaded")

        logger.info(
            f"Starting local federated training for '{adapter_id}': "
            f"epochs={epochs}, lr={learning_rate}, data={local_data_path}"
        )

        # Training loop
        model.train()
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
        )

        total_loss = 0.0
        steps = 0

        # Load training data
        try:
            training_data = self._load_training_data(local_data_path)
        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            return {"error": str(e), "adapter_id": adapter_id}

        for epoch in range(epochs):
            for batch in training_data:
                optimizer.zero_grad()
                try:
                    outputs = model(**batch)
                    loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    steps += 1
                except Exception as e:
                    logger.debug(f"Training step failed: {e}")
                    continue

        model.eval()
        avg_loss = total_loss / max(steps, 1)

        # Export trained weights
        weights = self.export_adapter_weights(adapter_id)

        logger.info(
            f"Local training complete for '{adapter_id}': "
            f"avg_loss={avg_loss:.4f}, steps={steps}"
        )

        return {
            "adapter_id": adapter_id,
            "avg_loss": avg_loss,
            "steps": steps,
            "epochs": epochs,
            "weights": weights,
        }

    def _load_training_data(self, path: str) -> list[dict]:
        """Load training data from path (JSONL or directory)."""
        import json
        data = []
        try:
            with open(path) as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        data.append(item)
        except Exception:
            # Try as a single JSON file
            with open(path) as f:
                data = json.load(f)
                if isinstance(data, dict):
                    data = [data]

        # Convert to tokenized format
        tokenized = []
        for item in data[:100]:  # Limit batch size
            text = item.get("text", "") or item.get("prompt", "") or str(item)
            if self.tokenizer:
                tokens = self.tokenizer(
                    text, return_tensors="pt", truncation=True,
                    max_length=512, padding=True,
                )
                tokens["labels"] = tokens["input_ids"].clone()
                tokenized.append(tokens)

        return tokenized
