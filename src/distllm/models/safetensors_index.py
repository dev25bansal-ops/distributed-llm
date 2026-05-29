"""Parse model.safetensors.index.json to map transformer layers to shard files."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class SafetensorsIndexError(Exception):
    """Raised when safetensors index parsing fails."""


class SafetensorsIndex:
    """Maps transformer layer parameters to safetensors shard files.

    A HuggingFace safetensors index (``model.safetensors.index.json``)
    contains a ``weight_map`` that maps every parameter key to the shard
    file it lives in. This class parses that index and provides helpers
    to determine which shard files are needed for a given layer range.

    Typical layer parameter keys follow one of three conventions::

        model.layers.5.self_attn.q_proj.weight   # Llama, Mistral, Qwen2
        transformer.h.5.self_attention.q_proj.weight  # GPT-J, Bloom
        model.block.5.layer.attention.q_proj.weight  # rare
    """

    LAYER_PATTERN = re.compile(r"(?:\.layers|\.block|\.h)\.(\d+)\.")

    def __init__(self, index_data: dict) -> None:
        self.metadata: dict = index_data.get("metadata", {})
        self.weight_map: dict[str, str] = index_data.get("weight_map", {})
        self.total_size: int = self.metadata.get("total_size", 0)

    # ---- Construction helpers ----

    @classmethod
    def from_file(cls, path: str | Path) -> SafetensorsIndex:
        """Parse from a local ``model.safetensors.index.json`` file."""
        with open(path) as f:
            return cls(json.load(f))

    @classmethod
    def from_hub(
        cls,
        model_name: str,
        revision: str = "main",
        token: str | None = None,
    ) -> SafetensorsIndex:
        """Download and parse the index from HuggingFace Hub."""
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=model_name,
            filename="model.safetensors.index.json",
            revision=revision,
            token=token,
        )
        return cls.from_file(path)

    @classmethod
    def from_cache(cls, model_path: str | Path) -> SafetensorsIndex | None:
        """Try to load the index from a local cached model directory."""
        for candidate in ("model.safetensors.index.json",):
            p = Path(model_path) / candidate
            if p.exists():
                return cls.from_file(p)
        return None

    # ---- Layer helpers ----

    @staticmethod
    def get_layer_number(param_key: str) -> int | None:
        """Extract the transformer layer number from a parameter key.

        Returns ``None`` for non-layer parameters such as embeddings,
        final norm, or LM head.
        """
        match = SafetensorsIndex.LAYER_PATTERN.search(param_key)
        if match:
            return int(match.group(1))
        return None

    @staticmethod
    def is_layer_param(param_key: str) -> bool:
        """Whether *param_key* belongs to a transformer layer block."""
        return SafetensorsIndex.get_layer_number(param_key) is not None

    def is_layer_in_range(
        self, param_key: str, start_layer: int, end_layer: int
    ) -> bool:
        """Check whether *param_key* falls within ``[start, end]``.

        Non-layer parameters (embeddings, norm, LM head) always return
        ``True`` because **every** node needs them.
        """
        layer_num = self.get_layer_number(param_key)
        if layer_num is not None:
            return start_layer <= layer_num <= end_layer
        return True

    # ---- Shard queries ----

    def get_shards_for_layer_range(
        self, start_layer: int, end_layer: int
    ) -> set[str]:
        """Return the minimum set of shard filenames for a layer range.

        Includes the index file itself plus every shard that contains at
        least one parameter key that falls inside ``[start, end]``.
        """
        needed: set[str] = set()
        for param_key, shard_file in self.weight_map.items():
            if self.is_layer_in_range(param_key, start_layer, end_layer):
                needed.add(shard_file)
        needed.add("model.safetensors.index.json")
        return needed

    def get_keys_for_layer_range(
        self, start_layer: int, end_layer: int
    ) -> set[str]:
        """Return parameter keys that belong to the target layer range."""
        return {
            k
            for k in self.weight_map
            if self.is_layer_in_range(k, start_layer, end_layer)
        }

    @property
    def all_shard_files(self) -> list[str]:
        """All unique shard file names listed in the index."""
        return sorted(set(self.weight_map.values()))

    # ---- Introspection ----

    @staticmethod
    def is_sharded_model(
        model_name: str,
        revision: str = "main",
        token: str | None = None,
    ) -> bool:
        """Check whether the model uses safetensors sharding.

        Returns ``False`` for models that use a single
        ``model.safetensors`` file.
        """
        try:
            from huggingface_hub import hf_hub_download

            hf_hub_download(
                repo_id=model_name,
                filename="model.safetensors.index.json",
                revision=revision,
                token=token,
            )
            return True
        except Exception:
            return False

    def summary(self) -> str:
        """Human-readable summary of the index."""
        layer_params = sum(1 for k in self.weight_map if self.is_layer_param(k))
        non_layer_params = len(self.weight_map) - layer_params
        return (
            f"SafetensorsIndex("
            f"shards={len(self.all_shard_files)}, "
            f"total_params={len(self.weight_map)}, "
            f"layer_params={layer_params}, "
            f"non_layer_params={non_layer_params}, "
            f"total_size={self.total_size})"
        )

    def __repr__(self) -> str:
        return self.summary()
