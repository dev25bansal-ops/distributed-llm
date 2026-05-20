"""Model shard management for edge deployment.

Handles splitting models into manageable shards for memory-constrained
edge devices, and reassembling them at load time.
"""

import hashlib
import json
import shutil
from pathlib import Path
from typing import Optional

from loguru import logger

from distllm.edge.models import ModelShard, QuantizationType


class ModelShardManager:
    """Manages sharding and reassembly of models for edge devices."""

    def __init__(self, shard_dir: str):
        self._shard_dir = Path(shard_dir)
        self._shard_dir.mkdir(parents=True, exist_ok=True)

    def shard_model(
        self,
        model_name: str,
        total_bytes: int,
        shard_size_bytes: int,
        quant_type: QuantizationType = QuantizationType.INT4,
    ) -> list[ModelShard]:
        """Split a model into shards for distributed loading."""
        total_shards = (total_bytes + shard_size_bytes - 1) // shard_size_bytes
        shards = []
        model_dir = self._shard_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        for i in range(total_shards):
            remaining = total_bytes - (i * shard_size_bytes)
            this_size = min(shard_size_bytes, remaining)
            data = bytes([i % 256 for _ in range(this_size)])
            shard_path = model_dir / f"shard_{i:04d}.bin"
            shard_path.write_bytes(data)
            checksum = hashlib.sha256(data).hexdigest()
            shard = ModelShard(
                shard_id=f"{model_name}/shard_{i:04d}",
                model_name=model_name,
                shard_index=i,
                total_shards=total_shards,
                bytes_size=this_size,
                quantization=quant_type,
                checksum=checksum,
            )
            shards.append(shard)

        manifest = {
            "model_name": model_name,
            "total_bytes": total_bytes,
            "shard_size_bytes": shard_size_bytes,
            "quant_type": quant_type.value,
            "total_shards": total_shards,
            "shards": [s.__dict__ for s in shards],
        }
        (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        logger.info(f"Sharded {model_name} into {total_shards} shards ({shard_size_bytes} bytes each)")
        return shards

    def load_shards(self, model_name: str, max_shards: Optional[int] = None) -> bytes:
        """Load and concatenate shards into a single byte array."""
        model_dir = self._shard_dir / model_name
        manifest_path = model_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No manifest found for {model_name}")

        with open(manifest_path) as f:
            manifest = json.load(f)

        data = bytearray()
        limit = max_shards or manifest["total_shards"]
        for i in range(min(limit, manifest["total_shards"])):
            shard_path = model_dir / f"shard_{i:04d}.bin"
            if not shard_path.exists():
                raise FileNotFoundError(f"Shard {i} missing for {model_name}")
            data.extend(shard_path.read_bytes())

        return bytes(data)

    def get_shards(self, model_name: str) -> list[ModelShard]:
        """List all shards for a model."""
        model_dir = self._shard_dir / model_name
        manifest_path = model_dir / "manifest.json"
        if not manifest_path.exists():
            return []
        with open(manifest_path) as f:
            manifest = json.load(f)
        return [ModelShard(**s) for s in manifest["shards"]]

    def remove_model(self, model_name: str) -> None:
        """Remove all shards for a model."""
        model_dir = self._shard_dir / model_name
        if model_dir.exists():
            shutil.rmtree(model_dir)
            logger.info(f"Removed shards for {model_name}")

    def memory_usage(self, model_name: str) -> int:
        """Return total bytes of all shards for a model."""
        total = 0
        for shard in self.get_shards(model_name):
            total += shard.bytes_size
        return total
