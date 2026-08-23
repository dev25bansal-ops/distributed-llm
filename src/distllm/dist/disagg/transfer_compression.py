"""KV cache compression during transfer between disaggregated prefill/decode nodes.

Provides three tiers of compression for KV cache transfer:

1. ``KVCacheCompressor`` — FP8 / INT8 quantization (2x reduction).
2. ``ProgressiveTransfer`` — send FP8 first, residual later.
3. ``TransferCompressionController`` — bandwidth-aware method selection.

Usage::

    from distllm.dist.disagg.transfer_compression import (
        KVCacheCompressor,
        ProgressiveTransfer,
        TransferCompressionController,
    )

    # Direct FP8 compression
    compressor = KVCacheCompressor(method="fp8")
    compressed_bytes, meta = compressor.compress(kv_cache)
    restored = compressor.decompress(compressed_bytes, meta)

    # Progressive transfer (FP8 first, residual later)
    pt = ProgressiveTransfer()
    quant, resid = pt.progressive_compress(kv_cache)
    approx = pt.progressive_decompress(quant)           # fast, lossy
    exact  = pt.progressive_decompress(quant, resid)    # full precision

    # Bandwidth-aware selection
    ctrl = TransferCompressionController()
    payload = ctrl.compress_for_transfer(kv_cache, bandwidth_bps=2_000_000_000)
    restored = ctrl.decompress(payload)
"""

from __future__ import annotations

import io
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from loguru import logger

# ---------------------------------------------------------------------------
# Bandwidth thresholds (bps)
# ---------------------------------------------------------------------------
# Above this threshold no compression is applied.
HIGH_BANDWIDTH_THRESHOLD: int = 10_000_000_000  # 10 Gbps
# Between this and HIGH_BANDWIDTH_THRESHOLD use FP8.
MEDIUM_BANDWIDTH_THRESHOLD: int = 1_000_000_000  # 1 Gbps
# Below MEDIUM_BANDWIDTH_THRESHOLD use progressive transfer.

# Literal type aliases
CompressMethod = str  # "fp8" | "int8" | "none"


def _fp8_available() -> bool:
    """Return True if the current torch build supports float8_e4m3fn."""
    return hasattr(torch, "float8_e4m3fn") and hasattr(
        torch, "float8_e5m2"
    )


# ---------------------------------------------------------------------------
# Payload container
# ---------------------------------------------------------------------------


@dataclass
class CompressedPayload:
    """Carries a compressed KV cache and reconstruction metadata.

    Attributes:
        method: Compression method used (``"fp8"``, ``"int8"``, ``"none"``,
            or ``"progressive"``).
        compressed_bytes: Serialised quantised (or raw) tensor data.
        metadata: Dictionary with shapes, dtypes and scale metadata needed
            to reconstruct the original tensors.
        residual_bytes: Optional residual data for progressive transfer.
    """

    method: str
    compressed_bytes: bytes
    metadata: dict[str, Any] = field(default_factory=dict)
    residual_bytes: Optional[bytes] = None


# ---------------------------------------------------------------------------
# KVCacheCompressor
# ---------------------------------------------------------------------------


class KVCacheCompressor:
    """Compresses and decompresses KV cache tensors for transfer.

    Applies FP8 or INT8 per-tensor quantization to each (key, value) pair
    in the cache, achieving approximately 2x reduction for FP8 and 2x for
    INT8 (vs FP16).

    The compressor produces a pair ``(compressed_bytes, metadata)`` so that
    the two can be sent separately over the wire — metadata is tiny and can
    travel ahead-of-band.

    Args:
        method: Default quantization method — ``"fp8"`` (default) or
            ``"int8"``.  Falls back to INT8 when FP8 is unavailable at
            runtime.
    """

    def __init__(self, method: CompressMethod = "fp8") -> None:
        if method not in ("fp8", "int8", "none"):
            raise ValueError(
                f"Unknown compression method '{method}'; "
                f"expected 'fp8', 'int8', or 'none'"
            )
        if method == "fp8" and not _fp8_available():
            logger.warning(
                "FP8 not available in this torch build; falling back to INT8"
            )
            method = "int8"
        self.method: str = method
        self._lock = threading.Lock()
        self._stats: dict[str, Any] = {
            "compress_calls": 0,
            "decompress_calls": 0,
            "total_original_bytes": 0,
            "total_compressed_bytes": 0,
            "method": method,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress(
        self,
        kv_cache: dict[str, tuple[torch.Tensor, torch.Tensor]],
        method: Optional[str] = None,
    ) -> tuple[bytes, dict[str, Any]]:
        """Compress a full KV cache.

        Args:
            kv_cache: Mapping of ``{layer_name: (key_tensor, value_tensor)}``.
                Each tensor has shape ``(num_heads, seq_len, head_dim)`` and
                dtype ``float16`` or ``bfloat16``.
            method: Override the instance-level default method.

        Returns:
            ``(compressed_bytes, metadata)``.  The bytes are self-contained
            quantised tensor data; the metadata dict holds shapes, dtypes
            and scale shapes needed for :meth:`decompress`.
        """
        method = method or self.method

        with self._lock:
            self._stats["compress_calls"] += 1

        original_bytes = sum(
            k.element_size() * k.numel() + v.element_size() * v.numel()
            for k, v in kv_cache.values()
        )
        with self._lock:
            self._stats["total_original_bytes"] += original_bytes

        if method == "none":
            return self._compress_none(kv_cache)

        per_layer: dict[str, dict[str, Any]] = {}
        layers_meta: dict[str, dict[str, Any]] = {}

        for layer_name, (key, value) in kv_cache.items():
            key = key.detach().contiguous()
            value = value.detach().contiguous()

            if method == "fp8":
                k_quant, k_scale = self._compress_fp8(key)
                v_quant, v_scale = self._compress_fp8(value)
            else:  # int8
                k_quant, k_scale = self._compress_int8(key)
                v_quant, v_scale = self._compress_int8(value)

            per_layer[layer_name] = {
                "k_quant": self._serialize_tensor(k_quant),
                "v_quant": self._serialize_tensor(v_quant),
                "k_scale": self._serialize_tensor(k_scale),
                "v_scale": self._serialize_tensor(v_scale),
            }
            layers_meta[layer_name] = {
                "key_shape": list(key.shape),
                "value_shape": list(value.shape),
                "key_dtype": str(key.dtype),
                "value_dtype": str(value.dtype),
                "key_scale_shape": list(k_scale.shape),
                "value_scale_shape": list(v_scale.shape),
            }

        buf = io.BytesIO()
        torch.save(per_layer, buf)
        compressed_bytes = buf.getvalue()

        metadata: dict[str, Any] = {
            "method": method,
            "layers": layers_meta,
        }

        with self._lock:
            self._stats["total_compressed_bytes"] += len(compressed_bytes)

        return compressed_bytes, metadata

    def decompress(
        self,
        compressed: bytes,
        metadata: dict[str, Any],
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Decompress a KV cache that was previously compressed with
        :meth:`compress`.

        Args:
            compressed: Byte stream from ``compress``.
            metadata: Metadata dict from ``compress``.

        Returns:
            Mapping of ``{layer_name: (key_tensor, value_tensor)}`` restored
            to the original dtype and shape.
        """
        with self._lock:
            self._stats["decompress_calls"] += 1

        method = metadata.get("method", self.method)

        if method == "none":
            return self._decompress_none(compressed, metadata)

        per_layer: dict[str, dict[str, Any]] = torch.load(
            io.BytesIO(compressed), map_location="cpu", weights_only=True
        )

        restored: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for layer_name, layer_meta in metadata.get("layers", {}).items():
            data = per_layer.get(layer_name, {})
            if not data:
                logger.warning("Missing layer %s in compressed data", layer_name)
                continue

            k_quant = self._deserialize_tensor(
                data["k_quant"],
                layer_meta.get("key_quant_dtype", None),
            )
            v_quant = self._deserialize_tensor(
                data["v_quant"],
                layer_meta.get("value_quant_dtype", None),
            )
            k_scale = self._deserialize_tensor(data["k_scale"])
            v_scale = self._deserialize_tensor(data["v_scale"])

            target_dtype = self._str_to_dtype(
                layer_meta.get("key_dtype", "torch.float16")
            )
            key = self._decompress_quantized(k_quant, k_scale, target_dtype)
            value = self._decompress_quantized(v_quant, v_scale, target_dtype)

            # Reshape to original dimensions if needed
            key = key.reshape(layer_meta["key_shape"])
            value = value.reshape(layer_meta["value_shape"])

            restored[layer_name] = (key, value)

        return restored

    # ------------------------------------------------------------------
    # Compression internals
    # ------------------------------------------------------------------

    def _compress_fp8(
        self, tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """FP8 quantization: map FP16 -> float8_e4m3fn with per-tensor scale.

        Returns ``(quantized, scale)``.
        """
        amax = tensor.abs().max().clamp(min=1e-12)
        scale = (amax / 448.0).float()
        quantized = (tensor.float() / scale).to(torch.float8_e4m3fn)
        return quantized, scale.reshape(1)

    def _compress_int8(
        self, tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """INT8 quantization: map FP16 -> int8 with per-tensor scale.

        Returns ``(quantized, scale)``.
        """
        amax = tensor.abs().max().clamp(min=1e-12)
        scale = (amax / 127.0).float()
        quantized = (tensor.float() / scale).clamp(-128.0, 127.0).to(torch.int8)
        return quantized, scale.reshape(1)

    def _compress_none(
        self,
        kv_cache: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[bytes, dict[str, Any]]:
        """Serialize the KV cache without any compression.

        Used by the controller when bandwidth is plentiful.
        """
        raw: dict[str, dict[str, Any]] = {}
        metadata_layers: dict[str, dict[str, Any]] = {}
        for layer_name, (key, value) in kv_cache.items():
            key_cpu = key.detach().contiguous().cpu()
            value_cpu = value.detach().contiguous().cpu()
            raw[layer_name] = {
                "key": self._serialize_tensor(key_cpu),
                "value": self._serialize_tensor(value_cpu),
            }
            metadata_layers[layer_name] = {
                "key_shape": list(key.shape),
                "value_shape": list(value.shape),
                "key_dtype": str(key.dtype),
                "value_dtype": str(value.dtype),
            }

        buf = io.BytesIO()
        torch.save(raw, buf)
        compressed_bytes = buf.getvalue()

        with self._lock:
            self._stats["total_compressed_bytes"] += len(compressed_bytes)

        return compressed_bytes, {
            "method": "none",
            "layers": metadata_layers,
        }

    def _decompress_none(
        self,
        compressed: bytes,
        metadata: dict[str, Any],
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Deserialize an uncompressed KV cache."""
        raw: dict[str, dict[str, Any]] = torch.load(
            io.BytesIO(compressed), map_location="cpu", weights_only=True
        )
        restored: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for layer_name, layer_meta in metadata.get("layers", {}).items():
            data = raw.get(layer_name, {})
            if not data:
                continue
            key = self._deserialize_tensor(
                data["key"],
                self._str_to_dtype(layer_meta.get("key_dtype", "torch.float16")),
            ).reshape(layer_meta["key_shape"])
            value = self._deserialize_tensor(
                data["value"],
                self._str_to_dtype(layer_meta.get("value_dtype", "torch.float16")),
            ).reshape(layer_meta["value_shape"])
            restored[layer_name] = (key, value)
        return restored

    @staticmethod
    def _decompress_quantized(
        quantized: torch.Tensor,
        scale: torch.Tensor,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Reverse FP8 or INT8 quantization."""
        restored = quantized.float() * scale.float()
        return restored.to(target_dtype)

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_tensor(tensor: torch.Tensor) -> bytes:
        """Serialize a single tensor to a byte string."""
        buf = io.BytesIO()
        torch.save(tensor, buf)
        return buf.getvalue()

    @staticmethod
    def _deserialize_tensor(
        data: bytes,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Deserialize a single tensor from a byte string.

        Args:
            data: Byte string from ``_serialize_tensor``.
            dtype: If provided, cast the tensor to this dtype after loading.

        Returns:
            Deserialised tensor on CPU.
        """
        tensor: torch.Tensor = torch.load(
            io.BytesIO(data), map_location="cpu", weights_only=True
        )
        if dtype is not None:
            tensor = tensor.to(dtype)
        return tensor

    @staticmethod
    def _str_to_dtype(dtype_str: str) -> torch.dtype:
        """Convert a string like ``"torch.float16"`` to ``torch.float16``."""
        mapping: dict[str, torch.dtype] = {
            "torch.float16": torch.float16,
            "torch.bfloat16": torch.bfloat16,
            "torch.float32": torch.float32,
            "torch.float64": torch.float64,
            "torch.int8": torch.int8,
            "torch.int16": torch.int16,
            "torch.int32": torch.int32,
            "torch.int64": torch.int64,
            "torch.uint8": torch.uint8,
            "torch.float8_e4m3fn": torch.float8_e4m3fn,
        }
        if dtype_str not in mapping:
            raise ValueError(f"Unrecognised dtype string: {dtype_str}")
        return mapping[dtype_str]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def compression_ratio(self) -> float:
        with self._lock:
            orig = self._stats["total_original_bytes"]
            comp = self._stats["total_compressed_bytes"]
        return comp / max(orig, 1)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                **self._stats,
                "compression_ratio": round(self.compression_ratio, 4),
                "method": self.method,
            }

    def __repr__(self) -> str:
        return (
            f"KVCacheCompressor(method={self.method}, "
            f"ratio={self.compression_ratio:.2f})"
        )


# ---------------------------------------------------------------------------
# ProgressiveTransfer
# ---------------------------------------------------------------------------


class ProgressiveTransfer:
    """Two-stage progressive KV cache transfer.

    Stage 1 (``quantized``) sends an FP8-quantised approximation of the
    KV cache, achieving ~50% reduction vs FP16.  The receiver can decode
    immediately for approximate generation.

    Stage 2 (``residual``) sends the residual
    ``original - dequantize(FP8)`` so the receiver can reconstruct the
    exact original cache.

    Usage::

        pt = ProgressiveTransfer()
        quant_payload, resid_payload = pt.progressive_compress(kv_cache)

        # Receiver can decode after stage 1 for lossy (fast) generation:
        approx_cache = pt.progressive_decompress(quant_payload)

        # Or wait for stage 2 for full precision:
        exact_cache = pt.progressive_decompress(quant_payload, resid_payload)
    """

    def __init__(self) -> None:
        self._compressor = KVCacheCompressor(method="fp8")
        self._stats: dict[str, Any] = {
            "progressive_compress_calls": 0,
            "progressive_decompress_calls": 0,
            "approximate_decompress_calls": 0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def progressive_compress(
        self,
        kv_cache: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Progressive compress a KV cache.

        Args:
            kv_cache: Mapping ``{layer_name: (key, value)}``.

        Returns:
            ``(quantized, residual)``.  Both are self-describing dicts that
            can be serialised independently (e.g. via ``pickle`` or
            ``torch.save``) and sent over the wire.

            - ``quantized`` contains FP8-quantised tensors and their scales,
              enough to reconstruct an approximate version.
            - ``residual`` contains the FP16 residual tensors needed for
              exact reconstruction.
        """
        self._stats["progressive_compress_calls"] += 1

        compressed_bytes, meta = self._compressor.compress(
            kv_cache, method="fp8"
        )

        quantized: dict[str, Any] = {
            "method": "fp8",
            "compressed_bytes": compressed_bytes,
            "metadata": meta,
        }

        # Compute residual: original - dequantised(FP8)
        approx = self.progressive_decompress(quantized)
        residual: dict[str, dict[str, bytes]] = {"method": "residual", "layers": {}}
        for layer_name, (orig_key, orig_val) in kv_cache.items():
            app_key, app_val = approx[layer_name]
            k_resid = orig_key.detach().contiguous().cpu().float() - app_key.float()
            v_resid = orig_val.detach().contiguous().cpu().float() - app_val.float()
            residual["layers"][layer_name] = {
                "k_residual": self._compressor._serialize_tensor(k_resid.half()),
                "v_residual": self._compressor._serialize_tensor(v_resid.half()),
                "key_shape": list(orig_key.shape),
                "value_shape": list(orig_val.shape),
            }

        return quantized, residual

    def progressive_decompress(
        self,
        quantized: dict[str, Any],
        residual: Optional[dict[str, Any]] = None,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Decompress a progressively compressed KV cache.

        Args:
            quantized: The first-stage payload from
                :meth:`progressive_compress`.
            residual: Optional second-stage payload.  When ``None`` the
                returned cache is an approximate FP8-only reconstruction.
                When provided the exact original is returned.

        Returns:
            Mapping ``{layer_name: (key_tensor, value_tensor)}``.
        """
        if residual is None:
            self._stats["approximate_decompress_calls"] += 1
        self._stats["progressive_decompress_calls"] += 1

        # Always decode the FP8 approximation first.
        approx = self._compressor.decompress(
            quantized["compressed_bytes"],
            quantized["metadata"],
        )

        if residual is None:
            return approx

        # Add residual for exact reconstruction.
        restored: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for layer_name, (app_key, app_val) in approx.items():
            resid_data = residual.get("layers", {}).get(layer_name)
            if resid_data is None:
                logger.warning(
                    "Missing residual for layer %s; returning approximate", layer_name
                )
                restored[layer_name] = (app_key, app_val)
                continue

            k_resid = self._compressor._deserialize_tensor(
                resid_data["k_residual"], torch.float16
            )
            v_resid = self._compressor._deserialize_tensor(
                resid_data["v_residual"], torch.float16
            )

            key = (app_key.float() + k_resid.float()).to(app_key.dtype)
            value = (app_val.float() + v_resid.float()).to(app_val.dtype)

            # Restore original shape (residual may have been flattened).
            key = key.reshape(resid_data["key_shape"])
            value = value.reshape(resid_data["value_shape"])

            restored[layer_name] = (key, value)

        return restored

    def stats(self) -> dict[str, Any]:
        return {**self._stats}

    def __repr__(self) -> str:
        return (
            f"ProgressiveTransfer(calls={self._stats['progressive_compress_calls']})"
        )


# ---------------------------------------------------------------------------
# TransferCompressionController
# ---------------------------------------------------------------------------


class TransferCompressionController:
    """Bandwidth-aware compression method selector.

    Selects the best compression strategy based on available bandwidth:

    =================  ===================================================
    Bandwidth          Strategy
    =================  ===================================================
    ``>= 10 Gbps``    No compression — send raw FP16 tensors.
    ``1-10 Gbps``     FP8 quantization — 2x reduction with negligible
                      quality loss.
    ``< 1 Gbps``      Progressive transfer — send FP8 first, residual
                      later so the receiver can start decoding faster.
    =================  ===================================================

    Usage::

        ctrl = TransferCompressionController()
        payload = ctrl.compress_for_transfer(kv_cache, bandwidth_bps=2e9)
        # payload.method == "fp8"
        restored = ctrl.decompress(payload)
    """

    def __init__(
        self,
        high_threshold: int = HIGH_BANDWIDTH_THRESHOLD,
        medium_threshold: int = MEDIUM_BANDWIDTH_THRESHOLD,
    ) -> None:
        self._high_threshold = high_threshold
        self._medium_threshold = medium_threshold
        self._compressor = KVCacheCompressor(method="fp8")
        self._progressive = ProgressiveTransfer()
        self._stats: dict[str, Any] = {
            "compress_calls": 0,
            "decompress_calls": 0,
            "method_counts": {"none": 0, "fp8": 0, "progressive": 0},
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress_for_transfer(
        self,
        kv_cache: dict[str, tuple[torch.Tensor, torch.Tensor]],
        bandwidth_bps: float,
    ) -> CompressedPayload:
        """Compress a KV cache using the method best suited for the given
        bandwidth.

        Args:
            kv_cache: Mapping ``{layer_name: (key, value)}``.
            bandwidth_bps: Available bandwidth in bits per second.

        Returns:
            A :class:`CompressedPayload` containing the compressed data,
            metadata, and optional residual.
        """
        self._stats["compress_calls"] += 1

        if bandwidth_bps >= self._high_threshold:
            return self._compress_high_bandwidth(kv_cache)
        elif bandwidth_bps >= self._medium_threshold:
            return self._compress_medium_bandwidth(kv_cache)
        else:
            return self._compress_low_bandwidth(kv_cache)

    def decompress(
        self,
        payload: CompressedPayload,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Decompress a :class:`CompressedPayload` previously produced by
        :meth:`compress_for_transfer`.

        Args:
            payload: The compressed payload.

        Returns:
            The original KV cache mapping.
        """
        self._stats["decompress_calls"] += 1

        if payload.method == "none":
            return self._compressor.decompress(
                payload.compressed_bytes, payload.metadata
            )
        elif payload.method == "fp8":
            return self._compressor.decompress(
                payload.compressed_bytes, payload.metadata
            )
        elif payload.method == "progressive":
            quantized: dict[str, Any] = {
                "method": "fp8",
                "compressed_bytes": payload.compressed_bytes,
                "metadata": payload.metadata,
            }
            residual: Optional[dict[str, Any]] = None
            if payload.residual_bytes is not None:
                buf = io.BytesIO(payload.residual_bytes)
                residual = torch.load(buf, map_location="cpu", weights_only=True)
            return self._progressive.progressive_decompress(quantized, residual)
        else:
            raise ValueError(f"Unknown payload method: {payload.method}")

    # ------------------------------------------------------------------
    # Per-tier compression
    # ------------------------------------------------------------------

    def _compress_high_bandwidth(
        self,
        kv_cache: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> CompressedPayload:
        """High bandwidth: send raw FP16 tensors (no compression)."""
        self._stats["method_counts"]["none"] += 1
        logger.debug("High bandwidth: no compression")
        compressed_bytes, metadata = self._compressor.compress(
            kv_cache, method="none"
        )
        return CompressedPayload(
            method="none",
            compressed_bytes=compressed_bytes,
            metadata=metadata,
        )

    def _compress_medium_bandwidth(
        self,
        kv_cache: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> CompressedPayload:
        """Medium bandwidth: FP8 quantization."""
        self._stats["method_counts"]["fp8"] += 1
        logger.debug("Medium bandwidth: FP8 compression")
        compressed_bytes, metadata = self._compressor.compress(
            kv_cache, method="fp8"
        )
        return CompressedPayload(
            method="fp8",
            compressed_bytes=compressed_bytes,
            metadata=metadata,
        )

    def _compress_low_bandwidth(
        self,
        kv_cache: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> CompressedPayload:
        """Low bandwidth: progressive transfer (FP8 + optional residual)."""
        self._stats["method_counts"]["progressive"] += 1
        logger.debug("Low bandwidth: progressive transfer")
        quantized, residual = self._progressive.progressive_compress(kv_cache)

        # Serialise the residual so it can be sent separately.
        residual_bytes: Optional[bytes] = None
        if residual:
            buf = io.BytesIO()
            torch.save(residual, buf)
            residual_bytes = buf.getvalue()

        return CompressedPayload(
            method="progressive",
            compressed_bytes=quantized["compressed_bytes"],
            metadata=quantized["metadata"],
            residual_bytes=residual_bytes,
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        return {**self._stats}

    def __repr__(self) -> str:
        counts = self._stats["method_counts"]
        return (
            f"TransferCompressionController("
            f"none={counts['none']}, fp8={counts['fp8']}, "
            f"progressive={counts['progressive']})"
        )


__all__ = [
    "CompressedPayload",
    "HIGH_BANDWIDTH_THRESHOLD",
    "MEDIUM_BANDWIDTH_THRESHOLD",
    "KVCacheCompressor",
    "ProgressiveTransfer",
    "TransferCompressionController",
]
