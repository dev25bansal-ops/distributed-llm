"""Compression negotiation, adaptive serialization, and GPU-direct tensor transfer."""

from __future__ import annotations

import enum
import threading
from typing import Any, Callable

import numpy as np
import torch
from loguru import logger

from distllm.dist.pipeline.serialization import (
    get_tensor_copy_stream,
    tensor_quantize,
    tensor_dequantize,
)


# ---------------------------------------------------------------------------
# 1.  Compression negotiation
# ---------------------------------------------------------------------------


class CompressionMethod(str, enum.Enum):
    """Supported compression methods, ordered by priority (highest first)."""

    ZSTD = "zstd"
    LZ4 = "lz4"
    NONE = "none"


# Priority list: higher index = lower priority
_COMPRESSION_PRIORITY: list[CompressionMethod] = [
    CompressionMethod.ZSTD,
    CompressionMethod.LZ4,
    CompressionMethod.NONE,
]

_COMPRESSION_PRIORITY_INDEX = {m: i for i, m in enumerate(_COMPRESSION_PRIORITY)}


def _best_common(
    local: set[CompressionMethod],
    remote: set[CompressionMethod],
) -> CompressionMethod:
    """Return the highest-priority method that both sides support."""
    common = local & remote
    if not common:
        return CompressionMethod.NONE
    # lowest index = highest priority
    return min(common, key=lambda m: _COMPRESSION_PRIORITY_INDEX[m])


class CompressionNegotiator:
    """Peers exchange supported compression methods; best common is cached per peer.

    Priority: zstd > lz4 > none.
    """

    def __init__(
        self,
        supported: set[CompressionMethod] | None = None,
    ) -> None:
        self._local_supported: set[CompressionMethod] = (
            supported
            if supported is not None
            else {CompressionMethod.ZSTD, CompressionMethod.LZ4, CompressionMethod.NONE}
        )
        self._lock = threading.Lock()
        # peer_id -> CompressionMethod
        self._cache: dict[str, CompressionMethod] = {}

    # -- public API ---------------------------------------------------------

    @property
    def local_supported(self) -> set[CompressionMethod]:
        """Return the set this instance supports."""
        return self._local_supported.copy()

    def negotiate(
        self,
        peer_id: str,
        remote_supported: set[CompressionMethod],
    ) -> CompressionMethod:
        """Determine best common method with *peer_id* and cache the result."""
        chosen = _best_common(self._local_supported, remote_supported)
        with self._lock:
            self._cache[peer_id] = chosen
        return chosen

    def get_cached(self, peer_id: str) -> CompressionMethod | None:
        """Return the previously negotiated method for *peer_id*, or *None*."""
        with self._lock:
            return self._cache.get(peer_id)

    def invalidate(self, peer_id: str) -> None:
        """Remove cached result for *peer_id* so a fresh negotiation is forced."""
        with self._lock:
            self._cache.pop(peer_id, None)

    def clear_cache(self) -> None:
        """Drop all cached negotiation results."""
        with self._lock:
            self._cache.clear()

    @staticmethod
    def default_supported() -> set[CompressionMethod]:
        return {CompressionMethod.ZSTD, CompressionMethod.LZ4, CompressionMethod.NONE}


# ---------------------------------------------------------------------------
# 2.  Adaptive serialization format
# ---------------------------------------------------------------------------


class SerializationFormat(str, enum.Enum):
    """Tag for the serialization format produced by AdaptiveSerializer."""

    RAW = "raw"  # raw bytes, no compression
    ZSTD = "zstd"  # zstd-compressed raw bytes
    FP8 = "fp8"  # FP8-quantised tensor data
    FP8_ZSTD = "fp8_zstd"  # FP8 quantisation followed by zstd


# Internal thresholds (bytes)
_SMALL_TENSOR_BYTES = 1_000_000  # 1 MB
_LARGE_TENSOR_BYTES = 100_000_000  # 100 MB


class AdaptiveSerializer:
    """Selects serialization format based on tensor size and type.

    Rules
    -----
    * Small tensors (< 1 MB)         : RAW (no overhead)
    * Medium tensors (1 – 100 MB)    : ZSTD
    * Large tensors (> 100 MB)       : FP8 + optional ZSTD
    """

    def __init__(
        self,
        small_threshold: int = _SMALL_TENSOR_BYTES,
        large_threshold: int = _LARGE_TENSOR_BYTES,
        zstd_level: int = 3,
        fp8_zstd_level: int = 1,
        compression_ctx: Any = None,
    ) -> None:
        """Initialize AdaptiveSerializer.

        Parameters
        ----------
        small_threshold:
            Tensors with ``element_size * numel <= small_threshold`` use RAW.
        large_threshold:
            Tensors larger than this use FP8 (+ZSTD).  Medium tensors use ZSTD.
        zstd_level:
            Compression level for medium tensors (pure ZSTD).
        fp8_zstd_level:
            Compression level for large tensors (FP8 + ZSTD).
        compression_ctx:
            Optional ZstdCompressionContext for dict-based / streaming zstd.
        """
        self._small_threshold = small_threshold
        self._large_threshold = large_threshold
        self._zstd_level = zstd_level
        self._fp8_zstd_level = fp8_zstd_level
        self._compression_ctx = compression_ctx
        self._import_zstd()

    # -- zstd lazy import ---------------------------------------------------

    _zstd: Any = None  # import-once cache

    @classmethod
    def _import_zstd(cls) -> None:
        if cls._zstd is not None:
            return
        try:
            import zstandard as _z  # type: ignore[import-untyped]

            cls._zstd = _z
        except ImportError:
            try:
                import zstd as _z  # type: ignore[import-untyped,no-redef]

                cls._zstd = _z
            except ImportError:
                cls._zstd = None  # type: ignore[assignment]

    # -- compress / decompress helpers --------------------------------------

    def _zstd_compress(self, data: bytes) -> bytes:
        if self._zstd is None:
            self._import_zstd()
        if self._zstd is None:
            logger.warning("zstd not available, falling back to raw")
            return data
        if hasattr(self._zstd, "compress"):
            return self._zstd.compress(data, self._zstd_level)  # type: ignore[no-any-return]
        # pyzstd / zstandard API
        return self._zstd.ZstdCompressor(level=self._zstd_level).compress(data)  # type: ignore[no-any-return]

    def _zstd_decompress(self, data: bytes) -> bytes:
        if self._zstd is None:
            self._import_zstd()
        if self._zstd is None:
            logger.warning("zstd not available, raw data returned unchanged")
            return data
        if hasattr(self._zstd, "decompress"):
            return self._zstd.decompress(data)  # type: ignore[no-any-return]
        return self._zstd.ZstdDecompressor().decompress(data)  # type: ignore[no-any-return]

    def _lz4_compress(self, data: bytes) -> bytes:
        try:
            import lz4.frame  # type: ignore[import-untyped]

            return lz4.frame.compress(data)  # type: ignore[no-any-return]
        except ImportError:
            logger.warning("lz4 not available, returning uncompressed")
            return data

    def _lz4_decompress(self, data: bytes) -> bytes:
        try:
            import lz4.frame

            return lz4.frame.decompress(data)  # type: ignore[no-any-return]
        except ImportError:
            logger.warning("lz4 not available, returning data unchanged")
            return data

    # -- public API ---------------------------------------------------------

    def choose_format(self, tensor: torch.Tensor) -> SerializationFormat:
        """Pick the best format for *tensor* without performing serialisation."""
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes < self._small_threshold:
            return SerializationFormat.RAW
        if nbytes > self._large_threshold:
            return SerializationFormat.FP8_ZSTD
        return SerializationFormat.ZSTD

    def serialize(self, tensor: torch.Tensor) -> tuple[SerializationFormat, bytes]:
        """Convert *tensor* to bytes, returning the format tag and payload.

        Returns
        -------
        (format, bytes)
        """
        fmt = self.choose_format(tensor)

        if fmt == SerializationFormat.RAW:
            return fmt, self._serialize_raw(tensor)

        if fmt == SerializationFormat.ZSTD:
            return fmt, self._zstd_compress(self._serialize_raw(tensor))

        # FP8_ZSTD: quantise to fp8, optionally compress
        quantized, scale = tensor_quantize(
            tensor,
            enabled=True,
            bits=8,
            use_fp8=True,
        )
        raw_quantized = self._serialize_raw(quantized)
        payload = (
            self._zstd_compress(raw_quantized)
            if self._fp8_zstd_level > 0
            else raw_quantized
        )
        # Store scale (if any) inline — 4-byte float32 header.
        if scale is not None:
            scale_bytes = np.array(scale, dtype=np.float32).tobytes()
        else:
            scale_bytes = b""
        return fmt, _pack_scale(scale_bytes, payload)

    def deserialize(
        self,
        fmt: SerializationFormat,
        data: bytes,
        dtype: torch.dtype | None = None,
        shape: list[int] | None = None,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Reconstruct a tensor from serialised *data*.

        Parameters
        ----------
        fmt:
            The format tag returned by :meth:`serialize`.
        data:
            The byte payload.
        dtype:
            Original dtype (needed when FP8 was used).
        shape:
            Original shape (needed when FP8 was used).
        device:
            Target device for the reconstructed tensor.

        **NOTE**:
        For ``FP8_ZSTD`` format the caller **must** supply *dtype* and *shape*
        because the quantised representation loses that metadata.
        """
        if fmt == SerializationFormat.RAW:
            return self._deserialize_raw(data, device=device)

        if fmt == SerializationFormat.ZSTD:
            raw = self._zstd_decompress(data)
            return self._deserialize_raw(raw, device=device)

        if fmt == SerializationFormat.FP8_ZSTD:
            scale_bytes, payload = _unpack_scale(data)
            if self._fp8_zstd_level > 0:
                payload = self._zstd_decompress(payload)
            quantized = self._deserialize_raw(payload, device=device)
            # serialize() stores a true FP8 (float8_e4m3fn) chunk with no
            # scale; the byte payload is reconstructed by _deserialize_raw as
            # uint8, so restore the FP8 dtype before dequantising — otherwise
            # the FP8 bytes are reinterpreted as uint8->float16 garbage and the
            # scale round-trip is lost (F-031).  When a scale header is present
            # the chunk is int8 and must be left as-is for scaling.
            if not scale_bytes and hasattr(torch, "float8_e4m3fn"):
                quantized = quantized.view(torch.float8_e4m3fn)
            scale = np.frombuffer(scale_bytes, dtype=np.float32)[0] if scale_bytes else None
            orig_dtype = dtype or torch.float16
            out = tensor_dequantize(quantized, scale, orig_dtype, use_fp8=True)
            # The quantised representation loses shape; restore it when the
            # caller supplied it (documented contract for FP8_ZSTD).
            if shape is not None and tuple(out.shape) != tuple(shape):
                out = out.reshape(shape)
            return out

        msg = f"Unknown serialisation format: {fmt}"
        raise ValueError(msg)

    # -- low-level helpers --------------------------------------------------

    @staticmethod
    def _serialize_raw(tensor: torch.Tensor) -> bytes:
        """Convert a tensor to flat uint8 bytes without any compression."""
        return bytes(
            memoryview(
                tensor.detach().contiguous().cpu().view(torch.uint8).numpy(force=True)
            )
        )

    @staticmethod
    def _deserialize_raw(
        data: bytes,
        dtype: torch.dtype = torch.uint8,
        device: str = "cpu",
    ) -> torch.Tensor:
        arr = np.frombuffer(data, dtype=np.uint8)
        return torch.from_numpy(arr).view(dtype).to(device)


def _pack_scale(scale: bytes, payload: bytes) -> bytes:
    """Prepend a 4-byte scale length header + scale bytes to *payload*."""
    return len(scale).to_bytes(4, "little") + scale + payload


def _unpack_scale(data: bytes) -> tuple[bytes, bytes]:
    """Reverse ``_pack_scale``: returns (scale_bytes, payload)."""
    scale_len = int.from_bytes(data[:4], "little")
    return data[4 : 4 + scale_len], data[4 + scale_len :]


# ---------------------------------------------------------------------------
# 3.  GPU-direct serialization
# ---------------------------------------------------------------------------


class GPUDirectSerializer:
    """Serialise tensors directly from GPU memory, avoiding a CPU staging copy.

    Uses CUDA streams for async device-to-host transfers; falls back to CPU
    serialisation when CUDA is unavailable.
    """

    def __init__(
        self,
        compression_fn: Callable[[bytes], bytes] | None = None,
        decompression_fn: Callable[[bytes], bytes] | None = None,
        stream_sync: bool = True,
    ) -> None:
        """Initialise GPUDirectSerializer.

        Parameters
        ----------
        compression_fn:
            Optional bytes->bytes compression applied after the GPU->CPU copy.
        decompression_fn:
            Optional bytes->bytes decompression before placing on the target
            device.
        stream_sync:
            If *True* (default), synchronise the CUDA copy stream before
            returning serialised data.  Set to *False* when the caller manages
            synchronisation externally.
        """
        self._compression_fn = compression_fn
        self._decompression_fn = decompression_fn
        self._stream_sync = stream_sync
        self._cuda_available: bool | None = None  # lazy probe

    # -- public API ---------------------------------------------------------

    def serialize_gpu(self, tensor: torch.Tensor) -> bytes:
        """Serialise *tensor* directly from GPU memory.

        If the tensor is already on CPU, or CUDA is not available, falls
        through to a simple CPU-based serialisation.
        """
        if not tensor.is_cuda or not self._cuda_supported():
            return self._serialize_cpu_fallback(tensor)

        copy_stream = get_tensor_copy_stream(tensor.device.index)
        with torch.cuda.stream(copy_stream):
            cpu_copy = tensor.to("cpu", non_blocking=True)

        if self._stream_sync and copy_stream is not None:
            copy_stream.synchronize()

        raw = bytes(
            memoryview(cpu_copy.contiguous().view(torch.uint8).numpy(force=True))
        )
        if self._compression_fn is not None:
            raw = self._compression_fn(raw)
        return raw

    def deserialize_gpu(
        self,
        data: bytes,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        """Reconstruct a tensor from *data* and place it on *device* (GPU).

        Falls back to CPU construction when CUDA is not available.
        """
        if not self._cuda_supported():
            arr = np.frombuffer(data, dtype=np.uint8)
            return torch.from_numpy(arr).view(dtype)

        if self._decompression_fn is not None:
            data = self._decompression_fn(data)

        # Stage on CPU first then push to GPU via the copy stream
        arr = np.frombuffer(data, dtype=np.uint8)
        cpu_tensor = torch.from_numpy(arr).view(dtype)

        dev = torch.device(device) if isinstance(device, str) else device
        if dev.type == "cuda":
            copy_stream = get_tensor_copy_stream(dev.index)
            with torch.cuda.stream(copy_stream):
                gpu_tensor = cpu_tensor.to(dev, non_blocking=True)
            if self._stream_sync and copy_stream is not None:
                copy_stream.synchronize()
            return gpu_tensor

        return cpu_tensor.to(dev)

    def serialize_gpu_async(
        self,
        tensor: torch.Tensor,
    ) -> tuple[Callable[[], bytes], torch.cuda.Stream | None]:
        """Start an async GPU->CPU copy and return a synchronisation function.

        Usage::

            finalize, stream = serializer.serialize_gpu_async(gpu_tensor)
            # … do other work …
            serialised_bytes = finalize()

        Returns
        -------
        (finalize_fn, stream)
        """
        if not tensor.is_cuda or not self._cuda_supported():
            data = self._serialize_cpu_fallback(tensor)
            return lambda: data, None

        copy_stream = get_tensor_copy_stream(tensor.device.index)
        with torch.cuda.stream(copy_stream):
            cpu_copy = tensor.to("cpu", non_blocking=True)

        def finalize() -> bytes:
            if copy_stream is not None:
                copy_stream.synchronize()
            raw = bytes(
                memoryview(cpu_copy.contiguous().view(torch.uint8).numpy(force=True))
            )
            if self._compression_fn is not None:
                raw = self._compression_fn(raw)
            return raw

        return finalize, copy_stream

    # -- internal helpers ---------------------------------------------------

    def _cuda_supported(self) -> bool:
        if self._cuda_available is None:
            self._cuda_available = torch.cuda.is_available()
        return self._cuda_available

    @staticmethod
    def _serialize_cpu_fallback(tensor: torch.Tensor) -> bytes:
        return bytes(
            memoryview(
                tensor.detach().contiguous().cpu().view(torch.uint8).numpy(force=True)
            )
        )


# ---------------------------------------------------------------------------
# 4.  SerializationController — combines all three components
# ---------------------------------------------------------------------------


class SerializationController:
    """Orchestrates compression negotiation, adaptive serialisation, and
    GPU-direct transfer for sending / receiving tensors between peers.
    """

    def __init__(
        self,
        negotiator: CompressionNegotiator | None = None,
        adaptive_serializer: AdaptiveSerializer | None = None,
        gpu_serializer: GPUDirectSerializer | None = None,
    ) -> None:
        self._negotiator = negotiator or CompressionNegotiator()
        self._adaptive = adaptive_serializer or AdaptiveSerializer()
        self._gpu = gpu_serializer or GPUDirectSerializer()

        # Optional: pluggable transport callbacks for send/recv.
        self._send_fn: Callable[[str, bytes], None] | None = None
        self._recv_fn: Callable[[str], bytes] | None = None

    # -- properties ---------------------------------------------------------

    @property
    def negotiator(self) -> CompressionNegotiator:
        return self._negotiator

    @property
    def adaptive_serializer(self) -> AdaptiveSerializer:
        return self._adaptive

    @property
    def gpu_serializer(self) -> GPUDirectSerializer:
        return self._gpu

    # -- transport wiring ---------------------------------------------------

    def set_transport(
        self,
        send: Callable[[str, bytes], None],
        recv: Callable[[str], bytes],
    ) -> None:
        """Register send/recv callbacks for moving bytes to/from a peer."""
        self._send_fn = send
        self._recv_fn = recv

    # -- high-level send / recv ---------------------------------------------

    def send_tensor(
        self,
        peer_id: str,
        tensor: torch.Tensor,
        *,
        method: CompressionMethod | None = None,
        force_cpu: bool = False,
    ) -> CompressionMethod:
        """Negotiate compression, serialise *tensor*, and transmit to *peer_id*.

        Parameters
        ----------
        peer_id:
            The remote peer identifier.
        tensor:
            The tensor to send.
        method:
            If provided, use this method directly instead of negotiating.
        force_cpu:
            If *True*, skip GPU-direct serialisation even when CUDA is
            available.

        Returns
        -------
        The compression method that was used.
        """
        # 1. Negotiate compression method.
        if method is not None:
            chosen = method
        else:
            cached = self._negotiator.get_cached(peer_id)
            if cached is not None:
                chosen = cached
            else:
                # The caller is expected to supply remote_supported out-of-band
                # (e.g. from a handshake message).  Here we fall back to the
                # local set as a reasonable default.
                chosen = self._negotiator.negotiate(
                    peer_id, self._negotiator.local_supported
                )

        # 2. Serialise.
        if chosen == CompressionMethod.NONE:
            # Use adaptive serializer for raw or GPU-direct for GPU tensors.
            if tensor.is_cuda and not force_cpu and torch.cuda.is_available():
                data = self._gpu.serialize_gpu(tensor)
            else:
                if tensor.is_cuda:
                    tensor = tensor.cpu()
                data = AdaptiveSerializer._serialize_raw(tensor)
            fmt_tag = SerializationFormat.RAW
        elif chosen == CompressionMethod.ZSTD:
            # Force CPU path for compression; GPU-direct with zstd is possible
            # but for simplicity we use AdaptiveSerializer.
            data = self._adaptive.serialize(tensor.cpu() if tensor.is_cuda else tensor)
            fmt_tag, data = data
            # ``serialize`` already returns ZSTD (medium) or FP8_ZSTD (large)
            # payloads that are zstd-compressed and, for FP8_ZSTD, scale-packed.
            # Do NOT re-compress or relabel those: re-compressing an already
            # zstd-compressed FP8 payload double-compresses it, and relabelling
            # it as plain ZSTD drops the scale so the receiver can never
            # dequantise (F-031).  Only small RAW payloads need wrapping so the
            # negotiated method is honoured.
            if fmt_tag == SerializationFormat.RAW:
                data = self._adaptive._zstd_compress(data)
                fmt_tag = SerializationFormat.ZSTD
        elif chosen == CompressionMethod.LZ4:
            raw = AdaptiveSerializer._serialize_raw(
                tensor.cpu() if tensor.is_cuda else tensor
            )
            data = self._adaptive._lz4_compress(raw)
            fmt_tag = SerializationFormat.RAW  # not a zstd format tag, but we
            # embed the method in the wire envelope; see _build_wire_envelope.
        else:
            raise ValueError(f"Unsupported compression method: {chosen}")

        # 3. Wrap in a simple wire envelope: 1-byte format + 1-byte method + payload.
        wire = self._build_wire_envelope(fmt_tag, chosen, data)

        # 4. Transmit.
        if self._send_fn is not None:
            self._send_fn(peer_id, wire)
        else:
            raise RuntimeError(
                "Transport not configured; call set_transport(send, recv) first."
            )

        return chosen

    def recv_tensor(
        self,
        peer_id: str,
        device: str = "cpu",
        *,
        dtype: torch.dtype | None = None,
        shape: list[int] | None = None,
    ) -> torch.Tensor:
        """Receive serialised data from *peer_id* and reconstruct a tensor.

        Parameters
        ----------
        peer_id:
            The remote peer identifier.
        device:
            Target device for the reconstructed tensor.
        dtype:
            Original dtype (needed when FP8 was used on the sender side).
        shape:
            Original shape (needed when FP8 was used).

        Returns
        -------
        The deserialised tensor.
        """
        if self._recv_fn is None:
            raise RuntimeError(
                "Transport not configured; call set_transport(send, recv) first."
            )

        wire = self._recv_fn(peer_id)
        fmt_tag, method, payload = self._parse_wire_envelope(wire)

        # Route to the right deserialisation path.
        if fmt_tag == SerializationFormat.RAW and method == CompressionMethod.LZ4:
            decompressed = self._adaptive._lz4_decompress(payload)
            return AdaptiveSerializer._deserialize_raw(decompressed, device=device)

        if fmt_tag == SerializationFormat.RAW:
            if method == CompressionMethod.NONE:
                return AdaptiveSerializer._deserialize_raw(payload, device=device)
            # ZSTD-only RAW (shouldn't normally happen)
            decompressed = self._adaptive._zstd_decompress(payload)
            return AdaptiveSerializer._deserialize_raw(decompressed, device=device)

        if fmt_tag == SerializationFormat.ZSTD:
            return self._adaptive.deserialize(
                SerializationFormat.ZSTD, payload, device=device
            )

        if fmt_tag == SerializationFormat.FP8_ZSTD:
            return self._adaptive.deserialize(
                SerializationFormat.FP8_ZSTD,
                payload,
                dtype=dtype,
                shape=shape,
                device=device,
            )

        raise ValueError(f"Unknown wire format tag: {fmt_tag}")

    # -- wire-protocol helpers ----------------------------------------------

    @staticmethod
    def _build_wire_envelope(
        fmt: SerializationFormat,
        method: CompressionMethod,
        payload: bytes,
    ) -> bytes:
        """Build a small wire envelope::

            [1 byte format enum][1 byte method enum][4 byte payload length][payload]
        """
        fmt_byte = max(1, ord(fmt.value[0])) & 0xFF
        method_byte = max(1, ord(method.value[0])) & 0xFF
        return (
            bytes([fmt_byte, method_byte])
            + len(payload).to_bytes(4, "little")
            + payload
        )

    @staticmethod
    def _parse_wire_envelope(
        wire: bytes,
    ) -> tuple[SerializationFormat, CompressionMethod, bytes]:
        """Reverse ``_build_wire_envelope``."""
        if len(wire) < 6:
            raise ValueError(f"Wire envelope too short: {len(wire)} bytes")
        # fmt_byte = wire[0]; method_byte = wire[1]
        payload_len = int.from_bytes(wire[2:6], "little")
        payload = wire[6 : 6 + payload_len]

        # Map back — heuristic based on first character of enum value.
        # In practice a proper 1-byte numeric enum would be cleaner, but
        # this keeps the wire format readable for debugging.
        fmt_map: dict[str, SerializationFormat] = {
            "r": SerializationFormat.RAW,
            "z": SerializationFormat.ZSTD,
            "f": SerializationFormat.FP8_ZSTD,
        }
        method_map: dict[str, CompressionMethod] = {
            "z": CompressionMethod.ZSTD,
            "l": CompressionMethod.LZ4,
            "n": CompressionMethod.NONE,
        }
        fmt = fmt_map.get(chr(wire[0]), SerializationFormat.RAW)
        method = method_map.get(chr(wire[1]), CompressionMethod.NONE)
        return fmt, method, payload

    # -- utility ------------------------------------------------------------

    def close(self) -> None:
        """Release resources held by sub-components (CUDA streams, etc.)."""
        from distllm.dist.pipeline.serialization import cleanup_tensor_copy_streams

        cleanup_tensor_copy_streams()
        self._negotiator.clear_cache()
