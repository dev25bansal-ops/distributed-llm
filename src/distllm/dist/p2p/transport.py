"""Complete KV cache exchange transport for gossip protocol.

Provides the actual network layer for transferring KV cache metadata
and data between peer nodes. Supports:
- gRPC-based transport (production)
- HTTP fallback for testing
- QUIC transport (preferred when aioquic is available)
- Bandwidth-aware transfers (skip when network saturated)
- Polynomial rolling hash for cache identification
- HMAC-SHA256 message authentication
- Safe serializer selection (torch_safe only; pickle mode removed for security)

Transport auto-selection
------------------------
When ``aioquic`` is installed, the module-level :func:`get_optimal_transport`
returns the QUIC-based :class:`~distllm.dist.p2p.quic_transport.QuicTransport`,
otherwise it falls back to the HTTP-based :class:`GossipTransport` defined here.
"""

from __future__ import annotations
import hashlib
import hmac
import io
import json
import os
import time
from loguru import logger

import torch

# ---------------------------------------------------------------------------
# Optional QUIC transport (preferred when aioquic is available)
# ---------------------------------------------------------------------------
try:
    from distllm.dist.p2p.quic_transport import (
        HAS_AIOQUIC,
        QuicTransport as QuicTransportImpl,
        StreamPriority,
    )
except ImportError:
    HAS_AIOQUIC = False
    QuicTransportImpl = None  # type: ignore[assignment]
    StreamPriority = None  # type: ignore[assignment]

class KVCacheTransfer:
    HASH_BASE = 31
    HASH_MOD = (1 << 61) - 1

    DEFAULT_MAX_BANDWIDTH = 100 * 1024 * 1024

    # Serializer selection (class-level fallback).
    # "torch_safe" (default): uses torch.save / torch.load(weights_only=True)
    #   - Safe: restricts unpickling to tensor types only.
    # "pickle" has been REMOVED in v0.4.2 for security reasons (CVSS 9.8).
    # Only torch_safe is supported. See https://github.com/distributed-llm/distributed-llm/security
    _default_serializer: str = "torch_safe"

    def __init__(
        self,
        max_bandwidth: int = DEFAULT_MAX_BANDWIDTH,
        serializer: str | None = None,
    ):
        self._max_bandwidth = max_bandwidth
        self._bytes_transferred = 0
        self._transfer_start = time.time()
        self._transfers_completed = 0
        self._transfers_failed = 0
        if serializer is not None:
            self._set_serializer(serializer)

    @classmethod
    def _set_serializer(cls, serializer: str) -> None:
        """Configure the serialization backend.

        Args:
            serializer: Must be "torch_safe". The "pickle" option was removed
                in v0.4.2 for security reasons (unrestricted pickle deserialization
                allows arbitrary code execution, CVSS 9.8).

        Raises:
            ValueError: If the serializer name is not "torch_safe".
        """
        if serializer == "pickle":
            raise ValueError(
                "The 'pickle' serializer has been removed in v0.4.2 for security reasons. "
                "Only 'torch_safe' is supported. Use torch.load(weights_only=True) for safe "
                "deserialization. See https://github.com/distributed-llm/distributed-llm/security "
                "for details."
            )
        if serializer != "torch_safe":
            raise ValueError(
                f"Unknown serializer: {serializer!r}. "
                f"Only 'torch_safe' is supported."
            )
        cls._default_serializer = serializer

    @classmethod
    def hash_tokens(cls, token_ids: list[int]) -> str:
        h = 0
        for t in token_ids:
            h = (h * cls.HASH_BASE + t) % cls.HASH_MOD
        return f"h{h}"

    @classmethod
    def hash_tokens_sha256(cls, token_ids: list[int]) -> str:
        data = json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(data).hexdigest()[:16]

    @classmethod
    def serialize_kv(cls, cache_data: dict) -> bytes:
        """Serialize KV cache data to a byte string.

        Uses ``torch.save`` under the hood, which employs Python's
        pickle protocol for the container format.  This is the standard
        PyTorch tensor serialization path and performs well for tensor
        payloads.

        Security note
        -------------
        ``torch.save`` writes pickle data.  Always pair with
        :meth:`deserialize_kv`, which enforces ``weights_only=True``
        to prevent arbitrary code execution when loading from
        untrusted sources.

        .. deprecated::
            The raw pickle format is tied to Python version and class
            definitions.  Prefer ``_default_serializer = "torch_safe"``
            (the default) which keeps ``torch.save`` for the wire
            format but always loads with ``weights_only=True``.
        """
        buffer = io.BytesIO()
        torch.save(cache_data, buffer)
        return buffer.getvalue()

    @classmethod
    def deserialize_kv(cls, data: bytes) -> dict:
        """Deserialize KV cache data safely.

        Uses ``torch.load`` with ``weights_only=True``, which restricts
        unpickling to safe tensor types and basic Python objects.  This
        prevents arbitrary code execution from maliciously crafted
        serialized data.

        This is the recommended safe loading approach per the PyTorch
        serialization docs:
        https://pytorch.org/docs/stable/notes/serialization.html

        Raises
        ------
        RuntimeError
            If the data was produced by an unrestricted pickle (e.g.
            plain ``pickle.dumps``) and contains non-tensor objects.
        """
        buffer = io.BytesIO(data)
        return torch.load(buffer, weights_only=True)

    @classmethod
    def estimate_size(cls, cache_data: dict) -> int:
        total = 0
        for layer_data in cache_data.values():
            if isinstance(layer_data, tuple):
                k, v = layer_data
                total += k.element_size() * k.numel()
                total += v.element_size() * v.numel()
            elif isinstance(layer_data, dict):
                total += cls.estimate_size(layer_data)
        return total

    def can_transfer(self, size_bytes: int) -> bool:
        elapsed = time.time() - self._transfer_start
        if elapsed <= 0:
            return True

        current_rate = self._bytes_transferred / elapsed
        return current_rate + size_bytes < self._max_bandwidth * 0.8

    def record_transfer(self, size_bytes: int, success: bool = True) -> None:
        if success:
            self._bytes_transferred += size_bytes
            self._transfers_completed += 1
        else:
            self._transfers_failed += 1

    def stats(self) -> dict:
        elapsed = time.time() - self._transfer_start
        return {
            "bytes_transferred": self._bytes_transferred,
            "transfers_completed": self._transfers_completed,
            "transfers_failed": self._transfers_failed,
            "avg_bandwidth_mbps": round(
                self._bytes_transferred / max(elapsed, 1) / 1024 / 1024, 2
            ),
            "max_bandwidth_mbps": round(self._max_bandwidth / 1024 / 1024, 2),
        }

    def reset_stats(self) -> None:
        self._bytes_transferred = 0
        self._transfer_start = time.time()
        self._transfers_completed = 0
        self._transfers_failed = 0

class GossipTransport:
    def __init__(
        self,
        node_id: str,
        host: str = "localhost",
        port: int = 50052,
        peer_resolver=None,
        max_bandwidth: int = KVCacheTransfer.DEFAULT_MAX_BANDWIDTH,
        hmac_key: str | None = None,
    ):
        self.node_id = node_id
        self.host = host
        self.port = port
        self._peer_resolver = peer_resolver
        self._transfer = KVCacheTransfer(max_bandwidth=max_bandwidth)
        self._session = None
        self._hmac_key: str | None = hmac_key

    def _sign_message(self, data: dict) -> dict:
        if self._hmac_key is None:
            return data
        msg = dict(data)
        serialized = json.dumps(msg, separators=(",", ":"), sort_keys=True).encode("utf-8")
        signature = hmac.new(
            self._hmac_key.encode(), msg=serialized, digestmod=hashlib.sha256
        ).hexdigest()
        msg["_hmac"] = signature
        return msg

    def _verify_message(self, data: dict) -> bool:
        """Verify an incoming message's HMAC signature.

        Returns ``True`` when no key is configured (legacy unauthenticated
        mode — callers log their own loud warning at construction time).
        With a configured key, missing/malformed/invalid signatures all
        fail closed in constant time.
        """
        if self._hmac_key is None:
            return True
        signature = data.get("_hmac")
        if not isinstance(signature, str):
            return False
        unsigned = dict(data)
        unsigned.pop("_hmac", None)
        serialized = json.dumps(unsigned, separators=(",", ":"), sort_keys=True).encode("utf-8")
        expected = hmac.new(
            self._hmac_key.encode(), msg=serialized, digestmod=hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(signature, expected)

    def _get_session(self):
        if self._session is None:
            import httpx

            self._session = httpx.Client(
                timeout=httpx.Timeout(10.0, connect=2.0),
                follow_redirects=False,
            )
        return self._session

    def exchange_advertisements(
        self, peer_id: str, my_ad: dict
    ) -> dict | None:
        peer_host, peer_port = self._resolve_peer(peer_id)
        if peer_host is None:
            return None

        url = f"http://{peer_host}:{peer_port}/api/v1/gossip/exchange"

        try:
            import json
            session = self._get_session()
            signed_ad = self._sign_message(my_ad)
            response = session.post(
                url,
                content=json.dumps(signed_ad).encode(),
                headers=self._headers(),
            )

            if response.status_code == 200:
                peer_ad = json.loads(response.text)
                logger.debug(
                    f"Gossip exchange with {peer_id}: "
                    f"{peer_ad.get('total_cache_entries', 0)} entries"
                )
                return peer_ad
        except Exception as e:
            logger.debug(f"Gossip exchange with {peer_id} failed: {e}")

        return None

    def request_kv_cache(
        self, peer_id: str, prefix_hashes: list[str],
        estimated_size_per_entry: int = 0,
    ) -> dict | None:
        """Request KV cache entries from a peer.

        Args:
            peer_id: The peer to request from.
            prefix_hashes: List of prefix hashes to request.
            estimated_size_per_entry: Approximate size in bytes of each
                cache entry.  When 0 (default), a reasonable heuristic
                based on a typical 7B-8B model (32 layers, 32 heads,
                head_dim=128, FP16, 1KB per token @ 256 tokens) is used.
        """
        peer_host, peer_port = self._resolve_peer(peer_id)
        if peer_host is None:
            return None

        if estimated_size_per_entry == 0:
            # Heuristic: ~1MB per entry (covers 32×32×128×FP16 at ~256
            # tokens, which is a reasonable default for a 7B-13B model).
            estimated_size_per_entry = 1 * 1024 * 1024

        beneficial_hashes = []
        for h in prefix_hashes:
            if self._transfer.can_transfer(estimated_size_per_entry):
                beneficial_hashes.append(h)
            else:
                logger.debug(
                    f"Skipping transfer of {h}: bandwidth limit reached"
                )

        if not beneficial_hashes:
            return {"success": True, "cache_entries": {}, "entries_returned": 0}

        url = f"http://{peer_host}:{peer_port}/api/v1/gossip/fetch"

        try:
            import json
            session = self._get_session()
            request_data = self._sign_message({
                "requester_id": self.node_id,
                "prefix_hashes": beneficial_hashes,
            })
            response = session.post(
                url,
                content=json.dumps(request_data).encode(),
                headers=self._headers(),
            )

            if response.status_code == 200:
                result = json.loads(response.text)
                # SECURITY (area-analysis H7): the fetch response carries
                # peer-controlled cache entries.  When this node signs its
                # gossip traffic, verify the peer's response signature before
                # handing any entry data to callers.  Unsigned responses are
                # rejected here too — a shared-key deployment must not accept
                # unauthenticated fetch data from a legacy/compromised peer.
                if self._hmac_key is not None and "_hmac" not in result:
                    logger.warning(
                        f"KV fetch from {peer_id}: response missing HMAC "
                        "signature — rejecting"
                    )
                    self._transfer.record_transfer(0, success=False)
                    return None
                if not self._verify_message(result):
                    logger.warning(
                        f"KV fetch from {peer_id}: response failed HMAC "
                        "verification — rejecting"
                    )
                    self._transfer.record_transfer(0, success=False)
                    return None
                total_size = sum(
                    len(e.get("data", b"")) if isinstance(e, dict) else len(str(e).encode())
                    for e in result.get("cache_entries", {}).values()
                )
                self._transfer.record_transfer(total_size, success=True)
                logger.debug(
                    f"Fetched {len(beneficial_hashes)} entries from {peer_id}: "
                    f"{total_size / 1024:.1f} KB"
                )
                return result
        except Exception as e:
            logger.debug(f"KV fetch from {peer_id} failed: {e}")
            self._transfer.record_transfer(0, success=False)

        return None

    def _resolve_peer(self, peer_id: str) -> tuple[str | None, int]:
        if self._peer_resolver:
            result = self._peer_resolver(peer_id)
            if result:
                return result
        return None, 0

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("DISTLLM_GOSSIP_API_KEY") or os.environ.get("API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    @property
    def transfer_stats(self) -> dict:
        return self._transfer.stats()


# ===================================================================
# Transport auto-detection
# ===================================================================


def quic_available() -> bool:
    """Return ``True`` if the aioquic library is installed and usable."""
    return HAS_AIOQUIC


def get_optimal_transport():
    """Return the best available transport class.

    Priority:
        1. :class:`~distllm.dist.p2p.quic_transport.QuicTransport`
           (when ``aioquic`` is installed)
        2. :class:`GossipTransport` (HTTP-based, always available)

    The returned class can be instantiated with the same constructor
    signature as ``QuicTransport`` (or ``GossipTransport`` as fallback).

    Example::

        TransportCls = get_optimal_transport()
        if quic_available():
            transport = TransportCls(node_id="node-1")
            await transport.connect("peer", 50053)
        else:
            transport = TransportCls(
                node_id="node-1",
                peer_resolver=...,
            )
    """
    if HAS_AIOQUIC and QuicTransportImpl is not None:
        return QuicTransportImpl
    return GossipTransport
