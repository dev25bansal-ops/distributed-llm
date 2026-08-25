"""gRPC client for connecting to remote worker nodes.

Creates and manages gRPC channels + stubs for NodeService RPCs.
Includes a channel pool to reuse connections and reduce overhead.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import threading
from dataclasses import dataclass

import grpc
from loguru import logger

from distllm.dist import node_pb2_grpc

class ChannelPool:
    """Thread-safe channel pool with reference counting.

    Encapsulates the pool so it can be reset between tests,
    avoiding global mutable state that leaks across test cases.
    """

    def __init__(self) -> None:
        self._pool: dict[str, tuple[grpc.Channel, int]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public query / mutation API
    # ------------------------------------------------------------------

    def increment_ref(self, target: str) -> tuple[grpc.Channel, int] | None:
        """Atomically look up *target* and increment its refcount.

        Returns the ``(channel, new_count)`` tuple, or *None* if
        *target* is not in the pool.
        """
        with self._lock:
            entry = self._pool.get(target)
            if entry is None:
                return None
            ch, count = entry
            self._pool[target] = (ch, count + 1)
            return (ch, count + 1)

    def decrement_ref(self, target: str) -> tuple[grpc.Channel, int] | None:
        """Atomically decrement the refcount for *target*.

        When the count reaches zero the underlying channel is closed
        and the entry is removed from the pool.  Returns the updated
        ``(channel, new_count)`` tuple, or *None* if the entry was
        removed (or was not present at all).
        """
        with self._lock:
            entry = self._pool.get(target)
            if entry is None:
                return None
            ch, count = entry
            if count <= 1:
                try:
                    ch.close()
                except Exception:
                    pass
                del self._pool[target]
                return None
            self._pool[target] = (ch, count - 1)
            return (ch, count - 1)

    def set(self, target: str, channel: grpc.Channel, ref_count: int = 1) -> None:
        """Insert or overwrite the entry for *target*."""
        with self._lock:
            self._pool[target] = (channel, ref_count)

    def ref_count(self, target: str) -> int | None:
        """Return the current reference count for *target*, or *None*."""
        with self._lock:
            entry = self._pool.get(target)
            return entry[1] if entry is not None else None

    def set_if_absent(self, target: str, new_channel: grpc.Channel) -> grpc.Channel:
        """Atomically insert *new_channel* for *target* if absent.

        If *target* already exists, increment its refcount, close
        *new_channel* and return the pooled channel.  If *target* is
        absent, insert *new_channel* with refcount 1 and return it.

        This handles the race when multiple threads create channels
        for the same target simultaneously.
        """
        with self._lock:
            existing = self._pool.get(target)
            if existing is not None:
                ch, count = existing
                self._pool[target] = (ch, count + 1)
                try:
                    new_channel.close()
                except Exception:
                    pass
                return ch
            self._pool[target] = (new_channel, 1)
            return new_channel

    def clear(self) -> None:
        """Close all tracked channels and empty the pool."""
        with self._lock:
            for ch, _ in self._pool.values():
                try:
                    ch.close()
                except Exception:
                    pass
            self._pool.clear()

    # ------------------------------------------------------------------
    # Container protocol (convenience for test assertions)
    # ------------------------------------------------------------------

    def __contains__(self, target: object) -> bool:
        if not isinstance(target, str):
            return False
        with self._lock:
            return target in self._pool

    def __len__(self) -> int:
        with self._lock:
            return len(self._pool)


# Channel-pool singleton -- shared by all NodeClient instances.
# Use reset_channel_pool() in test fixtures to guarantee isolation.
channel_pool = ChannelPool()


def reset_channel_pool() -> None:
    """Close all tracked channels and empty the pool.

    Call this from test ``autouse`` fixtures to prevent state leakage
    between test cases.
    """
    channel_pool.clear()


@dataclass
class NodeClient:
    """A connected synchronous gRPC client to a remote worker node."""
    channel: grpc.Channel
    stub: node_pb2_grpc.NodeServiceStub
    cluster_key: str | None = None
    _target: str = ""

    def close(self) -> None:
        try:
            channel_pool.decrement_ref(self._target)
            self.channel.close()
        except Exception:
            pass


@dataclass
class AsyncNodeClient:
    """A connected async gRPC (grpc.aio) client to a remote worker node."""
    channel: grpc.aio.Channel
    stub: node_pb2_grpc.NodeServiceStub
    cluster_key: str | None = None

    async def close(self) -> None:
        try:
            await self.channel.close()
        except Exception:
            pass


def create_node_client(host: str, port: int,
                       use_tls: bool = False,
                       ca_cert: str | None = None,
                       timeout_s: float = 5.0,
                       cluster_key: str | None = None) -> NodeClient:
    """Create a gRPC client connection to a worker node.

    Uses a channel pool to reuse connections when possible,
    reducing connection overhead for repeated calls.

    Args:
        host: Worker node hostname or IP.
        port: Worker node gRPC port.
        use_tls: Whether to use TLS encryption.
        ca_cert: Path to CA certificate for TLS.
        timeout_s: Connection timeout in seconds.
        cluster_key: Optional shared secret for node authentication.

    Returns:
        NodeClient with connected channel and stub.

    Raises:
        grpc.FutureTimeoutError: If connection times out.
    """
    target = f"{host}:{port}"

    # Check channel pool first
    entry = channel_pool.increment_ref(target)
    if entry is not None:
        channel, _ = entry
        stub = node_pb2_grpc.NodeServiceStub(channel)
        logger.debug(f"gRPC client reusing pooled connection to {target}")
        client = NodeClient(channel=channel, stub=stub, cluster_key=cluster_key)
        client._target = target
        return client

    MAX_MSG_SIZE = 100 * 1024 * 1024  # 100 MB — must match server setting
    channel_options = [
        ("grpc.max_send_message_length", MAX_MSG_SIZE),
        ("grpc.max_receive_message_length", MAX_MSG_SIZE),
        ("grpc.keepalive_time_ms", 30000),
        ("grpc.keepalive_timeout_ms", 5000),
        ("grpc.keepalive_permit_without_calls", True),
    ]

    if use_tls:
        if ca_cert:
            with open(ca_cert, 'rb') as f:
                creds = grpc.ssl_channel_credentials(f.read())
        else:
            creds = grpc.ssl_channel_credentials()
        channel = grpc.secure_channel(target, creds, options=channel_options)
    else:
        channel = grpc.insecure_channel(target, options=channel_options)

    grpc.channel_ready_future(channel).result(timeout=timeout_s)
    stub = node_pb2_grpc.NodeServiceStub(channel)
    logger.debug(f"gRPC client connected to {target}")

    # Atomically insert -- handles the race when another thread
    # created a channel for the same target simultaneously.
    channel = channel_pool.set_if_absent(target, channel)
    stub = node_pb2_grpc.NodeServiceStub(channel)
    logger.debug(f"gRPC client connected to {target}")

    client = NodeClient(channel=channel, stub=stub, cluster_key=cluster_key)
    client._target = target
    return client


async def create_async_node_client(host: str, port: int,
                                     use_tls: bool = False,
                                     ca_cert: str | None = None,
                                     timeout_s: float = 5.0,
                                     cluster_key: str | None = None) -> AsyncNodeClient:
    """Create an async gRPC client (grpc.aio) to a worker node.

    Uses ``grpc.aio`` for non-blocking I/O, allowing the asyncio event
    loop to handle other tasks while waiting for gRPC responses.
    """
    target = f"{host}:{port}"
    MAX_MSG_SIZE = 100 * 1024 * 1024
    channel_options = [
        ("grpc.max_send_message_length", MAX_MSG_SIZE),
        ("grpc.max_receive_message_length", MAX_MSG_SIZE),
    ]

    if use_tls:
        if ca_cert:
            with open(ca_cert, 'rb') as f:
                creds = grpc.ssl_channel_credentials(f.read())
        else:
            creds = grpc.ssl_channel_credentials()
        channel = grpc.aio.secure_channel(target, creds, options=channel_options)
    else:
        channel = grpc.aio.insecure_channel(target, options=channel_options)

    # grpc.aio's channel_ready() takes no timeout kwarg — wrap it.
    await asyncio.wait_for(channel.channel_ready(), timeout=timeout_s)
    stub = node_pb2_grpc.NodeServiceStub(channel)
    logger.debug(f"Async gRPC client connected to {target}")
    return AsyncNodeClient(channel=channel, stub=stub, cluster_key=cluster_key)


# ---------------------------------------------------------------------------
# Weight-transfer security helpers (SEC-A5)
#
# Model weights are high-value IP; they must travel over TLS when the rest of
# the pipeline does, and their integrity must be *authenticated* (HMAC keyed by
# the cluster secret), not just checksummed — a bare SHA-256 is recomputable by
# any on-path attacker.  These helpers are shared by request_layer_weights /
# request_layer_weights_stream and mirrored on the sender side by
# NodeServicer.TransferWeights(Stream) in node_service.py.
# ---------------------------------------------------------------------------

WEIGHTS_HMAC_METADATA_KEY = "x-weights-hmac-sha256"
LEGACY_CHECKSUM_METADATA_KEY = "x-checksum-sha256"


def resolve_pipeline_tls() -> tuple[bool, str | None]:
    """Resolve TLS settings for weight transfers from the shared env contract.

    Mirrors :meth:`PipelineOrchestrator.__init__`: ``DISTLLM_PIPELINE_TLS=1``
    enables TLS and ``DISTLLM_TLS_CA_CERT_FILE`` / ``DISTLLM_TLS_CA_CERT``
    point at the cluster CA.  Explicit arguments at call sites always win.
    """
    use_tls = os.environ.get("DISTLLM_PIPELINE_TLS", "0") == "1"
    ca_cert = (
        os.environ.get("DISTLLM_TLS_CA_CERT_FILE")
        or os.environ.get("DISTLLM_TLS_CA_CERT")
    )
    return use_tls, ca_cert


def compute_weights_hmac(cluster_key: str | None,
                         payload: bytes,
                         model_name: str = "",
                         start_layer: int = 0,
                         end_layer: int = 0) -> str | None:
    """Compute an HMAC-SHA256 over weight-transfer payload bytes.

    Keyed by the shared cluster secret so only holders of the key (i.e.
    nodes that already passed ``NodeServicer._check_auth``) can produce a
    tag a receiver will accept.  The layer range and model name are folded
    into the message so a captured tag cannot be replayed for a different
    request.

    Returns ``None`` when no cluster key is available (caller decides
    whether that is acceptable — weight receivers fail closed).
    """
    if not cluster_key:
        return None
    msg = payload + b"|weights-hmac-v1|" + model_name.encode(
        "utf-8", errors="replace") + b"|" + f"{start_layer}:{end_layer}".encode()
    return hmac.new(
        cluster_key.encode("utf-8"), msg, hashlib.sha256,
    ).hexdigest()


def _verify_weight_payload(payload: bytes,
                           cluster_key: str | None,
                           metadata_keys: dict[str, str],
                           context_desc: str,
                           model_name: str = "",
                           start_layer: int = 0,
                           end_layer: int = 0) -> bytes | None:
    """Authenticate + integrity-check received weight bytes. Fail closed.

    Policy:
      * The legacy bare SHA-256 checksum (``x-checksum-sha256``), when
        present, is still enforced — but it is never sufficient on its own
        because any on-path attacker can recompute it after tampering.
      * When the caller holds a ``cluster_key``, a valid
        ``x-weights-hmac-sha256`` tag under that key is REQUIRED: absence
        or mismatch rejects the payload outright.
      * Without any cluster key, HMAC verification is impossible; the
        payload is accepted with UNAUTHENTICATED integrity and a loud
        warning.  (Against a secured serving node this branch cannot yield
        a successful RPC anyway — the servicer rejects keyless requests.)

    Returns the payload on success, or ``None`` (after logging).
    """
    context = f"{context_desc} ({len(payload)} bytes)"

    legacy_checksum = metadata_keys.get(LEGACY_CHECKSUM_METADATA_KEY)
    if legacy_checksum:
        actual_checksum = hashlib.sha256(payload).hexdigest()
        if actual_checksum != legacy_checksum:
            logger.error(
                f"{context}: checksum mismatch — "
                f"expected {legacy_checksum}, got {actual_checksum}; "
                "rejecting corrupted transfer"
            )
            return None

    expected_hmac = metadata_keys.get(WEIGHTS_HMAC_METADATA_KEY)

    if cluster_key:
        if not expected_hmac:
            logger.error(
                f"{context}: peer sent no {WEIGHTS_HMAC_METADATA_KEY} tag — "
                "weight transfer rejected (fail-closed). Upgrade the serving "
                "node to a DistLLM version that signs weight payloads."
            )
            return None
        computed = compute_weights_hmac(
            cluster_key, payload, model_name, start_layer, end_layer,
        )
        if computed is None or not hmac.compare_digest(computed, expected_hmac):
            logger.error(
                f"{context}: HMAC verification failed — weight payload "
                "tampered with or cluster keys out of sync; rejecting transfer"
            )
            return None
        return payload

    if expected_hmac:
        # Tag present but we lack the key: we could be under a downgrade /
        # substitution attack against a signed peer. Refuse.
        logger.error(
            f"{context}: peer signed the payload but this node has no "
            "cluster key to verify it — refusing unverified weights"
        )
        return None

    logger.warning(
        f"{context}: no cluster key configured — weight integrity is "
        "UNAUTHENTICATED (checksum-only or unchecked). Configure "
        "DISTLLM_CLUSTER_KEY to enable HMAC-verified transfers."
    )
    return payload


def request_layer_weights(
    host: str, port: int,
    model_name: str, start_layer: int, end_layer: int,
    cluster_key: str | None = None,
    timeout_s: float = 120.0,
    use_tls: bool | None = None,
    ca_cert: str | None = None,
) -> bytes | None:
    """Request layer weights from a remote node via TransferWeights RPC.

    Args:
        host: Remote node host.
        port: Remote node gRPC port.
        model_name: Model name.
        start_layer: First layer index (inclusive).
        end_layer: Last layer index (exclusive).
        cluster_key: Shared cluster auth key.  Also used to key the
            required HMAC-SHA256 integrity tag on the received payload —
            transfers without a valid tag are rejected (fail-closed).
        timeout_s: Request timeout.
        use_tls: Encrypt the channel (model weights are high-value IP).
            Defaults to ``DISTLLM_PIPELINE_TLS`` env when not set explicitly;
            falls back to plaintext with a loud warning.
        ca_cert: CA cert path for verifying the node's certificate.

    Returns:
        Serialized state dict bytes, or None on failure.
    """
    from distllm.dist import node_pb2
    # Resolution mirrors PipelineOrchestrator: explicit arg > env var.
    cluster_key = cluster_key or os.environ.get("DISTLLM_CLUSTER_KEY") or None
    try:
        env_use_tls, env_ca_cert = resolve_pipeline_tls()
        effective_tls = use_tls if use_tls is not None else env_use_tls
        effective_ca = ca_cert or env_ca_cert
        if not effective_tls:
            logger.warning(
                f"Weight transfer {host}:{port} over PLAINTEXT gRPC — model "
                "weights are unencrypted on the wire. Set DISTLLM_PIPELINE_TLS=1"
                " (+ DISTLLM_TLS_CA_CERT_FILE) or pass use_tls=True."
            )
        client = create_node_client(
            host, port,
            use_tls=effective_tls, ca_cert=effective_ca,
            timeout_s=timeout_s, cluster_key=cluster_key,
        )
        req = node_pb2.TransferWeightsRequest(
            model_name=model_name,
            start_layer=start_layer,
            end_layer=end_layer,
            cluster_key=cluster_key or '',
        )
        resp, call = client.stub.TransferWeights.with_call(req)
        client.close()
        if resp.success and resp.state_dict_bytes:
            # Authenticated integrity: HMAC-SHA256(cluster_key, payload) sent
            # as trailing metadata by the serving node.  A bare SHA-256
            # checksum ("x-checksum-sha256") is NOT accepted as proof — any
            # on-path attacker can recompute it after tampering.
            metadata_keys: dict[str, str] = {}
            for key, value in call.trailing_metadata():
                k = key.decode("utf-8", errors="replace") if isinstance(key, bytes) else key
                v = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
                metadata_keys[k] = v
            context_desc = f"Weight transfer for {model_name} layers {start_layer}-{end_layer}"
            verified = _verify_weight_payload(
                resp.state_dict_bytes,
                cluster_key,
                metadata_keys,
                context_desc,
                model_name=model_name,
                start_layer=start_layer,
                end_layer=end_layer,
            )
            if verified is None:
                return None
            logger.info(f"Received weights for {model_name} layers {start_layer}-{end_layer} "
                         f"({len(resp.state_dict_bytes)} bytes)")
            return resp.state_dict_bytes
        logger.warning(f"Weight transfer failed: {resp.error_message}")
        return None
    except Exception as e:
        logger.warning(f"Weight transfer request failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Sync forward-path RPC timeout
#
# The sync ForwardPass call previously ran WITHOUT a deadline: a worker that
# accepted the connection but never replied blocked the pipeline forever.
# Every sync gRPC invocation now carries a deadline, resolved as
#     explicit argument > DISTLLM_RPC_TIMEOUT_S env > DEFAULT_RPC_TIMEOUT_S
# and a DEADLINE_EXCEEDED status is converted into a typed GRPCTimeoutError
# carrying the node identity instead of leaking a raw grpc.RpcError.
# ---------------------------------------------------------------------------

DEFAULT_RPC_TIMEOUT_S = 30.0


def resolve_rpc_timeout(timeout_s: float | None) -> float:
    """Resolve the effective sync-RPC timeout in seconds.

    Precedence: explicit argument, then the ``DISTLLM_RPC_TIMEOUT_S``
    environment variable, then :data:`DEFAULT_RPC_TIMEOUT_S` (30 s).
    Invalid or non-positive env values fall back to the default with a
    warning rather than silently disabling the deadline (fail-safe).
    """
    if timeout_s is not None:
        return timeout_s
    raw = os.environ.get("DISTLLM_RPC_TIMEOUT_S", "").strip()
    if raw:
        parsed: float | None = None
        try:
            parsed = float(raw)
        except ValueError:
            logger.warning(
                f"Invalid DISTLLM_RPC_TIMEOUT_S={raw!r} — "
                f"using default {DEFAULT_RPC_TIMEOUT_S}s"
            )
        if parsed is not None:
            if parsed > 0:
                return parsed
            logger.warning(
                f"DISTLLM_RPC_TIMEOUT_S={raw!r} is not positive — "
                f"using default {DEFAULT_RPC_TIMEOUT_S}s"
            )
    return DEFAULT_RPC_TIMEOUT_S


def forward_request(
    host: str,
    port: int,
    hidden_states: torch.Tensor,
    kv_cache: list | None = None,
    request_id: str = "",
    cluster_key: str | None = None,
    timeout_s: float | None = None,
    use_tls: bool = False,
    ca_cert: str | None = None,
) -> torch.Tensor:
    """Run a forward pass on a remote worker node via synchronous gRPC.

    Creates a temporary gRPC client, sends the ForwardPass RPC with
    the given hidden states and KV cache, and returns the output tensor.

    Args:
        host: Worker node hostname or IP.
        port: Worker node gRPC port.
        hidden_states: Input tensor to forward through the node's layers.
        kv_cache: Optional KV cache list of (key, value) tensor pairs.
        request_id: Request ID for tracking.
        cluster_key: Optional shared cluster auth key.
        timeout_s: RPC timeout in seconds.  When ``None`` (default),
            resolved from ``DISTLLM_RPC_TIMEOUT_S``, falling back to a
            30 s default.  Applied BOTH to connection establishment and
            as the gRPC deadline on the ForwardPass call itself, so a
            hung worker cannot block the pipeline indefinitely.
        use_tls: Encrypt the channel (activations/KV travel over the wire).
        ca_cert: Optional CA cert path for verifying the node's certificate.

    Returns:
        Output tensor from the remote node.

    Raises:
        GRPCTimeoutError: If the remote node does not respond within
            the effective timeout (gRPC DEADLINE_EXCEEDED).
        RuntimeError: If the remote node returns an error or no output.
    """
    from distllm.dist import node_pb2
    from distllm.errors.types import GRPCTimeoutError
    import time

    effective_timeout = resolve_rpc_timeout(timeout_s)
    client = create_node_client(
        host, port,
        use_tls=use_tls, ca_cert=ca_cert,
        timeout_s=effective_timeout, cluster_key=cluster_key,
    )
    try:
        kv_cache_pb = node_pb2.KVCacheProto()
        if kv_cache:
            from distllm.dist.pipeline.serialization import set_kv_cache_proto
            set_kv_cache_proto(kv_cache_pb, kv_cache)

        from distllm.dist.pipeline.serialization import to_proto_tensor
        req = node_pb2.ForwardPassRequest(
            request_id=request_id,
            hidden_states=to_proto_tensor(hidden_states),
            kv_cache=kv_cache_pb,
            use_cache=kv_cache is not None,
            cluster_key=cluster_key or "",
        )
        t0 = time.monotonic()
        try:
            resp, _ = client.stub.ForwardPass.with_call(
                req, timeout=effective_timeout,
            )
        except grpc.RpcError as e:
            if e.code() is grpc.StatusCode.DEADLINE_EXCEEDED:
                raise GRPCTimeoutError(
                    node_id=f"{host}:{port}",
                    timeout=effective_timeout,
                    host=host,
                    port=port,
                ) from e
            raise
        elapsed = (time.monotonic() - t0) * 1000

        if not resp.success:
            raise RuntimeError(
                f"Node {host}:{port} forward failed: {resp.error_message}"
            )

        from distllm.dist.pipeline.serialization import from_proto_tensor
        output = from_proto_tensor(resp.output, device=hidden_states.device.type)

        logger.debug(
            f"Forward pass {host}:{port} complete in {elapsed:.1f}ms "
            f"output shape={list(output.shape)}"
        )
        return output
    finally:
        client.close()


async def forward_request_async(
    host: str,
    port: int,
    hidden_states: torch.Tensor,
    kv_cache: list | None = None,
    request_id: str = "",
    cluster_key: str | None = None,
    timeout_s: float = 30.0,
    use_tls: bool = False,
    ca_cert: str | None = None,
) -> torch.Tensor:
    """Run a forward pass on a remote worker node via async gRPC (grpc.aio).

    Non-blocking variant of :func:`forward_request` for use in asyncio
    contexts (e.g. the micro-batched pipeline scheduler).

    Args:
        host: Worker node hostname or IP.
        port: Worker node gRPC port.
        hidden_states: Input tensor to forward through the node's layers.
        kv_cache: Optional KV cache list of (key, value) tensor pairs.
        request_id: Request ID for tracking.
        cluster_key: Optional shared cluster auth key.
        timeout_s: RPC timeout in seconds.
        use_tls: Encrypt the channel (activations/KV travel over the wire).
        ca_cert: Optional CA cert path for verifying the node's certificate.

    Returns:
        Output tensor from the remote node.

    Raises:
        RuntimeError: If the remote node returns an error or no output.
    """
    from distllm.dist import node_pb2
    import time

    client = await create_async_node_client(
        host, port,
        use_tls=use_tls, ca_cert=ca_cert,
        timeout_s=timeout_s, cluster_key=cluster_key,
    )
    try:
        kv_cache_pb = node_pb2.KVCacheProto()
        if kv_cache:
            from distllm.dist.pipeline.serialization import set_kv_cache_proto
            set_kv_cache_proto(kv_cache_pb, kv_cache)

        from distllm.dist.pipeline.serialization import to_proto_tensor
        req = node_pb2.ForwardPassRequest(
            request_id=request_id,
            hidden_states=to_proto_tensor(hidden_states),
            kv_cache=kv_cache_pb,
            use_cache=kv_cache is not None,
            cluster_key=cluster_key or "",
        )
        t0 = time.monotonic()
        resp: node_pb2.ForwardPassResponse = await client.stub.ForwardPass(req, timeout=timeout_s)
        elapsed = (time.monotonic() - t0) * 1000

        if not resp.success:
            raise RuntimeError(
                f"Node {host}:{port} forward failed: {resp.error_message}"
            )

        from distllm.dist.pipeline.serialization import from_proto_tensor
        output = from_proto_tensor(resp.output, device=hidden_states.device.type)

        logger.debug(
            f"Async forward pass {host}:{port} complete in {elapsed:.1f}ms "
            f"output shape={list(output.shape)}"
        )
        return output
    finally:
        await client.close()


def request_layer_weights_stream(
    host: str, port: int,
    model_name: str, start_layer: int, end_layer: int,
    cluster_key: str | None = None,
    timeout_s: float = 300.0,
    use_tls: bool | None = None,
    ca_cert: str | None = None,
) -> bytes | None:
    """Request layer weights via streaming TransferWeightsStream RPC.

    Collects all streamed chunks into a single bytes buffer, then verifies
    the sender's HMAC-SHA256(cluster_key, buffer) tag (trailing metadata)
    before returning anything — fail-closed.

    Args:
        host: Remote node host.
        port: Remote node gRPC port.
        model_name: Model name.
        start_layer: First layer index (inclusive).
        end_layer: Last layer index (exclusive).
        cluster_key: Shared cluster auth key.  Also keys the required
            payload HMAC; transfers without a valid tag are rejected.
        timeout_s: Request timeout.
        use_tls: Encrypt the channel (model weights are high-value IP).
            Defaults to ``DISTLLM_PIPELINE_TLS`` env when not set explicitly;
            falls back to plaintext with a loud warning.
        ca_cert: CA cert path for verifying the node's certificate.

    Returns:
        Complete serialized state dict bytes, or None on failure.
    """
    from distllm.dist import node_pb2
    # Resolution mirrors PipelineOrchestrator: explicit arg > env var.
    cluster_key = cluster_key or os.environ.get("DISTLLM_CLUSTER_KEY") or None
    try:
        env_use_tls, env_ca_cert = resolve_pipeline_tls()
        effective_tls = use_tls if use_tls is not None else env_use_tls
        effective_ca = ca_cert or env_ca_cert
        if not effective_tls:
            logger.warning(
                f"Streaming weight transfer {host}:{port} over PLAINTEXT "
                "gRPC — model weights are unencrypted on the wire. Set "
                "DISTLLM_PIPELINE_TLS=1 (+ DISTLLM_TLS_CA_CERT_FILE) or "
                "pass use_tls=True."
            )
        client = create_node_client(
            host, port,
            use_tls=effective_tls, ca_cert=effective_ca,
            timeout_s=timeout_s, cluster_key=cluster_key,
        )
        req = node_pb2.TransferWeightsRequest(
            model_name=model_name,
            start_layer=start_layer,
            end_layer=end_layer,
            cluster_key=cluster_key or '',
        )
        buffer = bytearray()
        expected_index = 0
        total_chunks = None
        seen_final = False
        responses = client.stub.TransferWeightsStream(req)
        for resp in responses:
            if not resp.success:
                logger.warning(f"Stream chunk error: {resp.error_message}")
                client.close()
                return None
            if not resp.state_dict_bytes:
                continue
            # F-049: verify chunk ordering + completeness so a reordered,
            # truncated, or duplicate stream is never assembled and loaded.
            if resp.chunk_index != expected_index:
                logger.warning(
                    f"Weight stream out of order: expected chunk {expected_index}, "
                    f"got {resp.chunk_index} — rejecting corrupted transfer"
                )
                client.close()
                return None
            if total_chunks is not None and resp.total_chunks != total_chunks:
                logger.warning("Weight stream total_chunks changed mid-transfer — rejecting")
                client.close()
                return None
            total_chunks = resp.total_chunks
            if resp.is_final_chunk and seen_final:
                logger.warning("Weight stream duplicated final chunk — rejecting")
                client.close()
                return None
            if resp.is_final_chunk:
                seen_final = True
            buffer.extend(resp.state_dict_bytes)
            expected_index += 1
        # Read trailing metadata while the call/channel is still open.
        trailing_fn = getattr(responses, "trailing_metadata", None)
        raw_metadata = ()
        if callable(trailing_fn):
            try:
                raw_metadata = trailing_fn() or ()
            except Exception as e:  # pragma: no cover - grpc internal errors
                logger.debug(f"Could not read trailing metadata: {e}")
        client.close()
        if not buffer:
            logger.warning("Streamed weight transfer returned empty buffer")
            return None
        # Completeness: every chunk up to total_chunks must have arrived.
        if total_chunks is None or expected_index < total_chunks or not seen_final:
            logger.warning(
                f"Weight stream incomplete: {expected_index}/{total_chunks} chunks, "
                f"final={seen_final} — rejecting truncated transfer"
            )
            return None

        # Authenticated integrity over the ASSEMBLED payload: the serving
        # node sends HMAC-SHA256(cluster_key, full_bytes) as trailing
        # metadata once the stream finishes.  Fail closed on absence or
        # mismatch — chunk ordering checks alone prove nothing about content.
        payload = bytes(buffer)
        context_desc = (
            f"Streaming weight transfer for {model_name} "
            f"layers {start_layer}-{end_layer}"
        )
        metadata_keys = {
            (key.decode("utf-8", errors="replace")
             if isinstance(key, bytes) else key):
            (value.decode("utf-8", errors="replace")
             if isinstance(value, bytes) else value)
            for key, value in raw_metadata
        }
        verified = _verify_weight_payload(
            payload,
            cluster_key,
            metadata_keys,
            context_desc,
            model_name=model_name,
            start_layer=start_layer,
            end_layer=end_layer,
        )
        if verified is None:
            return None

        logger.info(f"Streamed weights for {model_name} layers {start_layer}-{end_layer} "
                     f"({len(payload)} bytes, {expected_index}/{total_chunks} chunks)")
        return payload
    except Exception as e:
        logger.warning(f"Streaming weight transfer failed: {e}")
        return None
