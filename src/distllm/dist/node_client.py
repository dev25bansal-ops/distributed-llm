"""gRPC client for connecting to remote worker nodes.

Creates and manages gRPC channels + stubs for NodeService RPCs.
Includes a channel pool to reuse connections and reduce overhead.
"""

from __future__ import annotations

import asyncio
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


def request_layer_weights(
    host: str, port: int,
    model_name: str, start_layer: int, end_layer: int,
    cluster_key: str | None = None,
    timeout_s: float = 120.0,
) -> bytes | None:
    """Request layer weights from a remote node via TransferWeights RPC.

    Args:
        host: Remote node host.
        port: Remote node gRPC port.
        model_name: Model name.
        start_layer: First layer index (inclusive).
        end_layer: Last layer index (exclusive).
        cluster_key: Shared cluster auth key.
        timeout_s: Request timeout.

    Returns:
        Serialized state dict bytes, or None on failure.
    """
    import hashlib
    from distllm.dist import node_pb2
    try:
        client = create_node_client(host, port, timeout_s=timeout_s, cluster_key=cluster_key)
        req = node_pb2.TransferWeightsRequest(
            model_name=model_name,
            start_layer=start_layer,
            end_layer=end_layer,
            cluster_key=cluster_key or '',
        )
        resp, call = client.stub.TransferWeights.with_call(req)
        client.close()
        if resp.success and resp.state_dict_bytes:
            # Verify integrity via SHA-256 checksum sent as trailing metadata
            expected_checksum = None
            for key, value in call.trailing_metadata():
                if key == "x-checksum-sha256":
                    expected_checksum = value
                    break
            if expected_checksum:
                actual_checksum = hashlib.sha256(resp.state_dict_bytes).hexdigest()
                if actual_checksum != expected_checksum:
                    logger.error(
                        f"Weight transfer checksum mismatch for {model_name} "
                        f"layers {start_layer}-{end_layer}: "
                        f"expected {expected_checksum}, got {actual_checksum}"
                    )
                    return None
            logger.info(f"Received weights for {model_name} layers {start_layer}-{end_layer} "
                         f"({len(resp.state_dict_bytes)} bytes)")
            return resp.state_dict_bytes
        logger.warning(f"Weight transfer failed: {resp.error_message}")
        return None
    except Exception as e:
        logger.warning(f"Weight transfer request failed: {e}")
        return None


def forward_request(
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

    client = create_node_client(
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
        resp, _ = client.stub.ForwardPass.with_call(req)
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
) -> bytes | None:
    """Request layer weights via streaming TransferWeightsStream RPC.

    Collects all streamed chunks into a single bytes buffer.

    Args:
        host: Remote node host.
        port: Remote node gRPC port.
        model_name: Model name.
        start_layer: First layer index (inclusive).
        end_layer: Last layer index (exclusive).
        cluster_key: Shared cluster auth key.
        timeout_s: Request timeout.

    Returns:
        Complete serialized state dict bytes, or None on failure.
    """
    from distllm.dist import node_pb2
    try:
        client = create_node_client(host, port, timeout_s=timeout_s, cluster_key=cluster_key)
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
        for resp in client.stub.TransferWeightsStream(req):
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
        # Integrity: reject a stream producing no bytes (all-empty chunks dup).
        logger.info(f"Streamed weights for {model_name} layers {start_layer}-{end_layer} "
                     f"({len(buffer)} bytes, {expected_index}/{total_chunks} chunks)")
        return bytes(buffer)
    except Exception as e:
        logger.warning(f"Streaming weight transfer failed: {e}")
        return None
