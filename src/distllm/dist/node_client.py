"""gRPC client for connecting to remote worker nodes.

Creates and manages gRPC channels + stubs for NodeService RPCs.
Includes a channel pool to reuse connections and reduce overhead.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import grpc
from loguru import logger

from distllm.dist import node_pb2_grpc

# Channel pool: target -> (channel, ref_count)
_channel_pool: dict[str, tuple[grpc.Channel, int]] = {}
_channel_pool_lock = threading.Lock()


@dataclass
class NodeClient:
    """A connected synchronous gRPC client to a remote worker node."""
    channel: grpc.Channel
    stub: node_pb2_grpc.NodeServiceStub
    cluster_key: str | None = None
    _target: str = ""

    def close(self) -> None:
        try:
            with _channel_pool_lock:
                if self._target in _channel_pool:
                    ch, count = _channel_pool[self._target]
                    if count <= 1:
                        ch.close()
                        del _channel_pool[self._target]
                    else:
                        _channel_pool[self._target] = (ch, count - 1)
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
    with _channel_pool_lock:
        if target in _channel_pool:
            channel, count = _channel_pool[target]
            _channel_pool[target] = (channel, count + 1)
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

    # Add to pool
    with _channel_pool_lock:
        _channel_pool[target] = (channel, 1)

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

    await channel.channel_ready(timeout=timeout_s)
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

    Returns:
        Output tensor from the remote node.

    Raises:
        RuntimeError: If the remote node returns an error or no output.
    """
    from distllm.dist import node_pb2
    import time

    client = create_node_client(host, port, timeout_s=timeout_s, cluster_key=cluster_key)
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

    Returns:
        Output tensor from the remote node.

    Raises:
        RuntimeError: If the remote node returns an error or no output.
    """
    from distllm.dist import node_pb2
    import time

    client = await create_async_node_client(host, port, timeout_s=timeout_s, cluster_key=cluster_key)
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
        for resp in client.stub.TransferWeightsStream(req):
            if resp.success and resp.state_dict_bytes:
                buffer.extend(resp.state_dict_bytes)
            elif not resp.success:
                logger.warning(f"Stream chunk error: {resp.error_message}")
                client.close()
                return None
        client.close()
        if buffer:
            logger.info(f"Streamed weights for {model_name} layers {start_layer}-{end_layer} "
                         f"({len(buffer)} bytes)")
            return bytes(buffer)
        logger.warning("Streamed weight transfer returned empty buffer")
        return None
    except Exception as e:
        logger.warning(f"Streaming weight transfer failed: {e}")
        return None
