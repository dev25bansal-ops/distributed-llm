"""gRPC server for worker nodes — serves ForwardPass, HealthCheck, Profile RPCs.

Converts between protobuf wire format and internal tensor representations.
"""

from __future__ import annotations

import hmac
import time
import threading
from concurrent import futures

import grpc
import torch
from loguru import logger

from distllm.dist import node_pb2
from distllm.dist import node_pb2_grpc
from distllm.core.kv_cache import KVCache
from distllm.security.e2e import E2EEncryption, decrypt_tensor_payload, encrypt_tensor_payload


def tensor_to_proto(tensor: torch.Tensor) -> node_pb2.TensorProto:
    """Convert a torch tensor to protobuf TensorProto (raw bytes path)."""
    if tensor is None:
        return node_pb2.TensorProto(shape=[], dtype="none", raw_data=b"")
    t = tensor.detach()
    if t.is_cuda:
        t = t.to('cpu', non_blocking=True)
        torch.cuda.current_stream().synchronize()
    dtype_str = str(t.dtype)
    if t.dim() == 0:
        t = t.reshape(1)
    raw = bytes(memoryview(t.contiguous().view(torch.uint8).numpy(force=True)))
    return node_pb2.TensorProto(
        shape=list(t.shape),
        dtype=dtype_str,
        raw_data=raw,
    )


def tensor_from_proto(proto: node_pb2.TensorProto, device: str = "cpu") -> torch.Tensor:
    """Convert protobuf TensorProto back to a torch tensor.

    Raises:
        ValueError: If the payload size does not match the declared shape.
            Rejecting here (before any reshape/allocation) keeps a malformed
            or hostile proto from crashing with a cryptic torch error or
            from triggering a huge allocation (declared shape >> payload).
    """
    if not proto.shape:
        return torch.empty(0, device=device)
    dtype_map = {
        "torch.float32": torch.float32, "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16, "torch.int64": torch.int64,
        "torch.int32": torch.int32, "torch.uint8": torch.uint8,
        "torch.bool": torch.bool, "float32": torch.float32,
        "float16": torch.float16, "bfloat16": torch.bfloat16,
        "int64": torch.int64, "int32": torch.int32, "bool": torch.bool,
    }
    tdtype = dtype_map.get(proto.dtype, torch.float32)
    declared_numel = 1
    for dim in proto.shape:
        declared_numel *= dim
    import numpy as np
    if proto.raw_data:
        if len(proto.raw_data) != declared_numel * tdtype.itemsize:
            raise ValueError(
                f"TensorProto payload mismatch: {len(proto.raw_data)} bytes "
                f"cannot fill shape {list(proto.shape)} of {tdtype} "
                f"({declared_numel * tdtype.itemsize} bytes expected)"
            )
        arr = np.frombuffer(proto.raw_data, dtype=np.uint8)
        tensor = torch.from_numpy(arr).view(tdtype).reshape(list(proto.shape)).clone()
    else:
        if len(proto.data) != declared_numel:
            raise ValueError(
                f"TensorProto payload mismatch: {len(proto.data)} elements "
                f"cannot fill shape {list(proto.shape)} "
                f"({declared_numel} elements expected)"
            )
        tensor = torch.tensor(list(proto.data), dtype=tdtype).reshape(list(proto.shape))
    return tensor.to(device)


def kv_cache_to_proto(cache) -> node_pb2.KVCacheProto:
    """Convert internal KVCache to protobuf."""
    pb = node_pb2.KVCacheProto()
    if cache is None:
        return pb
    layers = cache.cache if hasattr(cache, 'cache') else cache
    if layers is None:
        return pb
    for k, v in layers:
        layer_pb = pb.layers.add()
        layer_pb.key_states.CopyFrom(tensor_to_proto(k))
        layer_pb.value_states.CopyFrom(tensor_to_proto(v))
    return pb


def kv_cache_from_proto(pb: node_pb2.KVCacheProto):
    """Convert protobuf back to internal KV cache list."""
    if pb is None or not pb.layers:
        return None
    cache = []
    for layer_pb in pb.layers:
        k = tensor_from_proto(layer_pb.key_states)
        v = tensor_from_proto(layer_pb.value_states)
        cache.append((k, v))
    return cache


class NodeServicer(node_pb2_grpc.NodeServiceServicer):
    """gRPC servicer that forwards requests to a WorkerNode's forward_fn."""

    MAX_BATCH_SIZE = 1024
    MAX_HIDDEN_DIM = 16384
    MAX_KV_SEQ_LEN = 131072  # Max sequence length in KV cache
    MAX_KV_LAYERS = 256  # Max number of KV cache layers
    PROFILE_RATE_LIMIT = 10  # Max Profile RPCs per second

    def __init__(self, worker_node, cluster_key: str | None = None,
                 e2e_encryption: E2EEncryption | None = None):
        self._node = worker_node
        self._cluster_key = cluster_key
        self._e2e = e2e_encryption
        self._profile_rate_tokens = self.PROFILE_RATE_LIMIT
        self._profile_rate_last = time.monotonic()
        self._profile_rate_lock = threading.Lock()

    def _check_auth(self, request) -> bool:
        """Validate cluster_key from request protobuf fields. Returns True if authorized.

        Uses constant-time comparison via :func:`hmac.compare_digest` to prevent
        timing side-channel attacks against the cluster key.

        Fails closed: a servicer without a configured ``cluster_key`` rejects
        every RPC.  A worker that must accept traffic must be started with the
        coordinator's cluster key.
        """
        if self._cluster_key is None:
            return False
        req_key = getattr(request, 'cluster_key', None)
        if not req_key:
            return False
        return hmac.compare_digest(req_key, self._cluster_key)

    def ForwardPass(self, request, context):
        if not self._check_auth(request):
            return node_pb2.ForwardPassResponse(
                success=False, error_message="authentication failed")
        t0 = time.monotonic()
        try:
            # Validate input sizes before processing
            if request.input_ids and len(request.input_ids) > self.MAX_BATCH_SIZE * 131072:
                return node_pb2.ForwardPassResponse(
                    success=False, error_message=f"input_ids too large: {len(request.input_ids)} tokens"
                )
            has_hidden = request.hidden_states and request.hidden_states.raw_data
            has_input_ids = request.input_ids and len(request.input_ids) > 0

            hidden_states = None
            input_ids = None
            attention_mask = None
            position_ids = None
            past_key_values = None

            if has_hidden:
                raw = decrypt_tensor_payload(request.hidden_states.raw_data, self._e2e)
                request.hidden_states.raw_data = raw
                hidden_states = tensor_from_proto(request.hidden_states, device=self._node._get_device())
                # Validate hidden state dimensions
                if hidden_states.dim() > 4 or (hidden_states.dim() >= 2 and hidden_states.shape[-1] > self.MAX_HIDDEN_DIM):
                    return node_pb2.ForwardPassResponse(
                        success=False, error_message=f"hidden_states too large: shape={list(hidden_states.shape)}"
                    )
                if hidden_states.shape[0] > self.MAX_BATCH_SIZE:
                    return node_pb2.ForwardPassResponse(
                        success=False, error_message=f"batch size too large: {hidden_states.shape[0]}"
                    )

            if has_input_ids:
                input_ids = torch.tensor([request.input_ids], dtype=torch.long)

            if request.attention_mask and request.attention_mask.raw_data:
                raw = decrypt_tensor_payload(request.attention_mask.raw_data, self._e2e)
                request.attention_mask.raw_data = raw
                attention_mask = tensor_from_proto(request.attention_mask, device=str(hidden_states.device) if hidden_states is not None else "cpu")

            if request.position_ids and request.position_ids.raw_data:
                raw = decrypt_tensor_payload(request.position_ids.raw_data, self._e2e)
                request.position_ids.raw_data = raw
                position_ids = tensor_from_proto(request.position_ids, device=str(hidden_states.device) if hidden_states is not None else "cpu")

            if request.kv_cache and request.kv_cache.layers:
                if len(request.kv_cache.layers) > self.MAX_KV_LAYERS:
                    return node_pb2.ForwardPassResponse(
                        success=False, error_message=f"kv_cache too many layers: {len(request.kv_cache.layers)}"
                    )
                for i, layer in enumerate(request.kv_cache.layers):
                    if layer.key_states and layer.key_states.shape:
                        seq_len = layer.key_states.shape[-2] if len(layer.key_states.shape) >= 2 else 0
                        if seq_len > self.MAX_KV_SEQ_LEN:
                            return node_pb2.ForwardPassResponse(
                                success=False, error_message=f"kv_cache layer {i} seq_len too large: {seq_len}"
                            )
                past_key_values = kv_cache_from_proto(request.kv_cache)

            result = self._node.forward_fn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                input_ids=input_ids,
            )
            if result is None:
                raise RuntimeError("forward_fn returned None — model may not be loaded")
            output, new_kv = result

            output_pb = tensor_to_proto(output)
            output_pb.raw_data = encrypt_tensor_payload(output_pb.raw_data, self._e2e)
            kv_pb = kv_cache_to_proto(new_kv)
            elapsed = (time.monotonic() - t0) * 1000

            return node_pb2.ForwardPassResponse(
                request_id=request.request_id,
                output=output_pb,
                kv_cache=kv_pb,
                success=True,
                processing_time_ms=round(elapsed, 2),
            )

        except Exception as e:
            logger.error(f"ForwardPass failed: {e}", exc_info=True)
            return node_pb2.ForwardPassResponse(
                request_id=request.request_id,
                success=False,
                error_message=str(e),
            )

    MAX_LAYER_RANGE = 512  # Sanity cap on requested layer range

    def TransferWeights(self, request, context):
        if not self._check_auth(request):
            return node_pb2.TransferWeightsResponse(success=False, error_message="authentication failed")
        try:
            node = self._node
            if node.partitioner is None or not hasattr(node.partitioner, 'full_model'):
                return node_pb2.TransferWeightsResponse(
                    success=False, error_message="model not loaded on this node")
            if request.start_layer < 0 or request.end_layer < request.start_layer or request.end_layer > self.MAX_LAYER_RANGE:
                return node_pb2.TransferWeightsResponse(
                    success=False, error_message=f"invalid layer range: {request.start_layer}-{request.end_layer}")
            import io, torch, hashlib
            model = node.partitioner.full_model
            state = model.state_dict()
            layer_keys = [k for k in state.keys()
                          if any(f"layers.{i}." in k
                                 for i in range(request.start_layer, request.end_layer))
                          or any(k.startswith(f"{i}.")
                                 for i in range(request.start_layer, request.end_layer))]
            if not layer_keys:
                return node_pb2.TransferWeightsResponse(
                    success=False, error_message=f"no layers found in range {request.start_layer}-{request.end_layer}")
            subset = {k: state[k] for k in layer_keys}
            buf = io.BytesIO()
            torch.save(subset, buf)
            state_dict_bytes = buf.getvalue()
            checksum = hashlib.sha256(state_dict_bytes).hexdigest()
            context.send_trailing_metadata((
                ("x-checksum-sha256", checksum),
            ))
            return node_pb2.TransferWeightsResponse(
                model_name=request.model_name,
                start_layer=request.start_layer,
                end_layer=request.end_layer,
                state_dict_bytes=state_dict_bytes,
                success=True,
            )
        except Exception as e:
            logger.error(f"TransferWeights failed: {e}")
            return node_pb2.TransferWeightsResponse(success=False, error_message=str(e))

    def TransferWeightsStream(self, request, context):
        if not self._check_auth(request):
            yield node_pb2.TransferWeightsResponse(success=False, error_message="authentication failed")
            return
        try:
            if request.start_layer < 0 or request.end_layer < request.start_layer or request.end_layer > self.MAX_LAYER_RANGE:
                yield node_pb2.TransferWeightsResponse(
                    success=False, error_message=f"invalid layer range: {request.start_layer}-{request.end_layer}")
                return
            node = self._node
            if node.partitioner is None or not hasattr(node.partitioner, 'full_model'):
                yield node_pb2.TransferWeightsResponse(
                    success=False, error_message="model not loaded on this node")
                return
            import io, torch as torch_mod
            model = node.partitioner.full_model
            state = model.state_dict()
            layer_keys = [k for k in state.keys()
                          if any(f"layers.{i}." in k
                                 for i in range(request.start_layer, request.end_layer))
                          or any(k.startswith(f"{i}.")
                                 for i in range(request.start_layer, request.end_layer))]
            if not layer_keys:
                yield node_pb2.TransferWeightsResponse(
                    success=False, error_message=f"no layers found in range {request.start_layer}-{request.end_layer}")
                return
            subset = {k: state[k] for k in layer_keys}
            buf = io.BytesIO()
            torch_mod.save(subset, buf)
            full_bytes = buf.getvalue()
            chunk_size = 1024 * 1024
            total_chunks = (len(full_bytes) + chunk_size - 1) // chunk_size
            for i in range(total_chunks):
                chunk = full_bytes[i * chunk_size:(i + 1) * chunk_size]
                is_final = (i == total_chunks - 1)
                yield node_pb2.TransferWeightsResponse(
                    model_name=request.model_name,
                    start_layer=request.start_layer,
                    end_layer=request.end_layer,
                    state_dict_bytes=chunk,
                    success=True,
                    chunk_index=i,
                    total_chunks=total_chunks,
                    is_final_chunk=is_final,
                )
        except Exception as e:
            logger.error(f"TransferWeightsStream failed: {e}")
            yield node_pb2.TransferWeightsResponse(success=False, error_message=str(e))

    def HealthCheck(self, request, context):
        if not self._check_auth(request):
            return node_pb2.HealthCheckResponse(healthy=False)
        try:
            node = self._node
            mem_used = 0
            mem_total = 0
            gpu_util = 0.0
            gpu_name = "cpu"
            if torch.cuda.is_available():
                mem_used = torch.cuda.memory_allocated()
                mem_total = torch.cuda.get_device_properties(0).total_memory
                gpu_util = torch.cuda.utilization()
                gpu_name = torch.cuda.get_device_name(0)
            return node_pb2.HealthCheckResponse(
                healthy=True,
                node_id=node.node_id,
                memory_used_bytes=mem_used,
                memory_total_bytes=mem_total,
                gpu_utilization=gpu_util,
                start_layer=node.start_layer,
                end_layer=node.end_layer,
                total_layers=node.total_layers,
                gpu_name=gpu_name,
                gpu_memory_total=mem_total,
                num_layers_loaded=len(node.partitioner.layers) if node.partitioner and node.partitioner.layers else 0,
            )
        except Exception as e:
            return node_pb2.HealthCheckResponse(healthy=False, error_message=str(e))

    def Profile(self, request, context):
        if not self._check_auth(request):
            return node_pb2.ProfileResponse(node_id=request.node_id)

        # Rate limit: token bucket to prevent resource scanning
        with self._profile_rate_lock:
            now = time.monotonic()
            elapsed = now - self._profile_rate_last
            self._profile_rate_last = now
            self._profile_rate_tokens = min(
                self.PROFILE_RATE_LIMIT,
                self._profile_rate_tokens + elapsed * self.PROFILE_RATE_LIMIT,
            )
            if self._profile_rate_tokens < 1:
                return node_pb2.ProfileResponse(
                    node_id=request.node_id,
                    gpu_name="rate_limited",
                )
            self._profile_rate_tokens -= 1

        try:
            node = self._node
            mem_total = 0
            mem_free = 0
            gpu_name = "cpu"
            sm_count = 0
            compute_tflops = 0.0
            mem_bw = 0.0
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                mem_total = props.total_memory
                mem_free = mem_total - torch.cuda.memory_allocated()
                gpu_name = props.name
                sm_count = props.multi_processor_count
                compute_tflops = round(
                    props.multi_processor_count * props.max_threads_per_multi_processor
                    * props.clock_rate * 2 * 1e-12, 2
                )
                mem_bw = round(
                    props.memory_bus_width * props.memory_clock_rate * 2 / 8 / 1e9, 2
                )
            return node_pb2.ProfileResponse(
                node_id=node.node_id,
                gpu_name=gpu_name,
                total_memory_bytes=mem_total,
                free_memory_bytes=mem_free,
                sm_count=sm_count,
                compute_tflops=compute_tflops,
                memory_bandwidth_gbps=mem_bw,
            )
        except Exception as e:
            return node_pb2.ProfileResponse(node_id=request.node_id)

    def AdvertiseModels(self, request, context):
        if not self._check_auth(request):
            return node_pb2.AdvertiseModelsResponse()
        try:
            node = self._node
            if node.partitioner is None:
                return node_pb2.AdvertiseModelsResponse()
            ad = node_pb2.ModelAdvertisement(
                model_name=node.model_name,
                start_layer=node.start_layer,
                end_layer=node.end_layer,
                total_layers=node.total_layers,
                node_id=node.node_id,
                host=node.coordinator_host if hasattr(node, 'coordinator_host') else 'localhost',
                port=node.port,
            )
            return node_pb2.AdvertiseModelsResponse(models=[ad])
        except Exception as e:
            logger.error(f"AdvertiseModels failed: {e}")
            return node_pb2.AdvertiseModelsResponse()


class NodeServer:
    """Manages the gRPC server lifecycle on a worker node."""

    def __init__(self, worker_node, port: int, max_workers: int = 4,
                 cluster_key: str | None = None):
        self._worker = worker_node
        self._port = port
        self._max_workers = max_workers
        self._cluster_key = cluster_key
        self._server: grpc.Server | None = None
        self._running = threading.Event()
        self._stopped = threading.Event()  # Only set in stop() — used by wait()

    MAX_MSG_SIZE = 512 * 1024 * 1024  # 512 MB — gRPC Cython int max on Windows (2GB fails to convert); accommodates large weight/KV transfers
    MAX_HIDDEN_DIM = 16384  # Sanity cap on hidden state dimensions
    MAX_BATCH_SIZE = 1024   # Sanity cap on batch dimension

    def start(self, use_tls: bool = False,
              cert_file: str | None = None,
              key_file: str | None = None,
              ca_cert: str | None = None) -> None:
        self._server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=self._max_workers),
            options=[
                ("grpc.max_send_message_length", self.MAX_MSG_SIZE),
                ("grpc.max_receive_message_length", self.MAX_MSG_SIZE),
                ("grpc.keepalive_time_ms", 30000),
                ("grpc.keepalive_timeout_ms", 10000),
                ("grpc.http2.min_time_between_pings_ms", 10000),
                ("grpc.http2.max_pings_without_data", 0),
            ],
        )
        servicer = NodeServicer(self._worker, cluster_key=self._cluster_key)
        node_pb2_grpc.add_NodeServiceServicer_to_server(servicer, self._server)

        # Register gRPC health probe service for K8s native health checks
        # Uses the standard grpc.health.v1.Health protocol
        try:
            from grpc_health.v1 import health_pb2_grpc, health
            health_servicer = health.HealthServicer()
            health_pb2_grpc.add_HealthServicer_to_server(health_servicer, self._server)
            # Mark the node service as serving
            health_servicer.set(
                "grpc.health.v1.Health",
                health._health_pb2.HealthCheckResponse.SERVING,
            )
            logger.debug("gRPC health probe service registered")
        except ImportError:
            logger.debug("grpcio-health not installed, K8s gRPC probes disabled")

        if use_tls and cert_file and key_file:
            with open(cert_file, 'rb') as f:
                cert = f.read()
            with open(key_file, 'rb') as f:
                key = f.read()
            creds = grpc.ssl_server_credentials([(key, cert)])
            self._server.add_secure_port(f'0.0.0.0:{self._port}', creds)
        else:
            self._server.add_insecure_port(f'0.0.0.0:{self._port}')

        # SECURITY: a keyless node servicer rejects every RPC (fail closed).
        # Warn loudly so operators notice the misconfiguration instead of
        # silently serving open (which previously allowed weight exfiltration).
        if not self._cluster_key:
            logger.warning(
                f"Node {self._worker.node_id}: no cluster_key configured — "
                f"the gRPC servicer will REJECT all RPCs (fail closed). "
                f"Set a cluster key to enable node-to-node traffic."
            )

        self._server.start()
        self._running.set()
        logger.info(f"Node {self._worker.node_id}: gRPC server started on port {self._port}")

    def stop(self, grace: float = 5.0) -> None:
        self._running.clear()
        self._stopped.set()
        if self._server:
            self._server.stop(grace)
            logger.info(f"Node {self._worker.node_id}: gRPC server stopped")

    def wait(self) -> None:
        """Block until the server is stopped.

        Uses a threading.Event that starts cleared and is set only
        when stop() is called.  This avoids the race where start()
        sets the event before wait() is called.
        """
        try:
            self._stopped.wait()
        except KeyboardInterrupt:
            self.stop()



class ClusterKeyInterceptor(grpc.ServerInterceptor):
    """gRPC server interceptor that validates cluster_key on every RPC.

    Intercepts all incoming requests and checks for a valid ``cluster_key``
    in the request metadata.  Rejects unauthenticated calls with
    ``UNAUTHENTICATED`` status.

    Usage::

        interceptor = ClusterKeyInterceptor(cluster_key="my-secret")
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=4),
            interceptors=[interceptor],
        )
    """

    def __init__(self, cluster_key: str):
        self._cluster_key = cluster_key

    def intercept_service(self, continuation, handler_call_details):
        """Intercept each RPC and validate the cluster_key."""
        # Extract cluster_key from metadata
        metadata = dict(handler_call_details.invocation_metadata)
        req_key = metadata.get("cluster-key") or metadata.get("x-cluster-key")

        if req_key is None:
            # No key provided — reject
            return self._unauthenticated_handler

        if not hmac.compare_digest(req_key, self._cluster_key):
            # Invalid key — reject
            return self._unauthenticated_handler

        # Valid key — continue to the actual handler
        return continuation(handler_call_details)

    @staticmethod
    def _unauthenticated_handler(request, context):
        """Handler that returns UNAUTHENTICATED for all RPCs."""
        context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid or missing cluster_key")
