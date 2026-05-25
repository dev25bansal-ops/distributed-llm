"""Data types for inter-node communication.

Replaces protobuf-generated node_pb2 with plain Python dataclasses.
These types are serialized via Ray object store (ray.put/ray.get)
or via the custom serializer in serializers.py for backward compat.
"""

from dataclasses import dataclass, field
from enum import IntEnum


class ErrorCode(IntEnum):
    UNKNOWN = 0
    MODEL_ERROR = 1
    OOM = 2
    TIMEOUT = 3
    INVALID_INPUT = 4
    NODE_UNREACHABLE = 5
    CIRCUIT_BREAKER_OPEN = 6


@dataclass
class TensorProto:
    shape: list[int] = field(default_factory=list)
    dtype: str = ""
    raw_data: bytes = b""
    data: list[float] = field(default_factory=list)
    scale: list[float] = field(default_factory=list)


@dataclass
class KVLayerCacheProto:
    key_states: TensorProto | None = None
    value_states: TensorProto | None = None


@dataclass
class KVCacheProto:
    layers: list[KVLayerCacheProto] = field(default_factory=list)
    quant_bits: int = 0


@dataclass
class ForwardPassRequestProto:
    request_id: str = ""
    input_ids: list[int] = field(default_factory=list)
    batch_size: int = 0
    seq_len: int = 0
    hidden_states: TensorProto | None = None
    attention_mask: TensorProto | None = None
    position_ids: TensorProto | None = None
    kv_cache: KVCacheProto | None = None
    use_cache: bool = False
    is_first_pass: bool = False
    draft_tokens: list[int] = field(default_factory=list)
    model_name: str = ""
    next_node_uri: str = ""
    is_p2p_forward: bool = False


@dataclass
class ForwardPassResponseProto:
    request_id: str = ""
    output: TensorProto | None = None
    kv_cache: KVCacheProto | None = None
    success: bool = True
    error_message: str = ""
    error_code: ErrorCode = ErrorCode.UNKNOWN
    is_logits: bool = False
    verified_tokens: list[int] = field(default_factory=list)
    speculative_score: float = 0.0
    processing_time_ms: float = 0.0


@dataclass
class NodeInfoProto:
    node_id: str = ""
    host: str = ""
    port: int = 0
    total_memory: int = 0
    available_memory: int = 0
    device_type: str = ""
    device_name: str = ""
    cluster_id: str = ""


@dataclass
class GPUInfoProto:
    gpu_id: int = 0
    name: str = ""
    total_memory: int = 0
    used_memory: int = 0
    free_memory: int = 0
    utilization: float = 0.0


@dataclass
class NodeRegistrationProto:
    node_info: NodeInfoProto | None = None
    num_layers: int = 0
    gpus: list[GPUInfoProto] = field(default_factory=list)
    expert_ids: list[int] = field(default_factory=list)


@dataclass
class RegistrationResponseProto:
    accepted: bool = False
    assigned_start_layer: int = 0
    assigned_end_layer: int = 0
    model_name: str = ""
    is_first_node: bool = False
    is_last_node: bool = False


@dataclass
class HealthCheckResponseProto:
    node_id: str = ""
    healthy: bool = True
    memory_used: int = 0
    memory_total: int = 0
    gpu_utilization: float = 0.0
    temperature: float = 0.0
    gpus: list[GPUInfoProto] = field(default_factory=list)


@dataclass
class MoEForwardRequestProto:
    hidden_states: TensorProto | None = None
    expert_ids: list[int] = field(default_factory=list)
    routing_weights: list[float] = field(default_factory=list)
    request_id: str = ""


@dataclass
class MoEForwardResponseProto:
    output: TensorProto | None = None
    success: bool = True
    error_message: str = ""
    processing_time_ms: float = 0.0


@dataclass
class InferenceRequestProto:
    request_id: str = ""
    prompt: str = ""
    max_tokens: int = 0
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False
    priority: int = 2


@dataclass
class LogitsResponseProto:
    request_id: str = ""
    generated_text: str = ""
    success: bool = True
    error_message: str = ""
    error_code: ErrorCode = ErrorCode.UNKNOWN


@dataclass
class TokenResponseProto:
    request_id: str = ""
    token: str = ""
    is_final: bool = False
    full_text: str = ""
    success: bool = True
    error_message: str = ""
