from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ErrorCode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    UNKNOWN: _ClassVar[ErrorCode]
    MODEL_ERROR: _ClassVar[ErrorCode]
    OOM: _ClassVar[ErrorCode]
    TIMEOUT: _ClassVar[ErrorCode]
    INVALID_INPUT: _ClassVar[ErrorCode]
    NODE_UNREACHABLE: _ClassVar[ErrorCode]
    CIRCUIT_BREAKER_OPEN: _ClassVar[ErrorCode]
UNKNOWN: ErrorCode
MODEL_ERROR: ErrorCode
OOM: ErrorCode
TIMEOUT: ErrorCode
INVALID_INPUT: ErrorCode
NODE_UNREACHABLE: ErrorCode
CIRCUIT_BREAKER_OPEN: ErrorCode

class Tensor(_message.Message):
    __slots__ = ("data", "shape", "dtype", "raw_data")
    DATA_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    RAW_DATA_FIELD_NUMBER: _ClassVar[int]
    data: _containers.RepeatedScalarFieldContainer[float]
    shape: _containers.RepeatedScalarFieldContainer[int]
    dtype: str
    raw_data: bytes
    def __init__(self, data: _Optional[_Iterable[float]] = ..., shape: _Optional[_Iterable[int]] = ..., dtype: _Optional[str] = ..., raw_data: _Optional[bytes] = ...) -> None: ...

class KVLayerCache(_message.Message):
    __slots__ = ("key_states", "value_states")
    KEY_STATES_FIELD_NUMBER: _ClassVar[int]
    VALUE_STATES_FIELD_NUMBER: _ClassVar[int]
    key_states: Tensor
    value_states: Tensor
    def __init__(self, key_states: _Optional[_Union[Tensor, _Mapping]] = ..., value_states: _Optional[_Union[Tensor, _Mapping]] = ...) -> None: ...

class KVCache(_message.Message):
    __slots__ = ("layers",)
    LAYERS_FIELD_NUMBER: _ClassVar[int]
    layers: _containers.RepeatedCompositeFieldContainer[KVLayerCache]
    def __init__(self, layers: _Optional[_Iterable[_Union[KVLayerCache, _Mapping]]] = ...) -> None: ...

class GPUInfo(_message.Message):
    __slots__ = ("gpu_id", "name", "total_memory", "used_memory", "free_memory", "utilization")
    GPU_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    TOTAL_MEMORY_FIELD_NUMBER: _ClassVar[int]
    USED_MEMORY_FIELD_NUMBER: _ClassVar[int]
    FREE_MEMORY_FIELD_NUMBER: _ClassVar[int]
    UTILIZATION_FIELD_NUMBER: _ClassVar[int]
    gpu_id: int
    name: str
    total_memory: int
    used_memory: int
    free_memory: int
    utilization: float
    def __init__(self, gpu_id: _Optional[int] = ..., name: _Optional[str] = ..., total_memory: _Optional[int] = ..., used_memory: _Optional[int] = ..., free_memory: _Optional[int] = ..., utilization: _Optional[float] = ...) -> None: ...

class ForwardPassRequest(_message.Message):
    __slots__ = ("request_id", "input_ids", "batch_size", "seq_len", "hidden_states", "attention_mask", "position_ids", "kv_cache", "use_cache", "is_first_pass", "draft_tokens", "model_name")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    INPUT_IDS_FIELD_NUMBER: _ClassVar[int]
    BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    SEQ_LEN_FIELD_NUMBER: _ClassVar[int]
    HIDDEN_STATES_FIELD_NUMBER: _ClassVar[int]
    ATTENTION_MASK_FIELD_NUMBER: _ClassVar[int]
    POSITION_IDS_FIELD_NUMBER: _ClassVar[int]
    KV_CACHE_FIELD_NUMBER: _ClassVar[int]
    USE_CACHE_FIELD_NUMBER: _ClassVar[int]
    IS_FIRST_PASS_FIELD_NUMBER: _ClassVar[int]
    DRAFT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    input_ids: _containers.RepeatedScalarFieldContainer[int]
    batch_size: int
    seq_len: int
    hidden_states: Tensor
    attention_mask: Tensor
    position_ids: Tensor
    kv_cache: KVCache
    use_cache: bool
    is_first_pass: bool
    draft_tokens: _containers.RepeatedScalarFieldContainer[int]
    model_name: str
    def __init__(self, request_id: _Optional[str] = ..., input_ids: _Optional[_Iterable[int]] = ..., batch_size: _Optional[int] = ..., seq_len: _Optional[int] = ..., hidden_states: _Optional[_Union[Tensor, _Mapping]] = ..., attention_mask: _Optional[_Union[Tensor, _Mapping]] = ..., position_ids: _Optional[_Union[Tensor, _Mapping]] = ..., kv_cache: _Optional[_Union[KVCache, _Mapping]] = ..., use_cache: bool = ..., is_first_pass: bool = ..., draft_tokens: _Optional[_Iterable[int]] = ..., model_name: _Optional[str] = ...) -> None: ...

class ForwardPassResponse(_message.Message):
    __slots__ = ("request_id", "output", "kv_cache", "success", "error_message", "error_code", "is_logits", "verified_tokens", "speculative_score", "processing_time_ms")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    KV_CACHE_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    ERROR_CODE_FIELD_NUMBER: _ClassVar[int]
    IS_LOGITS_FIELD_NUMBER: _ClassVar[int]
    VERIFIED_TOKENS_FIELD_NUMBER: _ClassVar[int]
    SPECULATIVE_SCORE_FIELD_NUMBER: _ClassVar[int]
    PROCESSING_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    output: Tensor
    kv_cache: KVCache
    success: bool
    error_message: str
    error_code: ErrorCode
    is_logits: bool
    verified_tokens: _containers.RepeatedScalarFieldContainer[int]
    speculative_score: float
    processing_time_ms: float
    def __init__(self, request_id: _Optional[str] = ..., output: _Optional[_Union[Tensor, _Mapping]] = ..., kv_cache: _Optional[_Union[KVCache, _Mapping]] = ..., success: bool = ..., error_message: _Optional[str] = ..., error_code: _Optional[_Union[ErrorCode, str]] = ..., is_logits: bool = ..., verified_tokens: _Optional[_Iterable[int]] = ..., speculative_score: _Optional[float] = ..., processing_time_ms: _Optional[float] = ...) -> None: ...

class NodeInfo(_message.Message):
    __slots__ = ("node_id", "host", "port", "total_memory", "available_memory", "device_type", "device_name", "cluster_id")
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    HOST_FIELD_NUMBER: _ClassVar[int]
    PORT_FIELD_NUMBER: _ClassVar[int]
    TOTAL_MEMORY_FIELD_NUMBER: _ClassVar[int]
    AVAILABLE_MEMORY_FIELD_NUMBER: _ClassVar[int]
    DEVICE_TYPE_FIELD_NUMBER: _ClassVar[int]
    DEVICE_NAME_FIELD_NUMBER: _ClassVar[int]
    CLUSTER_ID_FIELD_NUMBER: _ClassVar[int]
    node_id: str
    host: str
    port: int
    total_memory: int
    available_memory: int
    device_type: str
    device_name: str
    cluster_id: str
    def __init__(self, node_id: _Optional[str] = ..., host: _Optional[str] = ..., port: _Optional[int] = ..., total_memory: _Optional[int] = ..., available_memory: _Optional[int] = ..., device_type: _Optional[str] = ..., device_name: _Optional[str] = ..., cluster_id: _Optional[str] = ...) -> None: ...

class NodeRegistration(_message.Message):
    __slots__ = ("node_info", "num_layers", "gpus", "expert_ids")
    NODE_INFO_FIELD_NUMBER: _ClassVar[int]
    NUM_LAYERS_FIELD_NUMBER: _ClassVar[int]
    GPUS_FIELD_NUMBER: _ClassVar[int]
    EXPERT_IDS_FIELD_NUMBER: _ClassVar[int]
    node_info: NodeInfo
    num_layers: int
    gpus: _containers.RepeatedCompositeFieldContainer[GPUInfo]
    expert_ids: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, node_info: _Optional[_Union[NodeInfo, _Mapping]] = ..., num_layers: _Optional[int] = ..., gpus: _Optional[_Iterable[_Union[GPUInfo, _Mapping]]] = ..., expert_ids: _Optional[_Iterable[int]] = ...) -> None: ...

class RegistrationResponse(_message.Message):
    __slots__ = ("accepted", "assigned_start_layer", "assigned_end_layer", "model_name", "is_first_node", "is_last_node", "quantization", "expert_assignments")
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    ASSIGNED_START_LAYER_FIELD_NUMBER: _ClassVar[int]
    ASSIGNED_END_LAYER_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    IS_FIRST_NODE_FIELD_NUMBER: _ClassVar[int]
    IS_LAST_NODE_FIELD_NUMBER: _ClassVar[int]
    QUANTIZATION_FIELD_NUMBER: _ClassVar[int]
    EXPERT_ASSIGNMENTS_FIELD_NUMBER: _ClassVar[int]
    accepted: bool
    assigned_start_layer: int
    assigned_end_layer: int
    model_name: str
    is_first_node: bool
    is_last_node: bool
    quantization: QuantizationConfig
    expert_assignments: _containers.RepeatedCompositeFieldContainer[ExpertInfo]
    def __init__(self, accepted: bool = ..., assigned_start_layer: _Optional[int] = ..., assigned_end_layer: _Optional[int] = ..., model_name: _Optional[str] = ..., is_first_node: bool = ..., is_last_node: bool = ..., quantization: _Optional[_Union[QuantizationConfig, _Mapping]] = ..., expert_assignments: _Optional[_Iterable[_Union[ExpertInfo, _Mapping]]] = ...) -> None: ...

class ExpertInfo(_message.Message):
    __slots__ = ("expert_id", "layer_idx", "node_id")
    EXPERT_ID_FIELD_NUMBER: _ClassVar[int]
    LAYER_IDX_FIELD_NUMBER: _ClassVar[int]
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    expert_id: int
    layer_idx: int
    node_id: str
    def __init__(self, expert_id: _Optional[int] = ..., layer_idx: _Optional[int] = ..., node_id: _Optional[str] = ...) -> None: ...

class MoEForwardRequest(_message.Message):
    __slots__ = ("hidden_states", "expert_ids", "routing_weights", "request_id")
    HIDDEN_STATES_FIELD_NUMBER: _ClassVar[int]
    EXPERT_IDS_FIELD_NUMBER: _ClassVar[int]
    ROUTING_WEIGHTS_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    hidden_states: Tensor
    expert_ids: _containers.RepeatedScalarFieldContainer[int]
    routing_weights: _containers.RepeatedScalarFieldContainer[float]
    request_id: str
    def __init__(self, hidden_states: _Optional[_Union[Tensor, _Mapping]] = ..., expert_ids: _Optional[_Iterable[int]] = ..., routing_weights: _Optional[_Iterable[float]] = ..., request_id: _Optional[str] = ...) -> None: ...

class MoEForwardResponse(_message.Message):
    __slots__ = ("output", "success", "error_message", "processing_time_ms")
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    PROCESSING_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    output: Tensor
    success: bool
    error_message: str
    processing_time_ms: float
    def __init__(self, output: _Optional[_Union[Tensor, _Mapping]] = ..., success: bool = ..., error_message: _Optional[str] = ..., processing_time_ms: _Optional[float] = ...) -> None: ...

class QuantizationConfig(_message.Message):
    __slots__ = ("method", "bnb_4bit_compute_dtype", "bnb_4bit_quant_type", "bnb_4bit_use_double_quant", "llm_int8_threshold")
    METHOD_FIELD_NUMBER: _ClassVar[int]
    BNB_4BIT_COMPUTE_DTYPE_FIELD_NUMBER: _ClassVar[int]
    BNB_4BIT_QUANT_TYPE_FIELD_NUMBER: _ClassVar[int]
    BNB_4BIT_USE_DOUBLE_QUANT_FIELD_NUMBER: _ClassVar[int]
    LLM_INT8_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    method: str
    bnb_4bit_compute_dtype: str
    bnb_4bit_quant_type: str
    bnb_4bit_use_double_quant: bool
    llm_int8_threshold: float
    def __init__(self, method: _Optional[str] = ..., bnb_4bit_compute_dtype: _Optional[str] = ..., bnb_4bit_quant_type: _Optional[str] = ..., bnb_4bit_use_double_quant: bool = ..., llm_int8_threshold: _Optional[float] = ...) -> None: ...

class HealthCheckRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthCheckResponse(_message.Message):
    __slots__ = ("node_id", "healthy", "memory_used", "memory_total", "gpu_utilization", "temperature", "gpus")
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    HEALTHY_FIELD_NUMBER: _ClassVar[int]
    MEMORY_USED_FIELD_NUMBER: _ClassVar[int]
    MEMORY_TOTAL_FIELD_NUMBER: _ClassVar[int]
    GPU_UTILIZATION_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    GPUS_FIELD_NUMBER: _ClassVar[int]
    node_id: str
    healthy: bool
    memory_used: int
    memory_total: int
    gpu_utilization: float
    temperature: float
    gpus: _containers.RepeatedCompositeFieldContainer[GPUInfo]
    def __init__(self, node_id: _Optional[str] = ..., healthy: bool = ..., memory_used: _Optional[int] = ..., memory_total: _Optional[int] = ..., gpu_utilization: _Optional[float] = ..., temperature: _Optional[float] = ..., gpus: _Optional[_Iterable[_Union[GPUInfo, _Mapping]]] = ...) -> None: ...

class PingRequest(_message.Message):
    __slots__ = ("source_cluster_id", "timestamp")
    SOURCE_CLUSTER_ID_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    source_cluster_id: str
    timestamp: int
    def __init__(self, source_cluster_id: _Optional[str] = ..., timestamp: _Optional[int] = ...) -> None: ...

class PingResponse(_message.Message):
    __slots__ = ("node_id", "cluster_id", "timestamp", "latency_ms")
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    CLUSTER_ID_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    LATENCY_MS_FIELD_NUMBER: _ClassVar[int]
    node_id: str
    cluster_id: str
    timestamp: int
    latency_ms: float
    def __init__(self, node_id: _Optional[str] = ..., cluster_id: _Optional[str] = ..., timestamp: _Optional[int] = ..., latency_ms: _Optional[float] = ...) -> None: ...

class InferenceRequest(_message.Message):
    __slots__ = ("request_id", "prompt", "max_tokens", "temperature", "top_p", "stream", "priority")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    MAX_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    TOP_P_FIELD_NUMBER: _ClassVar[int]
    STREAM_FIELD_NUMBER: _ClassVar[int]
    PRIORITY_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    prompt: str
    max_tokens: int
    temperature: float
    top_p: float
    stream: bool
    priority: int
    def __init__(self, request_id: _Optional[str] = ..., prompt: _Optional[str] = ..., max_tokens: _Optional[int] = ..., temperature: _Optional[float] = ..., top_p: _Optional[float] = ..., stream: bool = ..., priority: _Optional[int] = ...) -> None: ...

class TokenResponse(_message.Message):
    __slots__ = ("request_id", "token", "is_final", "full_text")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    TOKEN_FIELD_NUMBER: _ClassVar[int]
    IS_FINAL_FIELD_NUMBER: _ClassVar[int]
    FULL_TEXT_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    token: str
    is_final: bool
    full_text: str
    def __init__(self, request_id: _Optional[str] = ..., token: _Optional[str] = ..., is_final: bool = ..., full_text: _Optional[str] = ...) -> None: ...

class LogitsResponse(_message.Message):
    __slots__ = ("request_id", "generated_text", "success", "error_message", "error_code")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    GENERATED_TEXT_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    ERROR_CODE_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    generated_text: str
    success: bool
    error_message: str
    error_code: ErrorCode
    def __init__(self, request_id: _Optional[str] = ..., generated_text: _Optional[str] = ..., success: bool = ..., error_message: _Optional[str] = ..., error_code: _Optional[_Union[ErrorCode, str]] = ...) -> None: ...

class ModelListResponse(_message.Message):
    __slots__ = ("models", "default_model")
    MODELS_FIELD_NUMBER: _ClassVar[int]
    DEFAULT_MODEL_FIELD_NUMBER: _ClassVar[int]
    models: _containers.RepeatedScalarFieldContainer[str]
    default_model: str
    def __init__(self, models: _Optional[_Iterable[str]] = ..., default_model: _Optional[str] = ...) -> None: ...

class GossipAdvertisement(_message.Message):
    __slots__ = ("node_id", "cache_prefixes", "total_cache_entries", "timestamp")
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    CACHE_PREFIXES_FIELD_NUMBER: _ClassVar[int]
    TOTAL_CACHE_ENTRIES_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    node_id: str
    cache_prefixes: _containers.RepeatedScalarFieldContainer[str]
    total_cache_entries: int
    timestamp: int
    def __init__(self, node_id: _Optional[str] = ..., cache_prefixes: _Optional[_Iterable[str]] = ..., total_cache_entries: _Optional[int] = ..., timestamp: _Optional[int] = ...) -> None: ...

class GossipRequest(_message.Message):
    __slots__ = ("requester_id", "target_node_id", "requested_prefixes")
    REQUESTER_ID_FIELD_NUMBER: _ClassVar[int]
    TARGET_NODE_ID_FIELD_NUMBER: _ClassVar[int]
    REQUESTED_PREFIXES_FIELD_NUMBER: _ClassVar[int]
    requester_id: str
    target_node_id: str
    requested_prefixes: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, requester_id: _Optional[str] = ..., target_node_id: _Optional[str] = ..., requested_prefixes: _Optional[_Iterable[str]] = ...) -> None: ...

class GossipResponse(_message.Message):
    __slots__ = ("success", "error_message", "cache_entries", "entries_returned")
    class CacheEntriesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: KVCache
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[KVCache, _Mapping]] = ...) -> None: ...
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    CACHE_ENTRIES_FIELD_NUMBER: _ClassVar[int]
    ENTRIES_RETURNED_FIELD_NUMBER: _ClassVar[int]
    success: bool
    error_message: str
    cache_entries: _containers.MessageMap[str, KVCache]
    entries_returned: int
    def __init__(self, success: bool = ..., error_message: _Optional[str] = ..., cache_entries: _Optional[_Mapping[str, KVCache]] = ..., entries_returned: _Optional[int] = ...) -> None: ...
