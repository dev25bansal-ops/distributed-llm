from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar

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
    def __init__(self, data: _Iterable[float] | None = ..., shape: _Iterable[int] | None = ..., dtype: str | None = ..., raw_data: bytes | None = ...) -> None: ...

class KVLayerCache(_message.Message):
    __slots__ = ("key_states", "value_states")
    KEY_STATES_FIELD_NUMBER: _ClassVar[int]
    VALUE_STATES_FIELD_NUMBER: _ClassVar[int]
    key_states: Tensor
    value_states: Tensor
    def __init__(self, key_states: Tensor | _Mapping | None = ..., value_states: Tensor | _Mapping | None = ...) -> None: ...

class KVCache(_message.Message):
    __slots__ = ("layers",)
    LAYERS_FIELD_NUMBER: _ClassVar[int]
    layers: _containers.RepeatedCompositeFieldContainer[KVLayerCache]
    def __init__(self, layers: _Iterable[KVLayerCache | _Mapping] | None = ...) -> None: ...

class ForwardPassRequest(_message.Message):
    __slots__ = ("request_id", "input_ids", "batch_size", "seq_len", "hidden_states", "attention_mask", "position_ids", "kv_cache", "use_cache", "is_first_pass")
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
    def __init__(self, request_id: str | None = ..., input_ids: _Iterable[int] | None = ..., batch_size: int | None = ..., seq_len: int | None = ..., hidden_states: Tensor | _Mapping | None = ..., attention_mask: Tensor | _Mapping | None = ..., position_ids: Tensor | _Mapping | None = ..., kv_cache: KVCache | _Mapping | None = ..., use_cache: bool = ..., is_first_pass: bool = ...) -> None: ...

class ForwardPassResponse(_message.Message):
    __slots__ = ("request_id", "output", "kv_cache", "success", "error_message", "error_code", "is_logits")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    KV_CACHE_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    ERROR_CODE_FIELD_NUMBER: _ClassVar[int]
    IS_LOGITS_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    output: Tensor
    kv_cache: KVCache
    success: bool
    error_message: str
    error_code: ErrorCode
    is_logits: bool
    def __init__(self, request_id: str | None = ..., output: Tensor | _Mapping | None = ..., kv_cache: KVCache | _Mapping | None = ..., success: bool = ..., error_message: str | None = ..., error_code: ErrorCode | str | None = ..., is_logits: bool = ...) -> None: ...

class NodeInfo(_message.Message):
    __slots__ = ("node_id", "host", "port", "total_memory", "available_memory", "device_type", "device_name")
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    HOST_FIELD_NUMBER: _ClassVar[int]
    PORT_FIELD_NUMBER: _ClassVar[int]
    TOTAL_MEMORY_FIELD_NUMBER: _ClassVar[int]
    AVAILABLE_MEMORY_FIELD_NUMBER: _ClassVar[int]
    DEVICE_TYPE_FIELD_NUMBER: _ClassVar[int]
    DEVICE_NAME_FIELD_NUMBER: _ClassVar[int]
    node_id: str
    host: str
    port: int
    total_memory: int
    available_memory: int
    device_type: str
    device_name: str
    def __init__(self, node_id: str | None = ..., host: str | None = ..., port: int | None = ..., total_memory: int | None = ..., available_memory: int | None = ..., device_type: str | None = ..., device_name: str | None = ...) -> None: ...

class NodeRegistration(_message.Message):
    __slots__ = ("node_info", "num_layers")
    NODE_INFO_FIELD_NUMBER: _ClassVar[int]
    NUM_LAYERS_FIELD_NUMBER: _ClassVar[int]
    node_info: NodeInfo
    num_layers: int
    def __init__(self, node_info: NodeInfo | _Mapping | None = ..., num_layers: int | None = ...) -> None: ...

class RegistrationResponse(_message.Message):
    __slots__ = ("accepted", "assigned_start_layer", "assigned_end_layer", "model_name", "is_first_node", "is_last_node")
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    ASSIGNED_START_LAYER_FIELD_NUMBER: _ClassVar[int]
    ASSIGNED_END_LAYER_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    IS_FIRST_NODE_FIELD_NUMBER: _ClassVar[int]
    IS_LAST_NODE_FIELD_NUMBER: _ClassVar[int]
    accepted: bool
    assigned_start_layer: int
    assigned_end_layer: int
    model_name: str
    is_first_node: bool
    is_last_node: bool
    def __init__(self, accepted: bool = ..., assigned_start_layer: int | None = ..., assigned_end_layer: int | None = ..., model_name: str | None = ..., is_first_node: bool = ..., is_last_node: bool = ...) -> None: ...

class HealthCheckRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthCheckResponse(_message.Message):
    __slots__ = ("node_id", "healthy", "memory_used", "memory_total", "gpu_utilization", "temperature")
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    HEALTHY_FIELD_NUMBER: _ClassVar[int]
    MEMORY_USED_FIELD_NUMBER: _ClassVar[int]
    MEMORY_TOTAL_FIELD_NUMBER: _ClassVar[int]
    GPU_UTILIZATION_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    node_id: str
    healthy: bool
    memory_used: int
    memory_total: int
    gpu_utilization: float
    temperature: float
    def __init__(self, node_id: str | None = ..., healthy: bool = ..., memory_used: int | None = ..., memory_total: int | None = ..., gpu_utilization: float | None = ..., temperature: float | None = ...) -> None: ...

class InferenceRequest(_message.Message):
    __slots__ = ("request_id", "prompt", "max_tokens", "temperature", "top_p", "stream")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    MAX_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    TOP_P_FIELD_NUMBER: _ClassVar[int]
    STREAM_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    prompt: str
    max_tokens: int
    temperature: float
    top_p: float
    stream: bool
    def __init__(self, request_id: str | None = ..., prompt: str | None = ..., max_tokens: int | None = ..., temperature: float | None = ..., top_p: float | None = ..., stream: bool = ...) -> None: ...

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
    def __init__(self, request_id: str | None = ..., token: str | None = ..., is_final: bool = ..., full_text: str | None = ...) -> None: ...

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
    def __init__(self, request_id: str | None = ..., generated_text: str | None = ..., success: bool = ..., error_message: str | None = ..., error_code: ErrorCode | str | None = ...) -> None: ...
