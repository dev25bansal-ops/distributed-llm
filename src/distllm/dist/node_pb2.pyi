from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TensorProto(_message.Message):
    __slots__ = ("shape", "dtype", "raw_data", "data")
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    RAW_DATA_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    shape: _containers.RepeatedScalarFieldContainer[int]
    dtype: str
    raw_data: bytes
    data: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, shape: _Optional[_Iterable[int]] = ..., dtype: _Optional[str] = ..., raw_data: _Optional[bytes] = ..., data: _Optional[_Iterable[float]] = ...) -> None: ...

class KVCacheProto(_message.Message):
    __slots__ = ("layers",)
    class LayerCache(_message.Message):
        __slots__ = ("key_states", "value_states", "key_scale", "value_scale")
        KEY_STATES_FIELD_NUMBER: _ClassVar[int]
        VALUE_STATES_FIELD_NUMBER: _ClassVar[int]
        KEY_SCALE_FIELD_NUMBER: _ClassVar[int]
        VALUE_SCALE_FIELD_NUMBER: _ClassVar[int]
        key_states: TensorProto
        value_states: TensorProto
        key_scale: TensorProto
        value_scale: TensorProto
        def __init__(self, key_states: _Optional[_Union[TensorProto, _Mapping]] = ..., value_states: _Optional[_Union[TensorProto, _Mapping]] = ..., key_scale: _Optional[_Union[TensorProto, _Mapping]] = ..., value_scale: _Optional[_Union[TensorProto, _Mapping]] = ...) -> None: ...
    LAYERS_FIELD_NUMBER: _ClassVar[int]
    layers: _containers.RepeatedCompositeFieldContainer[KVCacheProto.LayerCache]
    def __init__(self, layers: _Optional[_Iterable[_Union[KVCacheProto.LayerCache, _Mapping]]] = ...) -> None: ...

class ForwardPassRequest(_message.Message):
    __slots__ = ("request_id", "input_ids", "hidden_states", "attention_mask", "position_ids", "kv_cache", "use_cache", "is_first_pass", "draft_tokens", "batch_size", "seq_len", "is_last_pass", "model_name", "cluster_key")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    INPUT_IDS_FIELD_NUMBER: _ClassVar[int]
    HIDDEN_STATES_FIELD_NUMBER: _ClassVar[int]
    ATTENTION_MASK_FIELD_NUMBER: _ClassVar[int]
    POSITION_IDS_FIELD_NUMBER: _ClassVar[int]
    KV_CACHE_FIELD_NUMBER: _ClassVar[int]
    USE_CACHE_FIELD_NUMBER: _ClassVar[int]
    IS_FIRST_PASS_FIELD_NUMBER: _ClassVar[int]
    DRAFT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    SEQ_LEN_FIELD_NUMBER: _ClassVar[int]
    IS_LAST_PASS_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    CLUSTER_KEY_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    input_ids: _containers.RepeatedScalarFieldContainer[int]
    hidden_states: TensorProto
    attention_mask: TensorProto
    position_ids: TensorProto
    kv_cache: KVCacheProto
    use_cache: bool
    is_first_pass: bool
    draft_tokens: _containers.RepeatedScalarFieldContainer[int]
    batch_size: int
    seq_len: int
    is_last_pass: bool
    model_name: str
    cluster_key: str
    def __init__(self, request_id: _Optional[str] = ..., input_ids: _Optional[_Iterable[int]] = ..., hidden_states: _Optional[_Union[TensorProto, _Mapping]] = ..., attention_mask: _Optional[_Union[TensorProto, _Mapping]] = ..., position_ids: _Optional[_Union[TensorProto, _Mapping]] = ..., kv_cache: _Optional[_Union[KVCacheProto, _Mapping]] = ..., use_cache: bool = ..., is_first_pass: bool = ..., draft_tokens: _Optional[_Iterable[int]] = ..., batch_size: _Optional[int] = ..., seq_len: _Optional[int] = ..., is_last_pass: bool = ..., model_name: _Optional[str] = ..., cluster_key: _Optional[str] = ...) -> None: ...

class ForwardPassResponse(_message.Message):
    __slots__ = ("request_id", "output", "kv_cache", "success", "error_message", "error_code", "is_logits", "processing_time_ms", "cluster_key")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    KV_CACHE_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    ERROR_CODE_FIELD_NUMBER: _ClassVar[int]
    IS_LOGITS_FIELD_NUMBER: _ClassVar[int]
    PROCESSING_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    CLUSTER_KEY_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    output: TensorProto
    kv_cache: KVCacheProto
    success: bool
    error_message: str
    error_code: int
    is_logits: bool
    processing_time_ms: float
    cluster_key: str
    def __init__(self, request_id: _Optional[str] = ..., output: _Optional[_Union[TensorProto, _Mapping]] = ..., kv_cache: _Optional[_Union[KVCacheProto, _Mapping]] = ..., success: bool = ..., error_message: _Optional[str] = ..., error_code: _Optional[int] = ..., is_logits: bool = ..., processing_time_ms: _Optional[float] = ..., cluster_key: _Optional[str] = ...) -> None: ...

class HealthCheckRequest(_message.Message):
    __slots__ = ("node_id", "cluster_key")
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    CLUSTER_KEY_FIELD_NUMBER: _ClassVar[int]
    node_id: str
    cluster_key: str
    def __init__(self, node_id: _Optional[str] = ..., cluster_key: _Optional[str] = ...) -> None: ...

class PrivacySplitInfo(_message.Message):
    __slots__ = ("enabled", "prefix_layers", "suffix_layers")
    ENABLED_FIELD_NUMBER: _ClassVar[int]
    PREFIX_LAYERS_FIELD_NUMBER: _ClassVar[int]
    SUFFIX_LAYERS_FIELD_NUMBER: _ClassVar[int]
    enabled: bool
    prefix_layers: int
    suffix_layers: int
    def __init__(self, enabled: bool = ..., prefix_layers: _Optional[int] = ..., suffix_layers: _Optional[int] = ...) -> None: ...

class HealthCheckResponse(_message.Message):
    __slots__ = ("healthy", "node_id", "memory_used_bytes", "memory_total_bytes", "gpu_utilization", "start_layer", "end_layer", "total_layers", "gpu_name", "gpu_memory_total", "num_layers_loaded", "cluster_key", "privacy_split")
    HEALTHY_FIELD_NUMBER: _ClassVar[int]
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    MEMORY_USED_BYTES_FIELD_NUMBER: _ClassVar[int]
    MEMORY_TOTAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    GPU_UTILIZATION_FIELD_NUMBER: _ClassVar[int]
    START_LAYER_FIELD_NUMBER: _ClassVar[int]
    END_LAYER_FIELD_NUMBER: _ClassVar[int]
    TOTAL_LAYERS_FIELD_NUMBER: _ClassVar[int]
    GPU_NAME_FIELD_NUMBER: _ClassVar[int]
    GPU_MEMORY_TOTAL_FIELD_NUMBER: _ClassVar[int]
    NUM_LAYERS_LOADED_FIELD_NUMBER: _ClassVar[int]
    CLUSTER_KEY_FIELD_NUMBER: _ClassVar[int]
    PRIVACY_SPLIT_FIELD_NUMBER: _ClassVar[int]
    healthy: bool
    node_id: str
    memory_used_bytes: int
    memory_total_bytes: int
    gpu_utilization: float
    start_layer: int
    end_layer: int
    total_layers: int
    gpu_name: str
    gpu_memory_total: int
    num_layers_loaded: int
    cluster_key: str
    privacy_split: PrivacySplitInfo
    def __init__(self, healthy: bool = ..., node_id: _Optional[str] = ..., memory_used_bytes: _Optional[int] = ..., memory_total_bytes: _Optional[int] = ..., gpu_utilization: _Optional[float] = ..., start_layer: _Optional[int] = ..., end_layer: _Optional[int] = ..., total_layers: _Optional[int] = ..., gpu_name: _Optional[str] = ..., gpu_memory_total: _Optional[int] = ..., num_layers_loaded: _Optional[int] = ..., cluster_key: _Optional[str] = ..., privacy_split: _Optional[_Union[PrivacySplitInfo, _Mapping]] = ...) -> None: ...

class ProfileRequest(_message.Message):
    __slots__ = ("node_id", "cluster_key")
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    CLUSTER_KEY_FIELD_NUMBER: _ClassVar[int]
    node_id: str
    cluster_key: str
    def __init__(self, node_id: _Optional[str] = ..., cluster_key: _Optional[str] = ...) -> None: ...

class ProfileResponse(_message.Message):
    __slots__ = ("node_id", "gpu_name", "total_memory_bytes", "free_memory_bytes", "compute_tflops", "memory_bandwidth_gbps", "sm_count", "cluster_key")
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    GPU_NAME_FIELD_NUMBER: _ClassVar[int]
    TOTAL_MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
    FREE_MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
    COMPUTE_TFLOPS_FIELD_NUMBER: _ClassVar[int]
    MEMORY_BANDWIDTH_GBPS_FIELD_NUMBER: _ClassVar[int]
    SM_COUNT_FIELD_NUMBER: _ClassVar[int]
    CLUSTER_KEY_FIELD_NUMBER: _ClassVar[int]
    node_id: str
    gpu_name: str
    total_memory_bytes: int
    free_memory_bytes: int
    compute_tflops: float
    memory_bandwidth_gbps: float
    sm_count: int
    cluster_key: str
    def __init__(self, node_id: _Optional[str] = ..., gpu_name: _Optional[str] = ..., total_memory_bytes: _Optional[int] = ..., free_memory_bytes: _Optional[int] = ..., compute_tflops: _Optional[float] = ..., memory_bandwidth_gbps: _Optional[float] = ..., sm_count: _Optional[int] = ..., cluster_key: _Optional[str] = ...) -> None: ...

class TransferWeightsRequest(_message.Message):
    __slots__ = ("model_name", "start_layer", "end_layer", "cluster_key", "chunk_index", "total_chunks")
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    START_LAYER_FIELD_NUMBER: _ClassVar[int]
    END_LAYER_FIELD_NUMBER: _ClassVar[int]
    CLUSTER_KEY_FIELD_NUMBER: _ClassVar[int]
    CHUNK_INDEX_FIELD_NUMBER: _ClassVar[int]
    TOTAL_CHUNKS_FIELD_NUMBER: _ClassVar[int]
    model_name: str
    start_layer: int
    end_layer: int
    cluster_key: str
    chunk_index: int
    total_chunks: int
    def __init__(self, model_name: _Optional[str] = ..., start_layer: _Optional[int] = ..., end_layer: _Optional[int] = ..., cluster_key: _Optional[str] = ..., chunk_index: _Optional[int] = ..., total_chunks: _Optional[int] = ...) -> None: ...

class TransferWeightsResponse(_message.Message):
    __slots__ = ("model_name", "start_layer", "end_layer", "state_dict_bytes", "success", "error_message", "chunk_index", "total_chunks", "is_final_chunk")
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    START_LAYER_FIELD_NUMBER: _ClassVar[int]
    END_LAYER_FIELD_NUMBER: _ClassVar[int]
    STATE_DICT_BYTES_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    CHUNK_INDEX_FIELD_NUMBER: _ClassVar[int]
    TOTAL_CHUNKS_FIELD_NUMBER: _ClassVar[int]
    IS_FINAL_CHUNK_FIELD_NUMBER: _ClassVar[int]
    model_name: str
    start_layer: int
    end_layer: int
    state_dict_bytes: bytes
    success: bool
    error_message: str
    chunk_index: int
    total_chunks: int
    is_final_chunk: bool
    def __init__(self, model_name: _Optional[str] = ..., start_layer: _Optional[int] = ..., end_layer: _Optional[int] = ..., state_dict_bytes: _Optional[bytes] = ..., success: bool = ..., error_message: _Optional[str] = ..., chunk_index: _Optional[int] = ..., total_chunks: _Optional[int] = ..., is_final_chunk: bool = ...) -> None: ...

class AdvertiseModelsRequest(_message.Message):
    __slots__ = ("node_id", "cluster_key")
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    CLUSTER_KEY_FIELD_NUMBER: _ClassVar[int]
    node_id: str
    cluster_key: str
    def __init__(self, node_id: _Optional[str] = ..., cluster_key: _Optional[str] = ...) -> None: ...

class ModelAdvertisement(_message.Message):
    __slots__ = ("model_name", "start_layer", "end_layer", "total_layers", "node_id", "host", "port")
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    START_LAYER_FIELD_NUMBER: _ClassVar[int]
    END_LAYER_FIELD_NUMBER: _ClassVar[int]
    TOTAL_LAYERS_FIELD_NUMBER: _ClassVar[int]
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    HOST_FIELD_NUMBER: _ClassVar[int]
    PORT_FIELD_NUMBER: _ClassVar[int]
    model_name: str
    start_layer: int
    end_layer: int
    total_layers: int
    node_id: str
    host: str
    port: int
    def __init__(self, model_name: _Optional[str] = ..., start_layer: _Optional[int] = ..., end_layer: _Optional[int] = ..., total_layers: _Optional[int] = ..., node_id: _Optional[str] = ..., host: _Optional[str] = ..., port: _Optional[int] = ...) -> None: ...

class AdvertiseModelsResponse(_message.Message):
    __slots__ = ("models",)
    MODELS_FIELD_NUMBER: _ClassVar[int]
    models: _containers.RepeatedCompositeFieldContainer[ModelAdvertisement]
    def __init__(self, models: _Optional[_Iterable[_Union[ModelAdvertisement, _Mapping]]] = ...) -> None: ...
