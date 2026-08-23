"""Lightweight response validation for the DistLLM SDK.

Validates API responses against the expected type schemas without
requiring Pydantic. Catches API contract violations (missing fields,
wrong types) early with clear error messages.

Enabled via ``validate_responses=True`` on the client constructor::

    client = DistLLMClient(validate_responses=True)
    response = await client.chat_completions(...)  # validated
"""

from __future__ import annotations

import typing
from typing import Any, get_type_hints, get_origin, get_args

# Sentinel for unset defaults
_UNSET = object()


_TYPE_ALIASES: dict[type, tuple[type, ...]] = {
    str: (str,),
    int: (int,),
    float: (int, float),
    bool: (bool,),
    bytes: (bytes, bytearray),
    list: (list, tuple),
    dict: (dict,),
}


def _check_type(value: Any, expected: type, path: str) -> list[str]:
    """Return a list of type errors for *value* against *expected*."""
    errors: list[str] = []
    origin = get_origin(expected)
    args = get_args(expected)

    # Optional[X] = Union[X, None]
    if origin is type(typing.Union) or origin is type(typing.Optional):
        if value is None:
            return []
        inner = args[0]
        return _check_type(value, inner, path)

    # list[X]
    if origin is list:
        if not isinstance(value, (list, tuple)):
            return [f"{path}: expected list, got {type(value).__name__}"]
        inner = args[0] if args else typing.Any
        for i, item in enumerate(value):
            errors.extend(_check_type(item, inner, f"{path}[{i}]"))
        return errors

    # dict[K, V]
    if origin is dict:
        if not isinstance(value, dict):
            return [f"{path}: expected dict, got {type(value).__name__}"]
        return errors

    # Literal types
    if origin is type(typing.Literal):
        allowed = args
        if value not in allowed:
            return [f"{path}: expected one of {allowed}, got {value!r}"]
        return errors

    # Plain type
    allowed_types = _TYPE_ALIASES.get(expected, (expected,))
    if not isinstance(value, allowed_types):
        return [f"{path}: expected {expected.__name__}, got {type(value).__name__}"]

    return errors


def validate_dataclass(obj: Any, dataclass_type: type, path: str = "") -> list[str]:
    """Validate a dict or dataclass instance against a dataclass type.

    Args:
        obj: Dict or dataclass instance to validate.
        dataclass_type: A ``@dataclass`` type to validate against.
        path: Dot-separated field path (for nested error messages).

    Returns:
        List of validation error strings (empty if valid).
    """
    errors: list[str] = []
    hints = get_type_hints(dataclass_type)

    if not isinstance(obj, dict):
        # Already a dataclass instance — convert to dict for validation
        if hasattr(obj, "__dataclass_fields__"):
            obj = {f.name: getattr(obj, f.name) for f in dataclass_type.__dataclass_fields__.values()}
        else:
            return [f"{path}: expected dict or {dataclass_type.__name__}, got {type(obj).__name__}"]

    for field_name, field_type in hints.items():
        field_path = f"{path}.{field_name}" if path else field_name
        if field_name not in obj:
            errors.append(f"{field_path}: required field missing")
            continue
        value = obj[field_name]
        errors.extend(_check_type(value, field_type, field_path))

    return errors


def validate_response(data: dict, endpoint: str) -> list[str]:
    """Validate an API response dict against the expected schema for *endpoint*.

    Args:
        data: The JSON response dict from the API.
        endpoint: API path (e.g. ``/v1/chat/completions``).

    Returns:
        List of validation error strings (empty if valid).
    """
    from distllm_sdk.types import (
        ChatCompletionResponse, CompletionResponse,
        EmbeddingResponse, ModelList,
        BatchJob, BatchList,
        TranscriptionResponse, SpeechResponse,
        ImageGenerationResponse, ModerationResponse,
        FileInfo, FineTuningJob,
    )

    schema_map: dict[str, type] = {
        "/v1/chat/completions": ChatCompletionResponse,
        "/v1/completions": CompletionResponse,
        "/v1/embeddings": EmbeddingResponse,
        "/v1/models": ModelList,
        "/v1/batches": BatchList,
        "/v1/moderations": ModerationResponse,
        "/v1/audio/transcriptions": TranscriptionResponse,
        "/v1/audio/speech": SpeechResponse,
        "/v1/images/generations": ImageGenerationResponse,
        "/v1/files": list,
        "/health": dict,
    }

    schema_type = schema_map.get(endpoint)
    if schema_type is None:
        return []

    # Batch endpoints can be BatchJob or BatchList depending on whether an ID is present
    if endpoint.startswith("/v1/batches/"):
        schema_type = BatchJob
    if endpoint.startswith("/v1/files/"):
        schema_type = FileInfo
    if endpoint.startswith("/v1/fine_tuning/"):
        schema_type = FineTuningJob

    if schema_type is list:
        items = data.get("data", data)
        if isinstance(items, list):
            row_errors = []
            for i, item in enumerate(items):
                row_errors.extend(validate_dataclass(item, FileInfo, f"data[{i}]"))
            return row_errors
        return []

    if schema_type is dict:
        return []

    return validate_dataclass(data, schema_type)
