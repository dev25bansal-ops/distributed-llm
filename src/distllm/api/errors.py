"""Shared error response utilities for the API layer.

Provides standardized error responses used by all routers and middleware.
Maps DistLLM error types to OpenAI-compatible error codes.
"""

from fastapi import Request
from fastapi.responses import JSONResponse

from distllm.errors.types import (
    DistLLMError,
    NodeError,
    NodeUnreachableError,
    OOMError,
    GRPCTimeoutError,
    ModelError,
    ModelNotFoundError,
    ModelLoadError,
    ConfigError,
    ConfigValidationError,
    BatchError,
    BatchCapacityError,
    ConstraintError,
    ConstraintViolationError,
    InputValidationError,
    GatewayError,
    ProviderTimeoutError,
    CircuitBreakerError,
)

_HTTP_CODE_MAP: dict[int, str] = {
    400: "400",
    401: "401",
    403: "403",
    404: "404",
    408: "408",
    413: "413",
    422: "422",
    429: "429",
    500: "500",
    502: "502",
    503: "503",
    504: "504",
}

# Map DistLLM exception types to error codes
_EXCEPTION_CODE_MAP: dict[type, str] = {
    NodeUnreachableError: "node_unreachable",
    CircuitBreakerError: "circuit_breaker_open",
    OOMError: "out_of_memory",
    GRPCTimeoutError: "node_timeout",
    NodeError: "node_error",
    ModelNotFoundError: "model_not_found",
    ModelLoadError: "model_load_error",
    ModelError: "model_error",
    ConfigValidationError: "config_validation_error",
    ConfigError: "config_error",
    BatchCapacityError: "batch_capacity_exceeded",
    BatchError: "batch_error",
    ConstraintViolationError: "constraint_violation",
    ConstraintError: "constraint_error",
    InputValidationError: "invalid_request_error",
    ProviderTimeoutError: "provider_timeout",
    GatewayError: "gateway_error",
}


def get_error_code(exc: Exception | None = None, status_code: int = 500) -> str:
    """Get the error code for an exception or HTTP status.

    Checks exception type first (DistLLM hierarchy), then falls back
    to HTTP status code mapping.
    """
    if exc is not None:
        for exc_type, code in _EXCEPTION_CODE_MAP.items():
            if isinstance(exc, exc_type):
                return code
    return _HTTP_CODE_MAP.get(status_code, str(status_code))


def error_response(
    status_code: int,
    error: str,
    message: str,
    type: str = "api_error",
    request_id: str | None = None,
    retry_after: float | None = None,
    code: str | None = None,
    exc: Exception | None = None,
) -> JSONResponse:
    """Build a standardized API error response (OpenAI-compatible format).

    All API errors should use this function or the _error_response helper
    (which injects request_id from request.state) to ensure consistent format.

    OpenAI error format:
    {"error": {"message": "...", "type": "...", "param": null, "code": "..."}}

    Args:
        status_code: HTTP status code.
        error: Short error title (e.g. "Bad Request").
        message: Human-readable error detail.
        type: Error type category.
        request_id: Optional request ID for tracing.
        retry_after: Seconds until retry is allowed (for 429).
        code: Explicit error code (overrides auto-detection).
        exc: Original exception (used for auto error code detection).
    """
    error_code = code or get_error_code(exc=exc, status_code=status_code)
    error_body: dict = {
        "message": message,
        "type": type,
        "param": None,
        "code": error_code,
    }
    if retry_after is not None:
        error_body["retry_after"] = retry_after
    content: dict = {
        "error": error_body,
    }
    if request_id:
        content["request_id"] = request_id
    return JSONResponse(
        status_code=status_code,
        content=content,
    )


def error_response_from_request(
    status_code: int,
    error: str,
    message: str,
    type: str = "api_error",
    request: Request | None = None,
    code: str | None = None,
    exc: Exception | None = None,
) -> JSONResponse:
    """Build a standardized error response, injecting request_id from request.state."""
    request_id = getattr(request.state, "request_id", None) if request is not None else None
    return error_response(status_code, error, message, type, request_id, code=code, exc=exc)
