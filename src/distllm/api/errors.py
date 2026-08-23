"""Shared error response utilities for the API layer.

Provides standardized error responses used by all routers and middleware.
Maps DistLLM error types to OpenAI-compatible error codes.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

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


class ErrorCode(str, Enum):
    """Standardized error codes matching OpenAI API conventions.

    Every code is a string enum so it serialises to the JSON wire format
    without a custom encoder.
    """

    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    QUOTA_EXCEEDED = "quota_exceeded"
    MODEL_NOT_FOUND = "model_not_found"
    CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"
    INVALID_REQUEST = "invalid_request"
    AUTHENTICATION_ERROR = "authentication_error"
    SERVER_ERROR = "server_error"
    SERVICE_UNAVAILABLE = "service_unavailable"
    BATCH_PARTIAL_FAILURE = "batch_partial_failure"
    WEBHOOK_DELIVERY_FAILED = "webhook_delivery_failed"


class ErrorResponse(BaseModel):
    """Standardized error response matching OpenAI's error format.

    Wire format::

        {"error": {"message": "...", "type": "...",
                   "code": "...", "param": null, "request_id": "..."}}
    """

    error: dict = Field(
        default_factory=lambda: {
            "message": "",
            "type": "api_error",
            "code": str(ErrorCode.SERVER_ERROR),
            "param": None,
            "request_id": None,
        }
    )


class ErrorResponseBuilder:
    """Registry-based builder for :class:`ErrorResponse` objects.

    Usage::

        builder = ErrorResponseBuilder()
        builder.register("rate_limit", 429)
        builder.register("model_not_found", 404)
        response = builder.build(
            "rate_limit", message="Too many requests", request_id="abc-123"
        )
    """

    def __init__(self) -> None:
        self._registry: dict[str, int] = {}

    def register(self, error_type: str, status_code: int) -> None:
        """Register *error_type* to map to *status_code*."""
        self._registry[error_type] = status_code

    def build(
        self,
        error_type: str,
        message: str = "",
        type: str = "api_error",
        request_id: str | None = None,
        param: Any = None,
    ) -> ErrorResponse:
        """Build an :class:`ErrorResponse` for the registered *error_type*.

        Falls back to HTTP 500 if *error_type* has not been registered.
        """
        return ErrorResponse(
            error={
                "message": message,
                "type": type,
                "code": error_type,
                "param": param,
                "request_id": request_id,
            }
        )

    def build_json(
        self,
        error_type: str,
        message: str = "",
        type: str = "api_error",
        request_id: str | None = None,
        param: Any = None,
        status_code: int | None = None,
    ) -> JSONResponse:
        """Build a :class:`JSONResponse` with the appropriate HTTP status.

        Uses the registered status code for *error_type* unless
        *status_code* is explicitly provided.
        """
        code = status_code or self._registry.get(error_type, 500)
        error_obj = self.build(
            error_type,
            message=message,
            type=type,
            request_id=request_id,
            param=param,
        )
        return JSONResponse(
            status_code=code,
            content=error_obj.model_dump(),
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
