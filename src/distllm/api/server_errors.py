"""Error response models and exception handlers for the API server."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

from distllm.api.errors import (
    FEATURE_DISABLED_HEADER,
    FeatureDisabledError,
    error_response_from_request,
)
from distllm.errors import DistLLMError


class ErrorResponse(BaseModel):
    """Standardized error response format."""

    error: str
    message: str
    type: str = "api_error"
    code: str | None = None
    request_id: str | None = None


def _error_response(
    status_code: int,
    error: str,
    message: str,
    type: str = "api_error",
    request: Request | None = None,
    exc: Exception | None = None,
) -> JSONResponse:
    """Build a standardized error response."""
    return error_response_from_request(status_code, error, message, type, request, exc=exc)


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Convert Pydantic validation errors to OpenAI-compatible 422."""
    messages = []
    for err in exc.errors():
        loc = " -> ".join(str(p) for p in err.get("loc", []))
        messages.append(f"{loc}: {err.get('msg', '')}" if loc else err.get("msg", ""))
    return _error_response(
        status_code=422,
        error="Invalid Request",
        message="; ".join(messages),
        type="invalid_request_error",
        request=request,
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Convert HTTPException to structured error response."""
    return _error_response(
        status_code=exc.status_code,
        error=f"HTTP {exc.status_code}",
        message=exc.detail,
        type="http_error",
        request=request,
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for unhandled exceptions with structured response.

    A ``DistLLMError`` carries a deliberate, user-facing message (and a
    remediation hint). Replacing it with the generic "unexpected error ...
    try again later" text told callers to retry configuration problems that
    will never succeed until an operator acts — e.g.
    ``ConfigError("No nodes registered in the pipeline")`` became a
    transient-looking 500 whose cause existed only in the server log. Typed
    errors keep their message and remediation hint; everything else stays
    masked.
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    if isinstance(exc, DistLLMError):
        message = exc.user_message
        if exc.remediation_hint:
            message = f"{message} {exc.remediation_hint}"
        return _error_response(
            status_code=500,
            error="Internal Server Error",
            message=message,
            type="internal_error",
            request=request,
            exc=exc,
        )
    return _error_response(
        status_code=500,
        error="Internal Server Error",
        message="An unexpected error occurred. Please try again later.",
        type="internal_error",
        request=request,
        exc=exc,
    )


async def feature_disabled_exception_handler(
    request: Request, exc: FeatureDisabledError
) -> JSONResponse:
    """Render a feature-off 503 in the app's standard HTTPException shape.

    The body matches whatever the registered HTTPException handler produces
    (the structured envelope on the live app, ``{"detail": ...}`` under
    FastAPI's default), with the marker header the circuit breaker keys on.
    """
    response = await http_exception_handler(request, exc)
    response.headers[FEATURE_DISABLED_HEADER] = "1"
    return response


def register_exception_handlers(app: FastAPI) -> None:
    """Register all exception handlers on a FastAPI application instance."""
    app.exception_handler(RequestValidationError)(validation_exception_handler)
    # Specific subclass first: FeatureDisabledError is an HTTPException, so
    # without this entry Starlette's MRO lookup would hand it to the generic
    # handler and the marker header would never be attached.
    app.exception_handler(FeatureDisabledError)(feature_disabled_exception_handler)
    app.exception_handler(HTTPException)(http_exception_handler)
    app.exception_handler(Exception)(general_exception_handler)
