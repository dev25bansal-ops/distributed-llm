"""Shared error response utilities for the API layer.

Provides standardized error responses used by all routers and middleware.
"""

from fastapi import Request
from fastapi.responses import JSONResponse


def error_response(
    status_code: int,
    error: str,
    message: str,
    type: str = "api_error",
    request_id: str | None = None,
    retry_after: float | None = None,
) -> JSONResponse:
    """Build a standardized API error response (OpenAI-compatible format).

    All API errors should use this function or the _error_response helper
    (which injects request_id from request.state) to ensure consistent format.

    OpenAI error format:
    {"error": {"message": "...", "type": "...", "param": null, "code": "..."}}

    OpenAI error codes (mapped from status):
    401 -> "invalid_api_key"
    429 -> "rate_limit_exceeded"
    504 -> "timeout"
    503 -> "service_unavailable"
    400 -> "invalid_request_error"
    500 -> "internal_error"
    """
    code_map = {401: "invalid_api_key", 429: "rate_limit_exceeded", 504: "timeout",
                503: "service_unavailable", 400: "invalid_request_error", 500: "internal_error"}
    error_body: dict = {
        "message": message,
        "type": type,
        "param": None,
        "code": code_map.get(status_code, str(status_code)),
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
) -> JSONResponse:
    """Build a standardized error response, injecting request_id from request.state."""
    request_id = getattr(request.state, "request_id", None) if request is not None else None
    return error_response(status_code, error, message, type, request_id)
