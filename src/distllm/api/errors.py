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
    """Build a standardized API error response.

    All API errors should use this function or the _error_response helper
    (which injects request_id from request.state) to ensure consistent format.
    """
    content: dict = {
        "error": error,
        "message": message,
        "type": type,
        "code": str(status_code),
        "request_id": request_id,
    }
    if retry_after is not None:
        content["retry_after"] = retry_after
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
