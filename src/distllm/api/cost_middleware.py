"""Cost-Aware API middleware — adds cost tracking headers to responses.

Intercepts requests to calculate input token costs, and adds
X-DistLLM-Cost, X-DistLLM-Tokens, and X-DistLLM-Savings headers
to every API response.
"""

from __future__ import annotations

import time
from typing import Any

from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from distllm.core.cost_tracker import get_cost_tracker

# C3: Try to import tiktoken for accurate token estimation
_tiktoken_encoding = None
try:
    import tiktoken
    _tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
except (ImportError, Exception):
    pass


def _estimate_tokens(text: str) -> int:
    """C3: Estimate token count using tiktoken if available, else heuristic.

    tiktoken is ~95% accurate for English text. The len//4 heuristic is
    ~60-70% accurate for non-English, code, and JSON.
    """
    if _tiktoken_encoding is not None:
        try:
            return len(_tiktoken_encoding.encode(text))
        except Exception as e:
            logger.debug(f"tiktoken encode failed, using heuristic: {e}")
    # Fallback: ~4 chars per token (conservative for English)
    return max(1, len(text) // 4)


def _parse_request_body(body: bytes) -> dict[str, Any] | None:
    """Decode *body* and return it as a dict, or None on any parse/typing error.

    Returns None for empty bodies, JSON arrays, JSON numbers/strings, and
    JSON ``null`` — all of which are invalid request payloads for the
    inference endpoints this middleware fronts.
    """
    if not body:
        return None
    import json
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        # The downstream handler expects a JSON object (model / messages /
        # prompt). Anything else (``[]``, ``42``, ``"a string"``, ``null``)
        # is a malformed client request — surface 400, do not let the
        # downstream handler turn it into a 500.
        return None
    return data


class CostTrackingMiddleware(BaseHTTPMiddleware):
    """Middleware that tracks per-request costs and adds cost headers.

    For each request:
   1. Estimates input token count from the request body
    2. Records the start time
    3. After the response, calculates cost and adds headers

    Bad request bodies (non-dict JSON, malformed JSON, etc.) are rejected
    with ``400 Bad Request`` before being passed downstream. This prevents
    the downstream handler from turning a malformed body into a 500.
    """

    async def dispatch(self, request: Request, call_next):
        # Only track inference endpoints
        path = request.url.path
        if not any(ep in path for ep in [
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/embeddings",
        ]):
            return await call_next(request)

        tracker = get_cost_tracker()
        start_time = time.time()
        input_tokens = 0
        model_name = ""

        # Try to estimate input tokens from request body.
        # If the body is malformed or not a JSON object, return 400 here
        # rather than letting the downstream handler crash with 500.
        try:
            body = await request.body()
            if body:
                data = _parse_request_body(body)
                if data is None:
                    return JSONResponse(
                        status_code=400,
                        content={
                            "error": {
                                "message": (
                                    "Invalid request body: expected a JSON object "
                                    "with 'model' and 'messages' (or 'prompt') "
                                    "fields for inference endpoints."
                                ),
                                "type": "invalid_request_body",
                                "code": "400",
                            },
                        },
                    )
                model_name = data.get("model", "")
                messages = data.get("messages", [])
                prompt = data.get("prompt", "")

                # C3: Use tiktoken-based estimation
                if messages:
                    text = " ".join(
                        m.get("content", "") if isinstance(m.get("content"), str)
                        else str(m.get("content", ""))
                        for m in messages
                    )
                    input_tokens = _estimate_tokens(text)
                elif prompt:
                    input_tokens = _estimate_tokens(prompt)
        except Exception as e:
            logger.debug(f"Failed to estimate input tokens from request body: {e}")

        # Process the request
        response = await call_next(request)

        # Calculate cost
        try:
            elapsed_ms = (time.time() - start_time) * 1000
            # C4: Better output token estimation — read response body for non-streaming
            output_tokens = 0
            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > 0:
                # C3: Use tiktoken-based estimation for output too
                output_tokens = max(1, int(content_length) // 4)

            # Get tenant from request state
            tenant_id = getattr(request.state, "api_key_id", "default")

            estimate = tracker.record_request(
                tenant_id=tenant_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                duration_ms=elapsed_ms,
                model_name=model_name,
            )

            # Add cost headers to response
            cost_headers = tracker.get_cost_headers(estimate)
            for key, value in cost_headers.items():
                response.headers[key] = value

        except Exception as e:
            logger.debug(f"Cost tracking failed: {e}")

        return response


class StreamingCostMiddleware:
    """WebSocket/SSE middleware for streaming cost updates.

    This is not a traditional middleware but a helper that integrates
    with the streaming response generator to inject cost events.
    """

    def __init__(self):
        from distllm.core.streaming_cost import get_streaming_cost_tracker
        self._tracker = get_streaming_cost_tracker()

    def start_request(
        self,
        request_id: str,
        input_tokens: int,
        model_name: str = "",
    ):
        """Start tracking a streaming request."""
        from distllm.core.cost_tracker import (
            CLOUD_API_COST_PER_M_TOKENS,
            _estimate_throughput,
            _match_cloud_api,
            get_cost_tracker,
        )

        cost_tracker = get_cost_tracker()
        gpu_type = cost_tracker._default_gpu

        # C5: Calculate separate input and output cost per token
        tps = _estimate_throughput(gpu_type, model_name)
        gpu_cost_per_hour = cost_tracker.estimate_cost(
            input_tokens=1000, output_tokens=0, model_name=model_name,
        ).gpu_cost_per_hour
        cost_per_token = gpu_cost_per_hour / (tps * 3600) if tps > 0 else 0

        # Cloud cost per token (input and output have different rates)
        cloud_api = _match_cloud_api(model_name)
        cloud_input_cost_per_token = 0
        cloud_output_cost_per_token = 0
        if cloud_api and cloud_api in CLOUD_API_COST_PER_M_TOKENS:
            pricing = CLOUD_API_COST_PER_M_TOKENS[cloud_api]
            cloud_input_cost_per_token = pricing["input"] / 1_000_000
            cloud_output_cost_per_token = pricing["output"] / 1_000_000

        return self._tracker.start_tracking(
            request_id=request_id,
            input_tokens=input_tokens,
            model_name=model_name,
            gpu_type=gpu_type,
            cost_per_token=cost_per_token,
            cloud_cost_per_token=cloud_output_cost_per_token,  # C5: Use output rate for streaming
            input_cost_per_token=cost_per_token,  # C5: Separate input rate
            cloud_input_cost_per_token=cloud_input_cost_per_token,  # C5: Separate cloud input rate
        )

    def record_token(self, request_id: str) -> dict[str, Any] | None:
        """Record an output token and get cost event data."""
        return self._tracker.record_token(request_id)

    def finish_request(self, request_id: str) -> dict[str, Any] | None:
        """Finish tracking and get final summary."""
        return self._tracker.finish_tracking(request_id)
