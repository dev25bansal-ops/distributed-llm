"""OpenTelemetry tracing decorators for DistLLM.

Provides ``@otel_trace`` that instruments any async or sync function
with OpenTelemetry spans, auto-extracting attributes from the module,
class, and arguments.

Usage::

    from distllm.dist.otel import otel_trace

    @otel_trace
    async def prefill(self, input_ids: list[int]) -> None:
        ...

    @otel_trace("custom_span_name")
    def decode(self, ...):
        ...

The decorator is a no-op when OpenTelemetry is not installed.
"""

from __future__ import annotations

import functools
import inspect
import time
from typing import Any, Callable, Optional, TypeVar

from loguru import logger

F = TypeVar("F", bound=Callable[..., Any])

# Lazy import to avoid hard dependency on opentelemetry
_OTEL_AVAILABLE = False
_tracer_provider = None
_tracer = None


def _ensure_tracer() -> Any:
    global _OTEL_AVAILABLE, _tracer, _tracer_provider
    if _tracer is not None:
        return _tracer
    try:
        from opentelemetry import trace
        _tracer = trace.get_tracer("distllm", "0.4.1")
        _OTEL_AVAILABLE = True
    except ImportError:
        _tracer = None
        _OTEL_AVAILABLE = False
    return _tracer


def otel_trace(
    span_name: Optional[str] = None,
    attributes: Optional[dict[str, Any]] = None,
    record_exceptions: bool = True,
):
    """Decorator that wraps a function with an OpenTelemetry span.

    Args:
        span_name: Custom span name.  Defaults to ``{cls}.{method}``.
        attributes: Static attributes to set on every span.
        record_exceptions: If True, exceptions are recorded on the span.

    Usage::

        @otel_trace
        async def my_method(self, input_ids):
            ...

        @otel_trace("custom_span")
        def sync_fn():
            ...
    """
    def decorator(func: F) -> F:
        if not _ensure_tracer():
            # OpenTelemetry not installed — return function as-is
            return func

        qual_name = span_name or _qualname(func)

        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with _tracer.start_as_current_span(qual_name) as span:
                    _set_span_attributes(span, func, args, kwargs, attributes)
                    try:
                        result = await func(*args, **kwargs)
                        return result
                    except Exception as e:
                        if record_exceptions:
                            span.record_exception(e)
                            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))  # type: ignore[possibly-undefined]
                        raise
            return async_wrapper  # type: ignore[return-value]
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                with _tracer.start_as_current_span(qual_name) as span:
                    _set_span_attributes(span, func, args, kwargs, attributes)
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        if record_exceptions:
                            span.record_exception(e)
                            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))  # type: ignore[possibly-undefined]
                        raise
            return sync_wrapper  # type: ignore[return-value]
    return decorator


def _qualname(func: Callable) -> str:
    """Return the qualified name (e.g. ``ClassName.method_name``)."""
    parts = []
    if hasattr(func, "__qualname__"):
        parts.append(func.__qualname__)
    elif hasattr(func, "__name__"):
        parts.append(func.__name__)
    if hasattr(func, "__module__"):
        parts.insert(0, func.__module__)
    return ".".join(parts) if parts else func.__name__


def _set_span_attributes(
    span: Any,
    func: Callable,
    args: tuple,
    kwargs: dict,
    static_attrs: Optional[dict[str, Any]] = None,
) -> None:
    """Set span attributes from function arguments and static attrs."""
    from opentelemetry import trace  # noqa: F811

    if static_attrs:
        for k, v in static_attrs.items():
            span.set_attribute(k, v)

    # Extract common positional args
    sig = inspect.signature(func)
    param_names = list(sig.parameters.keys())
    for i, arg in enumerate(args):
        if i < len(param_names):
            name = param_names[i]
            if name in ("self", "cls"):
                continue
            if isinstance(arg, (str, int, float, bool)):
                span.set_attribute(f"arg.{name}", _serialize(arg))

    # Extract named kwargs
    for k, v in kwargs.items():
        if isinstance(v, (str, int, float, bool)):
            span.set_attribute(f"arg.{k}", _serialize(v))


def _serialize(val: Any) -> str:
    if isinstance(val, (list, tuple)):
        return ",".join(str(x) for x in val[:10])
    return str(val)
