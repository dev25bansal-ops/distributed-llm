"""Lightweight stub classes to replace MagicMock/AsyncMock/mock.patch in tests.

Usage rules:
- No MagicMock, AsyncMock, or mock.patch in new test code
- Every call is recorded for assertion via ._call_log / .called / .call_count
- .return_value on any stub attribute chain configures what that path returns
- _Raise wraps an exception to raise on call
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any


class _Raise:
    """Marker: when set as a return value, the stub raises the wrapped exception."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc


class _AttrStub:
    """Tracks attribute access on a parent _Stub.

    Each attribute in the chain produces a unique return-key stored on the
    parent _Stub.__dict__.  Methods like .assert_called(), .called,
    .call_count are real methods, not stubbed attributes.
    """

    def __init__(self, stub: _Stub, name: str) -> None:
        object.__setattr__(self, '_stub', stub)
        object.__setattr__(self, '_name', name)

    def _call_entries(self):
        return [(a, kw) for n, a, kw in self._stub._call_log if n == self._name]

    @property
    def called(self) -> bool:
        return any(n == self._name for n, _, _ in self._stub._call_log)

    @property
    def call_count(self) -> int:
        return sum(1 for n, _, _ in self._stub._call_log if n == self._name)

    def assert_called(self) -> None:
        assert self.called, f"Expected '{self._name}' to be called"

    def assert_not_called(self) -> None:
        assert not self.called, f"Expected '{self._name}' not to be called"

    def assert_called_once(self) -> None:
        count = self.call_count
        assert count == 1, f"Expected 1 call to '{self._name}', got {count}"

    def assert_called_once_with(self, *args: Any, **kwargs: Any) -> None:
        entries = self._call_entries()
        assert len(entries) == 1, f"Expected 1 call to '{self._name}', got {len(entries)}: {entries}"
        assert entries[0] == (args, kwargs), f"Expected ({args}, {kwargs}), got {entries[0]}"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        stub: _Stub = object.__getattribute__(self, '_stub')
        name: str = object.__getattribute__(self, '_name')
        stub._call_log.append((name, args, kwargs))
        rv_key = f'_ret_{name}'
        val = stub.__dict__.get(rv_key)
        if isinstance(val, _Raise):
            raise val._exc
        # Auto-vivification: if no explicit return value set,
        # return a fresh _Stub so downstream attribute access works
        if val is None and rv_key not in stub.__dict__:
            return _Stub()
        return val

    def __getattr__(self, name: str) -> _AttrStub:
        stub: _Stub = object.__getattribute__(self, '_stub')
        base: str = object.__getattribute__(self, '_name')
        return _AttrStub(stub, f'{base}.{name}')

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ('_stub', '_name'):
            object.__setattr__(self, name, value)
            return
        stub: _Stub = object.__getattribute__(self, '_stub')
        attr_name: str = object.__getattribute__(self, '_name')
        stub.__dict__[f'_ret_{attr_name}'] = value


class _AsyncAttrStub(_AttrStub):
    """Async variant: __call__ returns an awaitable coroutine."""

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        stub: _Stub = object.__getattribute__(self, '_stub')
        name: str = object.__getattribute__(self, '_name')
        stub._call_log.append((name, args, kwargs))
        rv_key = f'_ret_{name}'
        val = stub.__dict__.get(rv_key)
        if isinstance(val, _Raise):
            raise val._exc
        return val


class _Stub:
    """Lightweight test stub. Records all calls on attributes.

    Basic usage::

        stub = _Stub()
        stub.method.return_value = 42
        result = stub.method("arg")          # result == 42
        assert stub._call_log == [("method", ("arg",), {})]
        stub.method.assert_called_once_with("arg")

        # Raise on call
        stub.method.return_value = _Raise(ValueError("boom"))
        stub.method()                        # raises ValueError("boom")

        # Direct call (function stub)
        fn = _Stub()
        fn.return_value = "hello"
        fn("world")                          # returns "hello"
        fn.assert_called_once_with("world")

        # Data attributes via constructor kwargs
        obj = _Stub(success=True, name="foo")
        obj.success  # True
        obj.name     # "foo"
    """

    def __init__(self, **kwargs: Any) -> None:
        self._call_log: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        for k, v in kwargs.items():
            object.__setattr__(self, k, v)

    @property
    def called(self) -> bool:
        return bool(self._call_log)

    @property
    def call_count(self) -> int:
        return len(self._call_log)

    def assert_called(self) -> None:
        assert self.called, "Expected stub to be called"

    def assert_not_called(self) -> None:
        assert not self.called, "Expected stub not to be called"

    def assert_called_once(self) -> None:
        assert len(self._call_log) == 1, f"Expected 1 call, got {len(self._call_log)}"

    def assert_called_once_with(self, *args: Any, **kwargs: Any) -> None:
        assert len(self._call_log) == 1, f"Expected 1 call, got {len(self._call_log)}"
        assert self._call_log[0] == ('', args, kwargs), f"Expected ({args}, {kwargs}), got {self._call_log[0]}"

    def __getattr__(self, name: str) -> _AttrStub:
        return _AttrStub(self, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._call_log.append(('', args, kwargs))
        val = self.__dict__.get('return_value')
        if isinstance(val, _Raise):
            raise val._exc
        return val

    def __repr__(self) -> str:
        return f'_Stub(call_count={self.call_count})'


class _AsyncStub(_Stub):
    """Async variant: attribute access returns _AsyncAttrStub for awaitable calls."""

    def __getattr__(self, name: str) -> _AsyncAttrStub:
        return _AsyncAttrStub(self, name)


# ---------------------------------------------------------------------------
# Monkey-patching context managers (replace mock.patch/patch.object/patch.multiple)
# ---------------------------------------------------------------------------


@contextmanager
def _patch(module: Any, name: str, *, return_value: Any = None, side_effect: Any = None):
    """Monkey-patch ``module.name`` for the duration of the context.

    - ``return_value=X``: ``module.name`` becomes a callable that returns X.
    - ``side_effect=fn``: ``module.name`` becomes ``fn`` itself.
    - Neither: ``module.name`` becomes a ``_Stub()`` instance.

    Yields the replacement so you can configure it further.
    """
    old = getattr(module, name)
    if return_value is not None:
        replacement = lambda *a, **kw: return_value
    elif side_effect is not None:
        replacement = side_effect
    else:
        replacement = _Stub()
    setattr(module, name, replacement)
    try:
        yield replacement
    finally:
        setattr(module, name, old)


@contextmanager
def _patch_obj(obj: Any, attr_name: str):
    """Monkey-patch ``obj.attr_name`` with a ``_Stub`` for the duration of the context.

    Yields the stub for configuration.
    """
    old = getattr(obj, attr_name)
    stub = _Stub()
    setattr(obj, attr_name, stub)
    try:
        yield stub
    finally:
        setattr(obj, attr_name, old)


@contextmanager
def _patch_multiple(module: Any, **kwargs: Any):
    """Monkey-patch multiple attributes on a module for the duration of the context."""
    saved: dict[str, Any] = {}
    for name, value in kwargs.items():
        saved[name] = getattr(module, name)
        setattr(module, name, value)
    try:
        yield
    finally:
        for name, old_value in saved.items():
            setattr(module, name, old_value)
