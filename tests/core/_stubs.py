"""Lightweight stub classes to replace MagicMock/AsyncMock/patch in core tests.

Provides:
- ``_Stub`` — attribute-access mock that records calls.
- ``_AsyncStub`` — async variant supporting ``await stub()``.
- ``_Raise`` — marker to raise an exception on call.
- ``_Cycle`` — marker for iterable side effects.
- ``_patch(target, ...)`` — context manager for monkey-patching (supports dotted strings).
"""

from __future__ import annotations

import importlib
from contextlib import contextmanager
from typing import Any


class _Raise:
    """When set as .return_value on a stub, raises the wrapped exception on call."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc


class _Cycle:
    """When set as .side_effect, yields values from the iterable on each call."""

    def __init__(self, items: list[Any]) -> None:
        self._items = list(items)
        self._index = 0

    def next(self) -> Any:
        if self._index >= len(self._items):
            raise StopIteration("_Cycle exhausted")
        val = self._items[self._index]
        self._index += 1
        return val

    def reset(self) -> None:
        self._index = 0


def _call_with_side_effect(side_effect, args, kwargs, stub=None):
    """Invoke a side_effect callable, _Cycle, or raise an Exception.

    A list is auto-converted to a _Cycle (each call returns next element).
    When ``stub`` is provided, the conversion is cached on the stub dict.
    """
    if isinstance(side_effect, list):
        cycle = _Cycle(side_effect)
        if stub is not None:
            stub.__dict__["_se_"] = cycle  # cache so __call__ finds it directly
        return cycle.next()
    if isinstance(side_effect, _Cycle):
        return side_effect.next()
    if isinstance(side_effect, BaseException):
        raise side_effect
    return side_effect(*args, **kwargs)


class _AttrStub:
    """Attribute accessor on a _Stub. Every call is recorded in the parent's _call_log."""

    def __init__(self, stub: _Stub, name: str) -> None:
        object.__setattr__(self, "_stub", stub)
        object.__setattr__(self, "_name", name)

    def _call_entries(self):
        return [(a, kw) for n, a, kw in self._stub._call_log if n == self._name]

    @property
    def called(self) -> bool:
        return any(n == self._name for n, _, _ in self._stub._call_log)

    @property
    def call_count(self) -> int:
        return sum(1 for n, _, _ in self._stub._call_log if n == self._name)

    @property
    def call_args(self):
        entries = self._call_entries()
        if not entries:
            return ((), {})
        args, kwargs = entries[-1]
        return (args, kwargs)

    @property
    def call_args_list(self):
        return [(a, kw) for _, a, kw in self._stub._call_log if _ == self._name]

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

    def __getitem__(self, index: int) -> Any:
        if not self._call_entries():
            raise IndexError(f"'{self._name}' has no calls")
        args, kwargs = self._call_entries()[-1]
        if index == 0:
            return args
        if index == 1:
            return kwargs
        raise IndexError(f"index {index} out of range")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        stub: _Stub = object.__getattribute__(self, "_stub")
        name: str = object.__getattribute__(self, "_name")
        stub._call_log.append((name, args, kwargs))

        # Side-effect takes priority
        se_key = f"_se_{name}"
        se_val = stub.__dict__.get(se_key)
        if se_val is not None:
            # Convert list to _Cycle on first use (mutate the dict entry)
            if isinstance(se_val, list):
                cycle = _Cycle(se_val)
                stub.__dict__[se_key] = cycle
                se_val = cycle
            return _call_with_side_effect(se_val, args, kwargs, stub=stub)

        rv_key = f"_ret_{name}"
        val = stub.__dict__.get(rv_key)
        if isinstance(val, _Raise):
            raise val._exc
        if val is None and rv_key not in stub.__dict__:
            return _Stub()
        return val

    def __getattr__(self, name: str) -> _AttrStub:
        stub: _Stub = object.__getattribute__(self, "_stub")
        base: str = object.__getattribute__(self, "_name")
        return _AttrStub(stub, f"{base}.{name}")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("_stub", "_name"):
            object.__setattr__(self, name, value)
            return
        stub: _Stub = object.__getattribute__(self, "_stub")
        attr_name: str = object.__getattribute__(self, "_name")
        # "obj.child.return_value = x" → store as _ret_child = x (set the return value of child())
        # "obj.async_client.health_check = x" → store as _ret_async_client.health_check = x (set health_check on async_client)
        if name == "return_value":
            stub.__dict__[f"_ret_{attr_name}"] = value
        elif name == "side_effect":
            stub.__dict__[f"_se_{attr_name}"] = value
        else:
            dotted = f"{attr_name}.{name}" if attr_name else name
            stub.__dict__[f"_ret_{dotted}"] = value


class _AsyncAttrStub(_AttrStub):
    """Async variant: __call__ returns an awaitable coroutine."""

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        stub: _Stub = object.__getattribute__(self, "_stub")
        name: str = object.__getattribute__(self, "_name")
        stub._call_log.append((name, args, kwargs))

        se_key = f"_se_{name}"
        se_val = stub.__dict__.get(se_key)
        if se_val is not None:
            return _call_with_side_effect(se_val, args, kwargs)

        rv_key = f"_ret_{name}"
        val = stub.__dict__.get(rv_key)
        if isinstance(val, _Raise):
            raise val._exc
        return val


class _Stub:
    """Lightweight test stub. Records all calls.

    Constructor kwargs become data attributes::

        obj = _Stub(called=True, name="foo")
        obj.called   # True
        obj.name     # "foo"

    Attribute access returns auto-vivifying _AttrStub objects that record
    every call.  Set ``.return_value`` to control what the attribute returns::

        stub.method.return_value = 42
        stub.method("arg")  # returns 42
        stub.method.assert_called_once_with("arg")

    Set ``.side_effect`` to a callable or ``_Cycle`` to override the return::

        stub.fn.side_effect = lambda x: x.upper()
        stub.fn("hello")  # returns "HELLO"

    Direct calls on the stub itself::

        fn = _Stub()
        fn.return_value = "ok"
        fn("world")  # returns "ok"
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

    @property
    def call_args(self):
        if not self._call_log:
            return ((), {})
        _, args, kwargs = self._call_log[-1]
        return (args, kwargs)

    @property
    def call_args_list(self):
        return [(a, kw) for _, a, kw in self._call_log]

    def assert_called(self) -> None:
        assert self.called, "Expected stub to be called"

    def assert_not_called(self) -> None:
        assert not self.called, "Expected stub not to be called"

    def assert_called_once(self) -> None:
        assert len(self._call_log) == 1, f"Expected 1 call, got {len(self._call_log)}"

    def assert_called_once_with(self, *args: Any, **kwargs: Any) -> None:
        assert len(self._call_log) == 1, f"Expected 1 call, got {len(self._call_log)}"
        assert self._call_log[0] == ("", args, kwargs), f"Expected ({args}, {kwargs}), got {self._call_log[0]}"

    def reset_mock(self) -> None:
        """Clear the call log (compatible with MagicMock.reset_mock)."""
        self._call_log.clear()

    def __getattr__(self, name: str) -> _AttrStub:
        return _AttrStub(self, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._call_log.append(("", args, kwargs))
        se = self.__dict__.get("side_effect") or self.__dict__.get("_se_")
        if se is not None:
            # Convert list side_effect to _Cycle on first use
            if isinstance(se, list):
                cycle = _Cycle(se)
                self.__dict__["side_effect"] = cycle
                se = cycle
            return _call_with_side_effect(se, args, kwargs, stub=self)
        val = self.__dict__.get("return_value")
        if isinstance(val, _Raise):
            raise val._exc
        return val

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("_call_log",) or not name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            object.__setattr__(self, name, value)

    # ── Magic methods that MagicMock supports ──

    def __iter__(self):
        """Return an empty iterator (like MagicMock)."""
        return iter([])

    def __len__(self) -> int:
        return 0

    def __bool__(self) -> bool:
        return True

    def __await__(self):
        """Make _Stub awaitable (returns return_value configured on the stub)."""
        return self._await_impl().__await__()

    async def _await_impl(self):
        return getattr(self, 'return_value', None)

    def __enter__(self):
        """Context manager entry without recording a call."""
        return self

    def __exit__(self, *args):
        pass

    def __repr__(self) -> str:
        return f"_Stub(call_count={self.call_count})"


class _AsyncStub(_Stub):
    """Async variant: direct calls and attribute-access calls are awaitable."""

    def __getattr__(self, name: str) -> _AsyncAttrStub:
        return _AsyncAttrStub(self, name)

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._call_log.append(("", args, kwargs))
        se = self.__dict__.get("_se_")
        if se is not None:
            return _call_with_side_effect(se, args, kwargs)
        val = self.__dict__.get("return_value")
        if isinstance(val, _Raise):
            raise val._exc
        return val


# ---------------------------------------------------------------------------
# Monkey-patching context managers
# ---------------------------------------------------------------------------


class _patch:
    """Monkey-patch a dotted-name target like ``"httpx.post"``.

    Works as both a context manager and a decorator (injects the replacement
    as the first positional argument)::

        # Context manager
        with _patch("time.time") as mock_time:
            mock_time.return_value = 100.0

        # Decorator (injects mock as first param after self)
        @_patch("time.time")
        def test_something(self, mock_time):
            mock_time.return_value = 100.0
    """

    def __init__(self, target: str, *, return_value: Any = None,
                 side_effect: Any = None):
        self._target = target
        self._return_value = return_value
        self._side_effect = side_effect
        self._replacement = None

    def _setup(self):
        parts = self._target.split(".")
        name = parts[-1]
        # Walk from left to right to find the root module, then drill into attrs
        # e.g. "distllm.core.connection_pool.socket.create_connection"
        # → import distllm.core.connection_pool, then get .socket, then .create_connection
        for i in range(len(parts) - 1, 0, -1):
            mod_name = ".".join(parts[:i])
            try:
                mod = importlib.import_module(mod_name)
                break
            except ImportError:
                continue
        else:
            raise ImportError(f"Cannot resolve module from target '{self._target}'")

        # Walk remaining parts as attributes
        remaining = parts[i:-1]  # parts after the module, before the final name
        for attr in remaining:
            mod = getattr(mod, attr)
        old = getattr(mod, name)
        if self._return_value is not None:
            replacement = lambda *a, **kw: self._return_value  # noqa: E731
        elif self._side_effect is not None:
            replacement = self._side_effect
        else:
            replacement = _Stub()
        self._old = old
        self._name = name
        self._mod = mod
        setattr(mod, name, replacement)
        self._replacement = replacement
        return replacement

    def _teardown(self):
        if hasattr(self, '_mod') and hasattr(self, '_name'):
            setattr(self._mod, self._name, self._old)

    def __enter__(self):
        return self._setup()

    def __exit__(self, *args):
        self._teardown()

    def __call__(self, func):
        """Use as a decorator — wraps func with the patch and injects the replacement."""
        import functools
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            stub = self._setup()
            try:
                return func(*args, stub, **kwargs)
            finally:
                self._teardown()
        return wrapper


class _patch_obj:
    """Monkey-patch ``obj.attr_name`` with a ``_Stub`` for the duration.

    Works as both context manager and decorator (injects stub as first param).
    """

    def __init__(self, obj: Any, attr_name: str):
        self._obj = obj
        self._attr_name = attr_name

    def _setup(self):
        old = getattr(self._obj, self._attr_name)
        stub = _Stub()
        setattr(self._obj, self._attr_name, stub)
        self._old = old
        self._stub = stub
        return stub

    def _teardown(self):
        if hasattr(self, '_obj') and hasattr(self, '_attr_name'):
            setattr(self._obj, self._attr_name, self._old)

    def __enter__(self):
        return self._setup()

    def __exit__(self, *args):
        self._teardown()

    def __call__(self, func):
        """Use as a decorator — wraps func with the patch and injects the stub."""
        import functools
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            stub = self._setup()
            try:
                return func(*args, stub, **kwargs)
            finally:
                self._teardown()
        return wrapper


@contextmanager
def _patch_multiple(module: Any, **kwargs: Any):
    """Monkey-patch multiple attributes on a module for the duration."""
    saved: dict[str, Any] = {}
    for name, value in kwargs.items():
        saved[name] = getattr(module, name)
        setattr(module, name, value)
    try:
        yield
    finally:
        for name, old_value in saved.items():
            setattr(module, name, old_value)
