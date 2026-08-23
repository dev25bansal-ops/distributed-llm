"""Tests for Container -- lightweight dependency injection container.

Covers:
- Construction empty
- register and resolve
- register_factory (lazy singleton)
- resolve raises KeyError for missing
- resolve_optional returns None for missing
- has returns True/False
- clear removes all registrations
- registered returns names
- Chaining API (register returns self)
- Global get_container and reset_container

No MagicMock -- real dicts with protocol classes for interfaces.
"""

from __future__ import annotations

from typing import Any, Protocol

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_mod = load_module("distllm/core/di.py")
Container = _mod.Container
get_container = _mod.get_container
reset_container = _mod.reset_container


class ITokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...


class IModel(Protocol):
    def forward(self, x: Any) -> Any: ...


class _MockTokenizer:
    def encode(self: Any, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self: Any, ids: list[int]) -> str:
        return "".join(chr(i) if 0 <= i < 128 else "?" for i in ids)


class _MockModel:
    def __init__(self: Any, name: str = "default") -> None:
        self.name = name

    def forward(self: Any, x: Any) -> str:
        return f"output_{x}"


class TestContainerConstruction:
    """Construction and initial state."""

    def test_default_construction(self) -> None:
        c = Container()
        assert c._instances == {}
        assert c._factories == {}

    def test_has_returns_false_for_empty(self) -> None:
        c = Container()
        assert c.has(ITokenizer) is False

    def test_registered_empty(self) -> None:
        c = Container()
        assert c.registered() == []


class TestContainerRegister:
    """Register and resolve instances."""

    def test_register_and_resolve(self) -> None:
        c = Container()
        tokenizer = _MockTokenizer()
        c.register(ITokenizer, tokenizer)
        resolved = c.resolve(ITokenizer)
        assert resolved is tokenizer

    def test_register_returns_self(self) -> None:
        c = Container()
        result = c.register(ITokenizer, _MockTokenizer())
        assert result is c

    def test_has_after_register(self) -> None:
        c = Container()
        c.register(ITokenizer, _MockTokenizer())
        assert c.has(ITokenizer) is True

    def test_resolve_optional_returns_instance(self) -> None:
        c = Container()
        model = _MockModel()
        c.register(IModel, model)
        assert c.resolve_optional(IModel) is model

    def test_resolve_optional_returns_none(self) -> None:
        c = Container()
        assert c.resolve_optional(IModel) is None

    def test_resolve_unknown_raises(self) -> None:
        c = Container()
        with pytest.raises(KeyError):
            c.resolve(IModel)

    def test_registered_includes_name(self) -> None:
        c = Container()
        c.register(ITokenizer, _MockTokenizer())
        names = c.registered()
        assert "ITokenizer" in names


class TestContainerFactory:
    """Factory-based registration."""

    def test_register_factory(self) -> None:
        c = Container()
        c.register_factory(IModel, lambda: _MockModel(name="factory-model"))
        resolved = c.resolve(IModel)
        assert isinstance(resolved, _MockModel)
        assert resolved.name == "factory-model"

    def test_factory_is_singleton(self) -> None:
        c = Container()
        call_count: list[int] = []

        def factory() -> _MockModel:
            call_count.append(1)
            return _MockModel(name="singleton")

        c.register_factory(IModel, factory)
        m1 = c.resolve(IModel)
        m2 = c.resolve(IModel)
        assert m1 is m2
        assert len(call_count) == 1

    def test_register_overrides_factory(self) -> None:
        c = Container()
        c.register_factory(IModel, lambda: _MockModel(name="factory"))
        c.register(IModel, _MockModel(name="direct"))
        resolved = c.resolve(IModel)
        assert resolved.name == "direct"

    def test_register_factory_overrides_instance(self) -> None:
        c = Container()
        c.register(IModel, _MockModel(name="direct"))
        c.register_factory(IModel, lambda: _MockModel(name="factory"))
        resolved = c.resolve(IModel)
        assert resolved.name == "factory"


class TestContainerClear:
    """Clear and reset."""

    def test_clear_removes_all(self) -> None:
        c = Container()
        c.register(ITokenizer, _MockTokenizer())
        c.register_factory(IModel, lambda: _MockModel())
        c.clear()
        assert c.has(ITokenizer) is False
        assert c.has(IModel) is False
        assert c.registered() == []


class TestContainerGlobal:
    """Global container functions."""

    def test_get_container_returns_singleton(self) -> None:
        reset_container()
        c1 = get_container()
        c2 = get_container()
        assert c1 is c2

    def test_reset_container_clears_global(self) -> None:
        c1 = get_container()
        c1.register(ITokenizer, _MockTokenizer())
        reset_container()
        c2 = get_container()
        assert c2 is not c1
        assert c2.has(ITokenizer) is False
