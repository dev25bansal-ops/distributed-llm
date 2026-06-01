"""Simple dependency injection container.

Provides a lightweight DI container that supports:
- Direct instance registration
- Factory-based registration (lazy instantiation)
- Resolution by protocol/interface type
- Optional resolution (returns None if not registered)
- Thread-safe operations

No third-party DI library required.
"""

from __future__ import annotations

import threading
from typing import Any, Callable


class Container:
    """Thread-safe dependency injection container.

    Usage::

        container = Container()
        container.register(ITokenizer, mock_tokenizer)
        container.register_factory(INodeFactory, lambda: MyNodeFactory(...))

        tokenizer = container.resolve(ITokenizer)
        factory = container.resolve(INodeFactory)
        optional = container.resolve_optional(IMetricsExporter)
    """

    def __init__(self) -> None:
        self._instances: dict[type, Any] = {}
        self._factories: dict[type, Callable[[], Any]] = {}
        self._lock = threading.Lock()

    def register(self, interface: type, instance: Any) -> Container:
        """Register a concrete instance for an interface.

        Args:
            interface: The protocol or abstract class.
            instance: The concrete implementation.

        Returns:
            Self for chaining.
        """
        with self._lock:
            self._instances[interface] = instance
            self._factories.pop(interface, None)
        return self

    def register_factory(self, interface: type, factory: Callable[[], Any]) -> Container:
        """Register a factory function for lazy instantiation.

        The factory is called once on first ``resolve()`` and the result
        is cached as a singleton.

        Args:
            interface: The protocol or abstract class.
            factory: A callable that returns the implementation.

        Returns:
            Self for chaining.
        """
        with self._lock:
            self._factories[interface] = factory
            self._instances.pop(interface, None)
        return self

    def resolve(self, interface: type) -> Any:
        """Resolve an interface to its implementation.

        Args:
            interface: The protocol or abstract class to resolve.

        Returns:
            The registered implementation.

        Raises:
            KeyError: If no implementation is registered.
        """
        with self._lock:
            if interface in self._instances:
                return self._instances[interface]

            if interface in self._factories:
                factory = self._factories.pop(interface)
            else:
                factory = None

        # Call factory outside the lock to avoid deadlocks
        if factory is not None:
            instance = factory()
            with self._lock:
                self._instances[interface] = instance
            return instance

        raise KeyError(f"No implementation registered for {interface.__name__}")

    def resolve_optional(self, interface: type) -> Any | None:
        """Resolve an interface, returning None if not registered.

        Args:
            interface: The protocol or abstract class to resolve.

        Returns:
            The registered implementation, or None.
        """
        try:
            return self.resolve(interface)
        except KeyError:
            return None

    def has(self, interface: type) -> bool:
        """Check if an interface is registered."""
        with self._lock:
            return interface in self._instances or interface in self._factories

    def clear(self) -> None:
        """Clear all registrations."""
        with self._lock:
            self._instances.clear()
            self._factories.clear()

    def registered(self) -> list[str]:
        """Return names of all registered interfaces."""
        with self._lock:
            names = [t.__name__ for t in self._instances]
            names.extend(t.__name__ for t in self._factories)
            return sorted(names)


# Global singleton container
_container: Container | None = None
_container_lock = threading.Lock()


def get_container() -> Container:
    """Get or create the global DI container."""
    global _container
    if _container is None:
        with _container_lock:
            if _container is None:
                _container = Container()
    return _container


def reset_container() -> None:
    """Reset the global DI container (for testing)."""
    global _container
    with _container_lock:
        _container = None
