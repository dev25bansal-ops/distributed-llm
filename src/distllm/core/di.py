"""Simple dependency injection container.

Provides a lightweight DI container that supports:
- Direct instance registration
- Factory-based registration (lazy instantiation)
- Resolution by protocol/interface type
- Optional resolution (returns None if not registered)

No third-party DI library required.
"""

from typing import Any, Callable


class Container:
    """Simple dependency injection container.

    Usage:
        container = Container()
        container.register(ITokenizer, mock_tokenizer)
        container.register_factory(INodeFactory, lambda: MyNodeFactory(...))

        tokenizer = container.resolve(ITokenizer)
        factory = container.resolve(INodeFactory)
        optional = container.resolve_optional(IMetricsExporter)
    """

    def __init__(self):
        self._instances: dict[type, Any] = {}
        self._factories: dict[type, Callable[[], Any]] = {}

    def register(self, interface: type, instance: Any) -> "Container":
        """Register a concrete instance for an interface.

        Args:
            interface: The protocol or abstract class.
            instance: The concrete implementation.

        Returns:
            Self for chaining.
        """
        self._instances[interface] = instance
        self._factories.pop(interface, None)
        return self

    def register_factory(self, interface: type, factory: Callable[[], Any]) -> "Container":
        """Register a factory function for lazy instantiation.

        Args:
            interface: The protocol or abstract class.
            factory: A callable that returns the implementation.

        Returns:
            Self for chaining.
        """
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
        if interface in self._instances:
            return self._instances[interface]

        if interface in self._factories:
            instance = self._factories[interface]()
            self._instances[interface] = instance
            del self._factories[interface]
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
        return interface in self._instances or interface in self._factories

    def clear(self) -> None:
        """Clear all registrations."""
        self._instances.clear()
        self._factories.clear()
