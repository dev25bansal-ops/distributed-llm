"""Factory for instantiating vector-database providers."""

from __future__ import annotations

from typing import Any

from distllm.core.vectorstore.base import VectorDBInterface


# The registry maps provider names (case-insensitive) to implementation classes.
# The provider classes themselves do not import SDKs at module level — SDKs are
# loaded lazily inside the provider methods, so having a subset of SDKs
# installed is safe.
_PROVIDER_REGISTRY: dict[str, type[VectorDBInterface]] = {}

# Lazy-import the provider classes so that importing the factory does not
# trigger any SDK-level imports (the provider modules themselves only import
# SDKs inside their ``_ensure_*`` methods).


def _populate_registry() -> None:
    """Fill the provider registry (called once on first use)."""
    if _PROVIDER_REGISTRY:
        return

    from distllm.core.vectorstore.providers.pinecone import (
        _PineconeStore,  # noqa: PLC0415
    )
    from distllm.core.vectorstore.providers.qdrant import (
        _QdrantStore,  # noqa: PLC0415
    )
    from distllm.core.vectorstore.providers.weaviate import (
        _WeaviateStore,  # noqa: PLC0415
    )
    from distllm.core.vectorstore.providers.milvus import (
        _MilvusStore,  # noqa: PLC0415
    )

    _PROVIDER_REGISTRY.update(
        {
            "pinecone": _PineconeStore,
            "qdrant": _QdrantStore,
            "weaviate": _WeaviateStore,
            "milvus": _MilvusStore,
        }
    )


class VectorDBFactory:
    """Factory to instantiate a :class:`VectorDBInterface` by provider name.

    Configuration is resolved from ``config`` dict with environment-variable
    fallbacks documented per provider.

    Usage::

        store = VectorDBFactory.create("qdrant", {"url": "http://localhost:6333"})
        store.upsert(vectors=..., metadata=[{"id": "doc1"}])
        store.close()
    """

    @staticmethod
    def create(
        provider: str,
        config: dict[str, Any] | None = None,
    ) -> VectorDBInterface:
        """Create a vector-store client.

        Args:
            provider: One of ``"pinecone"``, ``"qdrant"``, ``"weaviate"``,
                ``"milvus"`` (case-insensitive).
            config: Provider-specific overrides.  Falls back to the
                environment variables documented per-provider.

        Returns:
            A ready-to-use vector-store instance.

        Raises:
            ValueError: Unknown provider name.
        """
        _populate_registry()

        key = provider.lower().strip()
        cls = _PROVIDER_REGISTRY.get(key)
        if cls is None:
            known = ", ".join(sorted(_PROVIDER_REGISTRY))
            raise ValueError(
                f"Unknown vector-db provider {provider!r}. "
                f"Known: {known}"
            )
        return cls(config or {})

    @staticmethod
    def list_providers() -> list[str]:
        """Return sorted list of registered provider names."""
        _populate_registry()
        return sorted(_PROVIDER_REGISTRY)
