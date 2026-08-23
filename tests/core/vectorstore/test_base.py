"""Tests for VectorDBInterface abstract base class.

Covers:
- Cannot instantiate the ABC directly
- Abstract method signatures exist on the class
- A concrete subclass with all methods compiles and runs
- Edge cases (missing methods, empty body) raise TypeError
"""

from __future__ import annotations

import pytest

from distllm.core.vectorstore.base import VectorDBInterface

VectorStore = VectorDBInterface


class TestVectorStoreABC:
    """VectorDBInterface is an abstract base class and cannot be instantiated directly."""

    def test_cannot_instantiate(self):
        """Direct instantiation must raise TypeError (abstract methods)."""
        with pytest.raises(TypeError, match="abstract"):
            VectorStore()  # type: ignore[abstract]

    def test_all_abstract_methods_exist(self):
        """All expected abstract method names must be present on the class."""
        expected = {"upsert", "query", "delete", "close"}
        for method in expected:
            assert hasattr(VectorStore, method), f"Missing abstract method: {method}"

    def test_concrete_subclass_is_instantiable(self):
        """A minimal concrete subclass implementing all abstract methods works."""
        class ConcreteStore(VectorStore):
            def upsert(self, vectors, metadata, *, namespace="", batch_size=None):
                return len(vectors)
            def query(self, vector, top_k=10, *, metadata_filter=None, namespace="", include_metadata=True):
                return []
            def delete(self, *, ids=None, metadata_filter=None, namespace=""):
                return len(ids or [])
            def close(self):
                pass

        store = ConcreteStore()
        assert isinstance(store, VectorStore)

    def test_concrete_subclass_methods_return_correct_types(self):
        """Verify return types from concrete subclass match the abstract signatures."""
        class ConcreteStore(VectorStore):
            def upsert(self, vectors, metadata, *, namespace="", batch_size=None):
                return len(vectors)
            def query(self, vector, top_k=10, *, metadata_filter=None, namespace="", include_metadata=True):
                return []
            def delete(self, *, ids=None, metadata_filter=None, namespace=""):
                return len(ids or [])
            def close(self):
                pass

        store = ConcreteStore()
        assert store.upsert([[0.1], [0.2]], [{"id": "id1"}, {"id": "id2"}]) == 2
        assert store.upsert([], []) == 0
        assert store.query([0.1, 0.2, 0.3]) == []
        assert store.query([0.1], top_k=5) == []
        assert store.delete(ids=["x"]) == 1
        assert store.delete(ids=[]) == 0


class TestVectorStoreEdgeCases:
    """Edge cases for the abstract interface contract."""

    def test_subclass_missing_one_method(self):
        """A subclass missing at least one abstract method must not be instantiable."""
        with pytest.raises(TypeError, match="abstract"):
            class BadStore(VectorStore):  # type: ignore[abstract]
                def upsert(self, vectors, metadata, *, namespace="", batch_size=None):
                    return len(vectors)
            BadStore()

    def test_subclass_empty_body_raises(self):
        """A subclass with no concrete methods must raise TypeError."""
        with pytest.raises(TypeError, match="abstract"):
            class EmptyStore(VectorStore):  # type: ignore[abstract]
                pass
            EmptyStore()