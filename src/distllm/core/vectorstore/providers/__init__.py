"""Provider implementations for supported vector databases.

Each module wraps a single vector-database SDK behind the
:class:`~distllm.core.vectorstore.base.VectorDBInterface` ABC.  SDKs are
imported lazily inside the provider methods so that having only a subset
of SDKs installed does not break imports.
"""
