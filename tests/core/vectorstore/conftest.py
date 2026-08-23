"""pytest configuration for vectorstore tests.

The previous version faked ``distllm.core.vectorstore`` via ``_import_helper``
to avoid a circular import that no longer exists — the real package imports
cleanly and exports ``VectorDBFactory`` / ``VectorDBInterface`` / ``RAGPipeline``.
Using the real package fixes the stale-name collection errors.
"""