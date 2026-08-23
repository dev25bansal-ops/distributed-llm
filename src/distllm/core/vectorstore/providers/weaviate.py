"""Weaviate vector store provider.

Requires the ``weaviate-client`` SDK (``pip install weaviate-client``).
"""

from __future__ import annotations

import os
from typing import Any

from loguru import logger

from distllm.core.vectorstore.base import VectorDBInterface


class _WeaviateStore(VectorDBInterface):
    """Weaviate_ vector store wrapper.

    .. _Weaviate: https://weaviate.io
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._url: str = (
            config.get("url")
            or os.environ.get("WEAVIATE_URL", "http://localhost:8080")
        )
        self._api_key: str | None = config.get("api_key") or os.environ.get(
            "WEAVIATE_API_KEY"
        )
        self._class_name: str = config.get(
            "class", config.get("collection", "Document")
        )
        self._client: Any = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import weaviate
            from weaviate.classes.config import Configure, DataType, Property
            from weaviate.classes.data import DataObject
        except ImportError as exc:
            raise ImportError(
                "Weaviate SDK not installed — run `pip install weaviate-client`"
            ) from exc

        auth = (
            weaviate.auth.AuthApiKey(api_key=self._api_key)
            if self._api_key
            else None
        )
        self._client = weaviate.connect_to_local(
            host=self._url, auth_credentials=auth
        )
        self._Property = Property
        self._DataType = DataType
        self._Configure = Configure
        self._DataObject = DataObject

        if not self._client.collections.exists(self._class_name):
            self._client.collections.create(
                name=self._class_name,
                vectorizer_config=Configure.Vectorizer.none(),
            )
        return self._client

    # ------------------------------------------------------------------
    # VectorDBInterface
    # ------------------------------------------------------------------

    def upsert(
        self,
        vectors: list[list[float]],
        metadata: list[dict[str, Any]],
        *,
        namespace: str = "",
        batch_size: int | None = None,
    ) -> int:
        client = self._ensure_client()
        coll = client.collections.get(self._class_name)
        if batch_size is not None:
            with coll.batch.fixed_size(batch_size=batch_size) as batch:
                for vec, meta in zip(vectors, metadata):
                    props = {k: v for k, v in meta.items() if k != "id"}
                    batch.add_object(
                        properties=props,
                        vector=vec,
                        uuid=meta.get("id"),
                    )
        else:
            with coll.batch.fixed_size() as batch:
                for vec, meta in zip(vectors, metadata):
                    props = {k: v for k, v in meta.items() if k != "id"}
                    batch.add_object(
                        properties=props,
                        vector=vec,
                        uuid=meta.get("id"),
                    )
        return len(vectors)

    def query(
        self,
        vector: list[float],
        top_k: int = 10,
        *,
        metadata_filter: dict[str, Any] | None = None,
        namespace: str = "",
        include_metadata: bool = True,
    ) -> list[dict[str, Any]]:
        if namespace:
            # Weaviate has no per-tenant namespace concept — silently ignoring
            # it would return cross-tenant records.
            raise ValueError("Weaviate provider does not support namespaces")
        client = self._ensure_client()
        coll = client.collections.get(self._class_name)

        kwargs: dict[str, Any] = {
            "query_vector": vector,
            "limit": top_k,
            "return_metadata": ["score"],
        }
        if include_metadata:
            kwargs["return_properties"] = True

        # Forward the metadata filter as a real Weaviate Filter instead of
        # silently dropping it (which returned unscoped results).
        if metadata_filter:
            from weaviate.classes.query import Filter as WFilter

            conds = [
                WFilter.by_property(str(k)).equal(v)
                for k, v in metadata_filter.items()
            ]
            if conds:
                filt = conds[0]
                for c in conds[1:]:
                    filt = filt & c
                kwargs["filters"] = filt

        resp = coll.query.near_vector(**kwargs)
        return [
            {
                "id": str(o.uuid),
                "score": o.metadata.score if o.metadata else 0.0,
                "metadata": o.properties if include_metadata else {},
            }
            for o in resp.objects
        ]

    def delete(
        self,
        *,
        ids: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        namespace: str = "",
    ) -> int:
        client = self._ensure_client()
        coll = client.collections.get(self._class_name)
        count = 0
        if ids:
            for uuid_str in ids:
                try:
                    coll.data.delete_by_id(uuid_str)
                    count += 1
                except Exception:
                    logger.warning("Weaviate delete miss for id %s", uuid_str)
        if metadata_filter:
            logger.warning(
                "Weaviate filter-based delete not yet implemented via "
                "generic interface — use ids instead"
            )
        return count

    def close(self) -> None:
        """Close the Weaviate client connection."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                logger.trace("Weaviate client close ignored")
            self._client = None
