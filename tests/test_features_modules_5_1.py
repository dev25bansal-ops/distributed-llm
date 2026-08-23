"""Regression tests — roadmap §5.1 new modules.

Fast / no external services:
- connection_pool half-open detection (real socket pair)
- resource_manager._tcp_health_check actually probes (and returns False for a closed peer)
- vector-store adapters import + fail closed when the driver is absent
- core.plugins.sandbox path re-exports the sandbox
"""

import socket
import threading

from distllm.core.connection_pool import ConnectionPool


# ── Health probes: real half-open detection ──

def _make_connected_pair():
    """Return (client_sock, server_sock) connected via loopback."""
    ls = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ls.bind(("127.0.0.1", 0))
    ls.listen(1)
    port = ls.getsockname()[1]
    cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    cli.connect(("127.0.0.1", port))
    srv, _ = ls.accept()
    ls.close()
    return cli, srv


def test_pool_half_open_detection_false_for_live_socket():
    cli, srv = _make_connected_pair()
    pool = ConnectionPool(max_size=2, validate_timeout=0.2)
    pool.put("127.0.0.1", 1, cli)  # bogus key; we test _is_half_open on the live sock
    assert pool._is_half_open(cli) is False
    srv.close(); cli.close(); pool.close_all()


def test_pool_half_open_detection_true_when_peer_closes():
    cli, srv = _make_connected_pair()
    # Peer closes -> client socket becomes half-open (readable EOF).
    srv.close()
    pool = ConnectionPool(validate_timeout=0.2)
    assert pool._is_half_open(cli) is True  # recv(1) returns b'' -> half-open
    cli.close(); pool.close_all()


def test_pool_detect_half_open_removes_from_pool():
    cli, srv = _make_connected_pair()
    pool = ConnectionPool(max_size=2, validate_timeout=0.2)
    pool.put("h", 9, cli)
    srv.close()  # make cli half-open
    detected = pool.detect_half_open("h", 9)
    assert detected is True
    # No healthy pooled connection remains.
    assert pool._pool.get(("h", 9), []) == []
    cli.close(); pool.close_all()


# ── resource_manager TCP health check actually probes ──
# The probe logic lives in ConnectionPool.probe_socket_alive (importable
# without the heavy resource_manager stack which needs pydantic/yaml).

def test_tcp_health_check_detects_closed_peer():
    cli, srv = _make_connected_pair()
    pool = ConnectionPool(max_size=2, validate_timeout=0.2)
    srv.close()  # peer gone -> probe must return False (not blindly True)
    ok = pool.probe_socket_alive(cli, timeout=0.5)
    assert ok is False
    cli.close(); pool.close_all()


def test_tcp_health_check_true_for_live_peer():
    cli, srv = _make_connected_pair()
    pool = ConnectionPool(max_size=2, validate_timeout=0.2)
    ok = pool.probe_socket_alive(cli, timeout=0.5)
    assert ok is True
    cli.close(); srv.close(); pool.close_all()


# ── Vector-store adapters: import + fail closed (no driver installed) ──

def test_chroma_store_fails_closed_without_driver():
    from distllm.core.vectorstore.chroma_store import ChromaStore

    store = ChromaStore(host="localhost", port=8000)
    try:
        store.upsert([[0.1, 0.2]], ["x"])
        raise AssertionError("expected RuntimeError (chromadb not installed)")
    except RuntimeError as e:
        assert "chromadb" in str(e)


def test_qdrant_store_fails_closed_without_driver():
    from distllm.core.vectorstore.qdrant_store import QdrantStore

    store = QdrantStore(host="localhost", port=6333)
    try:
        store.query([0.1, 0.2], top_k=3)
        raise AssertionError("expected RuntimeError (qdrant_client not installed)")
    except RuntimeError as e:
        assert "qdrant_client" in str(e)


def test_pgvector_store_fails_closed_without_driver():
    from distllm.core.vectorstore.pgvector_store import PGVectorStore

    store = PGVectorStore(connection_string="postgresql://localhost/test", vector_size=4)
    try:
        store.list_collections()
        raise AssertionError("expected RuntimeError (psycopg2 not installed)")
    except RuntimeError as e:
        assert "psycopg2" in str(e)


def test_vectorstore_adapters_subclass_base():
    from distllm.core.vectorstore.base import VectorDBInterface
    from distllm.core.vectorstore.chroma_store import ChromaStore
    from distllm.core.vectorstore.qdrant_store import QdrantStore
    from distllm.core.vectorstore.pgvector_store import PGVectorStore

    for cls in (ChromaStore, QdrantStore, PGVectorStore):
        assert issubclass(cls, VectorDBInterface)


# ── core.plugins.sandbox path ──

def test_plugins_sandbox_re_exports():
    from distllm.core.plugins.sandbox import (
        PluginCapability,
        PluginManifest,
        verify_manifest,
        run_sandboxed,
    )
    assert PluginCapability is not None
    assert PluginManifest is not None
    assert callable(verify_manifest)
    assert callable(run_sandboxed)
