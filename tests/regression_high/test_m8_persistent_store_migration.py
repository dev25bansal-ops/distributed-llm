"""Regression test for M8 (F7-core): empty PersistentStore migration registry.

BUG (pre-fix): ``distllm.core.persistent_store`` declared
``_SCHEMA_MIGRATIONS = {}`` (empty) with ``_SCHEMA_VERSION`` stuck at 1 and no
registered v1->v2 migration. That meant a store holding old (v1) data could
never be migrated to v2, so it would silently start at the wrong schema version
or fail to load old records with the new shape.

FIX: a real, *idempotent* migration ``_migrate_to_v2`` is registered for
version 2, bumping ``_SCHEMA_VERSION`` to 2. ``get_current_version()`` and
``apply_migrations()`` were added so callers can reconcile an on-disk schema
against the code without relying on ``initialize()`` side effects.

These tests FAIL on the pre-fix code (migration registry empty / version never
reaches 2 / no ``apply_migrations`` API) and PASS after the fix. The test uses
an in-memory SQLite backing, so no optional dependencies or disk files are
required.
"""

from __future__ import annotations

import sqlite3

import pytest

from distllm.core import persistent_store as ps


# --------------------------------------------------------------------------- #
# Helpers: build a *v1* database by hand (no v2 columns), exactly as an old
# install would have left it on disk.
# --------------------------------------------------------------------------- #
_V1_SCHEMA = """
CREATE TABLE jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT UNIQUE NOT NULL,
    type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    request_id TEXT,
    model TEXT,
    prompt TEXT,
    result TEXT,
    error TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    completed_at REAL,
    ttl REAL,
    metadata TEXT
);

CREATE TABLE sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT UNIQUE NOT NULL,
    user_id TEXT,
    created_at REAL NOT NULL,
    last_active REAL NOT NULL,
    metadata TEXT,
    expires_at REAL
);

CREATE TABLE schema_version (
    version INTEGER PRIMARY KEY
);
"""


def _make_v1_db(path: str):
    """Create a v1-shaped database at ``path`` and seed one job + one session.

    Returns ``(job_id, session_id)`` of the seeded rows.
    """
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_V1_SCHEMA)
        # Stamp the DB as schema v1 (the buggy code never advances past this).
        conn.execute("INSERT INTO schema_version (version) VALUES (1)")
        conn.execute(
            """
            INSERT INTO jobs
                (job_id, type, status, model, prompt, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "job-old-1",
                "completion",
                "completed",
                "gpt-x",
                "hello world",
                1_700_000_000.0,
                1_700_000_001.0,
                '{"src": "legacy"}',
            ),
        )
        conn.execute(
            """
            INSERT INTO sessions
                (session_id, user_id, created_at, last_active, metadata, expires_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "sess-old-1",
                "user-7",
                1_700_000_000.0,
                1_700_000_002.0,
                "{}",
                None,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return "job-old-1", "sess-old-1"


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


# --------------------------------------------------------------------------- #
# 1. The migration registry must be NON-EMPTY (the core F7-core bug).
# --------------------------------------------------------------------------- #
def test_migration_registry_is_not_empty():
    assert ps._SCHEMA_MIGRATIONS, "no migrations registered"
    assert 2 in ps._SCHEMA_MIGRATIONS, "v1->v2 migration missing from registry"
    # And the code's target version matches the registered migration.
    assert ps._SCHEMA_VERSION == 2


def test_get_current_version_api_exists():
    store = ps.PersistentStore(":memory:")
    # Fresh in-memory store: initialized to the current version.
    store.initialize()
    assert store.get_current_version() == ps._SCHEMA_VERSION


# --------------------------------------------------------------------------- #
# 2. A v1 store migrates to v2 and old records get the new shape.
# --------------------------------------------------------------------------- #
def test_apply_migrations_bumps_v1_to_v2_and_transforms_records(tmp_path):
    db = str(tmp_path / "legacy.db")
    job_id, session_id = _make_v1_db(db)

    store = ps.PersistentStore(db)
    new_version = store.apply_migrations()

    # Schema version is now 2.
    assert new_version == 2
    assert store.get_current_version() == 2

    # The v2 columns now exist on the on-disk schema...
    conn = sqlite3.connect(db)
    try:
        assert "priority" in _columns(conn, "jobs")
        assert "tenant" in _columns(conn, "sessions")
        # ...and the seeded legacy job is readable with the new column,
        # defaulted sensibly (priority 0, tenant 'default'). The added field
        # is observable through the raw record (the transform the migration
        # performs on old rows).
        legacy_priority = conn.execute(
            "SELECT priority FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()[0]
        legacy_tenant = conn.execute(
            "SELECT tenant FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
        assert legacy_priority == 0
        assert legacy_tenant == "default"
        # And the legacy record is still readable through the public store API.
        job = store.get_job(job_id)
        assert job is not None
        assert job.job_id == job_id
        assert job.metadata == {"src": "legacy"}
        sess = store.get_session(session_id)
        assert sess is not None
        assert sess["session_id"] == session_id
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# 3. Idempotency: applying migrations twice is a safe no-op.
# --------------------------------------------------------------------------- #
def test_apply_migrations_is_idempotent(tmp_path):
    db = str(tmp_path / "legacy2.db")
    job_id, session_id = _make_v1_db(db)

    store = ps.PersistentStore(db)
    v1 = store.apply_migrations()
    assert v1 == 2

    # Second call must NOT raise, must NOT bump the version further, and must
    # leave the single seeded record intact (no duplicate transforms / dup rows).
    v2 = store.apply_migrations()
    assert v2 == 2
    assert store.get_current_version() == 2

    conn = sqlite3.connect(db)
    try:
        job_count = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()[0]
        sess_count = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
        assert job_count == 1, "job row duplicated by re-running migration"
        assert sess_count == 1, "session row duplicated by re-running migration"
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# 4. A brand-new store also lands on v2 via apply_migrations (no legacy data).
# --------------------------------------------------------------------------- #
def test_fresh_store_initializes_at_v2():
    store = ps.PersistentStore(":memory:")
    store.apply_migrations()
    assert store.get_current_version() == 2
    # v2 columns present on a fresh store too, and writable.
    store.create_job(ps.JobRecord(job_id="j-new", type="completion"))
    # Use the store's own connection to verify the v2 column exists and can
    # carry a priority value (exercises the added column end-to-end).
    with store._transaction() as c:
        cols = _columns(c, "jobs")
        assert "priority" in cols
        c.execute("UPDATE jobs SET priority = 5 WHERE job_id = ?", ("j-new",))
        prio = c.execute(
            "SELECT priority FROM jobs WHERE job_id = ?", ("j-new",)
        ).fetchone()[0]
    assert prio == 5


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
