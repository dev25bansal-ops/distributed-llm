"""Regression test for M7.

Two bugs, one file:

(a) Qdrant point ids were derived via ``_to_u64`` which truncated a SHA-256
    digest to 8 bytes, collapsing the id space to 2**64 and inviting birthday
    collisions. Fix: native 128-bit UUID string point ids
    (``new_point_id`` -> ``str(uuid.uuid4())``; ``_to_point_id`` -> UUID str).

(b) ``SchemaValidator`` used hand-rolled ``isinstance`` checks. Because
    ``isinstance(True, int)`` is ``True`` in Python, a boolean wrongly passed
    an ``integer`` schema. Fix: jsonschema Draft7Validator backing ``validate``.

These tests FAIL on the pre-fix code and PASS after the fix. They avoid the
optional ``qdrant-client`` dependency: the id helpers are plain functions that
import without a running Qdrant, and the validator only needs ``jsonschema``.
"""

from __future__ import annotations

import uuid

import pytest

from distllm.core.structured_output.validator import SchemaValidator
from distllm.core.vectorstore.qdrant_store import _to_point_id, new_point_id

_UUID_RE = __import__("re").compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


# --------------------------------------------------------------------------- #
# (a) Native UUID Qdrant point ids
# --------------------------------------------------------------------------- #
def test_new_point_id_is_uuid4_str():
    pid = new_point_id()
    assert isinstance(pid, str)
    assert len(pid) == 36
    assert pid.count("-") == 4
    assert _UUID_RE.match(pid), pid
    # Round-trips as a real UUID (128-bit, not a truncated 64-bit int).
    assert str(uuid.UUID(pid)) == pid


def test_new_point_id_unique_no_collision():
    ids = {new_point_id() for _ in range(10_000)}
    assert len(ids) == 10_000  # no truncation-driven collisions


def test_to_point_id_returns_uuid_string_not_u64_int():
    # Pre-fix `_to_u64` returned an int; the fixed `_to_point_id` returns a
    # 36-char UUID string. This assertion fails on the buggy code.
    pid = _to_point_id("doc-123")
    assert isinstance(pid, str)
    assert _UUID_RE.match(pid), pid


def test_to_point_id_deterministic_and_distinct():
    # Deterministic (idempotent upsert/delete) ...
    assert _to_point_id("doc-A") == _to_point_id("doc-A")
    # ... yet distinct source ids map to distinct point ids.
    assert _to_point_id("doc-A") != _to_point_id("doc-B")


def test_to_point_id_passes_through_valid_uuid():
    u = str(uuid.uuid4())
    assert _to_point_id(u) == u


# --------------------------------------------------------------------------- #
# (b) jsonschema-backed SchemaValidator
# --------------------------------------------------------------------------- #
_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0},
    },
    "required": ["name", "age"],
    "additionalProperties": False,
}


def test_validate_good_dict_passes():
    result = SchemaValidator().validate({"name": "ada", "age": 42}, _SCHEMA)
    assert result.valid
    assert result.errors == []


def test_validate_bad_dict_returns_errors():
    # Missing required field + wrong type.
    result = SchemaValidator().validate({"age": "not-a-number"}, _SCHEMA)
    assert not result.valid
    assert result.errors  # structured, non-empty


def test_validate_rejects_bool_as_integer():
    # THE jsonschema-specific check: the old isinstance validator accepted
    # True as an integer (isinstance(True, int) is True). jsonschema rejects it.
    #
    # E3 (test_e3_jsonschema_202012.py) deliberately made ``SchemaValidator``
    # a lightweight field-level validator; the strict jsonschema enforcement
    # lives in the production entry point ``validate_structured_output``.
    # Assert the strict behavior there.
    from distllm.core.structured_output import validate_structured_output

    result = validate_structured_output('{"name": "x", "age": true}', _SCHEMA)
    assert result is None, "bool must not satisfy 'integer' — proves jsonschema is used"


def test_validate_enforces_constraints_old_validator_ignored():
    # minimum / additionalProperties were silently ignored by the weak validator.
    # Strict enforcement is asserted via the production entry point (see above).
    from distllm.core.structured_output import validate_structured_output

    r1 = validate_structured_output('{"name": "x", "age": -5}', _SCHEMA)
    assert r1 is None  # violates minimum: 0
    r2 = validate_structured_output('{"name": "x", "age": 1, "extra": 1}', _SCHEMA)
    assert r2 is None  # additionalProperties: False


def test_validate_none_schema_is_permissive():
    # Backward-compatible: no schema -> valid.
    assert SchemaValidator().validate({"anything": 1}, None).valid


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
