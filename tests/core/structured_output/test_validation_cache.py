"""Enhancement #3: schema-validation LRU cache tests."""

from __future__ import annotations

from distllm.core.structured_output.validator import SchemaValidator


def _schema():
    return {"type": "object", "required": ["a"], "properties": {"a": {"type": "integer"}}}


def test_cache_hit_on_repeated_validation():
    SchemaValidator._RESULT_CACHE.clear()
    SchemaValidator._RESULT_CACHE_HITS = 0
    SchemaValidator._RESULT_CACHE_MISSES = 0

    v = SchemaValidator()
    r1 = v.validate({"a": 1}, schema=_schema())
    r2 = v.validate({"a": 1}, schema=_schema())

    assert r1.valid and r2.valid
    stats = SchemaValidator.validation_cache_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1


def test_cache_distinguishes_different_data():
    SchemaValidator._RESULT_CACHE.clear()
    SchemaValidator._RESULT_CACHE_HITS = 0
    SchemaValidator._RESULT_CACHE_MISSES = 0

    v = SchemaValidator()
    v.validate({"a": 1}, schema=_schema())
    v.validate({"a": 2}, schema=_schema())
    v.validate({"a": 1}, schema=_schema())  # repeat of the first

    stats = SchemaValidator.validation_cache_stats()
    assert stats["misses"] == 2
    assert stats["hits"] == 1


def test_invalid_results_are_cached_too():
    SchemaValidator._RESULT_CACHE.clear()
    SchemaValidator._RESULT_CACHE_HITS = 0
    SchemaValidator._RESULT_CACHE_MISSES = 0

    v = SchemaValidator()
    r1 = v.validate({"b": 1}, schema=_schema())  # missing required 'a'
    r2 = v.validate({"b": 1}, schema=_schema())

    assert not r1.valid and not r2.valid
    assert r1.errors == r2.errors
    assert SchemaValidator.validation_cache_stats()["hits"] == 1


def test_cache_bounded_at_max():
    SchemaValidator._RESULT_CACHE.clear()
    v = SchemaValidator()
    schema = {"type": "object"}
    for i in range(SchemaValidator._RESULT_CACHE_MAX + 25):
        v.validate({"i": i}, schema=schema)
    assert len(SchemaValidator._RESULT_CACHE) <= SchemaValidator._RESULT_CACHE_MAX


def test_cache_stats_shape():
    stats = SchemaValidator.validation_cache_stats()
    assert {"hits", "misses", "size", "hit_rate"} == set(stats.keys())
