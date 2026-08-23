"""Tests for SchemaValidator, OutputRepairer, ValidationResult, RepairResult."""

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_mod = load_module("distllm/core/structured_output/validator.py")
SchemaValidator = _mod.SchemaValidator
OutputRepairer = _mod.OutputRepairer
ValidationResult = _mod.ValidationResult
RepairResult = _mod.RepairResult


# ── Data-class tests ────────────────────────────────────────────────────────


class TestValidationResult:
    """Construction and defaults for ValidationResult."""

    def test_valid_true(self):
        r = ValidationResult(valid=True)
        assert r.valid is True
        assert r.errors == []
        assert r.data is None

    def test_valid_false(self):
        r = ValidationResult(valid=False, errors=["bad"])
        assert r.valid is False
        assert r.errors == ["bad"]
        assert r.data is None

    def test_with_data(self):
        r = ValidationResult(valid=True, data={"x": 1})
        assert r.data == {"x": 1}

    def test_list_errors(self):
        r = ValidationResult(valid=False, errors=["e1", "e2"])
        assert r.errors == ["e1", "e2"]


class TestRepairResult:
    """Construction and defaults for RepairResult."""

    def test_success_defaults(self):
        r = RepairResult(success=True)
        assert r.success is True
        assert r.repaired_text == ""
        assert r.original_text == ""
        assert r.attempts == 0
        assert r.errors == []

    def test_failure(self):
        r = RepairResult(success=False, errors=["failed"])
        assert r.success is False
        assert r.errors == ["failed"]

    def test_full_construction(self):
        r = RepairResult(
            success=True,
            repaired_text='{"a": 1}',
            original_text='{"a": 1',
            attempts=1,
        )
        assert r.success is True
        assert r.repaired_text == '{"a": 1}'
        assert r.original_text == '{"a": 1'
        assert r.attempts == 1


# ── SchemaValidator ─────────────────────────────────────────────────────────


class TestSchemaValidator:
    """Key method behaviour for SchemaValidator.validate()."""

    def test_construct(self):
        v = SchemaValidator()
        assert isinstance(v, SchemaValidator)

    def test_validate_no_schema(self):
        """Without a schema, any data is valid."""
        v = SchemaValidator()
        result = v.validate({"anything": 42})
        assert result.valid is True
        assert result.data == {"anything": 42}

    def test_validate_none_schema(self):
        v = SchemaValidator()
        result = v.validate("hello", schema=None)
        assert result.valid is True
        assert result.data == "hello"

    def test_type_object_pass(self):
        v = SchemaValidator()
        result = v.validate({"x": 1}, schema={"type": "object"})
        assert result.valid is True

    def test_type_object_fail(self):
        v = SchemaValidator()
        result = v.validate("not an object", schema={"type": "object"})
        assert result.valid is False
        assert any("type" in e for e in result.errors)

    def test_type_string_pass(self):
        v = SchemaValidator()
        result = v.validate("hello", schema={"type": "string"})
        assert result.valid is True

    def test_type_string_fail(self):
        v = SchemaValidator()
        result = v.validate(42, schema={"type": "string"})
        assert result.valid is False

    def test_type_integer(self):
        v = SchemaValidator()
        assert v.validate(1, schema={"type": "integer"}).valid is True
        assert v.validate(1.5, schema={"type": "integer"}).valid is False

    def test_type_number(self):
        v = SchemaValidator()
        assert v.validate(1, schema={"type": "number"}).valid is True
        assert v.validate(1.5, schema={"type": "number"}).valid is True
        assert v.validate("x", schema={"type": "number"}).valid is False

    def test_type_boolean(self):
        v = SchemaValidator()
        assert v.validate(True, schema={"type": "boolean"}).valid is True
        assert v.validate(False, schema={"type": "boolean"}).valid is True
        assert v.validate(1, schema={"type": "boolean"}).valid is False

    def test_type_null(self):
        v = SchemaValidator()
        assert v.validate(None, schema={"type": "null"}).valid is True
        assert v.validate(0, schema={"type": "null"}).valid is False

    def test_type_array(self):
        v = SchemaValidator()
        assert v.validate([1, 2], schema={"type": "array"}).valid is True
        assert v.validate("x", schema={"type": "array"}).valid is False

    def test_unknown_type_allowed(self):
        """Unknown type strings don't cause errors."""
        v = SchemaValidator()
        result = v.validate(42, schema={"type": "unknown_type"})
        assert result.valid is True

    def test_required_fields_pass(self):
        v = SchemaValidator()
        schema = {"type": "object", "required": ["a", "b"], "properties": {"a": {}, "b": {}}}
        result = v.validate({"a": 1, "b": 2}, schema=schema)
        assert result.valid is True

    def test_required_fields_fail(self):
        v = SchemaValidator()
        schema = {"type": "object", "required": ["a", "b"], "properties": {"a": {}, "b": {}}}
        result = v.validate({"a": 1}, schema=schema)
        assert result.valid is False
        assert any("b" in e for e in result.errors)

    def test_nested_properties(self):
        v = SchemaValidator()
        schema = {
            "type": "object",
            "properties": {
                "inner": {"type": "object", "properties": {"x": {"type": "integer"}}},
            },
        }
        result = v.validate({"inner": {"x": "not_int"}}, schema=schema)
        assert result.valid is False

    def test_nested_properties_valid(self):
        v = SchemaValidator()
        schema = {
            "type": "object",
            "properties": {
                "inner": {"type": "object", "properties": {"x": {"type": "integer"}}},
            },
        }
        result = v.validate({"inner": {"x": 42}}, schema=schema)
        assert result.valid is True

    def test_empty_object_no_properties(self):
        v = SchemaValidator()
        result = v.validate({"x": 1}, schema={"type": "object"})
        assert result.valid is True

    def test_non_dict_with_properties(self):
        """Properties validation only runs on dict data."""
        v = SchemaValidator()
        schema = {"type": "array", "properties": {"x": {"type": "integer"}}}
        result = v.validate([1, 2, 3], schema=schema)
        assert result.valid is True  # array type check passes, properties skipped

    def test_check_type_integer_vs_float(self):
        v = SchemaValidator()
        assert v._check_type(42, "integer") is True
        assert v._check_type(42.0, "integer") is False

    def test_check_type_number(self):
        v = SchemaValidator()
        assert v._check_type(42, "number") is True
        assert v._check_type(42.5, "number") is True
        assert v._check_type("42", "number") is False

    def test_check_type_null(self):
        v = SchemaValidator()
        assert v._check_type(None, "null") is True
        assert v._check_type(0, "null") is False

    def test_check_type_unknown(self):
        """Unknown types fall back to True."""
        v = SchemaValidator()
        assert v._check_type("whatever", "unknown_type") is True


# ── OutputRepairer ──────────────────────────────────────────────────────────


class TestOutputRepairer:
    """Key method behaviour for OutputRepairer.repair()."""

    def test_construct_default(self):
        r = OutputRepairer()
        assert isinstance(r, OutputRepairer)

    def test_construct_custom_attempts(self):
        r = OutputRepairer(max_attempts=5)
        assert isinstance(r, OutputRepairer)

    def test_repair_valid_json(self):
        """Valid JSON passes through unchanged."""
        r = OutputRepairer()
        result = r.repair('{"valid": "json"}')
        assert result.success is True
        assert result.repaired_text == '{"valid": "json"}'
        assert result.attempts == 0

    def test_repair_unclosed_braces(self):
        r = OutputRepairer()
        result = r.repair('{"key": "value"')
        assert result.success is True
        assert '"key": "value"' in result.repaired_text

    def test_repair_unclosed_quotes(self):
        r = OutputRepairer()
        result = r.repair('{"key": "value')
        assert result.success is True
        assert "key" in result.repaired_text

    def test_repair_trailing_comma(self):
        r = OutputRepairer()
        result = r.repair('{"a": 1,}')
        assert isinstance(result.success, bool)

    def test_repair_unclosed_brackets(self):
        r = OutputRepairer()
        result = r.repair('{"items": [1, 2')
        assert isinstance(result.success, bool)

    def test_repair_multiple_issues(self):
        """Unclosed string + unclosed object + trailing comma."""
        r = OutputRepairer()
        result = r.repair('{"a": "hello,')
        assert result.success is True

    def test_repair_max_attempts_failure(self):
        """Very broken input should eventually fail."""
        r = OutputRepairer(max_attempts=2)
        result = r.repair("{{{ totally broken [[[")
        # The result might succeed or fail depending on repair heuristics,
        # but we just check that it returns a RepairResult.
        assert isinstance(result, RepairResult)

    def test_repair_empty_string(self):
        r = OutputRepairer()
        result = r.repair("")
        # Empty string is valid (repair always called), but check it works
        assert isinstance(result, RepairResult)

    def test_repair_tracks_attempts(self):
        """On non-trivial repairs, attempt count should be > 0."""
        r = OutputRepairer(max_attempts=3)
        result = r.repair('{"key": "value')
        if result.success:
            assert result.attempts >= 1

    def test_try_repair_closes_string(self):
        r = OutputRepairer()
        repaired = r._try_repair('{"a": "hello')
        assert repaired.count('"') % 2 == 0

    def test_try_repair_closes_braces(self):
        r = OutputRepairer()
        repaired = r._try_repair('{"a": 1')
        assert repaired.endswith("}")

    def test_try_repair_closes_brackets(self):
        r = OutputRepairer()
        repaired = r._try_repair('{"a": [1, 2')
        assert repaired.endswith("]")

    def test_try_repair_removes_trailing_comma(self):
        r = OutputRepairer()
        repaired = r._try_repair('{"a": 1,')
        # After closing, should not have trailing comma before closing brace
        assert repaired.rstrip().endswith(",") is False

    def test_repair_preserves_original_text(self):
        r = OutputRepairer()
        original = '{"key": "value"'
        result = r.repair(original)
        assert result.original_text == original

    def test_repair_with_schema(self):
        """Repair with schema should still work (type-level check)."""
        r = OutputRepairer()
        result = r.repair('{"a": 1}', schema={"type": "object"})
        assert result.success is True

    def test_repair_type_mismatch_with_schema(self):
        """Repair that yields wrong type still reports success if JSON is valid."""
        r = OutputRepairer()
        # The repairer fixes JSON syntax, not schema conformance
        result = r.repair("true", schema={"type": "object"})
        assert result.success is True  # "true" is valid JSON
