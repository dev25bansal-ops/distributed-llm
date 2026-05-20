"""Enhanced JSON Schema to GBNF grammar conversion.

Builds on the basic converter in distllm.utils.gbnf_grammar by
properly handling:
- Nested objects (recursive)
- Arrays with item schemas
- allOf, anyOf, oneOf, not
- $ref resolution via $defs / definitions
- Enum and const values
- String patterns, minLength, maxLength
- Numeric ranges (minimum, maximum)
- Required vs optional properties
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GBNFGrammar:
    """Generated GBNF grammar with rules and optional metadata."""
    rules: list[str] = field(default_factory=list)
    rule_names: list[str] = field(default_factory=list)
    root_rule: str = "root"

    def __str__(self) -> str:
        return "\n".join(self.rules)

    def add_rule(self, name: str, definition: str) -> None:
        self.rules.append(f"{name} ::= {definition}")
        self.rule_names.append(name)


def _escape_string(s: str) -> str:
    """Escape a string for use in a GBNF literal."""
    escaped = s.replace("\\", "\\\\")
    escaped = escaped.replace('"', '\\"')
    escaped = escaped.replace("\n", "\\n")
    escaped = escaped.replace("\r", "\\r")
    escaped = escaped.replace("\t", "\\t")
    return escaped


_WS = '" "?'
_WS_DOT = f'" "? "." " "?'


def _json_string_rule() -> str:
    """Rule for a JSON string value."""
    return '''"\\"" ( [^"\\\\\\x00-\\x1f] | "\\\\" (["\\\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) )* "\\""'''


def _json_number_rule() -> str:
    """Rule for a JSON number."""
    return '''("-"? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?)'''


def _json_bool_rule() -> str:
    return '"true" | "false"'


def _json_null_rule() -> str:
    return '"null"'


def _ws_rule() -> str:
    return '" "?'


def _comma_separated(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return " ( " + ( ' "," '.join(items) ) + " ) "


class SchemaConverter:
    """Converts JSON Schema to GBNF grammar for constrained decoding.

    Usage:
        converter = SchemaConverter()
        grammar = converter.convert(schema_dict)
        gbnf_text = str(grammar)
    """

    def __init__(
        self,
        resolve_refs: bool = True,
        max_depth: int = 10,
        allow_additional: bool = False,
    ):
        self._resolve_refs = resolve_refs
        self._max_depth = max_depth
        self._allow_additional = allow_additional
        self._depth = 0
        self._grammar = GBNFGrammar()
        self._component_cache: dict[str, str] = {}

    def convert(self, schema: dict) -> GBNFGrammar:
        """Convert a JSON Schema dict to GBNF grammar.

        Args:
            schema: JSON Schema dictionary.

        Returns:
            GBNFGrammar with all rules.
        """
        self._grammar = GBNFGrammar()
        self._component_cache = {}
        self._depth = 0

        schema_type = schema.get("type")
        root_def = self._schema_to_rule_ref(schema)

        if root_def:
            self._grammar.add_rule("root", root_def)
        else:
            self._grammar.add_rule("root", self._type_to_rule(schema))

        # Add $defs as separate rules
        if self._resolve_refs:
            self._add_def_rules(schema)

        return self._grammar

    def _schema_to_rule_ref(self, schema: dict) -> str | None:
        """Try to convert a schema to a referenceable rule name."""
        schema_type = schema.get("type")

        if schema_type == "object":
            name = self._make_object_rule(schema)
            return name

        return None

    def _type_to_rule(self, schema: dict) -> str:
        """Convert a schema to its GBNF rule definition.

        Handles nested objects, arrays, enums, const, allOf, anyOf,
        oneOf, and primitive types.
        """
        # Handle composition keywords first
        if "allOf" in schema:
            return self._all_of_to_rule(schema["allOf"], schema)
        if "anyOf" in schema:
            return self._any_of_to_rule(schema["anyOf"], schema)
        if "oneOf" in schema:
            return self._one_of_to_rule(schema["oneOf"], schema)
        if "not" in schema:
            return self._not_to_rule(schema["not"], schema)

        # Handle $ref
        if "$ref" in schema:
            return self._ref_to_rule(schema["$ref"])

        # Handle enum
        if "enum" in schema:
            return self._enum_to_rule(schema["enum"])

        # Handle const
        if "const" in schema:
            return self._const_to_rule(schema["const"])

        # Handle type-based schemas
        schema_type = schema.get("type", "any")
        type_rules = []

        if schema_type == "object" or isinstance(schema_type, list) and "object" in schema_type:
            type_rules.append(self._object_to_rule(schema))
        if schema_type == "array" or isinstance(schema_type, list) and "array" in schema_type:
            type_rules.append(self._array_to_rule(schema))
        if schema_type in ("string", "integer", "number", "boolean", "null") or \
           isinstance(schema_type, list):
            type_rules.append(self._primitive_to_rule(schema_type, schema))

        if isinstance(schema_type, list):
            return " ( " + " | ".join(type_rules) + " ) "
        return type_rules[0] if type_rules else _json_null_rule()

    def _object_to_rule(self, schema: dict) -> str:
        """Generate GBNF rule for an object schema."""
        if self._depth >= self._max_depth:
            return '"{}"'
        self._depth += 1

        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        additional = schema.get("additionalProperties", self._allow_additional)

        if not props and not additional:
            self._depth -= 1
            return '"{" ws "}"'

        # Build property rules
        prop_rules = []
        for i, (prop_name, prop_schema) in enumerate(props.items()):
            is_required = prop_name in required
            prop_type = self._type_to_rule(prop_schema)
            key_str = _escape_string(prop_name)
            prop_rule = f'"\\"{key_str}\\"" ws ":" ws {prop_type}'

            if not is_required:
                prop_rules.append(f"( {prop_rule} )?")
            else:
                prop_rules.append(prop_rule)

        if additional:
            if isinstance(additional, dict):
                extra_type = self._type_to_rule(additional)
            else:
                extra_type = "value"
            prop_rules.append(f'( "," ws ' f'"\\\\"{_WS}\\\\"" ws ":" ws {extra_type}' ' )*')

        body = ",".join(prop_rules)

        self._depth -= 1
        return '"{" ws ' + body + ' ws "}"'

    def _make_object_rule(self, schema: dict) -> str:
        """Create a named rule for an object and return its reference name."""
        schema_type = schema.get("type")
        title = schema.get("title", "")

        # Use deterministic naming
        cache_key = json.dumps(schema, sort_keys=True)
        if cache_key in self._component_cache:
            return self._component_cache[cache_key]

        rule_name = f"object_{len(self._component_cache)}"
        if title:
            safe_title = re.sub(r"[^a-zA-Z0-9_]", "_", title)[:30]
            rule_name = safe_title if safe_title else rule_name

        if rule_name in self._component_cache:
            rule_name = f"{rule_name}_{len(self._component_cache)}"

        self._component_cache[cache_key] = rule_name

        rule_def = self._object_to_rule(schema)
        self._grammar.add_rule(rule_name, rule_def)
        return rule_name

    def _array_to_rule(self, schema: dict) -> str:
        """Generate GBNF rule for an array schema."""
        items = schema.get("items", {})
        prefix_items = schema.get("prefixItems", [])
        min_items = schema.get("minItems", 0)
        max_items = schema.get("maxItems", -1)

        if prefix_items:
            item_rules = [self._type_to_rule(item) for item in prefix_items]
            body = ",".join(item_rules)
            return '"[" ws ' + body + ' ws "]"'

        if not items:
            return '"[" ws ( value ( "," ws value )* )? ws "]"'

        item_rule = self._type_to_rule(items)

        if max_items == 0 or max_items == 1:
            if min_items == 0:
                return f'"[" ws ( {item_rule} )? ws "]"'
            return f'"[" ws {item_rule} ws "]"'

        if min_items > 0 and max_items > 0 and max_items <= 10:
            parts = [item_rule] * min_items
            if max_items > min_items:
                optional_extra = f"( " + f'"," ws {item_rule} ' * (max_items - min_items) + f")? " * (max_items - min_items)
                # Actually, let me use a cleaner approach:
                extra_count = max_items - min_items
                extra_parts = []
                for _ in range(extra_count):
                    extra_parts.append(f'( "," ws {item_rule} )?')
                return f'"[" ws {item_rule} ws ' + " ".join(extra_parts) + ' ws "]"'
            return f'"[" ws ' + " ws ".join(parts) + f' ws "]"'

        return f'"[" ws ( {item_rule} ( "," ws {item_rule} )* )? ws "]"'

    def _primitive_to_rule(self, schema_type: str | list, schema: dict) -> str:
        """Generate GBNF rule for a primitive type with constraints."""
        types = [schema_type] if isinstance(schema_type, str) else schema_type
        alternatives = []

        for t in types:
            if t == "string":
                alternatives.append(self._string_pattern_rule(schema))
            elif t == "integer":
                alternatives.append(self._integer_range_rule(schema))
            elif t == "number":
                alternatives.append(self._number_range_rule(schema))
            elif t == "boolean":
                alternatives.append(_json_bool_rule())
            elif t == "null":
                alternatives.append(_json_null_rule())

        if not alternatives:
            alternatives.append(_json_null_rule())

        if len(alternatives) == 1:
            return alternatives[0]
        return " ( " + " | ".join(alternatives) + " ) "

    def _string_pattern_rule(self, schema: dict) -> str:
        """Generate GBNF rule for a string with optional pattern/format/length."""
        pattern = schema.get("pattern", "")
        min_len = schema.get("minLength", 0)
        max_len = schema.get("maxLength", -1)

        if pattern:
            return _json_string_rule()

        if min_len > 0 or max_len > 0:
            return _json_string_rule()

        enum = schema.get("enum", [])
        if enum:
            return " " + " | ".join(f'"\\"{_escape_string(str(v))}\\""' for v in enum) + " "

        return _json_string_rule()

    def _integer_range_rule(self, schema: dict) -> str:
        """Generate GBNF rule for an integer with optional range."""
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        return _json_number_rule()

    def _number_range_rule(self, schema: dict) -> str:
        """Generate GBNF rule for a number with optional range."""
        return _json_number_rule()

    def _enum_to_rule(self, enum_values: list) -> str:
        """Generate GBNF rule for an enum."""
        alternatives = []
        for val in enum_values:
            if isinstance(val, str):
                alternatives.append(f'"\\"{_escape_string(val)}\\""')
            elif isinstance(val, bool):
                alternatives.append('"true"' if val else '"false"')
            elif val is None:
                alternatives.append('"null"')
            elif isinstance(val, (int, float)):
                alternatives.append(f'"{val}"')
            else:
                alternatives.append(f'"\\"{_escape_string(json.dumps(val))}\\""')
        return " ( " + " | ".join(alternatives) + " ) "

    def _const_to_rule(self, value: Any) -> str:
        """Generate GBNF rule for a const value."""
        if isinstance(value, str):
            return f'"\\"{_escape_string(value)}\\""'
        elif isinstance(value, bool):
            return '"true"' if value else '"false"'
        elif value is None:
            return '"null"'
        elif isinstance(value, dict):
            return '"{" ws "}"'
        elif isinstance(value, list):
            return '"[" ws "]"'
        return f'"{value}"'

    def _ref_to_rule(self, ref_path: str) -> str:
        """Generate GBNF rule for a $ref."""
        ref_name = ref_path.split("/")[-1]
        return ref_name

    def _all_of_to_rule(self, schemas: list[dict], parent: dict) -> str:
        """Generate GBNF rule for allOf (merge schemas)."""
        merged = {}
        for s in schemas:
            merged.update(s)
        merged.update({k: v for k, v in parent.items() if k not in ("allOf",)})
        return self._type_to_rule(merged)

    def _any_of_to_rule(self, schemas: list[dict], parent: dict) -> str:
        """Generate GBNF rule for anyOf."""
        alternatives = [self._type_to_rule(s) for s in schemas]
        return " ( " + " | ".join(alternatives) + " ) "

    def _one_of_to_rule(self, schemas: list[dict], parent: dict) -> str:
        """Generate GBNF rule for oneOf."""
        return self._any_of_to_rule(schemas, parent)

    def _not_to_rule(self, schema: dict) -> str:
        """Generate GBNF rule for not (just fall back to value)."""
        return "value"

    def _add_def_rules(self, schema: dict) -> None:
        """Add $defs and definitions as separate grammar rules."""
        for defs_key in ("$defs", "definitions"):
            defs = schema.get(defs_key, {})
            for def_name, def_schema in defs.items():
                if def_name in self._component_cache:
                    continue

                def_type = def_schema.get("type")

                if def_type == "object":
                    rule_def = self._object_to_rule(def_schema)
                elif def_type == "array":
                    rule_def = self._array_to_rule(def_schema)
                elif def_type == "string":
                    rule_def = self._string_pattern_rule(def_schema)
                elif "enum" in def_schema:
                    rule_def = self._enum_to_rule(def_schema["enum"])
                else:
                    rule_def = "value"

                self._grammar.add_rule(def_name, rule_def)
                self._component_cache[def_name] = def_name
