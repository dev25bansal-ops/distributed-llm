"""GBNF grammar generation from JSON Schema."""

import json
from dataclasses import dataclass, field


@dataclass
class GBNFGrammar:
    rules: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return "\n".join(self.rules)


def json_schema_to_gbnf(schema_json: str) -> GBNFGrammar:
    schema = json.loads(schema_json) if isinstance(schema_json, str) else schema_json
    return _convert_schema(schema)


def generate_gbnf_for_json_schema(schema: dict, strict: bool = False) -> str:
    grammar = _convert_schema(schema)
    return str(grammar)


def _generate_rule_for_property(prop_name: str, prop_schema: dict) -> str:
    return _type_to_rule(prop_schema, prop_name)


def _generate_rule_for_ref(ref_path: str) -> str:
    ref_name = ref_path.split("/")[-1]
    return f"{ref_name} ::= value"


def _convert_schema(schema: dict) -> GBNFGrammar:
    rules = []
    schema_type = schema.get("type", "object")

    if schema_type == "object":
        rules.append('root ::= "{" ws')
        props = schema.get("properties", {})
        prop_rules = []
        for i, (name, prop_schema) in enumerate(props.items()):
            prop_type = _type_to_rule(prop_schema, name)
            key_rule = f'  "{json.dumps(name)[1:-1]}" ws ":" ws {prop_type}'
            if i < len(props) - 1:
                key_rule += ' "," ws'
            prop_rules.append(key_rule)
        if prop_rules:
            rules.append(" ".join(prop_rules))
        rules.append('"}" ws')
        rules.extend(_extra_rules(schema))
    elif schema_type in ("string", "integer", "number", "boolean"):
        rules.append(f'root ::= "{_primitive_example(schema_type)}"')
    else:
        rules.append('root ::= "null"')

    return GBNFGrammar(rules=rules)


def _type_to_rule(prop_schema: dict, name: str | None = None) -> str:
    prop_type = prop_schema.get("type", "string")
    if "enum" in prop_schema:
        alternatives = " | ".join(f'"{json.dumps(v)[1:-1]}"' for v in prop_schema["enum"])
        return f"( {alternatives} )"
    if "oneOf" in prop_schema:
        alternatives = " | ".join(_type_to_rule(s) for s in prop_schema["oneOf"])
        return f"( {alternatives} )"
    if "$ref" in prop_schema:
        return _generate_rule_for_ref(prop_schema["$ref"])
    return _primitive_rule(prop_type)


def _primitive_rule(prop_type: str) -> str:
    if prop_type == "string":
        return '"string"'
    elif prop_type == "integer":
        return '"0"'
    elif prop_type == "number":
        return '"0.0"'
    elif prop_type == "boolean":
        return '"true"'
    return '"null"'


def _primitive_example(prop_type: str) -> str:
    return {"string": "str", "integer": "0", "number": "0.0", "boolean": "true"}.get(prop_type, "null")


def _extra_rules(schema: dict) -> list[str]:
    rules = []
    defs = schema.get("$defs", {})
    for name, def_schema in defs.items():
        rules.append(f"{name} ::= value")
    return rules
