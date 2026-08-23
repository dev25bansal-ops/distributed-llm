"""Bootstrap and fixtures for distllm/utils tests."""

from __future__ import annotations

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

# Load modules once at conftest level so all tests import from here.
_gbnf = load_module("distllm/utils/gbnf_grammar.py")
_sched = load_module("distllm/utils/scheduling.py")

GBNFGrammar = _gbnf.GBNFGrammar
json_schema_to_gbnf = _gbnf.json_schema_to_gbnf
generate_gbnf_for_json_schema = _gbnf.generate_gbnf_for_json_schema
_type_to_rule = _gbnf._type_to_rule
_primitive_rule = _gbnf._primitive_rule
_primitive_example = _gbnf._primitive_example
_extra_rules = _gbnf._extra_rules

group_by_length = _sched.group_by_length
