"""Generate configuration reference documentation from Pydantic models.

Extracts field descriptions, types, defaults, and validators from
DistLLMSettings and outputs a structured markdown reference.
"""

from __future__ import annotations

import io
from typing import Any, get_type_hints

from pydantic import BaseModel


def _field_type_str(field_info: Any) -> str:
    """Get a human-readable type string from a Pydantic field."""
    try:
        ann = field_info.annotation
        if ann is None:
            return "any"
        if hasattr(ann, "__name__"):
            return ann.__name__
        return str(ann).replace("typing.", "")
    except Exception:
        return "any"


def _extract_model_fields(model: type[BaseModel], prefix: str = "") -> list[dict[str, Any]]:
    """Recursively extract fields from a Pydantic model and nested models."""
    rows: list[dict[str, Any]] = []
    for name, field_info in model.model_fields.items():
        full_name = f"{prefix}.{name}" if prefix else name
        field_type = _field_type_str(field_info)
        default = field_info.default
        description = field_info.description or ""

        # Check if this field is itself a BaseModel (nested section)
        ann = field_info.annotation
        if ann and isinstance(ann, type) and issubclass(ann, BaseModel):
            rows.append({
                "name": full_name,
                "type": "section",
                "default": "",
                "description": description,
            })
            rows.extend(_extract_model_fields(ann, prefix=full_name))
        else:
            rows.append({
                "name": full_name,
                "type": field_type,
                "default": repr(default) if default is not None else "",
                "description": description,
            })
    return rows


def generate_config_reference(settings_class: type[BaseModel]) -> str:
    """Generate a markdown config reference from a Pydantic settings class.

    Args:
        settings_class: The root settings BaseModel (e.g., DistLLMSettings).

    Returns:
        Markdown string with the configuration reference.
    """
    buf = io.StringIO()
    buf.write("# DistLLM Configuration Reference\n\n")
    buf.write("Auto-generated from Pydantic models. All fields can be set via:\n")
    buf.write("- `config.yaml` file\n")
    buf.write("- Environment variables (prefix: `DISTLLM__`, delimiter: `__`)\n")
    buf.write("- CLI arguments\n\n")
    buf.write("---\n\n")

    rows = _extract_model_fields(settings_class)

    current_section = ""
    for row in rows:
        if row["type"] == "section":
            current_section = row["name"]
            buf.write(f"## `{row['name']}`\n\n")
            if row["description"]:
                buf.write(f"{row['description']}\n\n")
            buf.write("| Field | Type | Default | Description |\n")
            buf.write("|-------|------|---------|-------------|\n")
        else:
            env_var = f"`DISTLLM__{row['name'].upper().replace('.', '__')}`"
            default_str = f"`{row['default']}`" if row["default"] else ""
            desc = row["description"]
            if desc:
                desc += f" Env: {env_var}"
            else:
                desc = f"Env: {env_var}"
            buf.write(f"| `{row['name']}` | `{row['type']}` | {default_str} | {desc} |\n")

    buf.write("\n---\n\n")
    buf.write("## Environment Variable Mapping\n\n")
    buf.write("Any config field can be overridden via environment variables:\n\n")
    buf.write("```\n")
    buf.write("config.yaml:  model.name -> DISTLLM__MODEL__NAME\n")
    buf.write("config.yaml:  coordinator.port -> DISTLLM__COORDINATOR__PORT\n")
    buf.write("config.yaml:  tls.enabled -> DISTLLM__TLS__ENABLED\n")
    buf.write("```\n")

    return buf.getvalue()
