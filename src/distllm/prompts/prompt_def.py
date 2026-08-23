from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


__all__ = ["SystemPromptDef", "_reg", "_PROMPTS", "register_prompt"]


@dataclass
class SystemPromptDef:
    id: str
    category: str
    name: str
    description: str
    prompt: str
    tags: list[str] = field(default_factory=list)
    version: int = 1


_PROMPTS: list[SystemPromptDef] = []


def register_prompt(defn: SystemPromptDef) -> None:
    _PROMPTS.append(defn)


def _reg(
    id: str, category: str, name: str, description: str, prompt: str, tags: list[str] | None = None
) -> SystemPromptDef:
    d = SystemPromptDef(id=id, category=category, name=name, description=description, prompt=prompt.strip(), tags=tags or [])
    register_prompt(d)
    return d
