from __future__ import annotations

from distllm.prompts.prompt_def import SystemPromptDef, _PROMPTS

SYSTEM_PROMPTS: dict[str, SystemPromptDef] = {p.id: p for p in _PROMPTS}


def get_prompt(prompt_id: str) -> SystemPromptDef | None:
    for p in _PROMPTS:
        if p.id == prompt_id:
            return p
    return None


def list_categories() -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for p in _PROMPTS:
        if p.category not in seen:
            seen.add(p.category)
            result.append(p.category)
    return result


def list_by_category(category: str) -> list[SystemPromptDef]:
    return [p for p in _PROMPTS if p.category == category]


def search_prompts(query: str) -> list[SystemPromptDef]:
    q = query.lower()
    results: list[SystemPromptDef] = []
    for p in _PROMPTS:
        if q in p.name.lower() or q in p.description.lower() or any(q in t.lower() for t in p.tags):
            results.append(p)
    return results
