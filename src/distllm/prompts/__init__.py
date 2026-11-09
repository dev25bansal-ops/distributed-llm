"""Prompt template engine and curated system prompt library."""

from distllm.prompts.engine import TemplateEngine
from distllm.prompts.library import (
    SYSTEM_PROMPTS,
    SystemPromptDef,
    get_prompt,
    list_by_category,
    list_categories,
    search_prompts,
)
from distllm.prompts.templates import (
    BUILTIN_TEMPLATES,
    alpaca_template,
    auto_detect_template,
    chatml_template,
    llama2_template,
    llama3_template,
    mistral_template,
    zephyr_template,
)

__all__ = [
    "TemplateEngine",
    "chatml_template",
    "llama2_template",
    "llama3_template",
    "mistral_template",
    "zephyr_template",
    "alpaca_template",
    "BUILTIN_TEMPLATES",
    "auto_detect_template",
    "SystemPromptDef",
    "SYSTEM_PROMPTS",
    "get_prompt",
    "list_categories",
    "list_by_category",
    "search_prompts",
]
