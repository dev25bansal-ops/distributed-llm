"""Prompt template engine and curated system prompt library."""

from distllm.prompts.engine import TemplateEngine
from distllm.prompts.templates import (
    chatml_template,
    llama2_template,
    llama3_template,
    mistral_template,
    zephyr_template,
    alpaca_template,
    BUILTIN_TEMPLATES,
    auto_detect_template,
)
from distllm.prompts.library import (
    SystemPromptDef,
    SYSTEM_PROMPTS,
    get_prompt,
    list_categories,
    list_by_category,
    search_prompts,
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
