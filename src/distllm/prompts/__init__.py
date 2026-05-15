"""Prompt template engine for chat models."""

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
]
