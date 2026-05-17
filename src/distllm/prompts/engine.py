"""Prompt template engine for chat models.

Supports built-in templates (ChatML, Llama-2/3, Mistral, etc.),
tokenizer.apply_chat_template() fallback, and custom template registration.
"""

from typing import Callable

from loguru import logger

from distllm.prompts.templates import (
    BUILTIN_TEMPLATES,
    auto_detect_template,
)


class TemplateEngine:
    """Formats chat messages into model-specific prompt strings.

    Priority:
    1. Explicitly registered custom template by name
    2. Built-in template by name
    3. tokenizer.apply_chat_template() if tokenizer has chat_template
    4. Fallback: naive "role: content" join
    """

    def __init__(
        self,
        template: str = "auto",
        tokenizer=None,
        custom_templates: dict[str, Callable] | None = None,
    ):
        self._template_name = template
        self._tokenizer = tokenizer
        self._custom_templates: dict[str, Callable] = custom_templates or {}

    def apply(
        self,
        messages: list[dict[str, str]],
        template_name: str | None = None,
        add_generation_prompt: bool = True,
    ) -> str:
        """Format messages into a prompt string.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            template_name: Override the engine's default template.
            add_generation_prompt: Whether to append generation prompt.

        Returns:
            Formatted prompt string.
        """
        if not messages:
            return ""

        name = template_name or self._template_name

        # 1. Custom registered template
        if name in self._custom_templates:
            return self._custom_templates[name](messages, add_generation_prompt)

        # 2. Built-in template
        if name in BUILTIN_TEMPLATES:
            return BUILTIN_TEMPLATES[name](messages, add_generation_prompt)

        # 3. Auto-detect from model name, then try built-in
        if name == "auto":
            detected = auto_detect_template(self._get_model_name())
            if detected in BUILTIN_TEMPLATES:
                return BUILTIN_TEMPLATES[detected](messages, add_generation_prompt)

        # 4. Fallback to tokenizer.apply_chat_template()
        if self._tokenizer is not None:
            chat_template = getattr(self._tokenizer, "chat_template", None)
            if chat_template is not None:
                try:
                    return self._tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=add_generation_prompt,
                    )
                except Exception as e:
                    logger.warning(f"tokenizer.apply_chat_template failed: {e}")

        # 5. Final fallback: naive join
        return self._fallback_format(messages)

    def register(self, name: str, template_func: Callable) -> None:
        """Register a custom template function.

        Args:
            name: Template name for lookup.
            template_func: Callable(messages, add_generation_prompt) -> str.
        """
        self._custom_templates[name] = template_func
        logger.info(f"Registered custom template: {name}")

    def list_templates(self) -> list[str]:
        """List all available template names."""
        return list(set(list(BUILTIN_TEMPLATES.keys()) + list(self._custom_templates.keys())))

    def set_tokenizer(self, tokenizer) -> None:
        """Update the tokenizer (used for apply_chat_template fallback)."""
        self._tokenizer = tokenizer

    def _get_model_name(self) -> str:
        """Get model name from tokenizer for auto-detection."""
        if self._tokenizer is None:
            return ""
        # Try to get model name from tokenizer
        name = getattr(self._tokenizer, "name_or_path", "")
        if not name:
            name = getattr(getattr(self._tokenizer, "tokenizer", None), "name_or_path", "")
        return name or ""

    @staticmethod
    def _fallback_format(messages: list[dict[str, str]]) -> str:
        """Naive fallback: role: content join."""
        return "\n".join(f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in messages)
