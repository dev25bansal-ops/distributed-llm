"""Tests for Feature 30: Prompt Template Engine."""

from unittest.mock import MagicMock

import pytest

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

# ChatML markers
IM_START = "\u003c\u007c"  # <|
IM_END = "\u007c\u003e"   # |>


class TestChatMLTemplate:
    def test_single_user_message(self):
        result = chatml_template([{"role": "user", "content": "Hello"}])
        assert "user" in result
        assert "Hello" in result

    def test_system_user_assistant(self):
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        result = chatml_template(messages)
        assert "system" in result
        assert "user" in result
        assert "assistant" in result

    def test_add_generation_prompt(self):
        result = chatml_template([{"role": "user", "content": "Hi"}], add_generation_prompt=True)
        assert "assistant" in result


class TestLlama2Template:
    def test_basic_instruction(self):
        result = llama2_template([{"role": "user", "content": "Hello"}])
        assert "[INST]" in result
        assert "[/INST]" in result
        assert "Hello" in result

    def test_system_prompt_with_instruction(self):
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
        ]
        result = llama2_template(messages)
        assert "[INST]" in result
        assert "<<SYS>>" in result
        assert "<</SYS>>" in result

    def test_assistant_response(self):
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        result = llama2_template(messages)
        assert "[INST]" in result
        assert "Hello!" in result


class TestLlama3Template:
    def test_basic_message(self):
        result = llama3_template([{"role": "user", "content": "Hello"}])
        assert "start_header_id" in result
        assert "user" in result
        assert "Hello" in result

    def test_multi_turn(self):
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        result = llama3_template(messages)
        assert "system" in result
        assert "user" in result
        assert "assistant" in result


class TestMistralTemplate:
    def test_instruction_format(self):
        result = mistral_template([{"role": "user", "content": "Hello"}])
        assert "[INST]" in result
        assert "[/INST]" in result

    def test_s_tag(self):
        result = mistral_template([{"role": "user", "content": "Hello"}])
        assert result.startswith("<s>")


class TestZephyrTemplate:
    def test_role_markers(self):
        result = zephyr_template([{"role": "user", "content": "Hi"}])
        assert "user" in result
        assert "Hi" in result

    def test_generation_prompt(self):
        result = zephyr_template([{"role": "user", "content": "Hi"}], add_generation_prompt=True)
        assert "assistant" in result


class TestAlpacaTemplate:
    def test_instruction_response(self):
        result = alpaca_template([{"role": "user", "content": "Hello"}])
        assert "### Instruction:" in result

    def test_assistant_response(self):
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        result = alpaca_template(messages)
        assert "### Instruction:" in result
        assert "### Response:" in result


class TestBuiltinTemplates:
    def test_all_templates_callable(self):
        messages = [{"role": "user", "content": "test"}]
        for name, func in BUILTIN_TEMPLATES.items():
            result = func(messages)
            assert isinstance(result, str)
            assert len(result) > 0

    def test_all_templates_include_user_content(self):
        messages = [{"role": "user", "content": "UNIQUE_TEST_TOKEN"}]
        for name, func in BUILTIN_TEMPLATES.items():
            result = func(messages)
            assert "UNIQUE_TEST_TOKEN" in result


class TestAutoDetect:
    def test_detects_llama3(self):
        assert auto_detect_template("meta-llama/Llama-3-8B") == "llama-3"

    def test_detects_llama2(self):
        assert auto_detect_template("meta-llama/Llama-2-7b-chat-hf") == "llama-2"

    def test_detects_mistral(self):
        assert auto_detect_template("mistralai/Mistral-7B") == "mistral"

    def test_detects_zephyr(self):
        assert auto_detect_template("HuggingFaceH4/zephyr-7b-beta") == "zephyr"

    def test_detects_chatml_for_qwen(self):
        assert auto_detect_template("Qwen/Qwen-7B-Chat") == "chatml"

    def test_returns_auto_for_unknown(self):
        assert auto_detect_template("unknown/model") == "auto"


class TestTemplateEngine:
    def test_explicit_template_by_name(self):
        engine = TemplateEngine(template="chatml")
        result = engine.apply([{"role": "user", "content": "Hi"}])
        assert "user" in result

    def test_auto_detect_with_tokenizer(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.name_or_path = "meta-llama/Llama-3-8B"
        # Llama-3 tokenizer may not have chat_template, so it falls back to built-in
        mock_tokenizer.chat_template = None

        engine = TemplateEngine(template="auto", tokenizer=mock_tokenizer)
        result = engine.apply([{"role": "user", "content": "Hi"}])
        assert len(result) > 0

    def test_tokenizer_apply_chat_template(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted by tokenizer"
        mock_tokenizer.chat_template = "some template"

        engine = TemplateEngine(template="unknown_template", tokenizer=mock_tokenizer)
        result = engine.apply([{"role": "user", "content": "Hi"}])
        assert result == "formatted by tokenizer"

    def test_fallback_format(self):
        engine = TemplateEngine(template="nonexistent")
        result = engine.apply([{"role": "user", "content": "Hi"}])
        assert result == "user: Hi"

    def test_empty_messages(self):
        engine = TemplateEngine(template="chatml")
        result = engine.apply([])
        assert result == ""

    def test_register_custom_template(self):
        engine = TemplateEngine(template="my_template")

        def custom_tmpl(messages, add_gen=True):
            return "CUSTOM: " + messages[0]["content"]

        engine.register("my_template", custom_tmpl)
        result = engine.apply([{"role": "user", "content": "hello"}])
        assert result == "CUSTOM: hello"

    def test_list_templates(self):
        engine = TemplateEngine(template="auto")
        templates = engine.list_templates()
        assert "chatml" in templates
        assert "llama-3" in templates

    def test_custom_template_overrides_builtin(self):
        engine = TemplateEngine(template="chatml")

        def override_tmpl(messages, add_gen=True):
            return "OVERRIDE"

        engine.register("chatml", override_tmpl)
        result = engine.apply([{"role": "user", "content": "test"}])
        assert result == "OVERRIDE"
