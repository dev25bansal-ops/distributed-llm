"""Tests for the built-in system prompt library."""

import pytest

from distllm.prompts.library import (
    SystemPromptDef,
    SYSTEM_PROMPTS,
    get_prompt,
    list_categories,
    list_by_category,
    search_prompts,
)


class TestSystemPromptDef:
    def test_minimal_creation(self):
        p = SystemPromptDef(id="t1", category="test", name="Test", description="A test", prompt="Hello")
        assert p.id == "t1"
        assert p.tags == []
        assert p.version == 1

    def test_full_creation(self):
        p = SystemPromptDef(
            id="t2", category="code", name="Test2", description="Another test",
            prompt="You are a test.", tags=["a", "b"], version=2,
        )
        assert p.version == 2
        assert p.tags == ["a", "b"]


class TestGetPrompt:
    def test_existing_prompt(self):
        p = get_prompt("code-review")
        assert p is not None
        assert p.id == "code-review"
        assert p.category == "code"

    def test_nonexistent_prompt(self):
        assert get_prompt("nonexistent-prompt-xyz") is None

    def test_empty_id(self):
        assert get_prompt("") is None


class TestListCategories:
    def test_returns_sorted_unique_categories(self):
        cats = list_categories()
        assert isinstance(cats, list)
        assert len(cats) > 0
        assert all(isinstance(c, str) for c in cats)
        assert len(cats) == len(set(cats))

    def test_includes_code_category(self):
        cats = list_categories()
        assert "code" in cats

    def test_includes_writing_category(self):
        cats = list_categories()
        assert "writing" in cats


class TestListByCategory:
    def test_code_category_returns_prompts(self):
        prompts = list_by_category("code")
        assert len(prompts) > 0
        for p in prompts:
            assert p.category == "code"

    def test_nonexistent_category_returns_empty(self):
        prompts = list_by_category("nonexistent-category-xyz")
        assert prompts == []

    def test_all_prompts_in_category_have_valid_ids(self):
        for cat in list_categories():
            prompts = list_by_category(cat)
            for p in prompts:
                assert p.id in SYSTEM_PROMPTS


class TestSearchPrompts:
    def test_search_by_name(self):
        results = search_prompts("Code Review")
        assert len(results) > 0
        assert any("code-review" in p.id for p in results)

    def test_search_by_description(self):
        results = search_prompts("debug")
        assert len(results) > 0

    def test_search_by_tag(self):
        results = search_prompts("security")
        assert len(results) > 0

    def test_search_no_match(self):
        results = search_prompts("xyznonexistent12345")
        assert results == []

    def test_search_empty_query(self):
        results = search_prompts("")
        assert len(results) == len(SYSTEM_PROMPTS)

    def test_search_case_insensitive(self):
        results_lower = search_prompts("summary")
        results_upper = search_prompts("SUMMARY")
        assert [p.id for p in results_lower] == [p.id for p in results_upper]


class TestSYSTEM_PROMPTS:
    def test_has_expected_count(self):
        assert len(SYSTEM_PROMPTS) >= 50

    def test_all_ids_are_unique(self):
        ids = [p.id for p in SYSTEM_PROMPTS.values()]
        assert len(ids) == len(set(ids))

    def test_all_prompts_have_required_fields(self):
        for prompt_id, p in SYSTEM_PROMPTS.items():
            assert p.id == prompt_id
            assert p.category, f"Prompt {prompt_id} missing category"
            assert p.name, f"Prompt {prompt_id} missing name"
            assert p.description, f"Prompt {prompt_id} missing description"
            assert p.prompt, f"Prompt {prompt_id} missing prompt content"

    def test_all_prompts_have_tags(self):
        for p in SYSTEM_PROMPTS.values():
            assert isinstance(p.tags, list)

    def test_all_prompts_have_positive_version(self):
        for p in SYSTEM_PROMPTS.values():
            assert p.version >= 1

    def test_each_category_has_at_least_one_prompt(self):
        for cat in list_categories():
            assert len(list_by_category(cat)) >= 1

    def test_no_empty_prompts(self):
        for p in SYSTEM_PROMPTS.values():
            assert len(p.prompt.strip()) > 0

    def test_no_unnamed_prompts(self):
        for p in SYSTEM_PROMPTS.values():
            assert len(p.name.strip()) > 0


class TestCLICommands:
    def test_cli_list_exists(self):
        from distllm.cli.prompts import prompt_app
        cmd_names = [cmd.name for cmd in prompt_app.registered_commands]
        assert "list" in cmd_names

    def test_cli_show_exists(self):
        from distllm.cli.prompts import prompt_app
        cmd_names = [cmd.name for cmd in prompt_app.registered_commands]
        assert "show" in cmd_names

    def test_cli_categories_exists(self):
        from distllm.cli.prompts import prompt_app
        cmd_names = [cmd.name for cmd in prompt_app.registered_commands]
        assert "categories" in cmd_names

    def test_cli_use_exists(self):
        from distllm.cli.prompts import prompt_app
        cmd_names = [cmd.name for cmd in prompt_app.registered_commands]
        assert "use" in cmd_names


class TestIntegration:
    def test_get_prompt_and_search_consistent(self):
        results = search_prompts("review")
        for r in results:
            assert get_prompt(r.id) is r

    def test_list_by_category_and_search_consistent(self):
        code_prompts = list_by_category("code")
        code_search = search_prompts("code")
        code_ids = {p.id for p in code_prompts}
        search_ids = {p.id for p in code_search}
        assert not code_ids.isdisjoint(search_ids)

    def test_categories_and_list_by_category_consistent(self):
        for cat in list_categories():
            prompts = list_by_category(cat)
            assert all(p.category == cat for p in prompts)
