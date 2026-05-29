"""Tests for the multi-model chat router (ModelRouter)."""

from distllm.config.settings import ChatRouterSettings, RouteRuleSettings
from distllm.core.model_router import ModelRouter


def _make_settings(
    name: str = "hybrid",
    default_model: str = "llama3",
    routes: list | None = None,
) -> ChatRouterSettings:
    return ChatRouterSettings(
        enabled=True,
        name=name,
        default_model=default_model,
        routes=routes or [],
    )


class TestModelRouterKeyword:
    """Keyword-based routing rules."""

    def setup_method(self):
        self.settings = _make_settings(routes=[
            RouteRuleSettings(name="code", match_type="keyword", match="write a function", target_model="codellama", priority=10),
            RouteRuleSettings(name="story", match_type="keyword", match="write a story", target_model="storywriter", priority=5),
            RouteRuleSettings(name="math", match_type="keyword", match="solve", target_model="mathgpt", priority=5),
        ])
        self.router = ModelRouter(self.settings)
        self.router.register_hybrid_name("hybrid")

    def test_keyword_match_routes_to_correct_model(self):
        assert self.router.resolve("write a function that sums numbers") == "codellama"

    def test_keyword_match_is_case_insensitive(self):
        assert self.router.resolve("Write A Function in Python") == "codellama"

    def test_second_rule_matches_when_first_does_not(self):
        assert self.router.resolve("write a story about dragons") == "storywriter"

    def test_falls_back_to_default_when_no_rule_matches(self):
        assert self.router.resolve("what is the weather today") == "llama3"

    def test_rule_priority_respected(self):
        msg = "write a function and write a story"
        assert self.router.resolve(msg) == "codellama"

    def test_available_models_filter_skips_unavailable(self):
        result = self.router.resolve("write a function", available_models=["storywriter", "llama3"])
        assert result == "llama3"

    def test_available_models_filter_passes_available(self):
        result = self.router.resolve("write a function", available_models=["codellama", "llama3"])
        assert result == "codellama"


class TestModelRouterRegex:
    """Regex-based routing rules."""

    def setup_method(self):
        self.settings = _make_settings(routes=[
            RouteRuleSettings(name="python", match_type="regex", match=r"\bdef\s+\w+\s*\(", target_model="pycoder", priority=10),
            RouteRuleSettings(name="email", match_type="regex", match=r"[\w\.-]+@[\w\.-]+\.\w+", target_model="emailer", priority=5),
        ])
        self.router = ModelRouter(self.settings)

    def test_regex_match_routes_correctly(self):
        assert self.router.resolve("def hello():") == "pycoder"

    def test_regex_match_email(self):
        assert self.router.resolve("contact me at user@example.com") == "emailer"

    def test_regex_no_match_falls_back(self):
        assert self.router.resolve("just a normal question") == "llama3"

    def test_regex_bad_pattern_does_not_crash(self):
        settings = _make_settings(routes=[
            RouteRuleSettings(name="bad", match_type="regex", match=r"[invalid", target_model="fallback", priority=10),
        ])
        router = ModelRouter(settings)
        assert router.resolve("any text") == "llama3"


class TestModelRouterWorkload:
    """Workload-type-based routing using WorkloadClassifier."""

    def setup_method(self):
        self.settings = _make_settings(routes=[
            RouteRuleSettings(name="code-route", match_type="workload", match="code", target_model="codellama", priority=10),
        ])
        self.router = ModelRouter(self.settings)

    def test_code_workload_routes_to_codellama(self):
        code_text = "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
        assert self.router.resolve(code_text) == "codellama"

    def test_non_code_workload_falls_back(self):
        assert self.router.resolve("The weather is nice today.") == "llama3"


class TestModelRouterEdgeCases:
    """Edge cases and special scenarios."""

    def test_empty_message_falls_back(self):
        router = ModelRouter(_make_settings())
        assert router.resolve("") == "llama3"

    def test_no_rules_configured(self):
        router = ModelRouter(_make_settings())
        assert router.resolve("anything") == "llama3"

    def test_empty_default_model(self):
        settings = _make_settings(default_model="")
        router = ModelRouter(settings)
        assert router.resolve("hello") == ""

    def test_hybrid_names_registration(self):
        router = ModelRouter(_make_settings())
        assert router.list_hybrid_models() == []
        router.register_hybrid_name("smart-router")
        router.register_hybrid_name("smart-router")
        assert router.list_hybrid_models() == ["smart-router"]

    def test_to_dict_serialization(self):
        settings = _make_settings(routes=[
            RouteRuleSettings(name="r1", match_type="keyword", match="hello", target_model="m1", priority=5),
        ])
        router = ModelRouter(settings)
        router.register_hybrid_name("hybrid")
        d = router.to_dict()
        assert d["enabled"] is True
        assert d["default_model"] == "llama3"
        assert d["hybrid_models"] == ["hybrid"]
        assert len(d["rules"]) == 1
        assert d["rules"][0]["name"] == "r1"

    def test_rule_with_no_name(self):
        settings = _make_settings(routes=[
            RouteRuleSettings(match_type="keyword", match="test", target_model="tester"),
        ])
        router = ModelRouter(settings)
        assert router.resolve("this is a test message") == "tester"
