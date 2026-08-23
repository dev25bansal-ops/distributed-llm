"""Tests for distllm.core.model_selector.

Covers:
    ModelProfile    -- dataclass fields and defaults
    ModelRecommendation -- recommendation dataclass
    DEFAULT_PROFILES     -- known model profiles
    ModelSelector        -- construction, scoring, selection, listing

Every test is deterministic (no network, no GPU, no time.sleep).
No MagicMock -- real objects only.
"""

from __future__ import annotations

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

# Bootstrap fake packages for distllm namespace
bootstrap_fake_packages()

# Load the source module
_sel_mod = load_module("distllm/core/model_selector.py")

# Re-export symbols for test readability
ModelProfile = _sel_mod.ModelProfile
ModelRecommendation = _sel_mod.ModelRecommendation
ModelSelector = _sel_mod.ModelSelector
DEFAULT_PROFILES = _sel_mod.DEFAULT_PROFILES

# ===================================================================
# MODEL PROFILE TESTS
# ===================================================================


class TestModelProfile:
    """ModelProfile dataclass -- construction, defaults, field access."""

    def test_minimal_construction(self) -> None:
        """A ModelProfile with just a name should get sensible defaults."""
        p = ModelProfile(name="test-model")
        assert p.name == "test-model"
        assert p.parameter_count == ""
        assert p.tasks == []
        assert p.quality_score == 0.0
        assert p.speed_score == 0.0
        assert p.cost_per_1m_tokens == 0.0
        assert p.context_length == 4096
        assert p.supports_tools is False
        assert p.supports_vision is False
        assert p.quantized is False

    def test_full_construction(self) -> None:
        """All fields should be settable via the constructor."""
        p = ModelProfile(
            name="custom-model",
            parameter_count="13B",
            tasks=["chat", "code"],
            quality_score=0.88,
            speed_score=0.75,
            cost_per_1m_tokens=0.45,
            context_length=16384,
            supports_tools=True,
            supports_vision=True,
            quantized=True,
        )
        assert p.name == "custom-model"
        assert p.parameter_count == "13B"
        assert p.tasks == ["chat", "code"]
        assert p.quality_score == 0.88
        assert p.speed_score == 0.75
        assert p.cost_per_1m_tokens == 0.45
        assert p.context_length == 16384
        assert p.supports_tools is True
        assert p.supports_vision is True
        assert p.quantized is True

    def test_tasks_default_factory_is_independent(self) -> None:
        """Each profile should get its own task list (not shared)."""
        p1 = ModelProfile(name="a")
        p2 = ModelProfile(name="b")
        p1.tasks.append("chat")
        assert "chat" not in p2.tasks

    def test_quality_score_zero(self) -> None:
        """quality_score of 0.0 should be valid."""
        p = ModelProfile(name="zero", quality_score=0.0)
        assert p.quality_score == 0.0

    def test_quality_score_one(self) -> None:
        """quality_score of 1.0 should be valid."""
        p = ModelProfile(name="perfect", quality_score=1.0)
        assert p.quality_score == 1.0

    def test_context_length_zero(self) -> None:
        """context_length can be set to 0."""
        p = ModelProfile(name="zero-ctx", context_length=0)
        assert p.context_length == 0


# ===================================================================
# MODEL RECOMMENDATION TESTS
# ===================================================================


class TestModelRecommendation:
    """ModelRecommendation dataclass -- fields and defaults."""

    def test_minimal_construction(self) -> None:
        """A recommendation with just name/confidence/reasoning."""
        r = ModelRecommendation(
            model_name="test-model",
            confidence=0.5,
            reasoning="good fit",
        )
        assert r.model_name == "test-model"
        assert r.confidence == 0.5
        assert r.reasoning == "good fit"
        assert r.alternatives == []

    def test_full_construction(self) -> None:
        """All fields including alternatives."""
        r = ModelRecommendation(
            model_name="best-model",
            confidence=0.95,
            reasoning="best overall match",
            alternatives=["alt-1", "alt-2"],
        )
        assert r.model_name == "best-model"
        assert r.confidence == 0.95
        assert r.reasoning == "best overall match"
        assert r.alternatives == ["alt-1", "alt-2"]

    def test_alternatives_default_is_independent(self) -> None:
        """Each recommendation should own its alternatives list."""
        r1 = ModelRecommendation(model_name="a", confidence=0.5, reasoning="")
        r2 = ModelRecommendation(model_name="b", confidence=0.5, reasoning="")
        r1.alternatives.append("x")
        assert "x" not in r2.alternatives

    def test_confidence_float_range(self) -> None:
        """Confidence should accept any float (clamped during selection)."""
        r = ModelRecommendation(model_name="m", confidence=0.0, reasoning="")
        assert r.confidence == 0.0
        r.confidence = 1.0
        assert r.confidence == 1.0


# ===================================================================
# DEFAULT PROFILES TESTS
# ===================================================================


class TestDefaultProfiles:
    """DEFAULT_PROFILES list -- structure and known entries."""

    def test_has_known_models(self) -> None:
        """The default profiles list should contain well-known models."""
        names = [p.name for p in DEFAULT_PROFILES]
        assert "meta-llama/Llama-3-70B-Instruct" in names
        assert "meta-llama/Llama-3-8B-Instruct" in names
        assert "mistralai/Mixtral-8x7B-Instruct-v0.1" in names
        assert "deepseek-ai/DeepSeek-Coder-V2-Instruct" in names

    def test_all_have_names(self) -> None:
        """Every default profile must have a non-empty name."""
        for p in DEFAULT_PROFILES:
            assert p.name, f"Profile missing name: {p}"

    def test_quality_scores_in_range(self) -> None:
        """All default quality scores should be between 0 and 1."""
        for p in DEFAULT_PROFILES:
            assert 0.0 <= p.quality_score <= 1.0, (
                f"{p.name}: quality_score={p.quality_score} out of range"
            )

    def test_speed_scores_in_range(self) -> None:
        """All default speed scores should be between 0 and 1."""
        for p in DEFAULT_PROFILES:
            assert 0.0 <= p.speed_score <= 1.0, (
                f"{p.name}: speed_score={p.speed_score} out of range"
            )

    def test_context_length_positive(self) -> None:
        """All default context lengths should be at least 1024."""
        for p in DEFAULT_PROFILES:
            assert p.context_length >= 1024, (
                f"{p.name}: context_length={p.context_length} too low"
            )

    def test_cost_non_negative(self) -> None:
        """All default cost values should be non-negative."""
        for p in DEFAULT_PROFILES:
            assert p.cost_per_1m_tokens >= 0, (
                f"{p.name}: negative cost"
            )


# ===================================================================
# MODEL SELECTOR: CONSTRUCTION
# ===================================================================


class TestModelSelectorConstruction:
    """ModelSelector construction and defaults."""

    def test_default_construction(self) -> None:
        """Without arguments, ModelSelector should use DEFAULT_PROFILES."""
        sel = ModelSelector()
        assert len(sel._profiles) == len(DEFAULT_PROFILES)
        assert sel._custom_profiles == []

    def test_custom_profiles_list(self) -> None:
        """A custom profile list should replace DEFAULT_PROFILES."""
        p = ModelProfile(name="my-model")
        sel = ModelSelector(profiles=[p])
        assert sel._profiles == [p]
        assert len(sel._profiles) == 1

    def test_empty_profiles_list_falls_back_to_defaults(self) -> None:
        """An empty/falsy profile list should fall back to DEFAULT_PROFILES."""
        sel = ModelSelector(profiles=[])
        # [] is falsy, so profiles or DEFAULT_PROFILES → DEFAULT_PROFILES
        assert sel._profiles == DEFAULT_PROFILES


# ===================================================================
# MODEL SELECTOR: ADD PROFILE
# ===================================================================


class TestModelSelectorAddProfile:
    """ModelSelector.add_profile() -- adding custom profiles."""

    def test_add_profile_appends(self) -> None:
        """add_profile should add to the custom profiles list."""
        sel = ModelSelector(profiles=[])
        p = ModelProfile(name="custom")
        sel.add_profile(p)
        assert p in sel._custom_profiles
        assert len(sel._custom_profiles) == 1

    def test_add_multiple_profiles(self) -> None:
        """Multiple profiles should all be added."""
        sel = ModelSelector(profiles=[])
        sel.add_profile(ModelProfile(name="a"))
        sel.add_profile(ModelProfile(name="b"))
        sel.add_profile(ModelProfile(name="c"))
        assert len(sel._custom_profiles) == 3

    def test_custom_profiles_combined_with_defaults(self) -> None:
        """Custom profiles should be considered alongside defaults during select."""
        sel = ModelSelector()
        sel.add_profile(ModelProfile(
            name="ultra-fast",
            tasks=["chat"],
            quality_score=0.5,
            speed_score=1.0,
            cost_per_1m_tokens=0.01,
        ))
        result = sel.select(quality="fast")
        assert result.model_name == "ultra-fast"


# ===================================================================
# MODEL SELECTOR: SCORE CANDIDATE
# ===================================================================


class TestModelSelectorScoreCandidate:
    """ModelSelector._score_candidate() -- internal scoring logic."""

    @pytest.fixture
    def sel(self) -> ModelSelector:
        return ModelSelector(profiles=[])

    def test_base_score(self, sel: ModelSelector) -> None:
        """A profile with no task match and empty quality string goes to balanced branch."""
        p = ModelProfile(name="base", tasks=[], quality_score=0.5, speed_score=0.5,
                         cost_per_1m_tokens=0)
        score = sel._score_candidate(p, task="", quality="", max_latency_ms=0)
        # empty quality -> else (balanced): (0.5+0.5)*0.15 = 0.15
        # cost=0 -> bypasses the cost check (>0)
        assert score == pytest.approx(0.65)

    def test_task_match_bonus(self, sel: ModelSelector) -> None:
        """Task matching should add 0.2 to the score."""
        p = ModelProfile(name="coder", tasks=["code"])
        score = sel._score_candidate(p, task="code", quality="", max_latency_ms=0)
        assert score == pytest.approx(0.7)  # 0.5 base + 0.2 task bonus

    def test_task_mismatch_penalty(self, sel: ModelSelector) -> None:
        """Task mismatch should subtract 0.1 from the score."""
        p = ModelProfile(name="coder", tasks=["code"])
        score = sel._score_candidate(p, task="chat", quality="", max_latency_ms=0)
        assert score == pytest.approx(0.4)  # 0.5 base - 0.1 penalty

    def test_quality_high_adds_quality_score_weighted(self, sel: ModelSelector) -> None:
        """High quality mode should add quality_score * 0.3."""
        p = ModelProfile(name="high-q", quality_score=0.8, speed_score=0.2)
        score = sel._score_candidate(p, task="", quality="high", max_latency_ms=0)
        # 0.5 base + 0.8*0.3 = 0.5 + 0.24 = 0.74
        assert score == pytest.approx(0.74)

    def test_quality_fast_adds_speed_score_weighted(self, sel: ModelSelector) -> None:
        """Fast quality mode should add speed_score * 0.3."""
        p = ModelProfile(name="fast-m", quality_score=0.2, speed_score=0.9)
        score = sel._score_candidate(p, task="", quality="fast", max_latency_ms=0)
        # 0.5 base + 0.9*0.3 = 0.5 + 0.27 = 0.77
        assert score == pytest.approx(0.77)

    def test_quality_balanced_averages_quality_and_speed(self, sel: ModelSelector) -> None:
        """Balanced mode should use (quality + speed) * 0.15."""
        p = ModelProfile(name="balanced", quality_score=0.6, speed_score=0.4)
        score = sel._score_candidate(p, task="", quality="balanced", max_latency_ms=0)
        # 0.5 base + (0.6 + 0.4) * 0.15 = 0.5 + 0.15 = 0.65
        assert score == pytest.approx(0.65)

    def test_cost_efficiency_lowers_score(self, sel: ModelSelector) -> None:
        """Higher cost should reduce the cost contribution."""
        p = ModelProfile(name="cheap", quality_score=0.5, speed_score=0.5,
                         cost_per_1m_tokens=0.5)
        score = sel._score_candidate(p, task="", quality="balanced", max_latency_ms=0)
        # cost_score = max(0, 1.0 - 0.5/2.0) = max(0, 0.75) = 0.75
        # 0.5 base + (0.5+0.5)*0.15 + 0.75*0.1 = 0.5 + 0.15 + 0.075 = 0.725
        assert score == pytest.approx(0.725)

    def test_cost_efficiency_expensive_model(self, sel: ModelSelector) -> None:
        """Very expensive model should get cost_score of 0."""
        p = ModelProfile(name="pricey", cost_per_1m_tokens=5.0)
        score = sel._score_candidate(p, task="", quality="", max_latency_ms=0)
        # cost_score = max(0, 1.0 - 5.0/2.0) = max(0, -1.5) = 0
        assert score == pytest.approx(0.5)  # base only

    def test_zero_cost_model(self, sel: ModelSelector) -> None:
        """Zero cost should not add cost bonus (cost_score guard: cost>0)."""
        p = ModelProfile(name="free", cost_per_1m_tokens=0.0)
        score = sel._score_candidate(p, task="", quality="", max_latency_ms=0)
        # cost == 0 so the cost check is skipped; no cost contribution
        # quality="" goes to else branch: (0.0+0.0)*0.15 = 0
        assert score == pytest.approx(0.5)  # base only


# ===================================================================
# MODEL SELECTOR: SELECT
# ===================================================================


class TestModelSelectorSelect:
    """ModelSelector.select() -- happy path, edge cases, error paths."""

    @pytest.fixture
    def sel(self) -> ModelSelector:
        """Selector with a known small set of profiles for deterministic tests."""
        return ModelSelector(profiles=[
            ModelProfile(
                name="fast-model", tasks=["chat", "code"],
                quality_score=0.3, speed_score=0.95,
                cost_per_1m_tokens=0.1,
            ),
            ModelProfile(
                name="quality-model", tasks=["chat", "analysis"],
                quality_score=0.95, speed_score=0.2,
                cost_per_1m_tokens=1.0,
            ),
            ModelProfile(
                name="mid-model", tasks=["chat", "code", "analysis"],
                quality_score=0.6, speed_score=0.6,
                cost_per_1m_tokens=0.5,
            ),
        ])

    def test_select_with_no_requirements(self, sel: ModelSelector) -> None:
        """With no requirements, the highest-scoring model should be selected."""
        result = sel.select()
        # fast-model: 0.5 + 0.2(task) + (0.3+0.95)*0.15 + 0.95*0.1 = ~0.98
        # mid-model: 0.5 + 0.2(task) + (0.6+0.6)*0.15 + 0.75*0.1 = ~0.955
        # quality-model: 0.5 - 0.1(no task) + (0.95+0.2)*0.15 + 0.5*0.1 = ~0.7225
        assert result.model_name == "fast-model"
        assert result.confidence > 0
        assert result.reasoning != ""
        assert len(result.alternatives) >= 1

    def test_select_high_quality(self, sel: ModelSelector) -> None:
        """High quality preference should pick the quality-model."""
        result = sel.select(quality="high")
        assert result.model_name == "quality-model"

    def test_select_fast(self, sel: ModelSelector) -> None:
        """Fast preference should pick the fast-model."""
        result = sel.select(quality="fast")
        assert result.model_name == "fast-model"

    def test_select_task_match_code(self, sel: ModelSelector) -> None:
        """Task=code should prefer models that specialize in code."""
        result = sel.select(task="code")
        # fast-model and mid-model both support code; fast-model has higher speed
        # but balanced mode weighs both.  Let the scoring determine.
        assert result.model_name in ("fast-model", "mid-model")

    def test_select_task_match_analysis(self, sel: ModelSelector) -> None:
        """Task=analysis should prefer models that support analysis."""
        result = sel.select(task="analysis")
        assert result.model_name in ("quality-model", "mid-model")

    def test_select_with_no_task(self, sel: ModelSelector) -> None:
        """Empty task string should not penalize any model."""
        result = sel.select(task="")
        assert result.model_name != ""

    def test_select_with_max_cost_filter(self, sel: ModelSelector) -> None:
        """max_cost_per_1m should exclude expensive models."""
        result = sel.select(max_cost_per_1m=0.3)
        # Only fast-model (0.1) fits under 0.3
        assert result.model_name == "fast-model"

    def test_select_with_context_length_filter(self, sel: ModelSelector) -> None:
        """context_length requirement should filter models with small context."""
        sel_with_ctx = ModelSelector(profiles=[
            ModelProfile(name="small-ctx", context_length=2048),
            ModelProfile(name="big-ctx", context_length=32768),
        ])
        result = sel_with_ctx.select(context_length=4096)
        assert result.model_name == "big-ctx"

    def test_select_with_requires_tools(self, sel: ModelSelector) -> None:
        """requires_tools should filter models that don't support tools."""
        sel_with_tools = ModelSelector(profiles=[
            ModelProfile(name="no-tools", supports_tools=False),
            ModelProfile(name="has-tools", supports_tools=True),
        ])
        result = sel_with_tools.select(requires_tools=True)
        assert result.model_name == "has-tools"

    def test_select_with_requires_vision(self, sel: ModelSelector) -> None:
        """requires_vision should filter models without vision support."""
        sel_with_vision = ModelSelector(profiles=[
            ModelProfile(name="no-vision", supports_vision=False),
            ModelProfile(name="has-vision", supports_vision=True),
        ])
        result = sel_with_vision.select(requires_vision=True)
        assert result.model_name == "has-vision"

    def test_select_with_available_models_filter(self, sel: ModelSelector) -> None:
        """available_models should restrict to the named models."""
        result = sel.select(available_models=["fast-model"])
        assert result.model_name == "fast-model"

    def test_select_available_models_no_match(self, sel: ModelSelector) -> None:
        """available_models with no match should return empty recommendation."""
        result = sel.select(available_models=["nonexistent-model"])
        assert result.model_name == ""
        assert result.confidence == 0.0
        assert "No models match" in result.reasoning

    def test_select_all_filters_combined(self, sel: ModelSelector) -> None:
        """Multiple filters combined should narrow candidates correctly."""
        result = sel.select(
            task="code",
            quality="high",
            max_cost_per_1m=0.2,
        )
        # Only fast-model (0.1) is under 0.2 cost limit
        assert result.model_name == "fast-model"

    def test_select_no_candidates_returns_empty(self, sel: ModelSelector) -> None:
        """When all models are filtered out, empty recommendation."""
        result = sel.select(
            max_cost_per_1m=0.01,
            requires_tools=True,
            requires_vision=True,
            context_length=1000000,
        )
        assert result.model_name == ""
        assert result.confidence == 0.0
        assert "No models match" in result.reasoning

    def test_select_confidence_clamped_to_one(self, sel: ModelSelector) -> None:
        """Confidence should not exceed 1.0."""
        # Create a profile that would get an extremely high score
        p = ModelProfile(
            name="overachiever",
            tasks=["chat"],
            quality_score=1.0,
            speed_score=1.0,
            cost_per_1m_tokens=0.01,
        )
        selector = ModelSelector(profiles=[p])
        result = selector.select(task="chat", quality="high")
        assert result.confidence <= 1.0
        assert result.confidence > 0


# ===================================================================
# MODEL SELECTOR: SELECT -- REASONING AND ALTERNATIVES
# ===================================================================


class TestModelSelectorReasoning:
    """The reasoning and alternatives in ModelRecommendation."""

    @pytest.fixture
    def sel(self) -> ModelSelector:
        return ModelSelector(profiles=[
            ModelProfile(
                name="best-chat", tasks=["chat", "code"],
                quality_score=0.9, speed_score=0.3,
                cost_per_1m_tokens=0.5,
            ),
            ModelProfile(
                name="second-chat", tasks=["chat", "analysis"],
                quality_score=0.7, speed_score=0.6,
                cost_per_1m_tokens=0.3,
            ),
            ModelProfile(
                name="third-chat", tasks=["chat"],
                quality_score=0.5, speed_score=0.8,
                cost_per_1m_tokens=0.2,
            ),
        ])

    def test_reasoning_mentions_task_match(self, sel: ModelSelector) -> None:
        """When task matches, reasoning should mention the task."""
        result = sel.select(task="code")
        assert "code" in result.reasoning

    def test_reasoning_high_quality(self, sel: ModelSelector) -> None:
        """High quality reasoning should mention the quality score."""
        result = sel.select(quality="high")
        assert "high quality" in result.reasoning or "score" in result.reasoning

    def test_reasoning_fast(self, sel: ModelSelector) -> None:
        """Fast quality reasoning should mention speed."""
        result = sel.select(quality="fast")
        assert "fast inference" in result.reasoning or "speed" in result.reasoning

    def test_reasoning_balanced_fallback(self, sel: ModelSelector) -> None:
        """No task and balanced quality should fall back to 'best overall match'."""
        result = sel.select()
        # If the model has no explicit task match and quality is balanced,
        # and no other flags triggered, reasoning is "best overall match"
        assert result.reasoning == "best overall match" or "overall" in result.reasoning

    def test_reasoning_quantized_model(self, sel: ModelSelector) -> None:
        """A quantized model should mention quantization."""
        sel_with_quant = ModelSelector(profiles=[
            ModelProfile(name="quant", quantized=True),
        ])
        result = sel_with_quant.select()
        assert "quantized" in result.reasoning

    def test_alternatives_contains_other_models(self, sel: ModelSelector) -> None:
        """Alternatives should list the next best models (up to 2)."""
        result = sel.select()
        assert len(result.alternatives) >= 1
        assert result.model_name not in result.alternatives

    def test_alternatives_limited_to_two(self, sel: ModelSelector) -> None:
        """The alternatives list should have at most 2 entries."""
        result = sel.select()
        assert len(result.alternatives) <= 2


# ===================================================================
# MODEL SELECTOR: LIST MODELS
# ===================================================================


class TestModelSelectorListModels:
    """ModelSelector.list_models() -- listing available profiles."""

    def test_list_models_with_defaults(self) -> None:
        """list_models should return all DEFAULT_PROFILES."""
        sel = ModelSelector()
        models = sel.list_models()
        assert len(models) == len(DEFAULT_PROFILES)
        assert models[0]["name"] == DEFAULT_PROFILES[0].name

    def test_list_models_structure(self) -> None:
        """Each listed model should be a dict with expected keys."""
        sel = ModelSelector(profiles=[
            ModelProfile(name="test", parameter_count="7B", tasks=["chat"],
                         quality_score=0.8, speed_score=0.7, cost_per_1m_tokens=0.5),
        ])
        models = sel.list_models()
        assert len(models) == 1
        entry = models[0]
        assert entry["name"] == "test"
        assert entry["parameters"] == "7B"
        assert entry["tasks"] == ["chat"]
        assert entry["quality"] == 0.8
        assert entry["speed"] == 0.7
        assert entry["cost_per_1m"] == 0.5

    def test_list_models_includes_custom(self) -> None:
        """Custom profiles added via add_profile should appear in list_models."""
        sel = ModelSelector(profiles=[ModelProfile(name="base")])
        sel.add_profile(ModelProfile(name="custom"))
        models = sel.list_models()
        names = [m["name"] for m in models]
        assert "base" in names
        assert "custom" in names

    def test_list_models_empty_falls_back_to_defaults(self) -> None:
        """An empty/falsy profiles list falls back to DEFAULT_PROFILES."""
        sel = ModelSelector(profiles=[])
        assert len(sel.list_models()) == len(DEFAULT_PROFILES)


# ===================================================================
# MODEL SELECTOR: EDGE CASES
# ===================================================================


class TestModelSelectorEdgeCases:
    """Edge cases and error handling."""

    def test_single_model(self) -> None:
        """With only one profile, it should always be selected."""
        sel = ModelSelector(profiles=[
            ModelProfile(name="only-one", tasks=["chat"]),
        ])
        result = sel.select()
        assert result.model_name == "only-one"
        assert result.alternatives == []  # no other models for alternatives

    def test_all_models_same_score(self) -> None:
        """All profiles with identical scores should pick the first."""
        sel = ModelSelector(profiles=[
            ModelProfile(name="first", quality_score=0.5, speed_score=0.5),
            ModelProfile(name="second", quality_score=0.5, speed_score=0.5),
        ])
        result = sel.select()
        assert result.model_name == "first"  # stable sort, first in list

    def test_very_large_context_length_filter(self) -> None:
        """A huge context length requirement should eliminate all models."""
        sel = ModelSelector(profiles=[
            ModelProfile(name="small", context_length=4096),
        ])
        result = sel.select(context_length=999999)
        assert result.model_name == ""

    def test_cost_filter_with_non_zero_value(self) -> None:
        """Setting max_cost_per_1m to a positive value should filter expensive models."""
        sel = ModelSelector(profiles=[
            ModelProfile(name="free", cost_per_1m_tokens=0.0),
            ModelProfile(name="paid", cost_per_1m_tokens=0.5),
        ])
        result = sel.select(max_cost_per_1m=0.25)
        assert result.model_name == "free"

    def test_returns_model_recommendation_instance(self) -> None:
        """select() should return a ModelRecommendation instance."""
        sel = ModelSelector(profiles=[
            ModelProfile(name="test"),
        ])
        result = sel.select()
        assert isinstance(result, ModelRecommendation)

    def test_extremely_expensive_model_penalized(self) -> None:
        """A model with extremely high cost_per_1m should score lower."""
        sel = ModelSelector(profiles=[
            ModelProfile(name="cheap", cost_per_1m_tokens=0.01),
            ModelProfile(name="expensive", cost_per_1m_tokens=100.0),
        ])
        result = sel.select()
        assert result.model_name == "cheap"

    def test_task_name_different_case(self) -> None:
        """Task matching should be case-sensitive per the list."""
        sel = ModelSelector(profiles=[
            ModelProfile(name="coder", tasks=["Code"]),  # Capitalized task
        ])
        result = sel.select(task="code")  # lowercase task
        # Case mismatch means no task bonus and a penalty
        assert result.model_name == "coder"  # still the only model
        # No task match message in reasoning
        assert "optimized for" not in result.reasoning

    def test_empty_available_models_uses_all(self) -> None:
        """An empty available_models list is falsy so no filter is applied."""
        sel = ModelSelector()
        result = sel.select(available_models=[])
        assert result.model_name != ""  # all models available

    def test_multiple_models_with_same_name_not_in_profiles(self) -> None:
        """available_models referencing non-existent names yields empty."""
        sel = ModelSelector()
        result = sel.select(available_models=["does-not-exist"])
        assert result.model_name == ""
