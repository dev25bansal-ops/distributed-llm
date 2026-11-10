"""Tests for framework integration adapters using real objects via load_module.

NOTE: Optional dependencies langchain-openai and llama-index-llms-openai are
not installed in the test env — their adapter functions raise ``ImportError``
at call time, which we test for directly.  haystack-ai IS installed in this
env, so its missing-dependency path is exercised by blocking the module in
``sys.modules``; its installed path is pinned by a construction test.
"""

from __future__ import annotations

import sys

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

# These adapters deliberately import their framework lazily, so the module
# imports fine without openai installed and the failure only appears at call
# time. The tests below construct a REAL client for the success paths, which
# needs the package. Skip them rather than fail: the absence path is covered
# separately, and a test suite should not demand an optional extra to be green.
pytestmark = pytest.mark.skipif(
    __import__("importlib").util.find_spec("openai") is None,
    reason="needs the [examples] extra (openai)",
)

_fa_mod = load_module("distllm/core/framework_adapters.py")
get_openai_client = _fa_mod.get_openai_client
get_async_openai_client = _fa_mod.get_async_openai_client
get_langchain_llm = _fa_mod.get_langchain_llm
get_llamaindex_llm = _fa_mod.get_llamaindex_llm
get_haystack_generator = _fa_mod.get_haystack_generator
list_frameworks = _fa_mod.list_frameworks
get_framework_adapter = _fa_mod.get_framework_adapter
FRAMEWORK_COMPAT = _fa_mod.FRAMEWORK_COMPAT


class TestGetOpenAIClient:
    """get_openai_client constructs an OpenAI client."""

    def test_returns_object(self) -> None:
        client = get_openai_client(base_url="http://localhost:8000")
        assert client is not None
        assert str(client.base_url).rstrip("/") == "http://localhost:8000/v1"

    def test_default_base_url(self) -> None:
        client = get_openai_client()
        assert "localhost" in str(client.base_url)

    def test_custom_api_key(self) -> None:
        client = get_openai_client(api_key="sk-custom")
        assert client.api_key == "sk-custom"

    def test_default_api_key(self) -> None:
        client = get_openai_client()
        assert client.api_key == "not-needed"


class TestGetAsyncOpenAIClient:
    """get_async_openai_client constructs an AsyncOpenAI client."""

    def test_returns_async_client(self) -> None:
        client = get_async_openai_client(base_url="http://localhost:8000")
        assert client is not None
        assert str(client.base_url).rstrip("/") == "http://localhost:8000/v1"


class TestGetLangChainLLM:
    """get_langchain_llm raises ImportError when langchain-openai missing."""

    def test_import_error_when_missing(self) -> None:
        with pytest.raises((ImportError, ModuleNotFoundError)):
            get_langchain_llm()


class TestGetLlamaIndexLLM:
    """get_llamaindex_llm raises ImportError when llama-index missing."""

    def test_import_error_when_missing(self) -> None:
        with pytest.raises((ImportError, ModuleNotFoundError)):
            get_llamaindex_llm()


class TestGetHaystackGenerator:
    """get_haystack_generator behavior for missing and installed haystack."""

    def test_import_error_when_missing(self) -> None:
        # Simulate haystack-ai being uninstalled: a None entry in sys.modules
        # makes any ``import haystack...`` raise ImportError immediately.
        saved = sys.modules.pop("haystack", None)
        sys.modules["haystack"] = None
        try:
            with pytest.raises((ImportError, ModuleNotFoundError)):
                get_haystack_generator()
        finally:
            del sys.modules["haystack"]
            if saved is not None:
                sys.modules["haystack"] = saved

    def test_constructs_generator_when_installed(self) -> None:
        # haystack-ai is installed in this env; the adapter must build a
        # working OpenAIGenerator (haystack 2.x requires a Secret api_key).
        pytest.importorskip("haystack")
        gen = get_haystack_generator(base_url="http://localhost:8000")
        assert type(gen).__name__ == "OpenAIGenerator"
        assert str(gen.client.base_url).rstrip("/") == "http://localhost:8000/v1"


class TestListFrameworks:
    """list_frameworks returns framework metadata."""

    def test_returns_list(self) -> None:
        frameworks = list_frameworks()
        assert isinstance(frameworks, list)
        assert len(frameworks) > 0

    def test_all_have_required_keys(self) -> None:
        frameworks = list_frameworks()
        for fw in frameworks:
            assert "name" in fw
            assert "package" in fw
            assert "class" in fw
            assert "features" in fw
            assert "example" in fw

    def test_includes_langchain(self) -> None:
        names = [fw["name"] for fw in list_frameworks()]
        assert "langchain" in names

    def test_includes_llamaindex(self) -> None:
        names = [fw["name"] for fw in list_frameworks()]
        assert "llamaindex" in names

    def test_includes_haystack(self) -> None:
        names = [fw["name"] for fw in list_frameworks()]
        assert "haystack" in names

    def test_includes_autogpt(self) -> None:
        names = [fw["name"] for fw in list_frameworks()]
        assert "autogpt" in names

    def test_includes_agency_swarm(self) -> None:
        names = [fw["name"] for fw in list_frameworks()]
        assert "agency_swarm" in names

    def test_includes_crewai(self) -> None:
        names = [fw["name"] for fw in list_frameworks()]
        assert "crewai" in names

    def test_includes_dify(self) -> None:
        names = [fw["name"] for fw in list_frameworks()]
        assert "dify" in names


class TestGetFrameworkAdapter:
    """get_framework_adapter dispatches to correct adapter."""

    def test_unknown_framework_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown framework"):
            get_framework_adapter("nonexistent")

    def test_langchain_raises_import_error(self) -> None:
        # The adapter exists but langchain-openai is not installed
        with pytest.raises((ImportError, ModuleNotFoundError, RuntimeError)):
            get_framework_adapter("langchain")

    def test_openai_adapter_succeeds(self) -> None:
        # OpenAI package IS installed (used by many things)
        client = get_framework_adapter("autogpt")
        assert client is not None
        assert "localhost" in str(client.base_url)


class TestFRAMEWORK_COMPAT:
    """FRAMEWORK_COMPAT dict structure."""

    def test_all_entries_have_adapter_or_example(self) -> None:
        for name, info in FRAMEWORK_COMPAT.items():
            if info.get("adapter") is None:
                assert "example" in info, f"{name} has no adapter or example"

    def test_all_entries_have_package(self) -> None:
        for name, info in FRAMEWORK_COMPAT.items():
            assert "package" in info, f"{name} missing package"

    def test_all_entries_have_features(self) -> None:
        for name, info in FRAMEWORK_COMPAT.items():
            assert isinstance(info.get("features"), list)
