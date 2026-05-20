"""Basic unit tests for distllm-langchain.

These tests use mocked HTTP to avoid requiring a running server.
"""

from unittest.mock import MagicMock, patch


def test_imports():
    """Verify all top-level imports work."""
    from distllm_langchain import DistLLMChat, DistLLM, DistLLMEmbeddings, DistLLMToolProvider
    assert DistLLMChat is not None
    assert DistLLM is not None
    assert DistLLMEmbeddings is not None
    assert DistLLMToolProvider is not None


def test_chat_model_init():
    """Verify DistLLMChat construction with minimal args."""
    from distllm_langchain.chat_models import DistLLMChat

    llm = DistLLMChat(model="test-model", base_url="http://localhost:8000")
    assert llm.model == "test-model"
    assert llm.base_url == "http://localhost:8000"
    assert llm.temperature == 0.7
    assert llm._llm_type == "distllm-chat"


def test_chat_model_message_conversion():
    """Verify LangChain message <-> DistLLM dict conversion."""
    from distllm_langchain.chat_models import _convert_message_to_dict, _convert_dict_to_message
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ChatMessage

    # HumanMessage
    d = _convert_message_to_dict(HumanMessage(content="hello"))
    assert d == {"role": "user", "content": "hello"}

    # AIMessage
    d = _convert_message_to_dict(AIMessage(content="world"))
    assert d == {"role": "assistant", "content": "world"}

    # SystemMessage
    d = _convert_message_to_dict(SystemMessage(content="be helpful"))
    assert d == {"role": "system", "content": "be helpful"}

    # ChatMessage (custom role)
    d = _convert_message_to_dict(ChatMessage(role="developer", content="code"))
    assert d == {"role": "developer", "content": "code"}

    # Dict -> message
    msg = _convert_dict_to_message({"role": "assistant", "content": "hi"})
    assert msg.content == "hi"


def test_llm_init():
    """Verify DistLLM construction."""
    from distllm_langchain.llms import DistLLM

    llm = DistLLM(model="test", base_url="http://localhost:8000")
    assert llm._llm_type == "distllm-llm"
    assert llm.model == "test"


def test_embeddings_init():
    """Verify DistLLMEmbeddings construction."""
    from distllm_langchain.embeddings import DistLLMEmbeddings

    emb = DistLLMEmbeddings(base_url="http://localhost:8000")
    assert emb.model == "distributed-llm"
    assert emb.base_url == "http://localhost:8000"


def test_embeddings_embed_documents_mocked():
    """Verify embed_documents calls the SDK correctly."""
    from distllm_langchain.embeddings import DistLLMEmbeddings

    emb = DistLLMEmbeddings(base_url="http://localhost:8000")
    mock_resp = MagicMock()
    mock_resp.data = [MagicMock(embedding=[0.1, 0.2, 0.3]), MagicMock(embedding=[0.4, 0.5, 0.6])]

    with patch.object(emb._client, "embeddings", return_value=mock_resp):
        result = emb.embed_documents(["hello", "world"])
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]


def test_tool_provider_defaults():
    """Verify DistLLMToolProvider returns default tools."""
    from distllm_langchain.tools import DistLLMToolProvider

    provider = DistLLMToolProvider(base_url="http://localhost:8000")
    tools = provider._default_tools()
    assert len(tools) == 3
    names = [t["name"] for t in tools]
    assert "distllm_chat" in names
    assert "distllm_complete" in names
    assert "distllm_embed" in names


def test_build_payload():
    """Verify payload building with various params."""
    from distllm_langchain.chat_models import DistLLMChat
    from langchain_core.messages import HumanMessage

    llm = DistLLMChat(model="m", base_url="http://localhost:8000")
    payload = llm._build_payload(
        [HumanMessage(content="hi")],
        stop=["\n"],
        kwargs={},
    )
    assert payload["model"] == "m"
    assert len(payload["messages"]) == 1
    assert payload["messages"][0]["role"] == "user"
    assert payload["messages"][0]["content"] == "hi"
    assert payload["stop"] == ["\n"]
    assert payload["temperature"] == 0.7
