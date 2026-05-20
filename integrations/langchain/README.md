# distllm-langchain

LangChain integration for DistLLM — use DistLLM as a drop-in `ChatModel`, `LLM`, `Embeddings`, and `Tool` provider in LangChain.

## Installation

```bash
pip install distllm-langchain
```

Or from source:

```bash
pip install -e integrations/langchain
```

## Quick Start

```python
from distllm_langchain import DistLLMChat
from langchain_core.messages import HumanMessage

llm = DistLLMChat(model="distributed-llm", base_url="http://localhost:8000")
response = llm.invoke([HumanMessage(content="Hello!")])
print(response.content)
```

## Streaming

```python
for chunk in llm.stream([HumanMessage(content="Tell me a story.")]):
    print(chunk.content, end="", flush=True)
```

## Embeddings

```python
from distllm_langchain import DistLLMEmbeddings

embeddings = DistLLMEmbeddings(base_url="http://localhost:8000")
vectors = embeddings.embed_documents(["Hello world", "Goodbye"])
print(len(vectors[0]))  # embedding dimension
```

## LLM (text completion)

```python
from distllm_langchain import DistLLM

llm = DistLLM(model="distributed-llm", base_url="http://localhost:8000")
result = llm.invoke("What is distributed inference?")
print(result)
```

## Tools

```python
from distllm_langchain import DistLLMToolProvider

provider = DistLLMToolProvider(base_url="http://localhost:8000")
tools = provider.get_tools()
```
