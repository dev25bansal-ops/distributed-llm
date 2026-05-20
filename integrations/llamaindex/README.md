# distllm-llamaindex

LlamaIndex integration for DistLLM — use DistLLM as a drop-in `LLM`, `Embeddings`, and `Tool` provider in LlamaIndex.

## Installation

```bash
pip install distllm-llamaindex
```

Or from source:

```bash
pip install -e integrations/llamaindex
```

## Quick Start

```python
from distllm_llamaindex import DistLLM

llm = DistLLM(model="distributed-llm", base_url="http://localhost:8000")
response = llm.complete("What is distributed inference?")
print(response.text)
```

## Chat

```python
from distllm_llamaindex import DistLLM
from llama_index.core.llms import ChatMessage, MessageRole

llm = DistLLM(model="distributed-llm", base_url="http://localhost:8000")
response = llm.chat([
    ChatMessage(role=MessageRole.USER, content="Hello!"),
])
print(response.message.content)
```

## Streaming

```python
for chunk in llm.stream_chat([
    ChatMessage(role=MessageRole.USER, content="Tell me a story.")
]):
    print(chunk.delta, end="", flush=True)
```

## Embeddings

```python
from distllm_llamaindex import DistLLMEmbeddings

embeddings = DistLLMEmbeddings(base_url="http://localhost:8000")
vectors = embeddings.get_text_embedding_batch(["Hello world", "Goodbye"])
print(len(vectors[0]))  # embedding dimension
```

## Tools

```python
from distllm_llamaindex import DistLLMToolProvider

provider = DistLLMToolProvider(base_url="http://localhost:8000")
tools = provider.get_tools()
```
