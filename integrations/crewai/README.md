# distllm-crewai

CrewAI integration for DistLLM — use DistLLM as a drop-in tool provider, LLM, embedder, and knowledge source in CrewAI.

## Installation

```bash
pip install distllm-crewai
```

Or from source:

```bash
pip install -e integrations/crewai
```

## Quick Start

```python
from distllm_crewai import DistLLMToolProvider, DistLLMCrewLLM

# Create a CrewAI-compatible LLM
llm = DistLLMCrewLLM(model="distributed-llm", base_url="http://localhost:8000")

# Discover tools from the DistLLM API
provider = DistLLMToolProvider(base_url="http://localhost:8000")
tools = provider.get_tools()
```

## Tools

```python
from distllm_crewai import DistLLMToolProvider

provider = DistLLMToolProvider(base_url="http://localhost:8000")
tools = provider.get_tools()

# Use tools in a CrewAI agent
from crewai import Agent

agent = Agent(
    role="Researcher",
    goal="Research topics using DistLLM",
    backstory="Expert researcher",
    tools=tools,
    llm=DistLLMCrewLLM(base_url="http://localhost:8000")
)
```

## LLM

```python
from distllm_crewai import DistLLMCrewLLM

llm = DistLLMCrewLLM(
    model="distributed-llm",
    base_url="http://localhost:8000",
    temperature=0.7,
)
```

## Embeddings

```python
from distllm_crewai import DistLLMCrewEmbedder

embedder = DistLLMCrewEmbedder(base_url="http://localhost:8000")
embedding = embedder.embed_text("Hello world")
```

## Knowledge Source

```python
from distllm_crewai import DistLLMKnowledgeSource

source = DistLLMKnowledgeSource(
    source_id="my-kb",
    base_url="http://localhost:8000"
)
```
