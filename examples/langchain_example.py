"""LangChain integration with Distributed LLM.

This example shows how to use LangChain with the DistLLM OpenAI-compatible API.

Requirements:
    pip install langchain-openai langchain-core

Usage:
    # Start the API server first:
    distllm-api --model roneneldan/TinyStories-1M --local

    # Then run this example:
    python examples/langchain_example.py
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


def main():
    # Connect to distributed LLM via OpenAI-compatible API
    llm = ChatOpenAI(
        model="distributed-llm",
        openai_api_base="http://localhost:8000/v1",
        openai_api_key="not-needed",  # Or set API_KEY env var
        temperature=0.7,
        max_tokens=256,
    )

    # Simple chat
    print("Testing LangChain + DistLLM integration...\n")

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Explain what distributed inference is in simple terms."),
    ]

    response = llm.invoke(messages)
    print("Response:")
    print(response.content)

    # Streaming example
    print("\n\nStreaming response:")
    for chunk in llm.stream([HumanMessage(content="Tell me a short story about a robot.")]):
        print(chunk.content, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
