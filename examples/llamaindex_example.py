"""LlamaIndex integration with Distributed LLM.

This example shows how to use LlamaIndex with the DistLLM OpenAI-compatible API.

Requirements:
    pip install llama-index-llms-openai llama-index-core

Usage:
    # Start the API server first:
    distllm-api --model roneneldan/TinyStories-1M --local

    # Then run this example:
    python examples/llamaindex_example.py
"""

from llama_index.llms.openai import OpenAI
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.prompts import PromptTemplate


def main():
    # Configure LlamaIndex to use DistLLM
    Settings.llm = OpenAI(
        model="distributed-llm",
        api_base="http://localhost:8000/v1",
        api_key="not-needed",
        temperature=0.7,
        max_tokens=256,
    )

    print("Testing LlamaIndex + DistLLM integration...\n")

    # Simple query
    response = Settings.llm.complete("What is the capital of France?")
    print("Response:")
    print(response.text)

    # RAG example with documents
    print("\n\nTesting RAG with documents:")

    # Create a simple index from text
    documents = SimpleDirectoryReader(input_files=[]).load_data()  # Empty for demo

    # Or create from string
    from llama_index.core import Document

    doc = Document(text="Distributed inference splits a large language model across multiple machines. "
                        "Each machine runs a portion of the model's layers, allowing us to run models "
                        "that are too large for a single GPU.")

    index = VectorStoreIndex.from_documents([doc])
    query_engine = index.as_query_engine()

    response = query_engine.query("How does distributed inference work?")
    print("\nRAG Response:")
    print(response)


if __name__ == "__main__":
    main()
