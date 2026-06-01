"""Haystack integration with Distributed LLM.

This example shows how to use Haystack (by deepset) with the DistLLM
OpenAI-compatible API for RAG pipelines, document search, and
question answering.

Requirements:
    pip install haystack-ai openai

Usage:
    # Start the API server first:
    distllm-api --model meta-llama/Llama-2-7b-hf --local

    # Then run this example:
    python examples/haystack_example.py
"""

from haystack import Pipeline, component
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack import Document


def main():
    print("Testing Haystack + DistLLM integration...\n")

    # Configure generator to use DistLLM
    generator = OpenAIGenerator(
        api_base_url="http://localhost:8000/v1",
        api_key="not-needed",
        model="distributed-llm",
        generation_kwargs={
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
        },
    )

    # Simple generation (no RAG)
    print("=== Simple Generation ===\n")
    result = generator.run(prompt="Explain distributed inference in 3 bullet points.")
    print("Response:")
    for reply in result["replies"]:
        print(reply)
    print()

    # RAG pipeline
    print("=== RAG Pipeline ===\n")

    # Create document store with sample documents
    document_store = InMemoryDocumentStore()
    documents = [
        Document(content="Distributed inference splits a large language model across multiple GPU devices. "
                         "Each device processes a portion of the model's layers, enabling models too large "
                         "for a single GPU to run efficiently."),
        Document(content="Pipeline parallelism assigns consecutive layers to different devices. "
                         "The output of one device becomes the input of the next, forming a pipeline. "
                         "This is simpler than tensor parallelism but may have pipeline bubbles."),
        Document(content="Tensor parallelism splits individual layers across devices. "
                         "Each device computes a portion of the layer's weights in parallel. "
                         "This provides better GPU utilization but requires high-bandwidth interconnects."),
        Document(content="Speculative decoding uses a smaller 'draft' model to generate candidate tokens "
                         "quickly, then a larger 'target' model verifies them in parallel. "
                         "This can speed up inference by 2-3x without changing output quality."),
    ]
    document_store.write_documents(documents)

    # Build RAG pipeline
    retriever = InMemoryBM25Retriever(document_store=document_store)

    prompt_template = """Answer the question based on the provided context.
If the context doesn't contain enough information, say so.

Context:
{% for doc in documents %}
  {{ doc.content }}
{% endfor %}

Question: {{ question }}
Answer:"""

    prompt_builder = PromptBuilder(template=prompt_template)

    rag_pipeline = Pipeline()
    rag_pipeline.add_component("retriever", retriever)
    rag_pipeline.add_component("prompt_builder", prompt_builder)
    rag_pipeline.add_component("generator", generator)

    rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder.prompt", "generator.prompt")

    # Run RAG query
    question = "What is speculative decoding and how does it speed up inference?"
    print(f"Question: {question}\n")

    result = rag_pipeline.run({
        "retriever": {"query": question},
        "prompt_builder": {"question": question},
    })

    print("RAG Response:")
    for reply in result["generator"]["replies"]:
        print(reply)


if __name__ == "__main__":
    main()
