"""Agency Swarm / AutoGPT integration with Distributed LLM.

This example shows how to use Agency Swarm with the DistLLM
OpenAI-compatible API for multi-agent orchestration.

Agency Swarm is a framework for creating collaborative AI agents
that can communicate and delegate tasks to each other.

Requirements:
    pip install agency-swarm openai

Usage:
    # Start the API server first:
    distllm-api --model meta-llama/Llama-2-70b-hf --local

    # Then run this example:
    python examples/agency_swarm_example.py
"""

from agency_swarm import Agent, Agency
from openai import OpenAI


def main():
    # Configure OpenAI client to point to DistLLM
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed",  # Or set API_KEY env var
    )

    print("Testing Agency Swarm + DistLLM integration...\n")

    # Create specialized agents
    researcher = Agent(
        name="Researcher",
        description="Expert at researching and analyzing technical topics",
        instructions="""You are a research analyst. When given a topic:
        1. Break down the topic into key components
        2. Analyze each component thoroughly
        3. Provide evidence-based insights
        4. Summarize findings clearly""",
        model="distributed-llm",
        client=client,
    )

    writer = Agent(
        name="Writer",
        description="Expert at writing clear, engaging content",
        instructions="""You are a technical writer. When given research findings:
        1. Organize the information logically
        2. Write in clear, concise language
        3. Use appropriate formatting
        4. Ensure accuracy and completeness""",
        model="distributed-llm",
        client=client,
    )

    reviewer = Agent(
        name="Reviewer",
        description="Expert at reviewing and improving content",
        instructions="""You are a content reviewer. When given text:
        1. Check for accuracy and completeness
        2. Identify areas for improvement
        3. Suggest specific edits
        4. Verify tone and style consistency""",
        model="distributed-llm",
        client=client,
    )

    # Create agency with agent communication
    agency = Agency(
        agents=[researcher, writer, reviewer],
        shared_instructions="""You are part of a collaborative team.
        Communicate clearly with other agents.
        Provide detailed outputs for handoffs.
        Ask for clarification when needed.""",
    )

    # Run a collaborative task
    print("Running collaborative task: Write a technical blog post about distributed inference\n")
    result = agency.get_completion(
        "Write a short technical blog post explaining how distributed LLM inference works. "
        "The Researcher should analyze the key concepts, the Writer should draft the post, "
        "and the Reviewer should provide feedback.",
    )

    print("Final Result:")
    print(result)


if __name__ == "__main__":
    main()
