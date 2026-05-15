"""CrewAI integration with Distributed LLM.

This example shows how to use CrewAI with the DistLLM OpenAI-compatible API.

Requirements:
    pip install crewai langchain-openai

Usage:
    # Start the API server first:
    distllm-api --model roneneldan/TinyStories-1M --local

    # Then run this example:
    python examples/crewai_example.py
"""

from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew, Process


def main():
    # Configure LLM for CrewAI
    llm = ChatOpenAI(
        model="distributed-llm",
        openai_api_base="http://localhost:8000/v1",
        openai_api_key="not-needed",
        temperature=0.7,
        max_tokens=256,
    )

    print("Testing CrewAI + DistLLM integration...\n")

    # Create agents
    researcher = Agent(
        role="Research Analyst",
        goal="Research and analyze the topic of distributed computing",
        backstory="You are an expert research analyst with deep knowledge of distributed systems.",
        llm=llm,
        verbose=True,
    )

    writer = Agent(
        role="Technical Writer",
        goal="Write clear and concise explanations based on research findings",
        backstory="You excel at transforming complex technical concepts into easy-to-understand content.",
        llm=llm,
        verbose=True,
    )

    # Create tasks
    research_task = Task(
        description="Research the benefits and challenges of distributed inference for LLMs",
        expected_output="A detailed analysis of distributed inference",
        agent=researcher,
    )

    writing_task = Task(
        description="Write a summary of the research findings in plain language",
        expected_output="A clear summary suitable for a technical blog post",
        agent=writer,
    )

    # Create and run the crew
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        process=Process.sequential,
    )

    result = crew.kickoff()
    print("\n\nFinal Result:")
    print(result)


if __name__ == "__main__":
    main()
