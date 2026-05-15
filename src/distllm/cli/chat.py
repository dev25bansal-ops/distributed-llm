"""Interactive chat command for DistLLM CLI."""

import httpx
from rich.console import Console
from rich.markdown import Markdown


def run_chat(
    model: str,
    host: str,
    port: int,
    max_tokens: int,
    temperature: float,
    console: Console,
):
    """Run interactive chat mode."""
    base_url = f"http://{host}:{port}"
    console.print(f"\n[bold blue]DistLLM Interactive Chat[/bold blue]")
    console.print(f"Model: {model}")
    console.print(f"Server: {base_url}")
    console.print("Type 'quit' or 'exit' to stop, 'clear' to reset\n")

    conversation = []

    try:
        with httpx.Client(timeout=120.0) as client:
            while True:
                try:
                    prompt = input("\n[bold You:[/bold] ").strip()
                except (EOFError, KeyboardInterrupt):
                    console.print("\nGoodbye!")
                    break

                if prompt.lower() in ("quit", "exit", "q"):
                    console.print("Goodbye!")
                    break

                if prompt.lower() == "clear":
                    conversation = []
                    console.print("Conversation cleared.")
                    continue

                if not prompt:
                    continue

                conversation.append({"role": "user", "content": prompt})

                try:
                    response = client.post(
                        f"{base_url}/v1/chat/completions",
                        json={
                            "model": model,
                            "messages": conversation,
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                        },
                    )
                    response.raise_for_status()

                    data = response.json()
                    assistant_msg = data["choices"][0]["message"]["content"]

                    console.print(f"\n[bold green]Assistant:[/bold green]")
                    console.print(Markdown(assistant_msg))

                    if "usage" in data:
                        tokens = data["usage"].get("completion_tokens", 0)
                        gen_time = data.get("generation_time", 0)
                        if gen_time > 0:
                            tps = tokens / gen_time
                            console.print(f"\n[dim]{tokens} tokens in {gen_time:.1f}s ({tps:.1f} tokens/s)[/dim]")

                    conversation.append({"role": "assistant", "content": assistant_msg})

                except httpx.HTTPError as e:
                    console.print(f"\n[red]Error:[/red] {e}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
