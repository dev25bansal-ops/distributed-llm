"""Interactive chat command for DistLLM CLI."""

import httpx
from rich.console import Console
from rich.markdown import Markdown

from distllm.cli.http_client import client_for


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
    console.print("\n[bold blue]DistLLM Interactive Chat[/bold blue]")
    console.print(f"Model: {model}")
    console.print(f"Server: {base_url}")
    console.print("Type 'quit' or 'exit' to stop, 'clear' to reset\n")

    # Fail before the prompt rather than after the first message: the old code
    # entered the loop, accepted input, then errored, which reads as though the
    # message had been sent.
    try:
        with client_for(host, port, timeout=10.0) as probe:
            probe.get("/health").raise_for_status()
    except (httpx.HTTPError, OSError):
        console.print(
            f"[red]Cannot reach the server at {base_url}.[/red]\n"
            "Start one with:\n"
            "  [cyan]distllm cluster start --model <model>[/cyan]\n"
            "or point at another server with [cyan]--host/--port[/cyan]."
        )
        return

    conversation = []

    try:
        with client_for(host, port, timeout=120.0) as client:
            while True:
                try:
                    # input() is not a Rich renderer: a markup prompt is shown
                    # to the user as the literal string "[bold You:[/bold]".
                    prompt = input("\nYou: ").strip()
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
                        "/v1/chat/completions",
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

                    console.print("\n[bold green]Assistant:[/bold green]")
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
        # CQ-08: failure is surfaced via console output; not propagated.
        console.print(f"[red]Error:[/red] {e}")
