"""Client interface for distributed LLM inference."""

import argparse
import time
from distllm.core.coordinator import Coordinator


def interactive_chat(coordinator):
    """Interactive chat mode with the distributed LLM."""
    print("=" * 60)
    print("Distributed LLM - Interactive Chat")
    print(f"Model: {coordinator.model_name}")
    print("Type 'quit' or 'exit' to stop, 'clear' to reset conversation")
    print("=" * 60)

    conversation = []

    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if prompt.lower() in ('quit', 'exit', 'q'):
            print("Goodbye!")
            break

        if prompt.lower() == 'clear':
            conversation = []
            print("Conversation cleared.")
            continue

        if not prompt:
            continue

        conversation.append({"role": "user", "content": prompt})
        full_prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation])

        print("\nAssistant: ", end="", flush=True)

        start_time = time.time()
        try:
            result = coordinator.generate(
                full_prompt,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
            )

            assistant_response = result[len(full_prompt):] if result.startswith(full_prompt) else result
            print(assistant_response.strip())

            elapsed = time.time() - start_time
            tokens = len(coordinator.tokenizer.encode(assistant_response))
            speed = tokens / elapsed if elapsed > 0 else 0
            print(f"\n[{tokens} tokens in {elapsed:.1f}s | {speed:.1f} tokens/s]")

            conversation.append({"role": "assistant", "content": assistant_response.strip()})

        except Exception as e:
            print(f"\nError: {e}")


def single_prompt(coordinator, prompt: str, max_tokens: int = 128, temperature: float = 0.7):
    """Generate text for a single prompt."""
    print(f"Model: {coordinator.model_name}")
    print(f"Prompt: {prompt}")
    print("\nGenerating...\n")

    start_time = time.time()
    result = coordinator.generate(
        prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
    )
    elapsed = time.time() - start_time

    print(result)

    tokens = len(coordinator.tokenizer.encode(result))
    speed = tokens / elapsed if elapsed > 0 else 0
    print(f"\n---\nGenerated {tokens} tokens in {elapsed:.1f}s ({speed:.1f} tokens/s)")


def show_health(coordinator):
    """Show health status of all nodes."""
    if not coordinator.nodes:
        print("No remote nodes registered.")
        if coordinator.local_partitioner:
            print("Running in local mode with full model loaded.")
        return

    print("Node Health Status:")
    print("-" * 50)
    health = coordinator.health_check()
    for node_id, status in health.items():
        node = coordinator.nodes[node_id]
        health_str = "HEALTHY" if status.get("healthy") else "UNHEALTHY"
        mem_used = status.get("memory_used", 0) / 1e9
        mem_total = status.get("memory_total", 0) / 1e9
        print(f"  {node_id}: {health_str}")
        print(f"    Layers: {node.start_layer}-{node.end_layer}")
        print(f"    Address: {node.host}:{node.port}")
        if mem_total > 0:
            print(f"    Memory: {mem_used:.1f}/{mem_total:.1f} GB")
        if "error" in status:
            print(f"    Error: {status['error']}")


def main():
    parser = argparse.ArgumentParser(description="Distributed LLM Client")
    parser.add_argument("--model", type=str, default="roneneldan/TinyStories-1M", help="Model name")
    parser.add_argument("--chat", action="store_true", help="Interactive chat mode")
    parser.add_argument("--prompt", type=str, help="Single prompt (non-interactive)")
    parser.add_argument("--health", action="store_true", help="Show node health")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--local", action="store_true", help="Load model locally (single-node mode)")

    args = parser.parse_args()

    coordinator = Coordinator(
        model_name=args.model,
        dtype=args.dtype,
    )

    if args.local:
        print(f"Loading model locally: {args.model}")
        coordinator.load_local_model()
    else:
        print(f"Client ready for model: {args.model}")
        print("Note: Use --local to load model on this machine, or register remote nodes.")

    if args.health:
        show_health(coordinator)
    elif args.prompt:
        single_prompt(coordinator, args.prompt, args.max_tokens, args.temperature)
    elif args.chat:
        interactive_chat(coordinator)
    else:
        print("\nUsage:")
        print(f"  python -m distllm.cli.client --model {args.model} --local --chat")
        print(f"  python -m distllm.cli.client --model {args.model} --local --prompt \"Hello world\"")
        print(f"  python -m distllm.cli.client --model {args.model} --health")


if __name__ == "__main__":
    main()
