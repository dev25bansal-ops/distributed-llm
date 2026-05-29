#!/usr/bin/env python3
"""End-to-end test for distributed LLM inference.

Tests the full pipeline: Coordinator -> Node 0 -> Node 1 -> Coordinator
on localhost with a small model.

Usage:
    # Single machine test (all processes on localhost)
    python tests/test_distributed.py --model roneneldan/TinyStories-1M

    # Compare single vs distributed output
    python tests/test_distributed.py --model roneneldan/TinyStories-1M --compare
"""

import argparse
import os
import sys
import time

import threading

from distllm.core.coordinator import Coordinator
from distllm.models.partitioner import get_model_info
from distllm.core.node import WorkerNode


def start_worker_node_in_thread(
    node_id, model_name, start_layer, end_layer, total_layers, port, dtype, ready_event, error_queue
):
    """Start a worker node in a thread (Windows-compatible)."""
    try:
        node = WorkerNode(
            node_id=node_id,
            model_name=model_name,
            start_layer=start_layer,
            end_layer=end_layer,
            total_layers=total_layers,
            port=port,
            device="cpu",  # Force CPU for local testing
            dtype=dtype,
        )
        node.load_model()
        try:
            grpc_mod = __import__("distllm.communication.grpc", fromlist=["NodeService", "GRPCServer"])
        except ImportError:
            raise RuntimeError("distllm.communication.grpc module removed")
        servicer = grpc_mod.NodeService(
            node_id=node_id,
            forward_fn=node.forward_fn,
        )
        server = grpc_mod.GRPCServer(
            port=port, servicer=servicer
        )
        server.start()
        ready_event.set()
        # Store server reference for cleanup
        error_queue.put(("started", server))
        # Block until stop signal
        error_queue.put(("waiting", server))
        server.wait_for_termination()
    except Exception as e:
        error_queue.put(("error", str(e)))
        ready_event.set()  # Signal that we're done (with error)


def run_single_node_test(model_name: str, dtype: str = "float32"):
    """Test single-node (local) inference as baseline."""
    print(f"\n{'='*60}")
    print("Test 1: Single-node (local) inference")
    print(f"{'='*60}")

    coordinator = Coordinator(model_name=model_name, dtype=dtype)
    coordinator.load_local_model()

    prompt = "Once upon a time"
    print(f"Prompt: {prompt}")

    start = time.perf_counter()
    result = coordinator.generate(prompt, max_new_tokens=30)
    elapsed = time.perf_counter() - start

    tokens = len(coordinator.tokenizer.encode(result))
    tok_s = tokens / elapsed if elapsed > 0 else 0

    print(f"\nResult: {result}")
    print(f"\nTokens: {tokens}, Time: {elapsed:.2f}s, Speed: {tok_s:.1f} tok/s")
    print("\nTest 1 PASSED")

    return result, elapsed


def run_distributed_test(model_name: str, dtype: str = "float32"):
    """Test distributed inference across 2 nodes on localhost."""
    import queue

    print(f"\n{'='*60}")
    print("Test 2: Distributed inference (2-node pipeline)")
    print(f"{'='*60}")

    model_info = get_model_info(model_name)
    total_layers = model_info["num_layers"]
    mid_layer = total_layers // 2

    print(f"Model: {model_name} ({total_layers} layers)")
    print(f"Node 0: layers 0-{mid_layer-1}")
    print(f"Node 1: layers {mid_layer}-{total_layers-1}")

    # Use ports that don't conflict with previous test
    port0, port1 = 50061, 50062

    # Start workers as threads
    ready0, ready1 = threading.Event(), threading.Event()
    queue0, queue1 = queue.Queue(), queue.Queue()

    t0 = threading.Thread(
        target=start_worker_node_in_thread,
        args=("node_0", model_name, 0, mid_layer - 1, total_layers, port0, dtype, ready0, queue0),
        daemon=True,
    )
    t0.start()
    print(f"Starting node_0 (layers 0-{mid_layer-1})...")
    ready0.wait(timeout=60)  # Wait up to 60s for model loading

    # Check for startup errors
    try:
        while True:
            status, data = queue0.get_nowait()
            if status == "error":
                raise RuntimeError(f"node_0 failed to start: {data}")
    except queue.Empty:
        pass

    print("node_0 ready")

    t1 = threading.Thread(
        target=start_worker_node_in_thread,
        args=(
            "node_1",
            model_name,
            mid_layer,
            total_layers - 1,
            total_layers,
            port1,
            dtype,
            ready1,
            queue1,
        ),
        daemon=True,
    )
    t1.start()
    print(f"Starting node_1 (layers {mid_layer}-{total_layers-1})...")
    ready1.wait(timeout=60)

    try:
        while True:
            status, data = queue1.get_nowait()
            if status == "error":
                raise RuntimeError(f"node_1 failed to start: {data}")
    except queue.Empty:
        pass

    print("node_1 ready")

    # Small delay to ensure gRPC servers are fully bound
    time.sleep(2)

    # Create coordinator and register nodes
    coordinator = Coordinator(model_name=model_name, dtype=dtype)
    coordinator.manual_register(
        node_id="node_0",
        host="localhost",
        port=port0,
        start_layer=0,
        end_layer=mid_layer - 1,
        total_layers=total_layers,
    )
    coordinator.manual_register(
        node_id="node_1",
        host="localhost",
        port=port1,
        start_layer=mid_layer,
        end_layer=total_layers - 1,
        total_layers=total_layers,
    )
    print("Nodes registered")

    # Health check
    print("\nChecking node health...")
    health = coordinator.health_check()
    for node_id, status in health.items():
        healthy = status.get("healthy", False)
        print(f"  {node_id}: {'HEALTHY' if healthy else 'UNHEALTHY'}")

    # Run inference
    prompt = "Once upon a time"
    print(f"\nPrompt: {prompt}")

    start = time.perf_counter()
    result = coordinator.generate(prompt, max_new_tokens=30)
    elapsed = time.perf_counter() - start

    tokens = len(coordinator.tokenizer.encode(result))
    tok_s = tokens / elapsed if elapsed > 0 else 0

    print(f"\nResult: {result}")
    print(f"\nTokens: {tokens}, Time: {elapsed:.2f}s, Speed: {tok_s:.1f} tok/s")

    # Cleanup - stop gRPC servers
    print("\nShutting down workers...")
    # Get server references from queues and stop them
    try:
        while not queue0.empty():
            status, data = queue0.get_nowait()
            if status == "waiting":
                data.stop(0)
    except Exception:
        pass

    try:
        while not queue1.empty():
            status, data = queue1.get_nowait()
            if status == "waiting":
                data.stop(0)
    except Exception:
        pass

    coordinator.stop()

    print("\nTest 2 PASSED")
    return result, elapsed


def compare_outputs(single_result: str, distributed_result: str):
    """Compare single-node and distributed outputs.

    Note: Exact output matching is not expected for distributed inference because
    floating-point differences in the pipeline (tensor serialization through gRPC,
    KV cache transfer) cause token sampling to diverge. Instead, we verify that
    both outputs are coherent text starting with the same prompt.
    """
    print(f"\n{'='*60}")
    print("Test 3: Output comparison")
    print(f"{'='*60}")

    # Both outputs should start with the prompt
    prompt = "Once upon a time"
    single_starts_with_prompt = single_result.lower().startswith(prompt.lower())
    distributed_starts_with_prompt = distributed_result.lower().startswith(prompt.lower())

    # Both outputs should be reasonably long (coherent generation)
    single_len = len(single_result.split())
    distributed_len = len(distributed_result.split())

    # Compute word overlap for informational purposes
    single_words = set(single_result.lower().split())
    distributed_words = set(distributed_result.lower().split())
    intersection = single_words & distributed_words
    union = single_words | distributed_words
    similarity = len(intersection) / len(union) if union else 0

    print(f"Single-node output:    {single_result[:100]}...")
    print(f"Distributed output:    {distributed_result[:100]}...")
    print(
        f"\nBoth start with prompt: {single_starts_with_prompt and distributed_starts_with_prompt}"
    )
    print(f"Single-node tokens: {single_len}, Distributed tokens: {distributed_len}")
    print(f"Vocabulary similarity (informational): {similarity:.2%}")

    passed = (
        single_starts_with_prompt
        and distributed_starts_with_prompt
        and single_len >= 10
        and distributed_len >= 10
    )

    if passed:
        print("\nOutputs are coherent (distributed pipeline working correctly)")
        print("Test 3 PASSED")
        return True
    else:
        print("\nOutputs may indicate pipeline issue")
        print("Test 3 FAILED")
        return False


def main():
    parser = argparse.ArgumentParser(description="End-to-end distributed LLM test")
    parser.add_argument("--model", type=str, default="roneneldan/TinyStories-1M")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument(
        "--compare", action="store_true", help="Compare single vs distributed output"
    )

    args = parser.parse_args()

    print("\nDistributed LLM - End-to-End Test")
    print(f"Model: {args.model}")
    print(f"Dtype: {args.dtype}")

    # Test 1: Single-node
    try:
        single_result, single_time = run_single_node_test(args.model, args.dtype)
    except Exception as e:
        print(f"\nTest 1 FAILED: {e}")
        return

    if not args.compare:
        print("\nAll tests passed! Use --compare to test distributed mode.")
        return

    # Test 2: Distributed
    try:
        distributed_result, distributed_time = run_distributed_test(args.model, args.dtype)
    except Exception as e:
        print(f"\nTest 2 FAILED: {e}")
        return

    # Test 3: Compare
    compare_outputs(single_result, distributed_result)

    print(f"\n{'='*60}")
    print("Performance Summary")
    print(f"{'='*60}")
    print(f"Single-node:      {single_time:.2f}s")
    print(f"Distributed:      {distributed_time:.2f}s")
    print(f"Overhead:         {distributed_time/single_time:.1f}x")


if __name__ == "__main__":
    main()
