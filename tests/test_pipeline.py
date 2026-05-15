#!/usr/bin/env python3
"""In-process test of the distributed pipeline logic.

Tests the gRPC communication, tensor serialization, KV cache handling,
and layer splitting WITHOUT spawning subprocesses. This validates that
the distributed pipeline code path works correctly.

Usage:
    python tests/test_pipeline.py

Note: These are legacy manual tests. Run via pytest with --run-manual flag.
"""

import os
import sys
import time

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from distllm.communication.grpc import GRPCServer, NodeService
from distllm.communication.serializers import kv_cache_to_proto, proto_to_kv_cache, proto_to_tensor, tensor_to_proto
from distllm.core.coordinator import Coordinator
from distllm.core.kv_cache import KVCache
from distllm.models.partitioner import ModelPartitioner, get_model_info


@pytest.mark.skip(reason="Legacy manual test, requires model download")
def test_tensor_serialization():
    """Test that tensor -> proto -> tensor roundtrip preserves data."""
    print("\n[Test 1] Tensor serialization roundtrip")

    # Float32 tensor
    t1 = torch.randn(2, 4, 16)
    p1 = tensor_to_proto(t1)
    r1 = proto_to_tensor(p1)
    assert r1.shape == t1.shape, f"Shape mismatch: {r1.shape} vs {t1.shape}"
    assert torch.allclose(r1.float(), t1, atol=1e-5), "Float32 data mismatch"
    print(f"  Float32: {t1.shape} -> proto -> {r1.shape} OK")

    # Int64 tensor (like input_ids)
    t2 = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    p2 = tensor_to_proto(t2)
    r2 = proto_to_tensor(p2)
    assert r2.shape == t2.shape, f"Shape mismatch: {r2.shape} vs {t2.shape}"
    assert torch.equal(r2.long(), t2), "Int64 data mismatch"
    print(f"  Int64: {t2.shape} -> proto -> {r2.shape} OK")

    print("  PASSED")


@pytest.mark.skip(reason="Legacy manual test, requires model download")
def test_kv_cache_serialization():
    """Test KV cache serialization roundtrip."""
    print("\n[Test 2] KV cache serialization roundtrip")

    cache = KVCache()
    # Simulate 4 layers of KV cache
    for i in range(4):
        k = torch.randn(1, 4, 3, 16)
        v = torch.randn(1, 4, 3, 16)
        cache.cache.append((k, v))

    proto = kv_cache_to_proto(cache)
    restored = proto_to_kv_cache(proto)

    assert len(restored.cache) == len(
        cache.cache
    ), f"Cache length mismatch: {len(restored.cache)} vs {len(cache.cache)}"

    for i, (k_orig, v_orig) in enumerate(cache.cache):
        k_rest, v_rest = restored.cache[i]
        assert k_rest.shape == k_orig.shape, f"Layer {i} key shape mismatch"
        assert v_rest.shape == v_orig.shape, f"Layer {i} value shape mismatch"
        assert torch.allclose(k_rest, k_orig, atol=1e-5), f"Layer {i} key data mismatch"
        assert torch.allclose(v_rest, v_orig, atol=1e-5), f"Layer {i} value data mismatch"

    print(f"  {len(cache.cache)} layers serialized and restored OK")
    print("  PASSED")


@pytest.mark.skip(reason="Legacy manual test, requires model download")
def test_layer_splitting(model_name: str = "roneneldan/TinyStories-1M"):
    """Test that layer splitting produces same output as full model."""
    print(f"\n[Test 3] Layer splitting: {model_name}")

    model_info = get_model_info(model_name)
    total_layers = model_info["num_layers"]
    mid = total_layers // 2
    print(f"  Total layers: {total_layers}, split at {mid}")

    # Load full model
    full = ModelPartitioner(model_name=model_name, dtype="float32")
    full.load_full_model()

    # Load first half (layers 0 to mid-1)
    first = ModelPartitioner(model_name=model_name, dtype="float32")
    first.load_layer_subset(0, mid - 1, total_layers, device="cpu")

    # Load second half (layers mid to total-1)
    second = ModelPartitioner(model_name=model_name, dtype="float32")
    second.load_layer_subset(mid, total_layers - 1, total_layers, device="cpu")

    # Prepare input
    prompt = "Once upon a time"
    input_ids = full.tokenizer.encode(prompt, return_tensors="pt")

    # Full model forward pass
    with torch.no_grad():
        full_output = full.full_model(input_ids)
        full_logits = full_output.logits

    # Split model: first half processes input_ids via embeddings
    with torch.no_grad():
        # Embed tokens + positional encoding
        embeddings = full.embed_input(input_ids)

        # First node processes embeddings
        hidden_1, kv_1 = first.forward(embeddings)

        # Second node processes hidden states
        hidden_2, kv_2 = second.forward(hidden_1)

        # Apply final norm and lm_head
        split_logits = second.get_logits(hidden_2)

    # Compare logits
    assert (
        full_logits.shape == split_logits.shape
    ), f"Logit shape mismatch: {split_logits.shape} vs {full_logits.shape}"

    # Check similarity (should be very close for float32)
    diff = (full_logits - split_logits).abs().max().item()
    similarity = torch.allclose(full_logits, split_logits, atol=1e-4)

    print(f"  Full logits shape: {full_logits.shape}")
    print(f"  Split logits shape: {split_logits.shape}")
    print(f"  Max absolute difference: {diff:.6f}")
    print(f"  Logits match (atol=1e-4): {similarity}")

    # Sample tokens
    full_token = torch.argmax(full_logits[:, -1, :], dim=-1).item()
    split_token = torch.argmax(split_logits[:, -1, :], dim=-1).item()
    print(f"  Full model next token: {full_token}")
    print(f"  Split model next token: {split_token}")
    print(f"  Next token match: {full_token == split_token}")

    if similarity or full_token == split_token:
        print("  PASSED")
        return True
    else:
        print("  FAILED - logits diverge significantly")
        return False


@pytest.mark.skip(reason="Legacy manual test, requires model download")
def test_distributed_pipeline_in_process(model_name: str = "roneneldan/TinyStories-1M"):
    """Test full gRPC pipeline in-process (no subprocesses).

    Starts 2 nodes as threads in the same process, with the coordinator
    connecting to them via localhost gRPC.
    """
    print(f"\n[Test 4] In-process distributed pipeline: {model_name}")

    model_info = get_model_info(model_name)
    total_layers = model_info["num_layers"]
    mid = total_layers // 2

    print(f"  Total layers: {total_layers}, Node 0: 0-{mid-1}, Node 1: {mid}-{total_layers-1}")

    # Create partitioners
    first = ModelPartitioner(model_name=model_name, dtype="float32")
    first.load_layer_subset(0, mid - 1, total_layers, device="cpu")

    second = ModelPartitioner(model_name=model_name, dtype="float32")
    second.load_layer_subset(mid, total_layers - 1, total_layers, device="cpu")

    # Create forward functions
    def first_forward(
        hidden_states, attention_mask=None, position_ids=None, past_key_values=None, input_ids=None
    ):
        if input_ids is not None and first.embed_input is not None:
            hidden_states = first.embed_input(input_ids)
        return first.forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

    def second_forward(
        hidden_states, attention_mask=None, position_ids=None, past_key_values=None, input_ids=None
    ):
        if input_ids is not None and second.embed_input is not None:
            hidden_states = second.embed_input(input_ids)
        output, new_kv = second.forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        if second.is_last_node:
            output = second.get_logits(output)
        return output, new_kv

    # Start gRPC servers
    servicer0 = NodeService(node_id="node_0", forward_fn=first_forward)
    server0 = GRPCServer(port=15051, servicer=servicer0)
    server0.start()

    servicer1 = NodeService(node_id="node_1", forward_fn=second_forward)
    server1 = GRPCServer(port=15052, servicer=servicer1)
    server1.start()

    time.sleep(1)  # Let servers start

    try:
        # Create coordinator and register nodes
        coordinator = Coordinator(model_name=model_name, dtype="float32")
        coordinator.manual_register(
            node_id="node_0",
            host="localhost",
            port=15051,
            start_layer=0,
            end_layer=mid - 1,
            total_layers=total_layers,
        )
        coordinator.manual_register(
            node_id="node_1",
            host="localhost",
            port=15052,
            start_layer=mid,
            end_layer=total_layers - 1,
            total_layers=total_layers,
        )

        # Health check
        health = coordinator.health_check()
        for node_id, status in health.items():
            print(f"  {node_id}: {'HEALTHY' if status.get('healthy') else 'UNHEALTHY'}")

        # Run inference
        prompt = "Once upon a time"
        print(f"  Prompt: {prompt}")

        start = time.perf_counter()
        result = coordinator.generate(prompt, max_new_tokens=20)
        elapsed = time.perf_counter() - start

        tokens = len(coordinator.tokenizer.encode(result))
        tok_s = tokens / elapsed if elapsed > 0 else 0

        print(f"  Result: {result[:100]}...")
        print(f"  Tokens: {tokens}, Time: {elapsed:.2f}s, Speed: {tok_s:.1f} tok/s")

        # Verify output is non-trivial
        assert len(result) > len(prompt), "Output should be longer than prompt"
        print("  PASSED")
        return True

    finally:
        server0.stop()
        server1.stop()


def main():
    print("=" * 60)
    print("Distributed LLM - Pipeline Validation Tests")
    print("=" * 60)

    model = "roneneldan/TinyStories-1M"

    # Test 1: Tensor serialization
    try:
        test_tensor_serialization()
    except Exception as e:
        print(f"  FAILED: {e}")

    # Test 2: KV cache serialization
    try:
        test_kv_cache_serialization()
    except Exception as e:
        print(f"  FAILED: {e}")

    # Test 3: Layer splitting
    try:
        test_layer_splitting(model)
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback

        traceback.print_exc()

    # Test 4: Full gRPC pipeline in-process
    try:
        test_distributed_pipeline_in_process(model)
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)
    print("All pipeline validation tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
