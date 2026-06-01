"""Multi-node distributed inference test.

Tests:
1. Single-node (coordinator only)
2. Multi-node on same machine (simulated)
3. Multi-node across network (requires second device)

Run: python tests/e2e/test_multi_node.py
"""

import sys
import time
import threading
import requests


def test_single_node():
    """Test 1: Single-node inference (coordinator only)."""
    print("\n[TEST 1] Single-node inference...")
    print("  Starting coordinator with local model...")

    from distllm.core.coordinator import Coordinator, CoordinatorConfig

    config = CoordinatorConfig(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        port=50050,
    )
    coord = Coordinator(config=config)
    coord.load_local_model()

    print(f"  Model loaded: {coord.model_name}")
    print(f"  Tokenizer: {type(coord.tokenizer).__name__}")

    # Test inference
    result = coord.generate(
        prompt="What is distributed computing?",
        max_new_tokens=30,
        temperature=0.7,
    )

    print(f"  Generated: {result[:100]}...")
    print("  [OK] Single-node inference works")
    return coord


def test_multi_node_simulation():
    """Test 2: Simulate multi-node on same machine."""
    print("\n[TEST 2] Multi-node simulation (same machine)...")
    print("  This simulates what happens with multiple workers:")
    print("  - Coordinator splits model layers across nodes")
    print("  - Each node processes its assigned layers")
    print("  - Results are combined for final output")

    from distllm.core.coordinator import Coordinator, CoordinatorConfig

    config = CoordinatorConfig(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        port=50050,
    )
    coord = Coordinator(config=config)

    # Simulate registering multiple nodes
    print("  Simulating 2-node cluster...")
    coord.manual_register(
        node_id="node-0",
        host="localhost",
        port=50051,
        start_layer=0,
        end_layer=13,
        total_layers=28,
    )
    coord.manual_register(
        node_id="node-1",
        host="localhost",
        port=50052,
        start_layer=14,
        end_layer=27,
        total_layers=28,
    )

    print(f"  Nodes registered: {list(coord.nodes.keys())}")
    print(f"  Node order: {coord.node_order}")
    print("  [OK] Multi-node setup simulated")


def test_network_connectivity():
    """Test 3: Check network connectivity to other devices."""
    print("\n[TEST 3] Network connectivity check...")

    import socket
    import subprocess

    # Get local IP
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"  Your IP: {local_ip}")

    # Check if we can reach common local IPs
    print("  Scanning local network for other devices...")
    found_devices = []

    for i in range(1, 20):
        ip = f"192.168.1.{i}"
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.1)
            result = sock.connect_ex((ip, 50050))
            if result == 0:
                found_devices.append(ip)
            sock.close()
        except:
            pass

    if found_devices:
        print(f"  Found devices with port 50050 open: {found_devices}")
    else:
        print("  No other devices found with port 50050 open")
        print("  To connect another device:")
        print(f"    1. On the other device, run: python -m distllm.cli.main system run --coordinator-host {local_ip}")
        print("    2. Make sure firewall allows port 50050")


def test_api_with_multi_node():
    """Test 4: API with multi-node setup."""
    print("\n[TEST 4] API multi-node test...")

    try:
        # Check if API is running
        resp = requests.get("http://127.0.0.1:8000/health", timeout=2)
        if resp.status_code == 200:
            data = resp.json()
            print(f"  API running: {data.get('status')}")
            print(f"  Nodes: {data.get('nodes', 0)}")
            print(f"  Model: {data.get('model', 'unknown')}")

            if data.get('nodes', 0) > 0:
                print("  [OK] Multi-node cluster is running")
            else:
                print("  [INFO] No worker nodes connected yet")
                print("  To add workers, run on other machines:")
                print("    python -m distllm.cli.main system run --coordinator-host <this-ip>")
        else:
            print("  [SKIP] API not running")
    except Exception as e:
        print(f"  [SKIP] API not accessible: {e}")


def main():
    """Run multi-node tests."""
    print("=" * 60)
    print("DistLLM Multi-Node Test")
    print("=" * 60)

    # Test 1: Single node
    try:
        coord = test_single_node()
    except Exception as e:
        print(f"  [FAIL] Single-node test failed: {e}")
        coord = None

    # Test 2: Multi-node simulation
    try:
        test_multi_node_simulation()
    except Exception as e:
        print(f"  [FAIL] Multi-node simulation failed: {e}")

    # Test 3: Network connectivity
    try:
        test_network_connectivity()
    except Exception as e:
        print(f"  [FAIL] Network test failed: {e}")

    # Test 4: API multi-node
    try:
        test_api_with_multi_node()
    except Exception as e:
        print(f"  [FAIL] API test failed: {e}")

    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print()
    print("To test REAL distributed inference:")
    print()
    print("1. On THIS machine (coordinator):")
    print("   $env:API_KEY='test-key-123'")
    print("   python -m distllm.cli.main system api --model Qwen/Qwen2.5-0.5B-Instruct --local")
    print()
    print("2. On ANOTHER machine (worker):")
    print("   pip install distllm")
    print("   python -m distllm.cli.main system run --coordinator-host 192.168.1.59")
    print()
    print("3. Test the distributed cluster:")
    print("   curl -X POST http://192.168.1.59:8000/v1/chat/completions \\")
    print("     -H 'Authorization: Bearer test-key-123' \\")
    print("     -H 'Content-Type: application/json' \\")
    print("     -d '{\"model\":\"Qwen/Qwen2.5-0.5B-Instruct\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}'")
    print()


if __name__ == "__main__":
    main()
