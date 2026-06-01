"""Quick test: 2-laptop distributed cluster.

Run this on BOTH laptops to verify connectivity and inference.

Usage:
    python tests/manual/quick_2node_test.py --mode coordinator
    python tests/manual/quick_2node_test.py --mode worker --coordinator 192.168.1.59
    python tests/manual/quick_2node_test.py --mode test --coordinator 192.168.1.59
"""

import argparse
import socket
import sys
import time


def get_local_ip():
    """Get the local Wi-Fi IP address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def check_port(host, port, timeout=3):
    """Check if a port is reachable."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        result = sock.connect_ex((host, port))
        return result == 0
    except Exception:
        return False
    finally:
        sock.close()


def test_coordinator(host, port=8000):
    """Test the coordinator API."""
    import httpx

    print(f"\n{'='*50}")
    print(f"  Testing Coordinator at {host}:{port}")
    print(f"{'='*50}\n")

    client = httpx.Client(base_url=f"http://{host}:{port}", timeout=30)

    # Test 1: Health
    try:
        r = client.get("/health")
        print(f"✅ Health: {r.json()}")
    except Exception as e:
        print(f"❌ Health failed: {e}")
        return False

    # Test 2: Models
    try:
        r = client.get("/v1/models")
        models = r.json().get("data", [])
        print(f"✅ Models: {len(models)} available")
        for m in models[:3]:
            print(f"   - {m['id']}")
    except Exception as e:
        print(f"⚠️  Models: {e}")

    # Test 3: Chat completion
    print(f"\n🔄 Sending test prompt...")
    try:
        start = time.time()
        r = client.post("/v1/chat/completions", json={
            "model": "distributed-llm",
            "messages": [{"role": "user", "content": "Say hello in one word."}],
            "max_tokens": 10,
            "temperature": 0.1,
        })
        elapsed = time.time() - start
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        tokens = data.get("usage", {}).get("completion_tokens", 0)
        print(f"✅ Response: {content}")
        print(f"   Tokens: {tokens}, Time: {elapsed:.2f}s, Speed: {tokens/elapsed:.1f} tok/s")
        return True
    except Exception as e:
        print(f"❌ Chat failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="2-Laptop DistLLM Test")
    parser.add_argument("--mode", choices=["info", "test"], default="info")
    parser.add_argument("--coordinator", default="192.168.1.59", help="Coordinator IP")
    parser.add_argument("--port", type=int, default=8000, help="API port")
    args = parser.parse_args()

    local_ip = get_local_ip()
    print(f"🖥️  This laptop: {local_ip} ({socket.gethostname()})")

    if args.mode == "info":
        print(f"\n📋 Network Info:")
        print(f"   Local IP: {local_ip}")
        print(f"   Hostname: {socket.gethostname()}")
        print(f"\n📋 To set up:")
        print(f"   1. On THIS laptop (coordinator):")
        print(f"      cd D:\\distributed-llm")
        print(f"      pip install -e . --no-deps")
        print(f"      distllm run --model roneneldan/TinyStories-1M --local --port 8000")
        print(f"\n   2. On OTHER laptop (worker):")
        print(f"      pip install distributed-llm")
        print(f"      distllm-node --coordinator {local_ip}:50050 --port 50051")
        print(f"\n   3. Test connection:")
        print(f"      python {sys.argv[0]} --mode test --coordinator {local_ip}")

    elif args.mode == "test":
        # Check connectivity
        print(f"\n🔍 Checking connectivity to {args.coordinator}:{args.port}...")
        if check_port(args.coordinator, args.port):
            print(f"✅ Port {args.port} is reachable")
            test_coordinator(args.coordinator, args.port)
        else:
            print(f"❌ Port {args.port} is NOT reachable")
            print(f"\n   Troubleshooting:")
            print(f"   1. Is the coordinator running on {args.coordinator}?")
            print(f"   2. Are both laptops on the same Wi-Fi?")
            print(f"   3. Is the firewall allowing port {args.port}?")
            print(f"      Run: netsh advfirewall firewall add rule name='DistLLM' dir=in action=allow protocol=tcp localport={args.port}")


if __name__ == "__main__":
    main()
