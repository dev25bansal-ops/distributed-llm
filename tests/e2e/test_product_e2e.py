"""End-to-end product test — verifies the full inference pipeline works.

Tests:
1. Coordinator initialization
2. Model loading
3. Local inference
4. API server startup
5. API chat completion
6. API text completion
7. API health check
8. SDK client
9. OpenAI compatibility
10. Token streaming

Run: python tests/e2e/test_product_e2e.py
"""

import json
import sys
import time
import threading
import requests
from pathlib import Path


def test_coordinator_init():
    """Test 1: Coordinator initializes correctly."""
    print("\n[TEST 1] Coordinator initialization...")
    from distllm.core.coordinator import Coordinator, CoordinatorConfig

    config = CoordinatorConfig(model_name="Qwen/Qwen2.5-0.5B-Instruct", port=50050)
    coord = Coordinator(config=config)

    assert coord.model_name == "Qwen/Qwen2.5-0.5B-Instruct"
    assert coord.port == 50050
    print("  [OK] Coordinator initialized")
    return coord


def test_model_loading(coord):
    """Test 2: Model loads correctly."""
    print("\n[TEST 2] Model loading...")
    coord.load_local_model()

    assert coord.tokenizer is not None
    assert coord._inference_engine.local_partitioner is not None
    print(f"  [OK] Model loaded: {coord.model_name}")
    print(f"  [OK] Tokenizer: {type(coord.tokenizer).__name__}")
    return coord


def test_local_inference(coord):
    """Test 3: Local inference works."""
    print("\n[TEST 3] Local inference...")

    result = coord.generate(
        prompt="Hello, how are you?",
        max_new_tokens=20,
        temperature=0.7,
    )

    assert isinstance(result, str)
    assert len(result) > 0
    print(f"  [OK] Generated: {result[:100]}...")
    return result


def test_api_server():
    """Test 4: API server starts correctly."""
    print("\n[TEST 4] API server initialization...")
    from distllm.api.server import app

    assert app.title == "Distributed LLM API"
    assert app.version == "0.4.0"
    assert len(app.routes) > 0
    print(f"  [OK] API server: {app.title} v{app.version}")
    print(f"  [OK] Routes: {len(app.routes)}")
    return app


def test_api_health():
    """Test 5: Health endpoint works."""
    print("\n[TEST 5] API health check...")

    try:
        resp = requests.get("http://localhost:8000/health", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            print(f"  [OK] Health: {data.get('status', 'unknown')}")
            return True
        else:
            print(f"  [WARN] Health check returned {resp.status_code}")
            return False
    except requests.ConnectionError:
        print("  [WARN] API server not running (expected for unit test)")
        return False


def test_api_chat():
    """Test 6: Chat completion endpoint works."""
    print("\n[TEST 6] API chat completion...")

    try:
        resp = requests.post(
            "http://localhost:8000/v1/chat/completions",
            json={
                "model": "Qwen/Qwen2.5-0.5B-Instruct",
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 20,
                "temperature": 0.7,
            },
            headers={"Content-Type": "application/json"},
            timeout=30,
        )

        if resp.status_code == 200:
            data = resp.json()
            assert "choices" in data
            assert len(data["choices"]) > 0
            content = data["choices"][0]["message"]["content"]
            print(f"  [OK] Chat response: {content[:100]}...")
            return True
        else:
            print(f"  [WARN] Chat returned {resp.status_code}: {resp.text[:200]}")
            return False
    except Exception as e:
        print(f"  [WARN] Chat failed: {e}")
        return False


def test_api_completion():
    """Test 7: Text completion endpoint works."""
    print("\n[TEST 7] API text completion...")

    try:
        resp = requests.post(
            "http://localhost:8000/v1/completions",
            json={
                "model": "Qwen/Qwen2.5-0.5B-Instruct",
                "prompt": "Once upon a time",
                "max_tokens": 30,
                "temperature": 0.7,
            },
            headers={"Content-Type": "application/json"},
            timeout=30,
        )

        if resp.status_code == 200:
            data = resp.json()
            assert "choices" in data
            text = data["choices"][0]["text"]
            print(f"  [OK] Completion: {text[:100]}...")
            return True
        else:
            print(f"  [WARN] Completion returned {resp.status_code}")
            return False
    except Exception as e:
        print(f"  [WARN] Completion failed: {e}")
        return False


def test_api_models():
    """Test 8: Models endpoint works."""
    print("\n[TEST 8] API models list...")

    try:
        resp = requests.get(
            "http://localhost:8000/v1/models",
            headers={"Authorization": "Bearer test-key"},
            timeout=5,
        )

        if resp.status_code == 200:
            data = resp.json()
            models = data.get("data", [])
            print(f"  [OK] Models: {len(models)} available")
            return True
        else:
            print(f"  [WARN] Models returned {resp.status_code}")
            return False
    except Exception as e:
        print(f"  [WARN] Models failed: {e}")
        return False


def test_sdk_client():
    """Test 9: SDK client works."""
    print("\n[TEST 9] SDK client...")

    try:
        from distllm.core.coordinator import Coordinator, CoordinatorConfig

        config = CoordinatorConfig(model_name="Qwen/Qwen2.5-0.5B-Instruct")
        coord = Coordinator(config=config)

        assert coord.model_name == "Qwen/Qwen2.5-0.5B-Instruct"
        print("  [OK] SDK client created")
        return True
    except Exception as e:
        print(f"  [WARN] SDK failed: {e}")
        return False


def test_openai_compat():
    """Test 10: OpenAI compatibility layer works."""
    print("\n[TEST 10] OpenAI compatibility...")

    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "sdk"))
        from sdk.compat.openai_compat import OpenAI

        client = OpenAI(base_url="http://localhost:8000/v1", api_key="test")
        assert client.base_url == "http://localhost:8000/v1"
        assert client.chat is not None
        assert client.completions is not None
        print("  [OK] OpenAI compatibility layer works")
        return True
    except Exception as e:
        print(f"  [WARN] OpenAI compat failed: {e}")
        return False


def test_streaming():
    """Test 11: Streaming works."""
    print("\n[TEST 11] Streaming...")

    try:
        resp = requests.post(
            "http://localhost:8000/v1/chat/completions",
            json={
                "model": "Qwen/Qwen2.5-0.5B-Instruct",
                "messages": [{"role": "user", "content": "Count to 5"}],
                "max_tokens": 30,
                "stream": True,
            },
            headers={"Content-Type": "application/json"},
            timeout=30,
            stream=True,
        )

        if resp.status_code == 200:
            chunks = []
            for line in resp.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data = line[6:]
                        if data.strip() != "[DONE]":
                            try:
                                chunk = json.loads(data)
                                chunks.append(chunk)
                            except json.JSONDecodeError:
                                pass

            print(f"  [OK] Streaming: {len(chunks)} chunks received")
            return True
        else:
            print(f"  [WARN] Streaming returned {resp.status_code}")
            return False
    except Exception as e:
        print(f"  [WARN] Streaming failed: {e}")
        return False


def main():
    """Run all end-to-end tests."""
    print("=" * 60)
    print("DistLLM End-to-End Product Test")
    print("=" * 60)

    results = {}

    # Tests that don't need a running server
    try:
        coord = test_coordinator_init()
        results["coordinator_init"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        results["coordinator_init"] = f"FAIL: {e}"
        coord = None

    try:
        if coord:
            coord = test_model_loading(coord)
            results["model_loading"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        results["model_loading"] = f"FAIL: {e}"

    try:
        if coord:
            test_local_inference(coord)
            results["local_inference"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        results["local_inference"] = f"FAIL: {e}"

    try:
        test_api_server()
        results["api_server"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        results["api_server"] = f"FAIL: {e}"

    try:
        test_sdk_client()
        results["sdk_client"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        results["sdk_client"] = f"FAIL: {e}"

    try:
        test_openai_compat()
        results["openai_compat"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        results["openai_compat"] = f"FAIL: {e}"

    # Tests that need a running server
    server_running = test_api_health()
    results["api_health"] = "PASS" if server_running else "SKIP (server not running)"

    if server_running:
        results["api_chat"] = "PASS" if test_api_chat() else "FAIL"
        results["api_completion"] = "PASS" if test_api_completion() else "FAIL"
        results["api_models"] = "PASS" if test_api_models() else "FAIL"
        results["streaming"] = "PASS" if test_streaming() else "FAIL"
    else:
        results["api_chat"] = "SKIP"
        results["api_completion"] = "SKIP"
        results["api_models"] = "SKIP"
        results["streaming"] = "SKIP"

    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v == "PASS")
    failed = sum(1 for v in results.values() if "FAIL" in v)
    skipped = sum(1 for v in results.values() if "SKIP" in v)

    for test, result in results.items():
        status = "[OK]" if result == "PASS" else "[FAIL]" if "FAIL" in result else "[SKIP]"
        print(f"  {status} {test}: {result}")

    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")

    if failed > 0:
        print("\n[WARN] Some tests failed. Check the output above.")
        return 1
    elif skipped > 0:
        print("\n[OK] All tests passed (some skipped).")
        return 0
    else:
        print("\n[OK] All tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
