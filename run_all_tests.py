#!/usr/bin/env python3
"""Run all three specified test steps and report PASS/FAIL."""
import subprocess
import sys
import os

os.chdir("D:/distributed-llm")
src_path = os.path.abspath("src")

results = {}

# Step 1: first batch
print("=== STEP 1: tests/core/test_kv_cache.py + test_speculative_decoder.py + test_event_bus.py ===")
r1 = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/core/test_kv_cache.py", "tests/core/test_speculative_decoder.py", "tests/core/test_event_bus.py", "-q", "--tb=short"],
    capture_output=True, text=True
)
print(r1.stdout)
if r1.stderr:
    print("STDERR:", r1.stderr[:500])
results["step1"] = "PASS" if r1.returncode == 0 else "FAIL"
print(f"Step 1 exit code: {r1.returncode}")

# Step 2: second batch
print("\n=== STEP 2: tests/core/test_agentic_router.py + test_autonomous_healer.py + test_prompt_library.py + tests/api/test_prompt_injection.py ===")
r2 = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/core/test_agentic_router.py", "tests/core/test_autonomous_healer.py", "tests/core/test_prompt_library.py", "tests/api/test_prompt_injection.py", "-q", "--tb=short"],
    capture_output=True, text=True
)
print(r2.stdout)
if r2.stderr:
    print("STDERR:", r2.stderr[:500])
results["step2"] = "PASS" if r2.returncode == 0 else "FAIL"
print(f"Step 2 exit code: {r2.returncode}")

# Step 3: imports
print("\n=== STEP 3: Import check ===")
r3 = subprocess.run(
    [sys.executable, "-c",
     "import sys; sys.path.insert(0,'src'); "
     "from distllm.core.agentic_router import AgenticRouter; "
     "from distllm.core.autonomous_healer import AutonomousHealer; "
     "from distllm.core.prompt_library import PromptRepository; "
     "from distllm.api.prompt_injection import PromptInjectionMiddleware, FastInjectionClassifier; "
     "print('IMPORTS OK')"],
    capture_output=True, text=True, cwd="D:/distributed-llm"
)
print(r3.stdout)
if r3.stderr:
    print("STDERR:", r3.stderr[:500])
results["step3"] = "PASS" if r3.returncode == 0 else "FAIL"
print(f"Step 3 exit code: {r3.returncode}")

# Summary
print("\n" + "=" * 50)
print("RESULTS SUMMARY")
print("=" * 50)
for step, result in results.items():
    print(f"{step}: {result}")
