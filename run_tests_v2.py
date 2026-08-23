import subprocess, sys, os
os.chdir("D:/distributed-llm")
results = {}

# Step 1
r1 = subprocess.run([sys.executable, "-m", "pytest", "tests/core/test_kv_cache.py", "tests/core/test_speculative_decoder.py", "tests/core/test_event_bus.py", "-q", "--tb=short"], capture_output=True, text=True)
with open("step1_output.txt", "w") as f:
    f.write(r1.stdout)
    if r1.stderr:
        f.write("\nSTDERR:\n" + r1.stderr[:1000])
    f.write(f"\nEXIT_CODE: {r1.returncode}\n")
results["step1"] = r1.returncode

# Step 2
r2 = subprocess.run([sys.executable, "-m", "pytest", "tests/core/test_agentic_router.py", "tests/core/test_autonomous_healer.py", "tests/core/test_prompt_library.py", "tests/api/test_prompt_injection.py", "-q", "--tb=short"], capture_output=True, text=True)
with open("step2_output.txt", "w") as f:
    f.write(r2.stdout)
    if r2.stderr:
        f.write("\nSTDERR:\n" + r2.stderr[:1000])
    f.write(f"\nEXIT_CODE: {r2.returncode}\n")
results["step2"] = r2.returncode

# Step 3
r3 = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0,'src'); from distllm.core.agentic_router import AgenticRouter; from distllm.core.autonomous_healer import AutonomousHealer; from distllm.core.prompt_library import PromptRepository; from distllm.api.prompt_injection import PromptInjectionMiddleware, FastInjectionClassifier; print('IMPORTS OK')"], capture_output=True, text=True, cwd="D:/distributed-llm")
with open("step3_output.txt", "w") as f:
    f.write(r3.stdout)
    if r3.stderr:
        f.write("\nSTDERR:\n" + r3.stderr[:1000])
    f.write(f"\nEXIT_CODE: {r3.returncode}\n")
results["step3"] = r3.returncode

# Summary
with open("summary.txt", "w") as f:
    for step, rc in results.items():
        status = "PASS" if rc == 0 else "FAIL"
        f.write(f"{step}: {status}\n")

print("DONE - results written to summary.txt")
