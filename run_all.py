#!/usr/bin/env python3
import subprocess, sys, os
os.chdir("D:/distributed-llm")

def run(cmd, logfile):
    with open(logfile, "w") as f:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        f.write(p.stdout)
        f.write(f"\nEXIT_CODE: {p.returncode}\n")
    return p.returncode

e1 = run([sys.executable, "-m", "pytest", "tests/core/test_kv_cache.py", "tests/core/test_speculative_decoder.py", "tests/core/test_event_bus.py", "-q", "--tb=short"], "out1.txt")
e2 = run([sys.executable, "-m", "pytest", "tests/core/test_agentic_router.py", "tests/core/test_autonomous_healer.py", "tests/core/test_prompt_library.py", "tests/api/test_prompt_injection.py", "-q", "--tb=short"], "out2.txt")
e3 = run([sys.executable, "-c", "import sys; sys.path.insert(0,'src'); from distllm.core.agentic_router import AgenticRouter; from distllm.core.autonomous_healer import AutonomousHealer; from distllm.core.prompt_library import PromptRepository; from distllm.api.prompt_injection import PromptInjectionMiddleware, FastInjectionClassifier; print('IMPORTS OK')"], "out3.txt")

with open("summary.txt", "w") as f:
    for name, ec in [("step1", e1), ("step2", e2), ("step3", e3)]:
        f.write(f"{name}: {'PASS' if ec == 0 else 'FAIL'}\n")
