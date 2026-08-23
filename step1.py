import os, sys
os.chdir("D:/distributed-llm")
exit_code = os.system(' '.join([sys.executable, '-m', 'pytest', 'tests/core/test_kv_cache.py', 'tests/core/test_speculative_decoder.py', 'tests/core/test_event_bus.py', '-q', '--tb=short']))
open("step1_done.txt","w").write(f"EXIT: {exit_code}\n")
sys.exit(exit_code)
