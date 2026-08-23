import ast
import sys

files = [
    "src/distllm/cli/client.py",
    "src/distllm/core/auto_discovery.py",
    "src/distllm/core/model_sizing.py",
]

all_ok = True
for f in files:
    try:
        with open(f) as fh:
            ast.parse(fh.read())
        print(f"OK: {f}")
    except SyntaxError as e:
        print(f"ERROR: {f}: {e}")
        all_ok = False

print("---")
print("ALL OK" if all_ok else "SOME FAILED")
sys.exit(0 if all_ok else 1)
