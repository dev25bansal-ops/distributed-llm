import ast
files = [
    "src/distllm/cli/client.py",
    "src/distllm/core/auto_discovery.py",
    "src/distllm/core/model_sizing.py",
]
for f in files:
    with open(f) as fh:
        ast.parse(fh.read())
    print(f"OK: {f}")
print("ALL SYNTAX OK")
