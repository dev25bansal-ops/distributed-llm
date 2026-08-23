import ast
import sys
import subprocess

files = [
    "src/distllm/cli/client.py",
    "src/distllm/core/auto_discovery.py",
    "src/distllm/core/model_sizing.py",
]

errors = []

for f in files:
    try:
        with open(f) as fh:
            code = fh.read()
        ast.parse(code)
        print(f"OK: {f}")
    except SyntaxError as e:
        print(f"ERROR: {f}: {e}")
        errors.append((f, str(e)))
    except FileNotFoundError as e:
        print(f"ERROR: {f}: File not found")
        errors.append((f, "File not found"))

print("\n--- SYNTAX CHECK SUMMARY ---")
if errors:
    print(f"FAILED: {len(errors)} file(s) with syntax errors")
    for f, e in errors:
        print(f"  {f}: {e}")
    sys.exit(1)
else:
    print("ALL OK - no syntax errors")

# Now check imports
print("\n--- IMPORT CHECK ---")
sys.path.insert(0, "src")

imports_to_try = [
    ("distllm.cli.client", ["DistLLMClient", "ClientConfig", "DistLLMError"]),
    ("distllm.core.auto_discovery", ["AutoDiscoverer", "DiscoveryConfig", "DiscoveredNode"]),
    ("distllm.core.model_sizing", ["estimate_model_size", "estimate_vram_gb", "model_info"]),
]

import_errors = []
for module_path, names in imports_to_try:
    try:
        mod = __import__(module_path, fromlist=names)
        print(f"OK: {module_path}")
        for name in names:
            if hasattr(mod, name):
                print(f"  - has {name}")
            else:
                print(f"  - MISSING {name}")
                import_errors.append((module_path, f"Missing attribute: {name}"))
    except ImportError as e:
        print(f"IMPORT ERROR: {module_path}: {e}")
        import_errors.append((module_path, str(e)))

if import_errors:
    print(f"\nIMPORT FAILURES: {len(import_errors)}")
    for mod, e in import_errors:
        print(f"  {mod}: {e}")
else:
    print("\nALL IMPORTS OK")
