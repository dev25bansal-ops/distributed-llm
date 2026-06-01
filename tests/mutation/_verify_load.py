"""Verify mutated module is actually loaded from tmp dir."""
import sys, os, tempfile, shutil, subprocess, ast, copy

def make_mutated_copy(source_path, out_dir):
    source = open(source_path).read()
    tree = ast.parse(source)
    
    # Apply a simple mutation: replace None with 0 at line 30 (return None -> return 0)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value is None and node.lineno == 30:
            node.value = 0
            break
    
    mutated = ast.unparse(tree)
    dest_dir = os.path.join(out_dir, 'distllm', 'core')
    os.makedirs(dest_dir, exist_ok=True)
    dest_file = os.path.join(dest_dir, 'latency_tracker.py')
    with open(dest_file, 'w') as f:
        f.write(mutated)
    return dest_file

# Create temp dir and write mutated version
tmp = tempfile.mkdtemp()
make_mutated_copy('src/distllm/core/latency_tracker.py', tmp)

# Run a subprocess with PYTHONPATH=tmp;src
env = os.environ.copy()
env['PYTHONPATH'] = tmp + os.pathsep + 'src'
env['PYTHONIOENCODING'] = 'utf-8'

verify_code = '''
import sys, os

# Clear cached module
for key in list(sys.modules.keys()):
    if 'latency_tracker' in key:
        del sys.modules[key]

import distllm.dist.latency as lt
print(f"MODULE_FILE: {lt.__file__}")

t = lt.LatencyTracker()
t.record("x", 1.0)
t.record("x", 2.0)
avg = t.get_avg("x")
print(f"AVG: {avg}")
print(f"EXPECTED_AVG: {1.5}")

# Check if mutated (returning 0 instead of None for empty)
avg2 = t.get_avg("nonexistent")
print(f"EMPTY_AVG: {avg2}")
print(f"EMPTY_IS_0: {avg2 == 0}")
'''

result = subprocess.run([sys.executable, '-c', verify_code],
    capture_output=True, text=True, timeout=30, env=env)

print(f"Return code: {result.returncode}")
print("STDOUT:")
print(result.stdout)
if result.stderr:
    print("STDERR (first 500 chars):")
    print(result.stderr[:500])

shutil.rmtree(tmp)

# Also verify that without the PYTHONPATH override, the normal module loads
print("\n--- Without override (normal) ---")
env2 = os.environ.copy()
env2['PYTHONPATH'] = 'src'
env2['PYTHONIOENCODING'] = 'utf-8'

result2 = subprocess.run([sys.executable, '-c', verify_code],
    capture_output=True, text=True, timeout=30, env=env2)

print(f"Return code: {result2.returncode}")
print(result2.stdout[:300])
