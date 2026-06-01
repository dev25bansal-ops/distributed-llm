"""Run all fuzz test harnesses.

Usage:
    python tests/fuzz/run_all.py                   # 500 iterations each, pytest-style
    python tests/fuzz/run_all.py --atheris          # coverage-guided via libFuzzer
    python tests/fuzz/run_all.py --iterations 2000  # custom iterations
"""
import argparse
import importlib.util
import os
import sys
import time


HERE = os.path.dirname(os.path.abspath(__file__))


HARNESSES = [
    ("grammar_parser", "fuzz_grammar_parser"),
    ("protobuf", "fuzz_protobuf_deserializer"),
    ("config_loader", "fuzz_config_loader"),
    ("plugin_installer", "fuzz_plugin_installer"),
    ("api_endpoints", "fuzz_api_endpoints"),
    ("cli_args", "fuzz_cli_args"),
]


def _load_module(name: str):
    """Load a harness module by filename (no packages needed)."""
    path = os.path.join(HERE, f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all fuzz harnesses")
    parser.add_argument("--atheris", action="store_true", help="Use atheris coverage-guided fuzzing")
    parser.add_argument("--iterations", type=int, default=500, help="Iterations per harness (pytest mode)")
    args = parser.parse_args()

    failures = 0
    total_start = time.time()

    for name, module_name in HARNESSES:
        print(f"\n{'='*60}")
        print(f"  Fuzz: {name}")
        print(f"{'='*60}")
        start = time.time()

        try:
            mod = _load_module(module_name)
            if args.atheris:
                import subprocess
                result = subprocess.run(
                    [sys.executable, os.path.join(HERE, f"{module_name}.py"), "--atheris"],
                    capture_output=True, text=True, timeout=120,
                )
                if result.returncode != 0:
                    print(f"  ! atheris process exited with code {result.returncode}")
                    for line in result.stderr.split("\n")[-5:]:
                        if line.strip():
                            print(f"    {line.strip()}")
                    failures += 1
                else:
                    elapsed = time.time() - start
                    print(f"  OK Completed in {elapsed:.1f}s")
            else:
                mod.pytest_fuzz(n=args.iterations)
                elapsed = time.time() - start
                print(f"  OK {args.iterations} iterations in {elapsed:.1f}s")
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            failures += 1

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    if failures:
        print(f"  FAIL {failures}/{len(HARNESSES)} harnesses failed ({total_elapsed:.1f}s)")
        sys.exit(1)
    else:
        print(f"  OK All {len(HARNESSES)} harnesses passed ({total_elapsed:.1f}s)")


if __name__ == "__main__":
    main()
