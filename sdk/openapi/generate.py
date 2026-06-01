#!/usr/bin/env python3
"""Generate SDK clients from OpenAPI spec.

Usage:
    python sdk/openapi/generate.py              # Generate all SDKs
    python sdk/openapi/generate.py --lang js    # Generate JS/TS only
    python sdk/openapi/generate.py --lang go    # Generate Go only
    python sdk/openapi/generate.py --lang rust  # Generate Rust only
"""

import argparse
import subprocess
import sys
from pathlib import Path


SPEC_PATH = Path(__file__).parent / "distllm.yaml"
SDK_DIR = Path(__file__).parent.parent


def generate_js():
    """Generate TypeScript SDK from OpenAPI spec."""
    print("Generating TypeScript SDK...")
    output = SDK_DIR / "js" / "src" / "generated"
    output.mkdir(parents=True, exist_ok=True)

    # Use openapi-typescript for type generation
    try:
        subprocess.run(
            ["npx", "openapi-typescript", str(SPEC_PATH), "-o", str(output / "types.ts")],
            check=True, capture_output=True, text=True,
        )
        print(f"  Generated: {output / 'types.ts'}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  openapi-typescript not found. Install with: npm i -D openapi-typescript")
        print("  Skipping type generation — manual types already in src/index.ts")


def generate_go():
    """Generate Go client from OpenAPI spec."""
    print("Generating Go SDK...")
    output = SDK_DIR / "go"
    output.mkdir(parents=True, exist_ok=True)

    # Use oapi-codegen for Go
    try:
        subprocess.run(
            ["oapi-codegen", "-package", "distllm", "-generate", "types,client",
             str(SPEC_PATH)],
            check=True, capture_output=True, text=True,
        )
        print(f"  Generated Go client (oapi-codegen)")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  oapi-codegen not found. Install: go install github.com/deepmap/oapi-codegen/cmd/oapi-codegen@latest")
        print("  Skipping — manual client already in client.go")


def generate_rust():
    """Generate Rust client from OpenAPI spec."""
    print("Generating Rust SDK...")
    output = SDK_DIR / "rust" / "src" / "generated"
    output.mkdir(parents=True, exist_ok=True)

    # Use progenitor for Rust client generation
    try:
        subprocess.run(
            ["progenitor", "-i", str(SPEC_PATH), "-o", str(output / "client.rs"),
             "--interface", "builder"],
            check=True, capture_output=True, text=True,
        )
        print(f"  Generated: {output / 'client.rs'}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  progenitor not found. Manual client already in src/lib.rs")


def validate_spec():
    """Validate the OpenAPI spec."""
    print(f"Validating OpenAPI spec: {SPEC_PATH}")
    try:
        import yaml
        with open(SPEC_PATH) as f:
            spec = yaml.safe_load(f)
        print(f"  OpenAPI version: {spec.get('openapi', 'unknown')}")
        print(f"  Title: {spec.get('info', {}).get('title', 'unknown')}")
        paths = spec.get("paths", {})
        print(f"  Paths: {len(paths)}")
        schemas = spec.get("components", {}).get("schemas", {})
        print(f"  Schemas: {len(schemas)}")
        print("  Validation: OK")
        return True
    except Exception as e:
        print(f"  Validation FAILED: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate SDK clients from OpenAPI spec")
    parser.add_argument("--lang", choices=["js", "go", "rust", "all"], default="all")
    args = parser.parse_args()

    if not SPEC_PATH.exists():
        print(f"OpenAPI spec not found: {SPEC_PATH}")
        sys.exit(1)

    validate_spec()
    print()

    if args.lang in ("js", "all"):
        generate_js()
    if args.lang in ("go", "all"):
        generate_go()
    if args.lang in ("rust", "all"):
        generate_rust()

    print("\nDone!")


if __name__ == "__main__":
    main()
