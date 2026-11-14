#!/usr/bin/env bash
# Security scan script for CI.
#
# Runs bandit (static analysis), safety (dependency vulns), and detect-secrets.
# Writes a plain-text report; pass --output FILE to also capture it to a file.
#
# Usage:
#     bash scripts/security_scan.sh
#     bash scripts/security_scan.sh --output report.txt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

OUTPUT_FILE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --output)
            OUTPUT_FILE="${2:-}"
            shift 2 || { echo "ERROR: --output needs a path" >&2; exit 2; }
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

# Mirror everything to the report file when requested (tee keeps stdout live).
if [ -n "$OUTPUT_FILE" ]; then
    exec > >(tee "$OUTPUT_FILE") 2>&1
fi

echo "============================================"
echo "  Security Scan: $(date)"
echo "============================================"
echo ""

ERRORS=0

# --- 1. Bandit: Static Analysis ---
echo "[1/3] Running bandit static analysis..."
cd "$PROJECT_DIR"
if command -v bandit &> /dev/null; then
    # JSON report goes to a temp file so repeated runs don't dirty the tree.
    BANDIT_JSON="$(mktemp)"
    bandit -r src/distllm/ -f json -o "$BANDIT_JSON" 2>/dev/null || true
    rm -f "$BANDIT_JSON"
    bandit -r src/distllm/ -x tests/ --skip B101 2>&1 || ERRORS=$((ERRORS + 1))
else
    echo "  SKIP: bandit not installed. Run: pip install bandit"
fi
echo ""

# --- 2. Safety: Dependency Vulnerabilities ---
echo "[2/3] Running safety dependency scan..."
if command -v safety &> /dev/null; then
    safety check --json 2>/dev/null || true
    safety check 2>&1 || ERRORS=$((ERRORS + 1))
else
    echo "  SKIP: safety not installed. Run: pip install safety"
fi
echo ""

# --- 3. Detect-Secrets: Hardcoded Secrets ---
echo "[3/3] Running detect-secrets scan..."
if command -v detect-secrets &> /dev/null; then
    if [ -f "$PROJECT_DIR/.secrets.baseline" ]; then
        # NOTE: `detect-secrets scan --baseline <file>` *updates* the baseline
        # in place and never fails on new secrets, so it cannot gate.  Take a
        # fresh scan and diff it against the committed baseline instead
        # (mirrors the CI gate in .github/workflows/ci.yml).
        cd "$PROJECT_DIR" || exit 1
        if ! detect-secrets scan src deploy > .secrets.new.baseline 2>/dev/null; then
            ERRORS=$((ERRORS + 1))
        elif ! python scripts/ci/check_secrets_baseline.py; then
            ERRORS=$((ERRORS + 1))
        fi
        rm -f .secrets.new.baseline
        cd - >/dev/null || true
    else
        echo "  No .secrets.baseline found. Run: detect-secrets scan > .secrets.baseline"
    fi
else
    echo "  SKIP: detect-secrets not installed. Run: pip install detect-secrets"
fi
echo ""

# --- Summary ---
echo "============================================"
if [ "$ERRORS" -gt 0 ]; then
    echo "  SCAN FAILED: $ERRORS tool(s) reported issues"
    echo "============================================"
    exit 1
else
    echo "  SCAN PASSED: No issues detected"
    echo "============================================"
    exit 0
fi
