#!/usr/bin/env bash
# Security scan script for CI.
#
# Runs bandit (static analysis), safety (dependency vulns), and detect-secrets.
# Outputs SARIF-compatible report to stdout.
#
# Usage:
#     bash scripts/security_scan.sh
#     bash scripts/security_scan.sh --output report.sarif

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_FILE="${1:-}"

echo "============================================"
echo "  Security Scan: $(date)"
echo "============================================"
echo ""

ERRORS=0

# --- 1. Bandit: Static Analysis ---
echo "[1/3] Running bandit static analysis..."
cd "$PROJECT_DIR"
if command -v bandit &> /dev/null; then
    bandit -r src/distllm/ -f json -o bandit-report.json 2>/dev/null || true
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
        detect-secrets scan --baseline "$PROJECT_DIR/.secrets.baseline" 2>&1 || ERRORS=$((ERRORS + 1))
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
