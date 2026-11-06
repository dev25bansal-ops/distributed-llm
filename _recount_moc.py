#!/usr/bin/env python3
"""Deterministic recount for docs/_map MOC aggregate counts.

Counts .py files and non-empty LOC per package tree, plus support-tree file
counts, and verifies every file/dir token listed in the living MOC tables
still exists on disk. Prints everything; edits are made by hand from this.
"""
import os
import re
import sys
import io

ROOT = r"D:\distributed-llm\.claude\worktrees\bubbly-pondering-whisper"

JUNK = {"__pycache__", ".git", "node_modules", "build", "dist", "out",
        "target", ".venv", "venv", ".mypy_cache", ".ruff_cache",
        ".pytest_cache", ".hypothesis", "certs", ".distllm", ".agent",
        ".claude", ".benchmarks", ".obsidian", ".devcontainer", ".github",
        ".pytest_cache", ".mypy_cache"}

def is_junk(d):
    return d in JUNK or d.startswith(".venv")

def walk_files(path, skip_junk=True):
    """Yield relative file paths under path, skipping junk dirs."""
    if not os.path.isdir(path):
        return
    for dirpath, dirnames, filenames in os.walk(path):
        if skip_junk:
            dirnames[:] = [d for d in dirnames if not is_junk(d)]
        for f in sorted(filenames):
            yield os.path.join(dirpath, f)

def loc_of(fp):
    try:
        with open(fp, "rb") as fh:
            data = fh.read()
        text = data.decode("utf-8", errors="replace")
    except OSError:
        return 0
    return sum(1 for ln in text.splitlines() if ln.strip())

def count_py(path):
    files = [f for f in walk_files(path) if f.endswith(".py")]
    return len(files), sum(loc_of(f) for f in files)

def count_all(path):
    files = list(walk_files(path))
    return len(files), sum(loc_of(f) for f in files)

def count_ext(path, ext):
    files = [f for f in walk_files(path) if f.endswith(ext)]
    return len(files)

def fmt(n):
    return f"{n:,}"

print("=" * 78)
print("A. src/distllm packages (immediate subdirs; .py count + non-empty LOC)")
print("=" * 78)
src = os.path.join(ROOT, "src", "distllm")
total_f = total_loc = 0
pkg_rows = []
for name in sorted(os.listdir(src)):
    p = os.path.join(src, name)
    if not os.path.isdir(p) or name == "__pycache__":
        continue
    nf, lc = count_py(p)
    pkg_rows.append((name, nf, lc))
    total_f += nf
    total_loc += lc
for name, nf, lc in pkg_rows:
    print(f"  {name:<16} {nf:>4} files  {fmt(lc):>8} LOC")
root_py = [f for f in os.listdir(src) if f.endswith(".py")]
root_loc = sum(loc_of(os.path.join(src, f)) for f in root_py)
print(f"  {'<root .py>':<16} {len(root_py):>4} files  {fmt(root_loc):>8} LOC  {root_py}")
total_f += len(root_py)
total_loc += root_loc
print(f"  {'TOTAL src/distllm':<16} {total_f:>4} files  {fmt(total_loc):>8} LOC")

print()
print("=" * 78)
print("B. tests/ (.py count + non-empty LOC)")
print("=" * 78)
troot = os.path.join(ROOT, "tests")
tf, tl = count_py(troot)
print(f"  TOTAL tests/ : {tf} .py files, {fmt(tl)} LOC")
root_tests = [f for f in os.listdir(troot) if f.endswith(".py")]
root_loc = sum(loc_of(os.path.join(troot, f)) for f in root_tests)
print(f"  tests root-level .py: {len(root_tests)} files, {fmt(root_loc)} LOC")
pkgs = []
for name in sorted(os.listdir(troot)):
    p = os.path.join(troot, name)
    if os.path.isdir(p) and name != "__pycache__":
        nf, lc = count_py(p)
        if nf or lc:
            pkgs.append((name, nf, lc))
print(f"  test packages (immediate subdirs with .py): {len(pkgs)}")
for name, nf, lc in pkgs:
    print(f"    {name:<20} {nf:>4} files  {fmt(lc):>8} LOC")

print()
print("=" * 78)
print("C. root support trees")
print("=" * 78)
for d in ["integrations", "sdk", "docs", "examples", "benchmarks", "scripts",
          "extensions", "website", "tauri", "deploy", "proto", "apps",
          "distllm", "distributed-llm"]:
    p = os.path.join(ROOT, d)
    if not os.path.exists(p):
        print(f"  {d:<14} MISSING at repo root")
        continue
    nf_all, lc_all = count_all(p)
    nf_py, lc_py = count_py(p)
    nf_md = count_ext(p, ".md")
    print(f"  {d:<14} all-files={nf_all:>4} (LOC {fmt(lc_all):>7})  .py={nf_py:>3}  .md={nf_md:>3}")
wf = [f for f in walk_files(os.path.join(ROOT, ".github", "workflows"))
      if f.endswith((".yml", ".yaml"))]
print(f"  .github/workflows yml/yaml: {len(wf)}")
gh_all, _ = count_all(os.path.join(ROOT, ".github"))
print(f"  .github total files: {gh_all}")

print()
print("=" * 78)
print("D. combined aggregates for _Project Overview Module map")
print("=" * 78)
def combined(label, paths):
    f = l = 0
    for p in paths:
        nf, lc = count_all(p)
        f += nf
        l += lc
    print(f"  {label:<55} {f:>4} files  {fmt(l):>8} LOC (all file types)")
combined("07 integrations/ + sdk/ + src/distllm/{integrations,sdk,prompts,plugins}",
         [os.path.join(ROOT, "integrations"), os.path.join(ROOT, "sdk"),
          os.path.join(src, "integrations"), os.path.join(src, "sdk"),
          os.path.join(src, "prompts"), os.path.join(src, "plugins")])
combined("08 tauri/ + website/ + extensions/ + apps/ + dash + ui",
         [os.path.join(ROOT, "tauri"), os.path.join(ROOT, "website"),
          os.path.join(ROOT, "extensions"), os.path.join(ROOT, "apps"),
          os.path.join(src, "dashboard"), os.path.join(src, "ui")])
combined("09 deploy/ + .github/ + proto/ + scripts/ + benchmarks/ + root packaging",
         [os.path.join(ROOT, "deploy"), os.path.join(ROOT, ".github"),
          os.path.join(ROOT, "proto"), os.path.join(ROOT, "scripts"),
          os.path.join(ROOT, "benchmarks")])
pkg_files = {n: (nf, lc) for n, nf, lc in pkg_rows}
def s(names):
    f = sum(pkg_files[n][0] for n in names)
    l = sum(pkg_files[n][1] for n in names)
    print(f"    {'+'.join(names):<52} {f:>4} files  {fmt(l):>8} LOC (.py)")
print("  Guide-to-packages combined rows:")
s(["cli", "config", "client"])
s(["backends", "models"])
s(["security", "compliance"])
s(["plugins", "prompts"])
s(["integrations", "sdk"])
s(["errors", "utils"])
s(["dashboard", "health", "ui"])

print()
print("=" * 78)
print("E. MOC table-row existence check (missing file/dir tokens)")
print("=" * 78)

tok_re = re.compile(r"`([^`]+)`")

# Section-aware check for per-file tables in 01/02/03/04/05/06/11.
# A "## " header mentioning `src/distllm/<pkg>/` or `<pkg>/` sets the base.
MOC_PKG = {
    "01 Core Engine.md": "core",
    "02 Distributed Layer.md": "dist",
    "03 API Server.md": "api",
    "04 CLI & Config.md": "cli",          # switches among cli/config/client
    "05 Backends & Models.md": "backends",
    "06 Security.md": "security",
    "11 Platform Services.md": "observability",
}
MULTI = {"04 CLI & Config.md": ["cli", "config", "client"],
         "05 Backends & Models.md": ["backends", "models"],
         "06 Security.md": ["security", "compliance"],
         "11 Platform Services.md": ["observability", "health", "cache", "cloud",
                                     "errors", "utils", "terraform", "verification",
                                     "benchmarks"]}

def check_moc(moc):
    path = os.path.join(ROOT, "docs", "_map", moc)
    pkgs = MULTI.get(moc, [MOC_PKG[moc]])
    cur = pkgs[0]
    hits = 0
    for i, line in enumerate(io.open(path, encoding="utf-8"), 1):
        if line.startswith("## "):
            for pk in pkgs:
                if re.search(r"`[^`]*" + re.escape(pk) + r"/`", line) or re.search(r"(?<![\w-])" + re.escape(pk) + r"/", line):
                    cur = pk
                    break
        if not line.strip().startswith("|"):
            continue
        for tok in tok_re.findall(line):
            t = tok.strip()
            if not t or t.endswith("…") or "*" in t or " " in t:
                continue
            # only plain single-token filenames (avoid prose in backticks)
            if not re.fullmatch(r"[\w./-]+\.py", t) and not re.fullmatch(r"[\w./-]+/", t):
                continue
            base = os.path.join(ROOT, "src", "distllm", cur)
            cand = os.path.join(base, t.rstrip("/"))
            if not os.path.exists(cand):
                print(f"  [{moc}] line {i} (base {cur}): MISSING {t!r}")
                hits += 1
    return hits

missing_total = 0
for moc in ["01 Core Engine.md", "02 Distributed Layer.md", "03 API Server.md",
            "04 CLI & Config.md", "05 Backends & Models.md", "06 Security.md",
            "11 Platform Services.md"]:
    missing_total += check_moc(moc)

# 07 Integrations: in-tree integrations/, prompts/, plugins/ tables
for base, start_marker in [(os.path.join(src, "integrations"), "In-tree `integrations/`"),
                           (os.path.join(src, "prompts"), "`prompts/`"),
                           (os.path.join(src, "plugins"), "`plugins/`")]:
    active = False
    for i, line in enumerate(io.open(os.path.join(ROOT, "docs", "_map", "07 Integrations.md"), encoding="utf-8"), 1):
        if line.startswith("## "):
            active = start_marker in line
        if active and line.strip().startswith("|"):
            for tok in tok_re.findall(line):
                t = tok.strip()
                if not t or t.endswith("…") or "*" in t or " " in t:
                    continue
                if not re.fullmatch(r"[\w./-]+\.py", t) and not re.fullmatch(r"[\w./-]+/", t):
                    continue
                cand = os.path.join(base, t.rstrip("/"))
                if not os.path.exists(cand):
                    print(f"  [07 Integrations.md] line {i} (base {os.path.basename(base)}): MISSING {t!r}")
                    missing_total += 1

# 00 Module Index: package dirs must exist
for i, line in enumerate(io.open(os.path.join(ROOT, "docs", "_map", "00 Module Index.md"), encoding="utf-8"), 1):
    if not line.strip().startswith("|"):
        continue
    for tok in tok_re.findall(line):
        t = tok.strip()
        if not t.endswith("/"):
            continue
        name = t.rstrip("/")
        if "/" in name:
            cand = os.path.join(ROOT, name)
        elif os.path.isdir(os.path.join(src, name)) or name in ("core","dist","api","cli","config","client","backends","models","security","compliance","integrations","prompts","plugins","sdk","observability","health","cache","cloud","errors","utils","terraform","verification","dashboard","ui","benchmarks"):
            cand = os.path.join(src, name)
        else:
            cand = os.path.join(ROOT, name)
        if not os.path.exists(cand):
            print(f"  [00 Module Index.md] line {i}: MISSING {t!r}")
            missing_total += 1

# 10 Tests: named test files at tests root
for i, line in enumerate(io.open(os.path.join(ROOT, "docs", "_map", "10 Tests.md"), encoding="utf-8"), 1):
    if not line.strip().startswith("|"):
        continue
    for tok in tok_re.findall(line):
        t = tok.strip()
        if re.fullmatch(r"[\w./-]+\.py", t):
            cand = os.path.join(troot, t)
            if not os.path.exists(cand):
                print(f"  [10 Tests.md] line {i}: MISSING {t!r}")
                missing_total += 1

print(f"  total missing tokens: {missing_total}")

print()
print("done.")
