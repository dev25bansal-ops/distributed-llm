"""
Custom AST-based mutation testing tool for Python.
Applies single mutations to source code and runs tests to check if they're caught.
"""

import ast
import copy
import os
import re
import subprocess
import sys
import tempfile
import time
import shutil
from pathlib import Path


MUTATION_OPERATORS = [
    "replace_comparison",
    "replace_bool_op",
    "negate_condition",
    "remove_not",
    "replace_arithmetic",
    "replace_true_false",
    "replace_none",
    "remove_break",
    "replace_dict_get",
    "replace_augmented_assign",
    # New operators for better coverage
    "remove_logging",
    "return_none",
    "replace_int_constant",
    "remove_decorator",
    "swap_arguments",
    "replace_string_empty",
]


class MutationPoint:
    def __init__(self, operator, description, target_type, target_lineno, target_col, transform_fn):
        self.operator = operator
        self.description = description
        self.target_type = target_type
        self.target_lineno = target_lineno
        self.target_col = target_col
        self.transform_fn = transform_fn


def find_mutation_points(source: str) -> list[MutationPoint]:
    """Find all mutation points by parsing AST and matching by position."""
    tree = ast.parse(source)

    _CMP_MAP = {
        ast.Eq: ("==", "!=", ast.NotEq),
        ast.NotEq: ("!=", "==", ast.Eq),
        ast.Lt: ("<", "<=", ast.LtE),
        ast.LtE: ("<=", "<", ast.Lt),
        ast.Gt: (">", ">=", ast.GtE),
        ast.GtE: (">=", ">", ast.Gt),
        ast.Is: ("is", "is not", ast.IsNot),
        ast.IsNot: ("is not", "is", ast.Is),
        ast.In: ("in", "not in", ast.NotIn),
        ast.NotIn: ("not in", "in", ast.In),
    }

    _ARITH_MAP = {
        ast.Add: ("+", "-", ast.Sub),
        ast.Sub: ("-", "+", ast.Add),
        ast.Mult: ("*", "/", ast.Div),
        ast.Div: ("/", "*", ast.Mult),
        ast.FloorDiv: ("//", "/", ast.Div),
        ast.Mod: ("%", "*", ast.Mult),
    }

    _AUG_MAP = {
        ast.Add: ("+=", "-=", ast.Sub),
        ast.Sub: ("-=", "+=", ast.Add),
        ast.Mult: ("*=", "/=", ast.Div),
        ast.Div: ("/=", "*=", ast.Mult),
    }

    lines = source.splitlines(keepends=True)
    points = []

    for node in ast.walk(tree):
        # replace_comparison (for Compare ops)
        if isinstance(node, ast.Compare):
            for idx, op in enumerate(node.ops):
                for cls, (orig, repl, new_cls) in _CMP_MAP.items():
                    if isinstance(op, cls):
                        desc = f"Replace {orig} with {repl} at line {node.lineno}"
                        points.append(MutationPoint(
                            "replace_comparison", desc,
                            ast.Compare, node.lineno, node.col_offset,
                            _make_compare_transform(node, idx, new_cls)))
                        break

        # replace_bool_op
        if isinstance(node, ast.BoolOp):
            orig = "and" if isinstance(node.op, ast.And) else "or"
            repl = "or" if isinstance(node.op, ast.And) else "and"
            new_cls = ast.Or if isinstance(node.op, ast.And) else ast.And
            desc = f"Replace {orig} with {repl} at line {node.lineno}"
            points.append(MutationPoint(
                "replace_bool_op", desc,
                ast.BoolOp, node.lineno, node.col_offset,
                lambda n: ast.BoolOp(op=new_cls(), values=n.values)))

        # negate_condition
        if isinstance(node, ast.If):
            desc = f"Negate condition at line {node.lineno}"
            points.append(MutationPoint(
                "negate_condition", desc,
                ast.If, node.lineno, node.col_offset,
                lambda n: ast.If(
                    test=ast.UnaryOp(op=ast.Not(), operand=n.test),
                    body=n.body, orelse=n.orelse)))

        # remove_not
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            desc = f"Remove 'not' at line {node.lineno}"
            points.append(MutationPoint(
                "remove_not", desc,
                ast.UnaryOp, node.lineno, node.col_offset,
                lambda n: n.operand))

        # replace_arithmetic
        if isinstance(node, ast.BinOp):
            for cls, (orig, repl, new_cls) in _ARITH_MAP.items():
                if isinstance(node.op, cls):
                    desc = f"Replace {orig} with {repl} at line {node.lineno}"
                    points.append(MutationPoint(
                        "replace_arithmetic", desc,
                        ast.BinOp, node.lineno, node.col_offset,
                        lambda n, nc=new_cls: ast.BinOp(left=n.left, op=nc(), right=n.right)))
                    break

        # replace_augmented_assign
        if isinstance(node, ast.AugAssign):
            for cls, (orig, repl, new_cls) in _AUG_MAP.items():
                if isinstance(node.op, cls):
                    desc = f"Replace {orig} with {repl} at line {node.lineno}"
                    points.append(MutationPoint(
                        "replace_augmented_assign", desc,
                        ast.AugAssign, node.lineno, node.col_offset,
                        lambda n, nc=new_cls: ast.AugAssign(target=n.target, op=nc(), value=n.value)))
                    break

        # replace_true_false
        if isinstance(node, ast.Constant) and node.value is True:
            desc = f"Replace True with False at line {node.lineno}"
            points.append(MutationPoint(
                "replace_true_false", desc,
                ast.Constant, node.lineno, node.col_offset,
                lambda n: ast.Constant(value=False)))
        elif isinstance(node, ast.Constant) and node.value is False:
            desc = f"Replace False with True at line {node.lineno}"
            points.append(MutationPoint(
                "replace_true_false", desc,
                ast.Constant, node.lineno, node.col_offset,
                lambda n: ast.Constant(value=True)))

        # replace_none
        if isinstance(node, ast.Constant) and node.value is None:
            col = getattr(node, "col_offset", 0)
            desc = f"Replace None with 0 at line {node.lineno}:{col}"
            points.append(MutationPoint(
                "replace_none", desc,
                ast.Constant, node.lineno, node.col_offset,
                lambda n: ast.Constant(value=0)))

        # remove_break
        if isinstance(node, ast.Break):
            desc = f"Remove break at line {node.lineno}"
            points.append(MutationPoint(
                "remove_break", desc,
                ast.Break, node.lineno, node.col_offset,
                lambda n: ast.Pass()))

        # replace_dict_get
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get":
            desc = f"Replace dict.get with [] at line {node.lineno}"
            if node.args:
                points.append(MutationPoint(
                    "replace_dict_get", desc,
                    ast.Call, node.lineno, node.col_offset,
                    lambda n: ast.Subscript(value=n.func.value, slice=n.args[0], ctx=ast.Load())))

        # remove_logging — remove logger.xxx() and print() calls
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("debug", "info", "warning", "error", "critical"):
                desc = f"Remove logging call .{node.func.attr}() at line {node.lineno}"
                points.append(MutationPoint(
                    "remove_logging", desc,
                    ast.Call, node.lineno, node.col_offset,
                    lambda n: ast.Constant(value=None)))

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
            desc = f"Remove print() at line {node.lineno}"
            points.append(MutationPoint(
                "remove_logging", desc,
                ast.Call, node.lineno, node.col_offset,
                lambda n: ast.Constant(value=None)))

        # return_none — replace return X with return None
        if isinstance(node, ast.Return) and node.value is not None:
            desc = f"Replace return value with None at line {node.lineno}"
            points.append(MutationPoint(
                "return_none", desc,
                ast.Return, node.lineno, node.col_offset,
                lambda n: ast.Return(value=ast.Constant(value=None))))

        # replace_int_constant — replace integer N with N+1, N-1, 0
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and node.value not in (0, 1, -1, True, False):
            for replacement, label in [(node.value + 1, "+1"), (node.value - 1, "-1"), (0, "0")]:
                desc = f"Replace int {node.value} with {replacement} ({label}) at line {node.lineno}"
                points.append(MutationPoint(
                    "replace_int_constant", desc,
                    ast.Constant, node.lineno, node.col_offset,
                    lambda n, r=replacement: ast.Constant(value=r)))

        # remove_decorator — remove @decorator
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.decorator_list:
            for dec in node.decorator_list:
                dec_name = ast.dump(dec)[:40]
                desc = f"Remove decorator @{dec_name} at line {dec.lineno}"
                points.append(MutationPoint(
                    "remove_decorator", desc,
                    type(dec), dec.lineno, dec.col_offset,
                    lambda n: ast.Constant(value=None)))

        # swap_arguments — swap first two arguments of a function call
        if isinstance(node, ast.Call) and len(node.args) >= 2:
            desc = f"Swap first 2 arguments at line {node.lineno}"
            points.append(MutationPoint(
                "swap_arguments", desc,
                ast.Call, node.lineno, node.col_offset,
                lambda n: ast.Call(func=n.func, args=[n.args[1], n.args[0]] + n.args[2:], keywords=n.keywords)))

        # replace_string_empty — replace string literal with empty string
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value != "":
            desc = f"Replace string '{node.value[:20]}...' with '' at line {node.lineno}"
            points.append(MutationPoint(
                "replace_string_empty", desc,
                ast.Constant, node.lineno, node.col_offset,
                lambda n: ast.Constant(value="")))

    # Deduplicate by description
    seen = set()
    unique = []
    for p in points:
        if p.description not in seen:
            seen.add(p.description)
            unique.append(p)
    unique.sort(key=lambda p: p.description)
    return unique


def _make_compare_transform(node, idx, new_cls):
    """Create a closure for compare operator replacement."""
    def transform(n):
        new_ops = list(n.ops)
        new_ops[idx] = new_cls()
        return ast.Compare(left=n.left, ops=new_ops, comparators=n.comparators)
    return transform


def _find_node(tree, target_type, lineno, col):
    """Find a node in tree by type and position."""
    for node in ast.walk(tree):
        if (type(node) is target_type and
            getattr(node, 'lineno', None) == lineno and
            getattr(node, 'col_offset', None) == col):
            return node
    return None


def apply_mutation(tree: ast.AST, point: MutationPoint) -> str:
    """Deep-copy tree, find target node by type+position, apply transform, unparse."""
    tree_copy = copy.deepcopy(tree)

    target_node = _find_node(tree_copy, point.target_type, point.target_lineno, point.target_col)
    if target_node is None:
        raise ValueError(f"Cannot find target node: {point.target_type} at line {point.target_lineno}:{point.target_col}")

    class _Visitor(ast.NodeTransformer):
        def __init__(self):
            self.done = False

        def visit(self, node):
            if self.done:
                return node
            if node is target_node:
                self.done = True
                result = point.transform_fn(node)
                if isinstance(result, ast.AST):
                    ast.copy_location(result, node)
                return result
            return super().visit(node)

    modified = _Visitor().visit(tree_copy)
    ast.fix_missing_locations(modified)
    return ast.unparse(modified)


def run_tests(test_path: str, project_root: str, timeout: int = 120) -> tuple:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src" + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-x", "--tb=short", "--no-header", "-q", test_path],
            capture_output=True, text=True, timeout=timeout, env=env, cwd=project_root,
        )
        return result.returncode == 0, result.stdout + result.stderr, time.time() - start
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT after {timeout}s", time.time() - start
    except Exception as e:
        return False, f"Error: {e}", time.time() - start


def mutate_and_test(source_path, test_path, point, tree, tmp_dir, baseline_time, project_root):
    with open(source_path, "r", encoding="utf-8") as f:
        original_source = f.read()

    try:
        mutated_source = apply_mutation(tree, point)
    except Exception as e:
        return {"point": point, "status": "ERROR", "error": f"Mutation application failed: {e}", "elapsed": 0}

    # Verify mutation actually changed the source
    if mutated_source == original_source:
        return {"point": point, "status": "ERROR", "error": "Mutation produced identical source (transform failed)", "elapsed": 0}

    rel_path = os.path.relpath(source_path, os.path.join(project_root, "src"))
    mutated_path = os.path.join(tmp_dir, rel_path)
    os.makedirs(os.path.dirname(mutated_path), exist_ok=True)
    with open(mutated_path, "w", encoding="utf-8") as f:
        f.write(mutated_source)

    # Create __init__.py files to make it a regular package (otherwise
    # Python skips the tmp dir in favor of the src dir which has __init__.py)
    rel_parts = Path(rel_path).parent.parts
    for i in range(1, len(rel_parts) + 1):
        init_dir = os.path.join(tmp_dir, *rel_parts[:i])
        init_file = os.path.join(init_dir, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w", encoding="utf-8") as f:
                f.write("")

    test_env = os.environ.copy()
    test_env["PYTHONPATH"] = tmp_dir + os.pathsep + "src" + os.pathsep + test_env.get("PYTHONPATH", "")
    test_env["PYTHONIOENCODING"] = "utf-8"

    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-x", "--tb=short", "--no-header", "-q",
             "-p", "no:cacheprovider", test_path],
            capture_output=True, text=True, timeout=baseline_time * 10 + 30, env=test_env, cwd=project_root,
        )
        elapsed = time.time() - start
        status = "KILLED" if result.returncode != 0 else "SURVIVED"
        return {"point": point, "status": status, "output": result.stdout + result.stderr, "elapsed": elapsed}
    except subprocess.TimeoutExpired:
        return {"point": point, "status": "TIMEOUT", "output": "TIMEOUT", "elapsed": time.time() - start}
    except Exception as e:
        return {"point": point, "status": "ERROR", "error": str(e), "elapsed": time.time() - start}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Custom mutation testing tool")
    parser.add_argument("source", help="Path to source file to mutate")
    parser.add_argument("test", help="Path to test file to run")
    parser.add_argument("--operators", nargs="*", choices=MUTATION_OPERATORS)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--list", dest="list_points", action="store_true")
    parser.add_argument("--id", type=int, default=None)
    args = parser.parse_args()

    source_path = Path(args.source).resolve()
    test_path = Path(args.test).resolve()
    project_root = str(Path(__file__).resolve().parent.parent.parent)

    if not source_path.exists():
        print(f"Error: source file not found: {source_path}")
        sys.exit(1)
    if not test_path.exists():
        print(f"Error: test file not found: {test_path}")
        sys.exit(1)

    with open(source_path, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source)
    all_points = find_mutation_points(source)
    if not all_points:
        print("No mutation points found.")
        sys.exit(0)

    if args.operators:
        points = [p for p in all_points if p.operator in args.operators]
    else:
        points = all_points

    if args.id is not None:
        if args.id < 0 or args.id >= len(points):
            print(f"Error: mutation ID {args.id} out of range (0-{len(points)-1})")
            sys.exit(1)
        points = [points[args.id]]

    if args.list_points:
        print(f"\nFound {len(points)} mutation points:")
        print(f"{'ID':<5} {'Operator':<28} Description")
        print("-" * 80)
        for i, p in enumerate(points):
            print(f"{i:<5} {p.operator:<28} {p.description}")
        sys.exit(0)

    print(f"\n{'='*60}")
    print(f"Source: {source_path.name}")
    print(f"Tests:  {test_path.name}")
    print(f"Points: {len(points)}")
    print(f"{'='*60}\n")

    print("1. Running baseline test suite...")
    baseline_passed, baseline_output, baseline_time = run_tests(str(test_path), project_root)
    if not baseline_passed:
        print(f"\n  BASELINE FAILED")
        print(f"  Output:\n{baseline_output[:2000]}")
        sys.exit(1)
    print(f"   Baseline passed in {baseline_time:.2f}s\n")

    if args.baseline_only:
        print("Baseline-only mode. Exiting.")
        sys.exit(0)

    tmp_dir = tempfile.mkdtemp(prefix="mutation_")
    results = {"KILLED": 0, "SURVIVED": 0, "TIMEOUT": 0, "ERROR": 0}
    survived = []
    errors = []

    print(f"2. Running mutation tests:\n")

    for i, point in enumerate(points):
        desc = point.description[:60]
        print(f"   [{i+1}/{len(points)}] {desc:<62}", end="", flush=True)

        result = mutate_and_test(str(source_path), str(test_path), point, tree, tmp_dir, baseline_time, project_root)
        status = result["status"]
        results[status] = results.get(status, 0) + 1

        if status == "KILLED":
            print(f"[KILLED] ({result['elapsed']:.1f}s)")
        elif status == "SURVIVED":
            print(f"[SURVIVED] ({result['elapsed']:.1f}s)")
            survived.append(result)
        elif status == "TIMEOUT":
            print(f"[TIMEOUT] ({result['elapsed']:.1f}s)")
            survived.append(result)
        else:
            print(f"[ERROR] ({result.get('error', 'unknown')})")
            errors.append(result)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    total = len(points)
    killed = results.get("KILLED", 0)
    survived_count = results.get("SURVIVED", 0)
    timeout_count = results.get("TIMEOUT", 0)
    error_count = results.get("ERROR", 0)

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  Total mutants : {total}")
    if total:
        print(f"  Killed        : {killed} ({killed/total*100:.1f}%)")
    else:
        print(f"  Killed        : 0")
    print(f"  Survived      : {survived_count}")
    print(f"  Timeout       : {timeout_count}")
    print(f"  Errors        : {error_count}")
    if total:
        print(f"  Mutation score: {killed/total*100:.1f}%")

    if survived:
        print(f"\n{'='*60}")
        print(f"SURVIVING MUTANTS")
        print(f"{'='*60}")
        for r in survived:
            print(f"  {r['point'].description}")

    if errors:
        print(f"\n{'='*60}")
        print(f"ERRORS")
        print(f"{'='*60}")
        for r in errors:
            print(f"  {r['point'].description}: {r.get('error', '')}")

    if survived_count > 0 or error_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
