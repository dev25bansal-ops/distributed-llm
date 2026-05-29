"""Unit tests for verification modules (direct file loading, no circ import).

Avoids distllm package import (which has circular import) by loading
individual .py files directly with exec() into fresh module namespaces,
pre-populating sys.modules with fake parent packages.
"""
import sys, json, hashlib, types, torch


def _load_as_module(filepath, dotted_name):
    """Execute filepath as a module with the given dotted_name.

    Pre-populates sys.modules with fake parent packages so that
    'from distllm.verification.comparator import ...' works inside
    the exec'd code without triggering the real distllm import.
    """
    # Ensure all parent packages exist as namespace holders
    parts = dotted_name.split(".")
    parent = None
    for i in range(1, len(parts) + 1):
        partial = ".".join(parts[:i])
        if partial not in sys.modules:
            pkg = types.ModuleType(partial)
            pkg.__path__ = []  # mark as package
            pkg.__package__ = partial
            pkg.__file__ = None
            # Chain parent relationship
            if parent is not None:
                child_name = parts[i - 1]
                setattr(parent, child_name, pkg)
                pkg.__package__ = parent.__name__ + "." + child_name
            if i < len(parts):
                pkg.__path__ = []  # it's a namespace package
            pkg.__name__ = partial
            sys.modules[partial] = pkg
        parent = sys.modules[partial]

    # Create the actual module and populate its namespace
    mod = types.ModuleType(dotted_name)
    mod.__file__ = filepath
    mod.__package__ = ".".join(parts[:-1]) if len(parts) > 1 else ""
    mod.__name__ = dotted_name
    # Wire it into parent
    sys.modules[dotted_name] = mod
    if parent is not None:
        setattr(parent, parts[-1], mod)

    with open(filepath) as f:
        src = f.read()
    exec(compile(src, filepath, "exec"), mod.__dict__)
    return mod


# Load modules: order matters due to dependencies
comp_mod = _load_as_module(
    "src/distllm/verification/comparator.py",
    "distllm.verification.comparator",
)
hash_mod = _load_as_module(
    "src/distllm/verification/hash_registry.py",
    "distllm.verification.hash_registry",
)
report_mod = _load_as_module(
    "src/distllm/verification/report.py",
    "distllm.verification.report",
)

# Verify the fake package structure works
assert sys.modules["distllm"] is not None
assert sys.modules["distllm.verification"] is not None
assert sys.modules["distllm.verification.comparator"] is comp_mod
assert comp_mod.compare_logits is not None
assert hash_mod.compute_output_hash is not None
assert report_mod.VerificationReport is not None
print("Module loading: OK")

# ============ TESTS ============
passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name}")

# === Comparator tests ===
g = torch.randn(2, 5, 100)
c = g.clone()
r = comp_mod.compare_logits(g, c)
check("logits identical cosim~1", r["cosine_sim"] > 0.9999)
check("logits identical kl_div=0", r["kl_div"] == 0.0)
check("logits identical max_abs=0", r["max_abs_diff"] == 0.0)

c2 = g * 1.1 + torch.randn_like(g) * 0.1
r2 = comp_mod.compare_logits(g, c2)
check("logits different cosim<1", r2["cosine_sim"] < 1.0)
check("logits different kl_div>0", r2["kl_div"] > 0.0)
check("logits different max_abs>0", r2["max_abs_diff"] > 0.0)

h_g = torch.randn(2, 10, 64)
h_c = h_g.clone()
r3 = comp_mod.compare_hidden_states(h_g, h_c)
check("hidden identical cosim~1", r3["cosine_sim"] > 0.9999)
check("hidden identical max_abs=0", r3["max_abs_diff"] == 0.0)
check("hidden identical rel_err=0", r3["relative_error"] == 0.0)

tk = comp_mod.compare_tokens([1,2,3,4], [1,2,3,4])
check("tokens exact=1", tk["exact_match"] == 1.0)
check("tokens edit=0", tk["edit_distance"] == 0.0)

tk2 = comp_mod.compare_tokens([1,2,3,4], [1,2,5,4])
check("tokens partial exact=0.75", tk2["exact_match"] == 0.75)
check("tokens partial edit=0.25", tk2["edit_distance"] == 0.25)

tx = comp_mod.compare_text("hello world", "hello world")
check("text exact=1", tx["exact_match"] == 1.0)
check("text overlap=1", tx["token_overlap"] == 1.0)

tx2 = comp_mod.compare_text("hello world", "goodbye world")
check("text diff exact=0", tx2["exact_match"] == 0.0)
check("text diff overlap<1", tx2["token_overlap"] < 1.0)

gold = {
    "token_exact_match": 1.0, "token_edit_distance": 0.0,
    "logit_cosine_sim": 1.0, "logit_kl_div": 0.0, "logit_max_abs_diff": 0.0,
}
comp = comp_mod.evaluate_comparison(gold)
check("eval pass=true", comp.pass_threshold == True)

bad = dict(gold, token_exact_match=0.5)
comp2 = comp_mod.evaluate_comparison(bad)
check("eval fail=false", comp2.pass_threshold == False)

# === Hash registry tests ===
t = torch.tensor([1.0, 2.0, 3.0])
h1 = hash_mod.compute_output_hash(t)
h2 = hash_mod.compute_output_hash(t.clone())
check("hash stable", h1 == h2)
check("hash different", h1 != hash_mod.compute_output_hash(torch.tensor([1.0, 2.0, 3.0, 4.0])))
check("text hash len=64", len(hash_mod.compute_text_hash("hello")) == 64)
check("token hash len=64", len(hash_mod.compute_token_ids_hash([1,2,3])) == 64)

registry = hash_mod.OutputHashRegistry()
ref_out = hash_mod.GenerationOutput(token_ids=[1,2,3], text="hello", step_logits=[torch.randn(1, 100)])
cand_out = hash_mod.GenerationOutput(token_ids=[1,2,3], text="hello", step_logits=[torch.randn(1, 100)])
registry.store_reference("p1", ref_out)
registry.store_candidate("p1", cand_out)
cr = registry.compare("p1")
check("hash token match", cr.get("token_ids") == True)
check("hash text match", cr.get("text") == True)

bad_cand = hash_mod.GenerationOutput(token_ids=[1,2,4], text="world", step_logits=[torch.randn(1, 100)])
registry.store_reference("p2", hash_mod.GenerationOutput(token_ids=[1,2,3], text="hello", step_logits=[]))
registry.store_candidate("p2", bad_cand)
cr2 = registry.compare("p2")
check("hash detect mismatch", cr2.get("token_ids") == False)

# === Report tests ===
report = report_mod.generate_report(
    comparisons=[comp, comp2],
    per_prompt_data=[
        {"prompt": "t1", "comparison": comp, "reference": ref_out, "candidate": cand_out},
        {"prompt": "t2", "comparison": comp2, "reference": ref_out, "candidate": bad_cand},
    ],
    hash_registry=registry,
    model_name="test-model",
    num_nodes=2,
)
s = report.summary()
check("report total=2", s["total"] == 2)
check("report passed=1", s["passed"] == 1)
check("report failed=1", s["failed"] == 1)

data = json.loads(report.to_json())
check("json prompts", "prompts" in data)
check("json summary", "summary" in data)
check("json total=2", data["summary"]["total"] == 2)
check("json passed=1", data["summary"]["passed"] == 1)

report2 = report_mod.VerificationReport()
s2 = report2.summary()
check("empty report total=0", s2["total"] == 0)
check("empty pass_rate=0", s2["pass_rate"] == 0.0)

# Test print doesn't crash
report.print_human_readable()
check("print no crash", True)

print("\n" + "=" * 40)
print(f"Results: {passed} passed, {failed} failed out of {passed+failed}")
if failed > 0:
    sys.exit(1)
print("ALL VERIFICATION TESTS PASSED")
