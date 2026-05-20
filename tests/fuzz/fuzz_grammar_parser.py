"""Fuzz the GBNF grammar parser with adversarial grammar strings.

Usage:
    python tests/fuzz/fuzz_grammar_parser.py          # 10k random iterations
    python tests/fuzz/fuzz_grammar_parser.py --atheris # atheris coverage-guided
"""
import os
import random
import string
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, SRC)


_TEST_CASES = 10000 if "--atheris" not in sys.argv else 0


def _random_identifier() -> str:
    length = random.randint(1, 20)
    return "".join(random.choices(string.ascii_letters + "_", k=length))


def _random_grammar_text() -> str:
    """Generate an adversarial GBNF grammar string."""
    rules = []
    n_rules = random.randint(1, 8)
    for _ in range(n_rules):
        name = _random_identifier()
        body = _random_expr()
        rules.append(f"{name} ::= {body}")
    return "\n".join(rules)


def _random_expr() -> str:
    kind = random.choice(["literal", "charclass", "ref", "seq", "alt", "group", "repeat", "plus", "opt", "dot"])
    if kind == "literal":
        chars = "".join(random.choices(string.printable, k=random.randint(0, 5)))
        return '"' + chars.replace("\\", "\\\\").replace('"', '\\"') + '"'
    elif kind == "charclass":
        chars = "".join(random.choices(string.printable, k=random.randint(0, 10)))
        return "[" + chars + "]"
    elif kind == "ref":
        return _random_identifier()
    elif kind == "seq":
        return " ".join(_random_expr() for _ in range(random.randint(1, 3)))
    elif kind == "alt":
        return " | ".join(_random_expr() for _ in range(random.randint(1, 3)))
    elif kind == "group":
        return "(" + _random_expr() + ")"
    elif kind == "repeat":
        return _random_expr() + "*"
    elif kind == "plus":
        return _random_expr() + "+"
    elif kind == "opt":
        return _random_expr() + "?"
    else:  # dot
        return "."


def _random_malformed() -> str:
    """Generate a deliberately malformed grammar string."""
    choices = [
        lambda: "::=",
        lambda: "\n" * random.randint(1, 10),
        lambda: " " * random.randint(0, 100),
        lambda: "\x00" * random.randint(1, 10),
        lambda: "\xff" * random.randint(1, 10),
        lambda: '"unclosed literal',
        lambda: "[unclosed charclass",
        lambda: "(unclosed group",
        lambda: "\\\\" * random.randint(1, 10),
        lambda: "".join(random.choices(string.printable, k=random.randint(1, 100))),
        lambda: "root ::= .*" * random.randint(1, 100),
    ]
    return random.choice(choices)()


def _run_one(data: bytes, *, allow_crash: bool = False) -> None:
    """Feed *data* through the parser and optionally the FSM."""
    try:
        text = data.decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return

    from distllm.core.grammar_decoder import GBNFParser, GBNFFSM

    try:
        parser = GBNFParser(text)
        rules = parser.parse()
    except Exception:
        if not allow_crash:
            raise
        return

    if not rules:
        return

    try:
        fsm = GBNFFSM(text, start_rule="root" if "root" in rules else list(rules.keys())[0])
        for _ in range(100):
            allowed = fsm.get_allowed_bytes()
            if not allowed:
                break
            byte_val = random.choice(list(allowed))
            fsm.transition(byte_val)
    except Exception:
        if not allow_crash:
            raise


def fuzz(data: bytes) -> None:
    """atheris-compatible fuzz target."""
    _run_one(data, allow_crash=True)


def pytest_fuzz(n: int = 500) -> None:
    """Run *n* random iterations — suitable for pytest."""
    for i in range(n):
        if random.random() < 0.5:
            text = _random_grammar_text()
        else:
            text = _random_malformed()
        _run_one(text.encode("utf-8", errors="replace"), allow_crash=True)


if __name__ == "__main__":
    if "--atheris" in sys.argv:
        import atheris
        atheris.Setup(sys.argv, fuzz)
        atheris.Fuzz()
    else:
        pytest_fuzz(_TEST_CASES)
        print(f"OK {_TEST_CASES} iterations completed, no crashes")
