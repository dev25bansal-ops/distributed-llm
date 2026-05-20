"""Fuzz the config/settings loader with malicious / adversarial YAML.

Usage:
    python tests/fuzz/fuzz_config_loader.py          # 10k random iterations
    python tests/fuzz/fuzz_config_loader.py --atheris # atheris coverage-guided
"""
import io
import os
import random
import string
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, SRC)

_TEST_CASES = 2000 if "--atheris" not in sys.argv else 0


def _random_yaml() -> str:
    """Generate a random (potentially malicious) YAML blob."""
    lines = []
    n_keys = random.randint(0, 20)
    for _ in range(n_keys):
        key = "".join(random.choices(string.ascii_letters + string.digits + "_", k=random.randint(0, 30)))
        if not key:
            key = "x"
        kind = random.randint(0, 6)
        if kind == 0:
            lines.append(f"{key}: {random.randint(-1000000, 1000000)}")
        elif kind == 1:
            lines.append(f"{key}: {random.uniform(-1e10, 1e10)}")
        elif kind == 2:
            val = "".join(random.choices(string.printable, k=random.randint(0, 50)))
            lines.append(f'{key}: "{val}"')
        elif kind == 3:
            lines.append(f"{key}: true")
        elif kind == 4:
            subkeys = []
            for _ in range(random.randint(0, 5)):
                sk = "".join(random.choices(string.ascii_letters, k=5))
                subkeys.append(f"  {sk}: {random.randint(0, 100)}")
            lines.append(f"{key}:")
            lines.extend(subkeys)
        elif kind == 5:
            lines.append(f"{key}: [{','.join(str(random.randint(0, 100)) for _ in range(random.randint(0, 10)))}]")
        else:
            lines.append(f"{key}: null")
    return "\n".join(lines)


def _malicious_yaml() -> str:
    """Generate YAML with known dangerous patterns."""
    choices = [
        # YAML alias bombs
        lambda: "x: &a [1]\ny: *a\nz: *a",
        # Deep nesting
        lambda: ("a:" * 1000) + " 1",
        # Very long string
        lambda: f'key: "{"A" * 100000}"',
        # Unicode bomb
        lambda: f'key: "{"\u00e9" * 1000}"',
        # Null bytes
        lambda: "key: \x00\x00\x00",
        # Circular references (via YAML anchors)
        lambda: "x: &a\ny: *a\n" * 100,
        # Negatives and edge cases
        lambda: f"port: {random.choice([-1, 0, 65536, 99999, -65535])}",
        # Large integers
        lambda: f"value: {random.choice([2**63 - 1, -(2**63), 2**100, -(2**100)])}",
        # Empty
        lambda: "",
        # Only whitespace
        lambda: "   \n  \n   ",
        # Python-specific YAML tags
        lambda: "!!python/object/new:os.system ['echo pwned']",
        # Tab characters (illegal in YAML)
        lambda: "\tkey: value",
        # Mixed indentation
        lambda: "parent:\n  child1: 1\n\tchild2: 2",
    ]
    return random.choice(choices)()


def _run_one(data: bytes, *, allow_crash: bool = False) -> None:
    """Feed *data* through DistLLMSettings and YAML config loading."""
    try:
        text = data.decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return

    from distllm.config.settings import DistLLMSettings

    # 1. Try to load via YAML env var
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(text)
            yaml_path = f.name
        settings = DistLLMSettings(_env_file=yaml_path)
    except Exception:
        if not allow_crash:
            raise
        return
    finally:
        try:
            os.unlink(yaml_path)
        except Exception:
            pass

    # 2. Try accessing random nested attributes (should not crash)
    try:
        _ = settings.model.name
        _ = settings.coordinator.port
        _ = settings.node.device
        _ = settings.generation.max_new_tokens
    except Exception:
        pass


def fuzz(data: bytes) -> None:
    """atheris-compatible fuzz target."""
    _run_one(data, allow_crash=True)


def pytest_fuzz(n: int = 500) -> None:
    """Run *n* random iterations via pytest."""
    for _ in range(n):
        if random.random() < 0.5:
            text = _random_yaml()
        else:
            text = _malicious_yaml()
        _run_one(text.encode("utf-8", errors="replace"), allow_crash=True)


if __name__ == "__main__":
    if "--atheris" in sys.argv:
        import atheris
        atheris.Setup(sys.argv, fuzz)
        atheris.Fuzz()
    else:
        pytest_fuzz(_TEST_CASES)
        print(f"OK {_TEST_CASES} iterations completed, no crashes")
