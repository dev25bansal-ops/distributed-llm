"""Fuzz the plugin installer with malicious / adversarial package names.

Usage:
    python tests/fuzz/fuzz_plugin_installer.py          # 10k random iterations
    python tests/fuzz/fuzz_plugin_installer.py --atheris # atheris coverage-guided
"""
import os
import random
import re
import string
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, SRC)

_TEST_CASES = 2000 if "--atheris" not in sys.argv else 0


def _random_package_name() -> str:
    """Generate a random package name (PEP 508-ish)."""
    length = random.randint(0, 50)
    chars = string.ascii_lowercase + string.digits + "-._"
    name = "".join(random.choices(chars, k=length))
    return name


def _random_package_spec() -> str:
    """Generate a random pip-style package specifier."""
    name = _random_package_name()

    kind = random.randint(0, 5)
    if kind == 0:
        return name  # plain
    elif kind == 1:
        extras = ",".join(random.choices(
            [f"{x}" for x in ["gpu", "cuda", "cpu", "test", "dev", "all"]],
            k=random.randint(0, 3),
        ))
        return f"{name}[{extras}]" if extras else name
    elif kind == 2:
        return f"{name}=={random.randint(0, 9)}.{random.randint(0, 99)}.{random.randint(0, 99)}"
    elif kind == 3:
        return f"{name}>={random.randint(0, 9)}.{random.randint(0, 99)}"
    elif kind == 4:
        return f"{name}<={random.randint(0, 9)}.{random.randint(0, 99)}"
    else:
        return f"{name}~={random.randint(0, 9)}.{random.randint(0, 99)}"


def _malicious_package_names() -> list[str]:
    """Return list of adversarial package name patterns."""
    return [
        "",
        " ",
        "\t",
        "\n",
        "\x00",
        "../",
        "../../",
        "/etc/passwd",
        "|",
        ";",
        "&&",
        "`cat /etc/passwd`",
        "$(cat /etc/passwd)",
        "{{config}}",
        "{%endif%}",
        "-" * 1000,
        "." * 1000,
        "a" * 10000,
        "pip install evil-package",
        "os.system('rm -rf /')",
        "{{7*7}}",
        "${7*7}",
        "<script>alert(1)</script>",
        "a\nb",
        "a\rb",
        "a\0b",
        "a:b",
        "a|b",
        "a>b",
        "a<b",
        "a&b",
        "\u0000",
        "\ufffe",
        "\U0010ffff",
        "..%2F..%2Fetc%2Fpasswd",
        "%00",
        "\x00\x00\x00\x00",
        "a" * 100 + "\x00" + "b" * 100,
    ]


def _run_one(data: bytes, *, allow_crash: bool = False) -> None:
    """Feed *data* through PluginInstaller logic."""
    try:
        text = data.decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return

    from distllm.plugins.installer import PluginInstaller, PluginInstallResult

    installer = PluginInstaller(plugin_registry_url="http://localhost:1/nonexistent")

    # Stub out pip subprocess calls to avoid actual network/subprocess I/O
    installer._pip_install = lambda spec: (False, ["pip disabled in fuzz"])
    installer._pip_uninstall = lambda spec: (False, ["pip disabled in fuzz"])
    installer._load_installed_metadata = lambda name: None

    # 1. Try install() with the package name
    try:
        result = installer.install(plugin_name=text)
    except Exception:
        if not allow_crash:
            raise
        return

    # 2. Try with version
    try:
        result = installer.install(plugin_name=text, version="1.0.0")
    except Exception:
        if not allow_crash:
            raise

    # 3. Try with extras
    try:
        result = installer.install(plugin_name=text, extras=["gpu", "cuda"])
    except Exception:
        if not allow_crash:
            raise

    # 4. Try list_installed() and uninstall() — should never throw
    try:
        _ = installer.list_installed()
        installer.uninstall(text)
    except Exception:
        if not allow_crash:
            raise


def fuzz(data: bytes) -> None:
    """atheris-compatible fuzz target."""
    _run_one(data, allow_crash=True)


def pytest_fuzz(n: int = 500) -> None:
    """Run *n* random iterations via pytest."""
    malicious = _malicious_package_names()
    for i in range(n):
        if random.random() < 0.5:
            text = _random_package_spec()
        else:
            text = random.choice(malicious)
        _run_one(text.encode("utf-8", errors="replace"), allow_crash=True)


if __name__ == "__main__":
    if "--atheris" in sys.argv:
        import atheris
        atheris.Setup(sys.argv, fuzz)
        atheris.Fuzz()
    else:
        pytest_fuzz(_TEST_CASES)
        print(f"OK {_TEST_CASES} iterations completed, no crashes")
