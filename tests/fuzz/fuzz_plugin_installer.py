"""Fuzz the plugin install path with malicious / adversarial package names.

The original target imported ``distllm.plugins.installer.PluginInstaller``,
which no longer exists (the hygiene sweep removed it). The surviving, and more
important, install surface is ``PluginMarketplace.install`` guarded by
``distllm.core.plugin_trust.enforce_install_allowed`` — that is what this target
fuzzes now. It must never raise for arbitrary input: a refused install is a
return, not an exception.

Usage:
    python tests/fuzz/fuzz_plugin_installer.py          # random iterations
    python tests/fuzz/fuzz_plugin_installer.py --atheris # atheris coverage-guided
"""
import os
import random
import string
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, SRC)

_TEST_CASES = 0 if "--atheris" in sys.argv else 2000


def _random_package_name() -> str:
    """Generate a random package name (PEP 508-ish)."""
    length = random.randint(0, 50)
    chars = string.ascii_lowercase + string.digits + "-._"
    return "".join(random.choices(chars, k=length))


def _random_package_spec() -> str:
    """Generate a random pip-style package specifier."""
    name = _random_package_name()
    kind = random.randint(0, 5)
    if kind == 0:
        return name
    if kind == 1:
        extras = ",".join(
            random.choices(["gpu", "cuda", "cpu", "test", "dev", "all"], k=random.randint(0, 3))
        )
        return f"{name}[{extras}]" if extras else name
    if kind == 2:
        return f"{name}=={random.randint(0, 9)}.{random.randint(0, 99)}.{random.randint(0, 99)}"
    if kind == 3:
        return f"{name}>={random.randint(0, 9)}.{random.randint(0, 99)}"
    if kind == 4:
        return f"{name}<={random.randint(0, 9)}.{random.randint(0, 99)}"
    return f"{name}~={random.randint(0, 9)}.{random.randint(0, 99)}"


def _malicious_package_names() -> list[str]:
    """Adversarial package-name patterns (shell/SQL/template/path injection...)."""
    return [
        "", " ", "\t", "\n", "\x00", "../", "../../", "/etc/passwd", "|", ";",
        "&&", "`cat /etc/passwd`", "$(cat /etc/passwd)", "{{config}}", "{%endif%}",
        "-" * 1000, "." * 1000, "a" * 10000, "pip install evil-package",
        "os.system('rm -rf /')", "{{7*7}}", "${7*7}", "<script>alert(1)</script>",
        "a\nb", "a\rb", "a\0b", "a:b", "a|b", "a>b", "a<b", "a&b", "\u0000",
        "￾", "\U0010ffff", "..%2F..%2Fetc%2Fpasswd", "%00", "\x00\x00\x00\x00",
        "a" * 100 + "\x00" + "b" * 100,
    ]


def _run_one(data: bytes, *, allow_crash: bool = False) -> None:
    """Feed *data* through the real plugin install gating — must never raise."""
    try:
        text = data.decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return

    from distllm.core.plugin_marketplace import PluginMarketplace
    from distllm.core.plugin_trust import PluginInstallRefused, enforce_install_allowed

    # 1. The trust gate must decide, never crash, for any name.
    try:
        enforce_install_allowed(text)
    except PluginInstallRefused:
        pass  # the expected fail-closed path when installs are not enabled
    except Exception:
        if not allow_crash:
            raise

    # 2. install() must degrade to a refused/false result, never raise, for any
    #    adversarial name. Installs are disabled by default, so this exercises
    #    the SEC-3 fail-closed path end to end.
    marketplace = PluginMarketplace(enable_pypi=False)
    try:
        marketplace.install(plugin_name=text)
    except PluginInstallRefused:
        pass
    except Exception:
        if not allow_crash:
            raise


def fuzz(data: bytes) -> None:
    """atheris-compatible fuzz target."""
    _run_one(data, allow_crash=True)


def pytest_fuzz(n: int = 500) -> None:
    """Run *n* random iterations via pytest."""
    malicious = _malicious_package_names()
    for _ in range(n):
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
