"""Fuzz-test harnesses for DistLLM.

Each module exposes a ``fuzz()`` function that atheris / a simple runner
can drive, plus an optional ``pytest_fuzz()`` that runs a fixed number of
random iterations suitable for pytest.
"""
