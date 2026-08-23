"""Pytest configuration for security_pkg tests.

Bootstraps fake packages for distllm namespace to avoid circular-import
chains, then loads modules via ``load_module()``.
"""
from tests._import_helper import bootstrap_fake_packages

bootstrap_fake_packages()
