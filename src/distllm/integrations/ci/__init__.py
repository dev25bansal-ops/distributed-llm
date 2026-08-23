"""CI/CD integration modules for DistLLM.

Provides connectors for GitLab CI and Jenkins to trigger evaluation
pipelines, fetch results, and manage build artifacts.

Export::

    from distllm.integrations.ci.gitlab import GitLabCIIntegration
    from distllm.integrations.ci.jenkins import JenkinsIntegration
"""

from __future__ import annotations

from distllm.integrations.ci._common import BuildInfo as BuildInfo
from distllm.integrations.ci._common import EvalResult as EvalResult
from distllm.integrations.ci.gitlab import GitLabCIIntegration as GitLabCIIntegration
from distllm.integrations.ci.jenkins import JenkinsIntegration as JenkinsIntegration

__all__ = [
    "BuildInfo",
    "EvalResult",
    "GitLabCIIntegration",
    "JenkinsIntegration",
]
