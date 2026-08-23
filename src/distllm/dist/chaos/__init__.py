"""Chaos engineering subpackage for distributed inference.

Provides fault injection, network partition simulation,
node failure testing, and resilience validation utilities.

Exports
-------
FaultInjector
    gRPC-interceptor-compatible fault injector for latency, error,
    and message-drop injection.
ChaosScenario
    Named, bounded chaos experiment with factory methods for common
    scenarios (node failure, network partition, latency spike, etc.).
SteadyStateChecker / SteadyStateMetrics
    Metrics-capture and comparison helpers for verifying the
    steady-state hypothesis before/after an experiment.
BlastRadiusControl
    Limits chaos to specific nodes, services, or methods, with
    maintenance mode support.
ChaosTemplate
    Generates Litmus and Chaos Mesh experiment YAML from scenarios.
"""

from __future__ import annotations

from distllm.dist.chaos.fault_injector import (
    BlastRadiusControl,
    ChaosScenario,
    ChaosScenarioType,
    ChaosTemplate,
    ErrorFault,
    FaultInjector,
    FaultType,
    LatencyFault,
    MessageDropFault,
    SteadyStateChecker,
    SteadyStateMetrics,
)

__all__ = [
    "BlastRadiusControl",
    "ChaosScenario",
    "ChaosScenarioType",
    "ChaosTemplate",
    "ErrorFault",
    "FaultInjector",
    "FaultType",
    "LatencyFault",
    "MessageDropFault",
    "SteadyStateChecker",
    "SteadyStateMetrics",
]
