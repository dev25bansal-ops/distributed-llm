"""Comprehensive CLI for cluster management — beyond partition/cli.py.

Covers cluster status, node management, federation control,
recovery triggers, power capping, and tenant SLO management.

Usage:
    distllm-admin cluster status
    distllm-admin nodes list
    distllm-admin nodes drain node-3
    distllm-admin federation peers
    distllm-admin recovery drill
    distllm-admin power status
    distllm-admin tenants list
    distllm-admin regions list
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import httpx
from loguru import logger


def _client(base_url: str, api_key: str | None = None) -> httpx.Client:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return httpx.Client(base_url=base_url, headers=headers, timeout=15.0)


def _print(data: Any, raw: bool = False) -> None:
    if raw:
        print(json.dumps(data, indent=2, default=str))
    elif isinstance(data, list):
        for item in data:
            print(json.dumps(item, indent=2, default=str))
            print("---")
    else:
        print(json.dumps(data, indent=2, default=str))


# ── Cluster ───────────────────────────────────────────────────────────

def cmd_cluster_status(args: argparse.Namespace) -> None:
    client = _client(args.url, args.api_key)
    resp = client.get("/admin/v1/cluster/status")
    resp.raise_for_status()
    _print(resp.json(), args.raw)


# ── Nodes ─────────────────────────────────────────────────────────────

def cmd_nodes_list(args: argparse.Namespace) -> None:
    client = _client(args.url, args.api_key)
    resp = client.get("/admin/v1/nodes")
    if resp.status_code == 401:
        print("Unauthorized — provide --api-key")
        return
    resp.raise_for_status()
    _print(resp.json(), args.raw)


def cmd_nodes_drain(args: argparse.Namespace) -> None:
    client = _client(args.url, args.api_key)
    resp = client.post(f"/admin/v1/nodes/{args.node_id}/drain")
    if resp.status_code == 200:
        print(f"Node {args.node_id} is now draining")
    else:
        print(f"Failed: {resp.status_code} {resp.text}")


def cmd_nodes_remove(args: argparse.Namespace) -> None:
    client = _client(args.url, args.api_key)
    resp = client.delete(f"/admin/v1/nodes/{args.node_id}")
    if resp.status_code == 200:
        print(f"Node {args.node_id} removed")
    else:
        print(f"Failed: {resp.status_code} {resp.text}")


# ── Federation ────────────────────────────────────────────────────────

def cmd_fed_peers(args: argparse.Namespace) -> None:
    client = _client(args.url, args.api_key)
    resp = client.get("/v1/federation/peers")
    resp.raise_for_status()
    _print(resp.json(), args.raw)


def cmd_fed_status(args: argparse.Namespace) -> None:
    client = _client(args.url, args.api_key)
    resp = client.get("/v1/federation/health")
    resp.raise_for_status()
    _print(resp.json(), args.raw)


# ── Recovery ──────────────────────────────────────────────────────────

def cmd_recovery_status(args: argparse.Namespace) -> None:
    client = _client(args.url, args.api_key)
    resp = client.get("/admin/v1/recovery/status")
    resp.raise_for_status()
    _print(resp.json(), args.raw)


def cmd_recovery_history(args: argparse.Namespace) -> None:
    client = _client(args.url, args.api_key)
    resp = client.get("/admin/v1/recovery/history")
    resp.raise_for_status()
    _print(resp.json(), args.raw)


def cmd_recovery_drill(args: argparse.Namespace) -> None:
    client = _client(args.url, args.api_key)
    resp = client.post("/admin/v1/recovery/drill")
    if resp.status_code == 200:
        result = resp.json()
        status = "PASSED" if result.get("passed") else "FAILED"
        print(f"Recovery drill {status}")
        print(f"  Time: {result.get('recovery_time_ms', 0):.0f}ms")
        print(f"  Sequences: {result.get('sequences_recovered', 0)} recovered, "
              f"{result.get('sequences_lost', 0)} lost")
        print(f"  Redistributions: {result.get('redistributions', 0)}")
        if result.get("failures"):
            for f in result["failures"]:
                print(f"  ⚠ {f}")
    else:
        print(f"Drill failed: {resp.status_code} {resp.text}")


# ── Power ─────────────────────────────────────────────────────────────

def cmd_power_status(args: argparse.Namespace) -> None:
    client = _client(args.url, args.api_key)
    resp = client.get("/admin/v1/power/status")
    resp.raise_for_status()
    _print(resp.json(), args.raw)


def cmd_power_autotune(args: argparse.Namespace) -> None:
    client = _client(args.url, args.api_key)
    resp = client.post("/admin/v1/power/auto-tune")
    if resp.status_code == 200:
        print("Auto-tune complete:")
        _print(resp.json(), args.raw)
    elif resp.status_code == 202:
        print("Auto-tune started (async)")
    else:
        print(f"Failed: {resp.status_code} {resp.text}")


# ── Tenants ───────────────────────────────────────────────────────────

def cmd_tenants_list(args: argparse.Namespace) -> None:
    client = _client(args.url, args.api_key)
    resp = client.get("/admin/v1/tenants")
    resp.raise_for_status()
    _print(resp.json(), args.raw)


def cmd_tenants_get(args: argparse.Namespace) -> None:
    client = _client(args.url, args.api_key)
    resp = client.get(f"/admin/v1/tenants/{args.tenant_id}")
    if resp.status_code == 200:
        _print(resp.json(), args.raw)
    else:
        print(f"Tenant {args.tenant_id} not found")


# ── Regions ───────────────────────────────────────────────────────────

def cmd_regions_list(args: argparse.Namespace) -> None:
    client = _client(args.url, args.api_key)
    params = {"min_gpu_memory_gb": args.min_gpu_memory_gb}
    resp = client.get("/api/v1/regions", params=params)
    resp.raise_for_status()
    _print(resp.json(), args.raw)


# ── Deployments ───────────────────────────────────────────────────────

def cmd_deploy_list(args: argparse.Namespace) -> None:
    client = _client(args.url, args.api_key)
    params = {}
    if args.tenant_id:
        params["tenant_id"] = args.tenant_id
    if args.status:
        params["status"] = args.status
    resp = client.get("/api/v1/provisioning/deployments", params=params)
    resp.raise_for_status()
    _print(resp.json(), args.raw)


# ── Parser ────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DistLLM cluster administration CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--url", default="http://localhost:8000",
                        help="Coordinator URL")
    parser.add_argument("--api-key", default=None,
                        help="API key for authenticated endpoints")
    parser.add_argument("--raw", action="store_true",
                        help="Output raw JSON")

    sub = parser.add_subparsers(dest="command", required=True)

    # cluster
    p = sub.add_parser("cluster", help="Cluster commands")
    p_sub = p.add_subparsers(dest="subcommand", required=True)
    p_sub.add_parser("status", help="Cluster status").set_defaults(func=cmd_cluster_status)

    # nodes
    p = sub.add_parser("nodes", help="Node management")
    p_sub = p.add_subparsers(dest="subcommand", required=True)
    p_sub.add_parser("list", help="List nodes").set_defaults(func=cmd_nodes_list)
    pn = p_sub.add_parser("drain", help="Drain a node")
    pn.add_argument("node_id")
    pn.set_defaults(func=cmd_nodes_drain)
    pn = p_sub.add_parser("remove", help="Remove a node")
    pn.add_argument("node_id")
    pn.set_defaults(func=cmd_nodes_remove)

    # federation
    p = sub.add_parser("federation", help="Federation commands")
    p_sub = p.add_subparsers(dest="subcommand", required=True)
    p_sub.add_parser("peers", help="List peers").set_defaults(func=cmd_fed_peers)
    p_sub.add_parser("status", help="Federation health").set_defaults(func=cmd_fed_status)

    # recovery
    p = sub.add_parser("recovery", help="Recovery commands")
    p_sub = p.add_subparsers(dest="subcommand", required=True)
    p_sub.add_parser("status", help="Recovery status").set_defaults(func=cmd_recovery_status)
    p_sub.add_parser("history", help="Recovery history").set_defaults(func=cmd_recovery_history)
    p_sub.add_parser("drill", help="Run recovery drill").set_defaults(func=cmd_recovery_drill)

    # power
    p = sub.add_parser("power", help="Power management")
    p_sub = p.add_subparsers(dest="subcommand", required=True)
    p_sub.add_parser("status", help="GPU power status").set_defaults(func=cmd_power_status)
    p_sub.add_parser("auto-tune", help="Auto-tune power limits").set_defaults(func=cmd_power_autotune)

    # tenants
    p = sub.add_parser("tenants", help="Tenant SLO management")
    p_sub = p.add_subparsers(dest="subcommand", required=True)
    p_sub.add_parser("list", help="List tenants").set_defaults(func=cmd_tenants_list)
    pt = p_sub.add_parser("get", help="Get tenant details")
    pt.add_argument("tenant_id")
    pt.set_defaults(func=cmd_tenants_get)

    # regions
    p = sub.add_parser("regions", help="Cloud regions")
    p_sub = p.add_subparsers(dest="subcommand", required=True)
    pr = p_sub.add_parser("list", help="List available regions")
    pr.add_argument("--min-gpu-memory-gb", type=float, default=80.0)
    pr.set_defaults(func=cmd_regions_list)

    # deployments
    p = sub.add_parser("deployments", help="Deployment management")
    p_sub = p.add_subparsers(dest="subcommand", required=True)
    pd = p_sub.add_parser("list", help="List deployments")
    pd.add_argument("--tenant-id", default=None)
    pd.add_argument("--status", default=None)
    pd.set_defaults(func=cmd_deploy_list)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except httpx.HTTPStatusError as e:
        print(f"HTTP {e.response.status_code}: {e.response.text}", file=sys.stderr)
        sys.exit(1)
    except httpx.ConnectError as e:
        print(f"Connection failed: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
