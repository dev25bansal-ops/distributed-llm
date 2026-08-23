"""Tests for distllm.dist.admin_cli -- CLI argument parsing and command structure.

Tests the parser construction, argument parsing, and error behavior of
command handlers without requiring a running coordinator server.
"""

from __future__ import annotations

import argparse
import sys

import httpx
import pytest

from distllm.dist.admin_cli import (
    build_parser,
    cmd_cluster_status,
    cmd_deploy_list,
    cmd_fed_peers,
    cmd_fed_status,
    cmd_nodes_drain,
    cmd_nodes_list,
    cmd_nodes_remove,
    cmd_power_autotune,
    cmd_power_status,
    cmd_recovery_drill,
    cmd_recovery_history,
    cmd_recovery_status,
    cmd_regions_list,
    cmd_tenants_get,
    cmd_tenants_list,
    main,
)


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


class TestBuildParser:
    """build_parser() constructs a correct argument parser."""

    def test_returns_argument_parser(self) -> None:
        """Returns an ArgumentParser instance."""
        parser = build_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    # -- Global arguments --------------------------------------------------

    def test_default_url(self) -> None:
        """Default --url is http://localhost:8000."""
        parser = build_parser()
        args = parser.parse_args(["cluster", "status"])
        assert args.url == "http://localhost:8000"

    def test_custom_url(self) -> None:
        """--url overrides the default coordinator URL."""
        parser = build_parser()
        args = parser.parse_args(
            ["--url", "http://example.com:9000", "cluster", "status"]
        )
        assert args.url == "http://example.com:9000"

    def test_default_api_key_is_none(self) -> None:
        """--api-key defaults to None."""
        parser = build_parser()
        args = parser.parse_args(["cluster", "status"])
        assert args.api_key is None

    def test_custom_api_key(self) -> None:
        """--api-key is forwarded to args."""
        parser = build_parser()
        args = parser.parse_args(["--api-key", "sk-test", "cluster", "status"])
        assert args.api_key == "sk-test"

    def test_raw_default_false(self) -> None:
        """--raw defaults to False."""
        parser = build_parser()
        args = parser.parse_args(["cluster", "status"])
        assert args.raw is False

    def test_raw_flag(self) -> None:
        """--raw sets raw=True."""
        parser = build_parser()
        args = parser.parse_args(["--raw", "cluster", "status"])
        assert args.raw is True

    # -- Cluster subcommand ------------------------------------------------

    def test_cluster_status_routing(self) -> None:
        """cluster status routes to cmd_cluster_status."""
        parser = build_parser()
        args = parser.parse_args(["cluster", "status"])
        assert args.func is cmd_cluster_status

    # -- Nodes subcommands -------------------------------------------------

    def test_nodes_list_routing(self) -> None:
        """nodes list routes to cmd_nodes_list."""
        parser = build_parser()
        args = parser.parse_args(["nodes", "list"])
        assert args.func is cmd_nodes_list

    def test_nodes_drain_routing(self) -> None:
        """nodes drain <id> routes to cmd_nodes_drain with node_id."""
        parser = build_parser()
        args = parser.parse_args(["nodes", "drain", "node-3"])
        assert args.func is cmd_nodes_drain
        assert args.node_id == "node-3"

    def test_nodes_remove_routing(self) -> None:
        """nodes remove <id> routes to cmd_nodes_remove with node_id."""
        parser = build_parser()
        args = parser.parse_args(["nodes", "remove", "node-7"])
        assert args.func is cmd_nodes_remove
        assert args.node_id == "node-7"

    # -- Federation subcommands --------------------------------------------

    def test_federation_peers_routing(self) -> None:
        """federation peers routes to cmd_fed_peers."""
        parser = build_parser()
        args = parser.parse_args(["federation", "peers"])
        assert args.func is cmd_fed_peers

    def test_federation_status_routing(self) -> None:
        """federation status routes to cmd_fed_status."""
        parser = build_parser()
        args = parser.parse_args(["federation", "status"])
        assert args.func is cmd_fed_status

    # -- Recovery subcommands ----------------------------------------------

    def test_recovery_status_routing(self) -> None:
        """recovery status routes to cmd_recovery_status."""
        parser = build_parser()
        args = parser.parse_args(["recovery", "status"])
        assert args.func is cmd_recovery_status

    def test_recovery_history_routing(self) -> None:
        """recovery history routes to cmd_recovery_history."""
        parser = build_parser()
        args = parser.parse_args(["recovery", "history"])
        assert args.func is cmd_recovery_history

    def test_recovery_drill_routing(self) -> None:
        """recovery drill routes to cmd_recovery_drill."""
        parser = build_parser()
        args = parser.parse_args(["recovery", "drill"])
        assert args.func is cmd_recovery_drill

    # -- Power subcommands -------------------------------------------------

    def test_power_status_routing(self) -> None:
        """power status routes to cmd_power_status."""
        parser = build_parser()
        args = parser.parse_args(["power", "status"])
        assert args.func is cmd_power_status

    def test_power_autotune_routing(self) -> None:
        """power auto-tune routes to cmd_power_autotune."""
        parser = build_parser()
        args = parser.parse_args(["power", "auto-tune"])
        assert args.func is cmd_power_autotune

    # -- Tenants subcommands -----------------------------------------------

    def test_tenants_list_routing(self) -> None:
        """tenants list routes to cmd_tenants_list."""
        parser = build_parser()
        args = parser.parse_args(["tenants", "list"])
        assert args.func is cmd_tenants_list

    def test_tenants_get_routing(self) -> None:
        """tenants get <id> routes to cmd_tenants_get with tenant_id."""
        parser = build_parser()
        args = parser.parse_args(["tenants", "get", "tenant-xyz"])
        assert args.func is cmd_tenants_get
        assert args.tenant_id == "tenant-xyz"

    # -- Regions subcommands -----------------------------------------------

    def test_regions_list_routing(self) -> None:
        """regions list routes to cmd_regions_list."""
        parser = build_parser()
        args = parser.parse_args(["regions", "list"])
        assert args.func is cmd_regions_list

    def test_regions_list_default_gpu_memory(self) -> None:
        """regions list default min_gpu_memory_gb is 80.0."""
        parser = build_parser()
        args = parser.parse_args(["regions", "list"])
        assert args.min_gpu_memory_gb == 80.0

    def test_regions_list_custom_gpu_memory(self) -> None:
        """--min-gpu-memory-gb accepts a custom float."""
        parser = build_parser()
        args = parser.parse_args(
            ["regions", "list", "--min-gpu-memory-gb", "120.5"]
        )
        assert args.min_gpu_memory_gb == 120.5

    # -- Deployments subcommands -------------------------------------------

    def test_deploy_list_routing(self) -> None:
        """deployments list routes to cmd_deploy_list."""
        parser = build_parser()
        args = parser.parse_args(["deployments", "list"])
        assert args.func is cmd_deploy_list

    def test_deploy_list_defaults(self) -> None:
        """deployments list has None defaults for tenant_id and status."""
        parser = build_parser()
        args = parser.parse_args(["deployments", "list"])
        assert args.tenant_id is None
        assert args.status is None

    def test_deploy_list_filters(self) -> None:
        """--tenant-id and --status filters are passed through."""
        parser = build_parser()
        args = parser.parse_args(
            ["deployments", "list", "--tenant-id", "acme", "--status", "active"]
        )
        assert args.tenant_id == "acme"
        assert args.status == "active"

    # -- Error cases -------------------------------------------------------

    def test_missing_command_exits(self) -> None:
        """Parser exits when no command is provided."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_missing_subcommand_exits(self) -> None:
        """Parser exits when no subcommand is provided."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["cluster"])

    def test_invalid_subcommand_exits(self) -> None:
        """Parser exits for an unknown subcommand."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["cluster", "nonexistent"])

    # -- Routing completeness ----------------------------------------------

    def test_all_subcommands_route_to_unique_funcs(self) -> None:
        """Every known subcommand maps to a distinct function."""
        parser = build_parser()
        routes = [
            (["cluster", "status"], cmd_cluster_status),
            (["nodes", "list"], cmd_nodes_list),
            (["nodes", "drain", "x"], cmd_nodes_drain),
            (["nodes", "remove", "x"], cmd_nodes_remove),
            (["federation", "peers"], cmd_fed_peers),
            (["federation", "status"], cmd_fed_status),
            (["recovery", "status"], cmd_recovery_status),
            (["recovery", "history"], cmd_recovery_history),
            (["recovery", "drill"], cmd_recovery_drill),
            (["power", "status"], cmd_power_status),
            (["power", "auto-tune"], cmd_power_autotune),
            (["tenants", "list"], cmd_tenants_list),
            (["tenants", "get", "x"], cmd_tenants_get),
            (["regions", "list"], cmd_regions_list),
            (["deployments", "list"], cmd_deploy_list),
        ]
        seen: set[object] = set()
        for argv, expected_func in routes:
            args = parser.parse_args(argv)
            assert args.func is expected_func, (
                f"{' '.join(argv)} routed to {args.func}, expected {expected_func}"
            )
            assert args.func not in seen, f"{args.func} used by multiple subcommands"
            seen.add(args.func)


# ---------------------------------------------------------------------------
# Command handler tests
# ---------------------------------------------------------------------------


class TestCommandHandlers:
    """Command handlers raise ConnectError when coordinator is unreachable."""

    OFF_URL = "http://127.0.0.1:1"

    @staticmethod
    def _make_args(**extra: object) -> argparse.Namespace:
        """Build a Namespace with base fields and extras."""
        base: dict[str, object] = {
            "url": TestCommandHandlers.OFF_URL,
            "api_key": None,
            "raw": False,
        }
        base.update(extra)
        return argparse.Namespace(**base)

    def test_cmd_cluster_status(self) -> None:
        """cmd_cluster_status raises ConnectError without a server."""
        with pytest.raises(httpx.ConnectError):
            cmd_cluster_status(self._make_args())

    def test_cmd_nodes_list(self) -> None:
        """cmd_nodes_list raises ConnectError without a server."""
        with pytest.raises(httpx.ConnectError):
            cmd_nodes_list(self._make_args())

    def test_cmd_nodes_drain(self) -> None:
        """cmd_nodes_drain raises ConnectError without a server."""
        with pytest.raises(httpx.ConnectError):
            cmd_nodes_drain(self._make_args(node_id="node-3"))

    def test_cmd_nodes_remove(self) -> None:
        """cmd_nodes_remove raises ConnectError without a server."""
        with pytest.raises(httpx.ConnectError):
            cmd_nodes_remove(self._make_args(node_id="node-3"))

    def test_cmd_fed_peers(self) -> None:
        """cmd_fed_peers raises ConnectError without a server."""
        with pytest.raises(httpx.ConnectError):
            cmd_fed_peers(self._make_args())

    def test_cmd_fed_status(self) -> None:
        """cmd_fed_status raises ConnectError without a server."""
        with pytest.raises(httpx.ConnectError):
            cmd_fed_status(self._make_args())

    def test_cmd_recovery_status(self) -> None:
        """cmd_recovery_status raises ConnectError without a server."""
        with pytest.raises(httpx.ConnectError):
            cmd_recovery_status(self._make_args())

    def test_cmd_recovery_history(self) -> None:
        """cmd_recovery_history raises ConnectError without a server."""
        with pytest.raises(httpx.ConnectError):
            cmd_recovery_history(self._make_args())

    def test_cmd_recovery_drill(self) -> None:
        """cmd_recovery_drill raises ConnectError without a server."""
        with pytest.raises(httpx.ConnectError):
            cmd_recovery_drill(self._make_args())

    def test_cmd_power_status(self) -> None:
        """cmd_power_status raises ConnectError without a server."""
        with pytest.raises(httpx.ConnectError):
            cmd_power_status(self._make_args())

    def test_cmd_power_autotune(self) -> None:
        """cmd_power_autotune raises ConnectError without a server."""
        with pytest.raises(httpx.ConnectError):
            cmd_power_autotune(self._make_args())

    def test_cmd_tenants_list(self) -> None:
        """cmd_tenants_list raises ConnectError without a server."""
        with pytest.raises(httpx.ConnectError):
            cmd_tenants_list(self._make_args())

    def test_cmd_tenants_get(self) -> None:
        """cmd_tenants_get raises ConnectError without a server."""
        with pytest.raises(httpx.ConnectError):
            cmd_tenants_get(self._make_args(tenant_id="tenant-1"))

    def test_cmd_regions_list(self) -> None:
        """cmd_regions_list raises ConnectError without a server."""
        with pytest.raises(httpx.ConnectError):
            cmd_regions_list(self._make_args(min_gpu_memory_gb=80.0))

    def test_cmd_deploy_list(self) -> None:
        """cmd_deploy_list raises ConnectError without a server."""
        with pytest.raises(httpx.ConnectError):
            cmd_deploy_list(self._make_args(tenant_id=None, status=None))


# ---------------------------------------------------------------------------
# Main entry point tests
# ---------------------------------------------------------------------------


class TestMainEntryPoint:
    """main() entry point error handling."""

    def test_connect_error(self, capsys) -> None:
        """main exits with code 1 on connection failure."""
        old_argv = sys.argv
        try:
            sys.argv = ["prog", "cluster", "status"]
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 1
            err = capsys.readouterr().err
            assert "Connection failed" in err
        finally:
            sys.argv = old_argv

    def test_parse_error(self) -> None:
        """main exits on parse error (missing command)."""
        old_argv = sys.argv
        try:
            sys.argv = ["prog"]
            with pytest.raises(SystemExit):
                main()
        finally:
            sys.argv = old_argv
