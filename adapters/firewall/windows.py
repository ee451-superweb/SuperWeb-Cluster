"""Windows firewall implementation for the UDP discovery rule."""

from __future__ import annotations

import subprocess

from core.types import FirewallStatus
from core.constants import WINDOWS_FIREWALL_DATA_PLANE_RULE_NAME, WINDOWS_FIREWALL_RULE_NAME
from core.tracing import trace_function


def _inbound_rule_name() -> str:
    """Return the Windows firewall rule name for inbound UDP discovery traffic."""
    return f"{WINDOWS_FIREWALL_RULE_NAME}-Inbound"


def _outbound_rule_name() -> str:
    """Return the Windows firewall rule name for outbound UDP discovery traffic."""
    return f"{WINDOWS_FIREWALL_RULE_NAME}-Outbound"


def _data_plane_inbound_rule_name() -> str:
    """Return the Windows firewall rule name for inbound artifact data-plane traffic."""
    return f"{WINDOWS_FIREWALL_DATA_PLANE_RULE_NAME}-Inbound"


def _run_firewall_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run one Windows firewall command and capture its output."""
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


@trace_function
def ensure_rules(discovery_port: int, is_admin_user: bool, data_plane_port: int = 0) -> FirewallStatus:
    """Ensure the UDP discovery and (optionally) TCP data-plane rules exist on Windows."""

    if not is_admin_user:
        return FirewallStatus(
            supported=True,
            applied=False,
            needs_admin=True,
            backend="windows",
            message="Windows firewall changes require administrator privileges.",
        )

    # Windows multicast is sensitive to both firewall direction and network
    # profile selection, so we create a matched pair of UDP rules that apply
    # on any profile.
    commands = [
        [
            "netsh",
            "advfirewall",
            "firewall",
            "add",
            "rule",
            f"name={_inbound_rule_name()}",
            "dir=in",
            "action=allow",
            "protocol=UDP",
            f"localport={discovery_port}",
            "profile=any",
        ],
        [
            "netsh",
            "advfirewall",
            "firewall",
            "add",
            "rule",
            f"name={_outbound_rule_name()}",
            "dir=out",
            "action=allow",
            "protocol=UDP",
            f"remoteport={discovery_port}",
            "profile=any",
        ],
    ]
    if data_plane_port > 0:
        commands.append(
            [
                "netsh",
                "advfirewall",
                "firewall",
                "add",
                "rule",
                f"name={_data_plane_inbound_rule_name()}",
                "dir=in",
                "action=allow",
                "protocol=TCP",
                f"localport={data_plane_port}",
                "profile=any",
            ]
        )

    completions = [_run_firewall_command(command) for command in commands]
    success = all(completed.returncode == 0 for completed in completions)
    outputs = [(completed.stdout or completed.stderr).strip() for completed in completions]
    output = "; ".join(part for part in outputs if part)

    return FirewallStatus(
        supported=True,
        applied=success,
        needs_admin=False,
        backend="windows",
        message=(
            output
            or (
                "Windows inbound/outbound firewall rules ensured."
                if success
                else "Windows firewall rule creation failed."
            )
        ),
    )


@trace_function
def cleanup_rules(discovery_port: int, is_admin_user: bool, data_plane_port: int = 0) -> FirewallStatus:
    """Delete the UDP discovery and (optionally) TCP data-plane rules on Windows."""

    del discovery_port

    if not is_admin_user:
        return FirewallStatus(
            supported=True,
            applied=False,
            needs_admin=True,
            backend="windows",
            message="Windows firewall cleanup requires administrator privileges.",
        )

    commands = [
        [
            "netsh",
            "advfirewall",
            "firewall",
            "delete",
            "rule",
            f"name={_inbound_rule_name()}",
        ],
        [
            "netsh",
            "advfirewall",
            "firewall",
            "delete",
            "rule",
            f"name={_outbound_rule_name()}",
        ],
    ]
    if data_plane_port > 0:
        commands.append(
            [
                "netsh",
                "advfirewall",
                "firewall",
                "delete",
                "rule",
                f"name={_data_plane_inbound_rule_name()}",
            ]
        )

    completions = [_run_firewall_command(command) for command in commands]
    success = all(completed.returncode == 0 for completed in completions)
    outputs = [(completed.stdout or completed.stderr).strip() for completed in completions]
    output = "; ".join(part for part in outputs if part)

    return FirewallStatus(
        supported=True,
        applied=success,
        needs_admin=False,
        backend="windows",
        message=output or "Windows firewall rule cleanup completed.",
    )

