"""Windows firewall implementation for the UDP discovery rule."""

from __future__ import annotations

import subprocess

from common.types import FirewallStatus
from constants import WINDOWS_FIREWALL_RULE_NAME
from trace_utils import trace_function


def _inbound_rule_name() -> str:
    return f"{WINDOWS_FIREWALL_RULE_NAME}-Inbound"


def _outbound_rule_name() -> str:
    return f"{WINDOWS_FIREWALL_RULE_NAME}-Outbound"


def _run_firewall_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False)


@trace_function
def ensure_rules(discovery_port: int, is_admin_user: bool) -> FirewallStatus:
    """Ensure the kickoff UDP discovery rule exists on Windows."""

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
def cleanup_rules(discovery_port: int, is_admin_user: bool) -> FirewallStatus:
    """Delete the kickoff UDP discovery rule on Windows."""

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
