"""Firewall adapter entry points."""

from __future__ import annotations

from common.types import FirewallStatus, PlatformInfo
from trace_utils import trace_function


@trace_function
def ensure_rules(platform_info: PlatformInfo, discovery_port: int) -> FirewallStatus:
    """Apply kickoff firewall rules when supported."""

    # The adapter is selected purely from normalized platform detection so the
    # rest of the program does not need OS-specific branches.
    if platform_info.platform_name == "windows":
        from .windows import ensure_rules as windows_ensure_rules

        return windows_ensure_rules(discovery_port=discovery_port, is_admin_user=platform_info.is_admin)

    if platform_info.platform_name in {"linux", "wsl"}:
        from .linux import ensure_rules as linux_ensure_rules

        return linux_ensure_rules(discovery_port=discovery_port, is_admin_user=platform_info.is_admin)

    if platform_info.platform_name == "macos":
        from .macos import ensure_rules as macos_ensure_rules

        return macos_ensure_rules(discovery_port=discovery_port, is_admin_user=platform_info.is_admin)

    return FirewallStatus(
        supported=False,
        applied=False,
        needs_admin=False,
        backend="none",
        message="Firewall adapter unavailable for this platform.",
    )


@trace_function
def cleanup_rules(platform_info: PlatformInfo, discovery_port: int) -> FirewallStatus:
    """Remove kickoff firewall rules when supported."""

    if platform_info.platform_name == "windows":
        from .windows import cleanup_rules as windows_cleanup_rules

        return windows_cleanup_rules(discovery_port=discovery_port, is_admin_user=platform_info.is_admin)

    if platform_info.platform_name in {"linux", "wsl"}:
        from .linux import cleanup_rules as linux_cleanup_rules

        return linux_cleanup_rules(discovery_port=discovery_port, is_admin_user=platform_info.is_admin)

    if platform_info.platform_name == "macos":
        from .macos import cleanup_rules as macos_cleanup_rules

        return macos_cleanup_rules(discovery_port=discovery_port, is_admin_user=platform_info.is_admin)

    return FirewallStatus(
        supported=False,
        applied=False,
        needs_admin=False,
        backend="none",
        message="Firewall cleanup unavailable for this platform.",
    )
