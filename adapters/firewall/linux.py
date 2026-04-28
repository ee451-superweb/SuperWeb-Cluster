"""Linux firewall skeleton for the kickoff version."""

from core.types import FirewallStatus
from core.constants import MSG_NOT_IMPLEMENTED
from core.tracing import trace_function


@trace_function
def ensure_rules(discovery_port: int, is_admin_user: bool, data_plane_port: int = 0) -> FirewallStatus:
    """Placeholder Linux firewall implementation."""

    del discovery_port, is_admin_user, data_plane_port
    return FirewallStatus(
        supported=False,
        applied=False,
        needs_admin=False,
        backend="linux",
        message=MSG_NOT_IMPLEMENTED,
    )


@trace_function
def cleanup_rules(discovery_port: int, is_admin_user: bool, data_plane_port: int = 0) -> FirewallStatus:
    """Placeholder Linux firewall cleanup implementation."""

    del discovery_port, is_admin_user, data_plane_port
    return FirewallStatus(
        supported=False,
        applied=False,
        needs_admin=False,
        backend="linux",
        message=MSG_NOT_IMPLEMENTED,
    )

