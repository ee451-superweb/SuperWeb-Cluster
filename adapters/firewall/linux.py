"""Linux firewall skeleton for the kickoff version."""

from common.types import FirewallStatus
from constants import MSG_NOT_IMPLEMENTED
from trace_utils import trace_function


@trace_function
def ensure_rules(discovery_port: int, is_admin_user: bool) -> FirewallStatus:
    """Placeholder Linux firewall implementation."""

    del discovery_port, is_admin_user
    return FirewallStatus(
        supported=False,
        applied=False,
        needs_admin=False,
        backend="linux",
        message=MSG_NOT_IMPLEMENTED,
    )


@trace_function
def cleanup_rules(discovery_port: int, is_admin_user: bool) -> FirewallStatus:
    """Placeholder Linux firewall cleanup implementation."""

    del discovery_port, is_admin_user
    return FirewallStatus(
        supported=False,
        applied=False,
        needs_admin=False,
        backend="linux",
        message=MSG_NOT_IMPLEMENTED,
    )
