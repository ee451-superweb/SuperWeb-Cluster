"""Fallback helpers for manual peer entry when automatic discovery is unavailable.

Use this module after multicast discovery has failed and the user should be
allowed to type a host or ``host:port`` pair manually.
"""

from __future__ import annotations

from core.types import DiscoveryResult
from core.constants import MSG_MANUAL_CANCELLED, MSG_MANUAL_PROMPT
from wire.discovery_protocol import normalize_manual_address
from core.tracing import trace_function


@trace_function
def prompt_manual_address(default_port: int) -> DiscoveryResult:
    """Use this when discovery should fall back to plain-text manual address input.

    Args: default_port port to use when the user supplies only a host value.
    Returns: A DiscoveryResult describing the chosen manual endpoint or why input failed.
    """

    # Manual input is intentionally plain text in the kickoff version so the
    # user can recover even when automatic discovery is unavailable.
    try:
        user_input = input(MSG_MANUAL_PROMPT)
    except EOFError:
        return DiscoveryResult(success=False, source="manual", message=MSG_MANUAL_CANCELLED)

    if not user_input.strip():
        return DiscoveryResult(success=False, source="manual", message=MSG_MANUAL_CANCELLED)

    try:
        host, port = normalize_manual_address(user_input, default_port)
    except ValueError as exc:
        return DiscoveryResult(success=False, source="manual", message=str(exc))

    return DiscoveryResult(
        success=True,
        peer_address=host,
        peer_port=port,
        source="manual",
        message=f"Using manual peer {host}:{port}.",
    )


