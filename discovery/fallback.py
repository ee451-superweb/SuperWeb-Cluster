"""Manual fallback when multicast discovery fails."""

from __future__ import annotations

from common.types import DiscoveryResult
from app.constants import MSG_MANUAL_CANCELLED, MSG_MANUAL_PROMPT
from wire.discovery import normalize_manual_address
from app.trace_utils import trace_function


@trace_function
def prompt_manual_address(default_port: int) -> DiscoveryResult:
    """Prompt for a manual host or host:port value."""

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


