"""Sprint 1 discovery flow."""

from __future__ import annotations

from common.types import DiscoveryResult
from app.config import AppConfig
from discovery import multicast
from app.trace_utils import trace_function


@trace_function
def discover_peer(config: AppConfig) -> DiscoveryResult:
    """Send discovery and wait for an announce reply."""

    try:
        # The sender binds an ephemeral local UDP port so the announce side can
        # reply directly to that source address.
        endpoint = multicast.create_sender(config)
    except OSError as exc:
        return DiscoveryResult(
            success=False,
            message=f"Unable to create discovery sender socket: {exc}.",
        )

    try:
        multicast.send_discover(endpoint, config, config.node_name)
        return multicast.recv_announce(endpoint, config)
    finally:
        multicast.close(endpoint)


@trace_function
def announce_peer(config: AppConfig) -> DiscoveryResult:
    """Wait for one main-node query and reply with main-node details."""

    try:
        # The receiver joins the multicast group and waits for a discover probe.
        endpoint = multicast.create_receiver(config)
    except OSError as exc:
        return DiscoveryResult(
            success=False,
            message=f"Unable to create discovery receiver socket: {exc}.",
        )

    try:
        discovered = multicast.recv_discover(endpoint, config.buffer_size)
        if discovered is None:
            return DiscoveryResult(success=False, message="No main-node query packet received.")

        target, _message = discovered
        local_host = multicast.send_announce(endpoint, target, config, config.node_name)
        return DiscoveryResult(
            success=True,
            peer_address=target[0],
            peer_port=target[1],
            source="mdns",
            message=f"Reported main-node availability from {local_host}:{config.tcp_port}.",
        )
    finally:
        multicast.close(endpoint)


@trace_function
def run_pairing(config: AppConfig) -> DiscoveryResult:
    """Dispatch discovery behavior based on the configured role."""

    if config.role == "announce":
        return announce_peer(config)
    return discover_peer(config)

