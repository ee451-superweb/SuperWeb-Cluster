"""Run the high-level discovery role flow for one process.

Use this module when a process has already chosen a discovery role and needs a
single function that performs either the discover-side browse flow or the
main-node announce-side reply flow.
"""

from __future__ import annotations

from core.types import DiscoveryResult
from core.config import AppConfig
from discovery import multicast
from core.tracing import trace_function


@trace_function
def discover_peer(config: AppConfig) -> DiscoveryResult:
    """Use this when a node wants to discover the current main node.

    Args: config discovery settings and local node metadata.
    Returns: A DiscoveryResult describing the discovered peer or the failure cause.
    """

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
    """Use this when a main node wants to answer one discovery probe.

    Args: config discovery settings and the local main-node TCP endpoint info.
    Returns: A DiscoveryResult describing the peer that was answered or the failure cause.
    """

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
    """Use this when callers want role-based discovery behavior behind one function.

    Args: config whose ``role`` decides whether to discover or announce.
    Returns: The DiscoveryResult produced by the selected discovery branch.
    """

    if config.role == "announce":
        return announce_peer(config)
    return discover_peer(config)

