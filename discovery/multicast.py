"""Create, use, and close UDP multicast sockets for discovery traffic.

Use this module when discovery code needs concrete UDP socket operations such as
sending the browse query, receiving announce packets, or joining/leaving the
multicast group.
"""

from __future__ import annotations

import logging
import socket
import sys
import time
from dataclasses import dataclass

from adapters import network
from core.constants import LOGGER_NAME
from core.types import DiscoveryResult
from core.config import AppConfig
from wire.discovery_protocol import (
    describe_discovery_message,
    build_announce_message,
    build_discover_message,
    parse_announce_message,
    parse_discover_message,
)
from core.tracing import trace_function


@dataclass(slots=True)
class MulticastSocket:
    """Bundle one UDP socket with the multicast membership bytes needed to leave it."""

    sock: socket.socket
    membership: bytes | None = None

    @trace_function
    def close(self) -> None:
        """Use this when discovery code is done with the socket wrapper.

        Args: self wrapper whose socket and multicast membership should be cleaned up.
        Returns: None after the socket is closed and membership is dropped when present.
        """

        network.drop_multicast_membership(self.sock, self.membership)
        network.safe_close(self.sock)


@trace_function
def create_sender(config: AppConfig) -> MulticastSocket:
    """Use this when a node wants to send the multicast discover probe.

    Args: config discovery settings including timeout, group, port, and TTL.
    Returns: A sender socket wrapper bound to an ephemeral UDP port for replies.
    """

    sock = network.create_udp_socket(reuse_addr=False)
    # Bind to an ephemeral port so replies come back to this specific process.
    sock.bind(("", 0))
    interface_ip = network.resolve_multicast_interface_ip(config.multicast_group, config.udp_port)
    network.configure_multicast_sender(
        sock,
        config.multicast_ttl,
        interface_ip=interface_ip,
    )
    network.set_socket_timeout(sock, config.discovery_timeout)
    return MulticastSocket(sock=sock)


@trace_function
def create_receiver(config: AppConfig) -> MulticastSocket:
    """Use this when a main node wants to listen for multicast discovery probes.

    Args: config discovery settings including group, port, and timeout.
    Returns: A receiver socket wrapper joined to the configured multicast group.
    """

    # macOS requires SO_REUSEPORT to share the standard mDNS port 5353 with
    # other listeners such as mDNSResponder and third-party apps.
    sock = network.create_udp_socket(reuse_port=sys.platform == "darwin")
    interface_ip = network.resolve_multicast_interface_ip(config.multicast_group, config.udp_port)
    # Joining the multicast group is what lets the announce role hear the shared
    # discovery probe without knowing the sender ahead of time.
    membership = network.configure_multicast_receiver(
        sock,
        group=config.multicast_group,
        port=config.udp_port,
        interface_ip=interface_ip,
    )
    network.set_socket_timeout(sock, config.discovery_timeout)
    return MulticastSocket(sock=sock, membership=membership)


@trace_function
def send_discover(endpoint: MulticastSocket, config: AppConfig, node_name: str) -> None:
    """Use this after creating a sender socket to emit one discover packet.

    Args: endpoint sender socket wrapper, config discovery settings, and node_name sender name for protocol helpers.
    Returns: None after one UDP discover packet is sent.
    """

    payload = build_discover_message(node_name)
    endpoint.sock.sendto(payload, (config.multicast_group, config.udp_port))


@trace_function
def recv_announce(endpoint: MulticastSocket, config: AppConfig) -> DiscoveryResult:
    """Use this on the discovering side while waiting for one announce reply.

    Args: endpoint sender socket wrapper and config containing timeout/buffer settings.
    Returns: A DiscoveryResult describing the discovered main node or timeout/failure.
    """

    original_timeout = endpoint.sock.gettimeout()
    deadline = None if original_timeout is None else time.monotonic() + original_timeout

    try:
        while True:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return DiscoveryResult(success=False, message="Discovery timed out.")
                endpoint.sock.settimeout(remaining)

            packet = recv_packet(endpoint, config.buffer_size)
            if packet is None:
                return DiscoveryResult(success=False, message="Discovery timed out.")

            addr, data = packet
            payload = parse_announce_message(data)
            if payload is None:
                continue
            logging.getLogger(LOGGER_NAME).info(
                "UDP multicast announce from %s: %s",
                addr,
                describe_discovery_message(data),
            )

            return DiscoveryResult(
                success=True,
                peer_address=payload.host,
                peer_port=payload.port,
                source="mdns",
                message=f"Received scheduler announcement from {payload.node_name}.",
            )
    finally:
        endpoint.sock.settimeout(original_timeout)


def recv_packet(endpoint: MulticastSocket, buffer_size: int) -> tuple[tuple[str, int], bytes] | None:
    """Use this low-level helper when any discovery flow needs one UDP datagram.

    Args: endpoint socket wrapper to read from and buffer_size maximum datagram size.
    Returns: ``(addr, data)`` for one packet, or ``None`` when the socket times out.
    """

    try:
        data, addr = endpoint.sock.recvfrom(buffer_size)
    except socket.timeout:
        return None

    return addr, data


@trace_function
def recv_discover(endpoint: MulticastSocket, buffer_size: int) -> tuple[tuple[str, int], bytes] | None:
    """Use this on the announce side while waiting for a valid discover query.

    Args: endpoint receiver socket wrapper and buffer_size maximum datagram size.
    Returns: ``(addr, message)`` for the first valid discover packet, or ``None`` on timeout.
    """

    original_timeout = endpoint.sock.gettimeout()
    deadline = None if original_timeout is None else time.monotonic() + original_timeout

    try:
        while True:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                endpoint.sock.settimeout(remaining)

            packet = recv_packet(endpoint, buffer_size)
            if packet is None:
                return None

            addr, message = packet
            if parse_discover_message(message):
                logging.getLogger(LOGGER_NAME).info(
                    "UDP multicast discover from %s: %s",
                    addr,
                    describe_discovery_message(message),
                )
                return addr, message
    finally:
        endpoint.sock.settimeout(original_timeout)


@trace_function
def send_announce(
    endpoint: MulticastSocket,
    target: tuple[str, int],
    config: AppConfig,
    node_name: str,
) -> str:
    """Use this after receiving a discover packet to reply directly to that peer.

    Args: endpoint receiver socket wrapper, target sender address, config listener settings, and node_name announced main-node name.
    Returns: The local host address embedded into the announce payload.
    """

    host = network.resolve_local_ip(remote_host=target[0], remote_port=max(target[1], 1))
    payload = build_announce_message(host, config.tcp_port, node_name)
    endpoint.sock.sendto(payload, target)
    return host


@trace_function
def describe_packet(message: bytes) -> str:
    """Use this for logs that should summarize one raw discovery datagram.

    Args: message raw UDP discovery packet bytes.
    Returns: A human-readable description of the parsed packet contents.
    """

    return describe_discovery_message(message)


@trace_function
def close(endpoint: MulticastSocket | None) -> None:
    """Use this helper when callers want a nullable-safe multicast close.

    Args: endpoint optional socket wrapper to close.
    Returns: None after the wrapper is closed when it exists.
    """

    if endpoint is None:
        return
    endpoint.close()


