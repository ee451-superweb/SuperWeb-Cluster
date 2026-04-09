"""Low-level UDP multicast helpers."""

from __future__ import annotations

import socket
import sys
import time
from dataclasses import dataclass

from adapters import network
from common.types import DiscoveryResult
from config import AppConfig
from protocol import (
    describe_discovery_message,
    build_announce_message,
    build_discover_message,
    parse_announce_message,
    parse_discover_message,
)
from trace_utils import trace_function


@dataclass(slots=True)
class MulticastSocket:
    """Socket wrapper with optional multicast membership tracking."""

    sock: socket.socket
    membership: bytes | None = None

    @trace_function
    def close(self) -> None:
        """Best-effort close of the underlying socket."""

        network.drop_multicast_membership(self.sock, self.membership)
        network.safe_close(self.sock)


@trace_function
def create_sender(config: AppConfig) -> MulticastSocket:
    """Create a sender socket that can also receive a unicast announce reply."""

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
    """Create a multicast listener socket."""

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
    """Send a multicast discover packet."""

    payload = build_discover_message(node_name)
    endpoint.sock.sendto(payload, (config.multicast_group, config.udp_port))


@trace_function
def recv_announce(endpoint: MulticastSocket, config: AppConfig) -> DiscoveryResult:
    """Wait for an announce reply on a sender socket."""

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

            return DiscoveryResult(
                success=True,
                peer_address=payload.host,
                peer_port=payload.port,
                source="mdns",
                message=f"Received scheduler announcement from {payload.node_name}.",
            )
    finally:
        endpoint.sock.settimeout(original_timeout)


@trace_function
def recv_packet(endpoint: MulticastSocket, buffer_size: int) -> tuple[tuple[str, int], bytes] | None:
    """Receive one UDP packet."""

    try:
        data, addr = endpoint.sock.recvfrom(buffer_size)
    except socket.timeout:
        return None

    return addr, data


@trace_function
def recv_discover(endpoint: MulticastSocket, buffer_size: int) -> tuple[tuple[str, int], bytes] | None:
    """Wait for a main-node query packet and return the sender address."""

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
    """Send an announce packet back to the discovering peer."""

    host = network.resolve_local_ip(remote_host=target[0], remote_port=max(target[1], 1))
    payload = build_announce_message(host, config.tcp_port, node_name)
    endpoint.sock.sendto(payload, target)
    return host


@trace_function
def describe_packet(message: bytes) -> str:
    """Return a human-readable summary of a discovery packet."""

    return describe_discovery_message(message)


@trace_function
def close(endpoint: MulticastSocket | None) -> None:
    """Best-effort close helper."""

    if endpoint is None:
        return
    endpoint.close()
