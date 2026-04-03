"""Network helpers used by multicast discovery."""

from __future__ import annotations

import socket
import struct
import uuid

from trace_utils import trace_function


@trace_function
def create_udp_socket(*, reuse_addr: bool = True, reuse_port: bool = False) -> socket.socket:
    """Create a UDP socket with common kickoff defaults."""

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    if reuse_addr:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if reuse_port and hasattr(socket, "SO_REUSEPORT"):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    return sock


@trace_function
def set_socket_timeout(sock: socket.socket, timeout: float) -> None:
    """Apply a socket timeout."""

    sock.settimeout(timeout)


@trace_function
def configure_multicast_sender(
    sock: socket.socket,
    ttl: int,
    interface_ip: str = "",
) -> None:
    """Apply sender-side multicast settings."""

    if not 0 <= ttl <= 255:
        raise ValueError("multicast TTL must be between 0 and 255")
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, struct.pack("B", ttl))
    if interface_ip:
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(interface_ip))


@trace_function
def configure_multicast_receiver(
    sock: socket.socket,
    group: str,
    port: int,
    bind_host: str = "",
    interface_ip: str = "",
) -> bytes:
    """Bind a UDP socket and join the given multicast group."""

    # Binding on the discovery port makes the process eligible to receive the
    # multicast datagrams sent to that group.
    sock.bind((bind_host, port))
    membership_interface = interface_ip or "0.0.0.0"
    membership = struct.pack(
        "4s4s",
        socket.inet_aton(group),
        socket.inet_aton(membership_interface),
    )
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, membership)
    return membership


@trace_function
def drop_multicast_membership(
    sock: socket.socket,
    membership: bytes | None,
) -> None:
    """Leave the multicast group if membership is active."""

    if membership is None:
        return
    try:
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_DROP_MEMBERSHIP, membership)
    except OSError:
        return


@trace_function
def safe_close(sock: socket.socket | None) -> None:
    """Best-effort socket close."""

    if sock is None:
        return
    try:
        sock.close()
    except OSError:
        return


@trace_function
def resolve_local_ip(remote_host: str = "8.8.8.8", remote_port: int = 80) -> str:
    """Best-effort local IP detection without sending user data."""

    probe = None
    try:
        # UDP connect is used here only to ask the OS which outbound interface
        # it would choose, which gives us a usable local IP for announce packets.
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect((remote_host, remote_port))
        return probe.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        safe_close(probe)


@trace_function
def resolve_multicast_interface_ip(group: str, port: int) -> str:
    """Pick a usable local interface IP for multicast traffic."""

    primary_ip = resolve_local_ip()
    if not primary_ip.startswith("127."):
        return primary_ip

    interface_ip = resolve_local_ip(remote_host=group, remote_port=max(port, 1))
    if not interface_ip.startswith("127."):
        return interface_ip

    return ""


@trace_function
def get_local_mac_address() -> str:
    """Return the current machine MAC address in a printable form."""

    node = uuid.getnode()
    octets = [(node >> shift) & 0xFF for shift in range(40, -1, -8)]
    return ":".join(f"{octet:02x}" for octet in octets)
