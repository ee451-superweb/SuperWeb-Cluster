"""Create and configure sockets used by multicast discovery traffic.

Use this module when discovery code needs low-level socket helpers for UDP
creation, multicast membership, timeout setup, local-interface selection, and
safe socket cleanup.
"""

from __future__ import annotations

import socket
import struct
import uuid

from core.tracing import trace_function


@trace_function
def create_udp_socket(*, reuse_addr: bool = True, reuse_port: bool = False) -> socket.socket:
    """Use this when discovery code needs a fresh UDP socket with reuse options.

    Args: reuse_addr whether to enable ``SO_REUSEADDR`` and reuse_port whether to request ``SO_REUSEPORT`` when supported.
    Returns: A newly created UDP socket configured with the requested reuse flags.
    """

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    if reuse_addr:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if reuse_port and hasattr(socket, "SO_REUSEPORT"):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    return sock


@trace_function
def set_socket_timeout(sock: socket.socket, timeout: float) -> None:
    """Use this when a discovery socket should enforce bounded blocking calls.

    Args: sock target socket and timeout timeout value in seconds.
    Returns: None after the timeout is applied to the socket.
    """

    sock.settimeout(timeout)


@trace_function
def configure_multicast_sender(
    sock: socket.socket,
    ttl: int,
    interface_ip: str = "",
) -> None:
    """Use this when one socket should send multicast traffic to the discovery group.

    Args: sock UDP socket to configure, ttl multicast hop limit, and interface_ip optional outbound interface IP.
    Returns: None after multicast sender options are applied.
    """

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
    """Use this when one socket should listen for discovery traffic on a multicast group.

    Args: sock UDP socket to bind, group multicast address, port UDP port, bind_host optional bind address, interface_ip optional join interface address.
    Returns: The packed membership bytes needed later to leave the multicast group.
    """

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
    """Use this during cleanup when a multicast listener should leave its joined group.

    Args: sock UDP socket and membership packed membership bytes returned by join setup.
    Returns: None after the leave attempt completes or is skipped.
    """

    if membership is None:
        return
    try:
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_DROP_MEMBERSHIP, membership)
    except OSError:
        return


@trace_function
def safe_close(sock: socket.socket | None) -> None:
    """Use this helper whenever socket cleanup should ignore close-time errors.

    Args: sock optional socket object to close.
    Returns: None after the socket is closed when present.
    """

    if sock is None:
        return
    try:
        sock.close()
    except OSError:
        return


@trace_function
def resolve_local_ip(remote_host: str = "8.8.8.8", remote_port: int = 80) -> str:
    """Use this when discovery code needs a likely outbound local IPv4 address.

    Args: remote_host and remote_port probe endpoint used only for interface selection.
    Returns: The detected local IPv4 address, or ``127.0.0.1`` as a fallback.
    """

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
    """Use this when multicast sockets need the best available local interface IP.

    Args: group multicast address and port UDP port used for the interface probe.
    Returns: A non-loopback interface IP when found, otherwise an empty string.
    """

    primary_ip = resolve_local_ip()
    if not primary_ip.startswith("127."):
        return primary_ip

    interface_ip = resolve_local_ip(remote_host=group, remote_port=max(port, 1))
    if not interface_ip.startswith("127."):
        return interface_ip

    return ""


@trace_function
def get_local_mac_address() -> str:
    """Use this when diagnostics or registration want a printable local MAC address.

    Args: selfless helper with no caller-supplied inputs.
    Returns: The local MAC address formatted as colon-separated lowercase hex octets.
    """

    node = uuid.getnode()
    octets = [(node >> shift) & 0xFF for shift in range(40, -1, -8)]
    return ":".join(f"{octet:02x}" for octet in octets)

