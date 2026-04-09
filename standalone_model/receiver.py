#!/usr/bin/env python3
"""Minimal mDNS responder for standalone testing."""

from __future__ import annotations

import argparse
import socket
import sys

from mdns_minimal import (
    DEFAULT_SERVICE_NAME,
    MDNS_GROUP,
    MDNS_PORT,
    build_announcement,
    create_listener_socket,
    drop_membership,
    make_host_name,
    make_instance_name,
    packet_requests_service,
    resolve_local_ip,
    safe_close,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Listen on mDNS and reply to one service query.")
    parser.add_argument("--name", default="standalone-node", help="Instance label used in PTR/SRV/TXT records.")
    parser.add_argument("--service", default=DEFAULT_SERVICE_NAME, help="DNS-SD service name to answer.")
    parser.add_argument("--port", type=int, default=52020, help="TCP port to advertise in the SRV record.")
    parser.add_argument("--host-ip", default="", help="IPv4 address to publish in the A record.")
    parser.add_argument("--interface-ip", default="", help="Specific local interface IPv4 for multicast membership.")
    parser.add_argument("--timeout", type=float, default=1.0, help="Socket timeout per receive attempt.")
    parser.add_argument("--once", action="store_true", help="Exit after the first successful reply.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    interface_ip = args.interface_ip or resolve_local_ip()
    host_ip = args.host_ip or interface_ip
    if host_ip.startswith("127."):
        print("warning: host IP resolved to loopback; remote peers will not be able to connect", file=sys.stderr)

    instance_name = make_instance_name(args.name, args.service)
    host_name = make_host_name(args.name)
    txt_values = [
        f"instance={args.name}",
        f"port={args.port}",
        f"host={host_ip}",
    ]

    sock = None
    membership = None

    try:
        sock, membership = create_listener_socket(timeout=args.timeout, interface_ip=args.interface_ip)
        listen_host, listen_port = sock.getsockname()
        print(f"listening on {listen_host or '0.0.0.0'}:{listen_port}")
        print(f"joined multicast group {MDNS_GROUP}:{MDNS_PORT} via {args.interface_ip or '(default interface)'}")
        print(f"advertising {instance_name} -> {host_ip}:{args.port}")

        while True:
            try:
                packet, addr = sock.recvfrom(2048)
            except socket.timeout:
                continue

            if not packet_requests_service(packet, args.service):
                continue

            print(f"received matching query from {addr[0]}:{addr[1]}")
            reply = build_announcement(
                service_name=args.service,
                instance_name=instance_name,
                host_name=host_name,
                host_ip=host_ip,
                service_port=args.port,
                txt_values=txt_values,
            )
            sock.sendto(reply, addr)
            print(f"sent unicast reply to {addr[0]}:{addr[1]}")

            if args.once:
                return 0
    except KeyboardInterrupt:
        print("stopped by user")
        return 130
    finally:
        if sock is not None:
            drop_membership(sock, membership)
        safe_close(sock)


if __name__ == "__main__":
    raise SystemExit(main())
