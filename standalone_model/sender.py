#!/usr/bin/env python3
"""Minimal mDNS sender that looks for one service announcement."""

from __future__ import annotations

import argparse
import socket
import time

from mdns_minimal import (
    DEFAULT_SERVICE_NAME,
    MDNS_GROUP,
    MDNS_PORT,
    build_ptr_query,
    create_sender_socket,
    parse_announcement,
    resolve_local_ip,
    safe_close,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send one mDNS PTR query and wait for a reply.")
    parser.add_argument("--service", default=DEFAULT_SERVICE_NAME, help="DNS-SD service name to query.")
    parser.add_argument("--interface-ip", default="", help="Specific local interface IPv4 used for multicast sends.")
    parser.add_argument("--timeout", type=float, default=1.5, help="Seconds to wait for a reply per attempt.")
    parser.add_argument("--attempts", type=int, default=3, help="Number of query retries before giving up.")
    parser.add_argument("--interval", type=float, default=1.0, help="Delay between attempts.")
    parser.add_argument("--ttl", type=int, default=255, help="Multicast TTL.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    interface_ip = args.interface_ip or resolve_local_ip()
    query = build_ptr_query(args.service, request_unicast_response=True)

    sock = None
    try:
        sock = create_sender_socket(
            timeout=args.timeout,
            interface_ip=args.interface_ip,
            ttl=args.ttl,
        )
        local_host, local_port = sock.getsockname()
        print(f"sender socket {local_host or '0.0.0.0'}:{local_port}")
        print(f"multicast target {MDNS_GROUP}:{MDNS_PORT}")
        print(f"multicast interface {args.interface_ip or interface_ip or '(default interface)'}")

        for attempt in range(1, args.attempts + 1):
            print(f"attempt {attempt}/{args.attempts}: sending PTR query for {args.service}")
            sock.sendto(query, (MDNS_GROUP, MDNS_PORT))

            deadline = time.monotonic() + args.timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break

                sock.settimeout(remaining)
                try:
                    packet, addr = sock.recvfrom(2048)
                except socket.timeout:
                    break

                announcement = parse_announcement(packet, args.service)
                if announcement is None:
                    continue

                print(f"reply from {addr[0]}:{addr[1]}")
                print(f"instance {announcement.instance_name}")
                print(f"host {announcement.host_name}")
                print(f"service endpoint {announcement.host_ip}:{announcement.port}")
                if announcement.txt_values:
                    print(f"txt {announcement.txt_values}")
                return 0

            if attempt < args.attempts:
                time.sleep(args.interval)

        print("no matching mDNS response received")
        return 1
    except KeyboardInterrupt:
        print("stopped by user")
        return 130
    finally:
        safe_close(sock)


if __name__ == "__main__":
    raise SystemExit(main())
